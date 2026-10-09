"""Roll forward to the latest committed checkpoint, without touching other workloads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import boto3
from botocore.config import Config as AWSConfig
from botocore.exceptions import ClientError
from common import *

BUCKET, PREFIX = CONFIG['artifact_root'][5:].split('/',1)
PREFIX += '/checkpoints/'
STOP = False

def halt(signum, frame):
    global STOP
    STOP = True

def pause(seconds):
    until = time.monotonic()+seconds
    while not STOP and time.monotonic()<until: time.sleep(min(1,until-time.monotonic()))

def client():
    return boto3.Session(profile_name=CONFIG['aws_profile']).client('s3',region_name='us-east-1',
        config=AWSConfig(max_pool_connections=24,connect_timeout=10,read_timeout=60,retries={'max_attempts':3}))

def snapshot(s3, step):
    prefix=PREFIX+f'step={step}.ckpt/'
    entries=[]
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix=prefix):
        for obj in page.get('Contents',[]):
            name=obj['Key'][len(prefix):]
            if re.fullmatch(r'__\d+_\d+\.distcp|\.metadata|meta\.pt',name):
                entries.append(dict(name=name,key=obj['Key'],bytes=obj['Size'],etag=obj['ETag']))
    return sorted(entries,key=lambda o:o['name'])

def committed(objects):
    names={o['name'] for o in objects}
    return {'.metadata','meta.pt'}<=names and len(names)>2 and all(o['bytes']>0 for o in objects)

def discover(s3):
    steps=[]
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix=PREFIX,Delimiter='/'):
        for item in page.get('CommonPrefixes',[]):
            m=re.search(r'/step=(\d+)\.ckpt/$',item['Prefix'])
            if m: steps.append(int(m[1]))
    for step in sorted(steps,reverse=True):
        objects=snapshot(s3,step)
        if committed(objects):
            atomic(ROOT/'state/discovery.json',dict(updated_at=time.time(),latest_committed=step,
                   observed_steps=sorted(steps),selection='latest complete after finishing the active evaluation'))
            return step,objects
    raise RuntimeError('No committed checkpoint found')

def download(s3, step, objects):
    dest=ROOT/'checkpoints'/f'step-{step}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DOWNLOAD.json').exists():
        saved=read(dest/'DOWNLOAD.json')
        if all((dest/o['name']).is_file() for o in objects):
            assert [{k:o[k] for k in ('name','key','bytes','etag')} for o in saved['objects']]==objects
            def unchanged(obj):
                path=dest/obj['name']
                return path.stat().st_size==obj['bytes'] and sha(path)==obj['sha256']
            with ThreadPoolExecutor(max_workers=8) as pool:
                if all(pool.map(unchanged,saved['objects'])):return dest
    required=sum(o['bytes'] for o in objects if not (dest/o['name']).exists())
    assert shutil.disk_usage(ROOT).free > required + (CONFIG['min_free_gib']<<30),'Insufficient scratch for checkpoint'
    status('downloading',active_step=step,downloaded_objects=0,total_objects=len(objects),error=None)
    def get(obj):
        if STOP: raise InterruptedError('Stopping')
        path=dest/obj['name'];receipt=path.with_name(path.name+'.receipt.json')
        head=s3.head_object(Bucket=BUCKET,Key=obj['key'],IfMatch=obj['etag'])
        identity=dict(obj,version=head.get('VersionId'))
        if receipt.exists() and path.exists():
            old=read(receipt)
            if all(old.get(k)==v for k,v in identity.items()) and path.stat().st_size==obj['bytes'] and sha(path)==old['sha256']:
                return old
        args=dict(Bucket=BUCKET,Key=obj['key'],IfMatch=obj['etag'])
        if identity['version']:args['VersionId']=identity['version']
        stream=s3.get_object(**args)['Body'];temporary=path.with_name(path.name+'.partial');h=hashlib.sha256()
        try:
            with temporary.open('wb') as handle:
                for block in stream.iter_chunks(chunk_size=8<<20):
                    if STOP:raise InterruptedError('Stopping')
                    handle.write(block);h.update(block)
        finally:stream.close()
        assert temporary.stat().st_size==obj['bytes'],'Incomplete download'
        identity['sha256']=h.hexdigest();temporary.replace(path);atomic(receipt,identity)
        return identity
    proofs=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(get,obj) for obj in objects]
        for future in as_completed(futures):
            proofs.append(future.result())
            if len(proofs)%24==0 or len(proofs)==len(objects):
                status('downloading',active_step=step,downloaded_objects=len(proofs),total_objects=len(objects))
    assert snapshot(s3,step)==objects,'Checkpoint changed while downloading'
    atomic(dest/'DOWNLOAD.json',dict(uri=CONFIG['artifact_root']+f'/checkpoints/step={step}.ckpt/',
        objects=sorted(proofs,key=lambda o:o['name']),status='SNAPSHOT_DOWNLOADED'))
    return dest

def cleanup_checkpoint(s3, step, work, checkpoint):
    """Delete only our reproducible shards after result verification and S3 identity checks."""
    assert work==BASE or work.parent==ROOT/'runs'
    assert checkpoint==BASE/'checkpoint' or checkpoint.parent==ROOT/'checkpoints'
    complete=read(work/'COMPLETE.json')
    assert complete['status']=='PASS' and complete['outputs']==2722
    assert complete['plan_sha256']==sha(work/'plan.json')
    shards=list(checkpoint.glob('*.distcp'))
    if not shards:return
    original={o['name']:o for o in read(checkpoint/'DOWNLOAD.json')['objects']}
    remote={o['name']:o for o in snapshot(s3,step)}
    for path in shards:
        expected=original[path.name]
        assert re.fullmatch(r'__\d+_\d+\.distcp',path.name)
        assert path.stat().st_size==expected['bytes']
        assert remote[path.name]['etag']==expected['etag'] and remote[path.name]['bytes']==expected['bytes']
    proof=dict(step=step,verified_outputs=2722,bytes=sum(p.stat().st_size for p in shards),
        files=[p.name for p in shards],remote_identity_verified=True,
        control_and_metadata_retained=True,status='verified_before_deletion')
    atomic(checkpoint/'SHARD_CLEANUP.json',proof)
    for path in shards:path.unlink()
    proof.update(status='deleted_after_output_and_remote_verification',completed_at=time.time())
    atomic(checkpoint/'SHARD_CLEANUP.json',proof)

def initialize_encoding():
    if (ENC/'plan.json').exists():return
    status('preparing_encoding',error=None)
    inventory=[]
    candidates=[p for model in ['Qwen3.8-27B','FLUX.2-klein-4B-vae'] for p in (MODELS/model).rglob('*') if p.is_file()]
    def asset(path):return dict(name=str(path.relative_to(MODELS)),bytes=path.stat().st_size,
                               mtime_ns=path.stat().st_mtime_ns,sha256=sha(path))
    with ThreadPoolExecutor(max_workers=8) as pool:inventory=list(pool.map(asset,sorted(candidates)))
    identity={str(p.relative_to(HANDOFF)):sha(p) for p in sorted(HANDOFF.rglob('*.py'))}
    identity.update({'runtime.json':sha(HANDOFF/'runtime.json'),'rolling_runtime.py':sha(ROOT/'source/rolling_runtime.py')})
    p=dict(schema=1,kind='native-grouped-shared-qwen-vae',identity=identity,
           config_sha256=sha(ENC/'training.json'),manifest_sha256=sha(ENC/'manifest.jsonl'),
           expected_samples=1361,source_samples=1364,model_files=inventory)
    atomic(ENC/'plan.json',p)
    repair_encoding_audit()

def repair_encoding_audit():
    # Idempotent: plan creation can be interrupted before the audit copy.
    audit=read(BASE/'audit.json');audit['plan_sha256']=sha(ENC/'plan.json')
    atomic(ENC/'audit.json',audit)

def setup_work(step):
    work=ROOT/'runs'/f'step-{step}';work.mkdir(parents=True,exist_ok=True)
    for name in ['training.json','manifest.jsonl']:
        dest=work/name
        if dest.exists():assert sha(dest)==sha(BASE/name)
        else:shutil.copyfile(BASE/name,dest)
    return work

def idle_gpus():
    return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()

def stop_own_container(stage,step,rank):
    subprocess.run(['docker','stop','--time','20',docker_name(stage,step,rank)],env=DOCKER_ENV,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)

def stage(stage_name,step,count=1,ids=None):
    gpu=stage_name in ('encode','infer')
    if gpu and not idle_gpus():raise RuntimeError('GPUs occupied by another process; waiting without stopping it')
    status(stage_name,active_step=step,workers=count,canary=bool(ids),error=None)
    workers=[]
    try:
        for rank in range(count):
            logpath=ROOT/'logs'/f'{stage_name}-{step}-g{rank}-{time.time_ns()}.log'
            log=logpath.open('w')
            proc=subprocess.Popen(worker_command(stage_name,step,rank,count,ids),stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            workers.append((proc,log,rank))
        heartbeat=0
        while any(p.poll() is None for p,_,_ in workers):
            if STOP:raise InterruptedError('Stopping')
            if any(p.poll() not in (None,0) for p,_,_ in workers):raise RuntimeError(f'{stage_name} worker failed; see this evaluation logs')
            if time.monotonic()-heartbeat>30:
                status(stage_name,active_step=step,workers=count,canary=bool(ids),
                       worker_pids=[p.pid for p,_,_ in workers])
                heartbeat=time.monotonic()
            pause(5)
        if any(p.returncode!=0 for p,_,_ in workers):raise RuntimeError(f'{stage_name} failed')
    finally:
        for p,_,rank in workers:
            if p.poll() is None:
                stop_own_container(stage_name,step,rank)
                p.terminate()
        for p,log,_ in workers:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()

def verify_features():
    rows=[json.loads(line) for line in (ENC/'manifest.jsonl').read_text().splitlines()]
    ready=[r for r in rows if r['status']=='ready'];contract=sha(ENC/'plan.json')
    def check(row):
        path=ENC/'bundles'/f"{row['request_id']}.pt";receipt=read(path.with_suffix('.json'))
        assert receipt['plan_sha256']==contract and receipt['request_id']==row['request_id']
        assert sha(path)==receipt['sha256']
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,ready))
    atomic(ENC/'COMPLETE.json',dict(status='PASS',samples=len(ready),plan_sha256=contract))

def evaluate(s3,step,objects):
    atomic(ROOT/'state/active.json',dict(step=step,selected_at=time.time()))
    # Baseline results and metadata remain available; its old shards are a recoverable cache.
    if not (ENC/'COMPLETE.json').exists():cleanup_checkpoint(s3,22000,BASE,BASE/'checkpoint')
    checkpoint=download(s3,step,objects)
    work=setup_work(step)
    stage('prepare',step)
    with ExitStack() as stack:
        for gpu in CONFIG['gpus']:
            handle=stack.enter_context((Path(CONFIG['gpu_locks'])/f'inference-gpu-{gpu}.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not idle_gpus():raise RuntimeError('Waiting for local GPUs; other tasks remain unchanged')
        ids=read(BASE/'audit.json')['canary_ids']
        if not (work/'CANARY_COMPLETE.json').exists():
            if not (ENC/'COMPLETE.json').exists():stage('encode',step,len(ids),ids)
            stage('infer',step,len(ids),ids)
            for rid in ids:
                for cfg in [2,3]:
                    path=work/'results'/f'cfg{cfg}'/f'{rid}.png';receipt=read(path.with_suffix('.json'))
                    assert receipt['plan_sha256']==sha(work/'plan.json') and sha(path)==receipt['image_sha256']
                    assert receipt['finite_latents'] and receipt['step']==step and receipt['weights']=='EMA'
            atomic(work/'CANARY_COMPLETE.json',dict(status='PASS',ids=ids,outputs=len(ids)*2,plan_sha256=sha(work/'plan.json')))
        if not (ENC/'COMPLETE.json').exists():
            stage('encode',step,len(CONFIG['gpus']))
            status('verifying_encoding',active_step=step)
            verify_features()
        if not (work/'COMPLETE.json').exists():stage('infer',step,len(CONFIG['gpus']))
    stage('verify',step)
    cleanup_checkpoint(s3,step,work,checkpoint)
    atomic(ROOT/'state/active.json',dict(step=None,last_completed=step))
    status('waiting_for_checkpoint',active_step=None,last_completed=step,error=None)

def main():
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initialize_encoding();repair_encoding_audit()
        while not STOP:
            try:
                s3=client();latest,objects=discover(s3)
                completed=[22000]+[int(p.parent.name.split('-')[1]) for p in (ROOT/'runs').glob('step-*/COMPLETE.json')
                                   if read(p).get('status')=='PASS']
                for finished in completed:
                    if finished==22000:continue
                    cleanup_checkpoint(s3,finished,ROOT/'runs'/f'step-{finished}',ROOT/'checkpoints'/f'step-{finished}')
                active=read(ROOT/'state/active.json').get('step') if (ROOT/'state/active.json').exists() else None
                if active and active not in completed:
                    step=active;objects=snapshot(s3,step)
                elif latest>max(completed):step=latest
                else:
                    status('waiting_for_checkpoint',active_step=None,last_completed=max(completed),latest_committed=latest,error=None)
                    pause(CONFIG['poll_seconds']);continue
                assert committed(objects),'Selected checkpoint is incomplete'
                status('selected_checkpoint',active_step=step,latest_committed=latest,last_completed=max(completed),error=None)
                evaluate(s3,step,objects)
            except InterruptedError:break
            except Exception as exc:
                status('retry_wait',error=str(exc),error_type=type(exc).__name__)
                if isinstance(exc,ClientError) and exc.response['Error']['Code'] in ('ExpiredToken','InvalidToken','ExpiredTokenException'):
                    try:
                        subprocess.run(['infra','auth','aws','--flavor','prod','--role','core-cn-readwrite'],
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45,check=True)
                    except (subprocess.SubprocessError,OSError):pass
                pause(CONFIG['poll_seconds'])
        status('stopped')

if __name__=='__main__':main()
