"""Incrementally archive verified evaluation outputs and scoring inputs in a dedicated S3 prefix."""
from concurrent.futures import ThreadPoolExecutor
import base64
import fcntl
import hashlib
import io
import mimetypes
import signal
import threading
import time

import boto3
from botocore.config import Config as AWSConfig
from botocore.exceptions import ClientError
from PIL import Image
from common import *

ARCHIVE = CONFIG['artifact_root'] + '/evaluations/test1364-native-grouped-rolling-20260927/'
BUCKET, PREFIX = ARCHIVE[5:].split('/', 1)
OWNER = 'large15-test1364-rolling-20260927'
DATASET = 's3://core-cn-oss.canva.com/usr/josephbai/editing-mds-filtered/craft-v1-editing-0924/test/'
STOP = False


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()


def client():
    return boto3.Session(profile_name=CONFIG['aws_profile']).client('s3', region_name='us-east-1',
        config=AWSConfig(max_pool_connections=16,connect_timeout=10,read_timeout=60,retries={'max_attempts':3}))


class Store:
    def __init__(self, s3):
        self.s3 = s3
        self.confirmed = {}
        self.lock = threading.Lock()
        self.ledger_path = ROOT/'state/s3-objects.json'
        self.ledger = read(self.ledger_path) if self.ledger_path.exists() else {}
        self.output_cache = {}

    def save(self):
        with self.lock:
            atomic(self.ledger_path, self.ledger)

    def put(self, relative, body, digest, size, content_type, mutable=False):
        assert relative and not relative.startswith('/') and '..' not in relative.split('/')
        if STOP:
            raise InterruptedError('Stopping archive service')
        if self.confirmed.get(relative) == digest:
            return self.ledger[relative]
        key = PREFIX + relative
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        old = None
        # Only our explicitly mutable indexes may be replaced. Other objects are immutable.
        if relative in self.ledger or mutable:
            try:
                old = self.s3.head_object(Bucket=BUCKET,Key=key,ChecksumMode='ENABLED')
            except ClientError as e:
                if e.response['Error']['Code'] not in ('404','NoSuchKey','NotFound'):
                    raise
        def matches(head):
            return (head['ContentLength']==size and head.get('ChecksumSHA256')==checksum
                and head.get('Metadata',{}).get('sha256')==digest
                and head.get('Metadata',{}).get('evaluation')==OWNER)
        if old and not matches(old):
            assert mutable, f'S3 object differs; refusing overwrite: {relative}'
            assert old.get('Metadata',{}).get('evaluation')==OWNER, 'S3 index has another owner'
        if not old or not matches(old):
            args=dict(Bucket=BUCKET,Key=key,Body=body,ContentLength=size,ContentType=content_type,
                ChecksumSHA256=checksum,Metadata={'sha256':digest,'evaluation':OWNER})
            args.update({'IfMatch':old['ETag']} if old else {'IfNoneMatch':'*'})
            try:
                self.s3.put_object(**args)
            except ClientError as e:
                if e.response['Error']['Code'] not in ('PreconditionFailed','412'):
                    raise
            old=self.s3.head_object(Bucket=BUCKET,Key=key,ChecksumMode='ENABLED')
            assert matches(old), f'S3 checksum/content conflict: {relative}'
        proof=dict(sha256=digest,bytes=size,etag=old['ETag'],version_id=old.get('VersionId'),
                   checksum_sha256=old['ChecksumSHA256'])
        with self.lock:
            self.ledger[relative]=proof
            self.confirmed[relative]=digest
        return proof

    def data(self, relative, value, content_type='application/json', mutable=False):
        payload=value if isinstance(value,bytes) else encoded(value)
        return self.put(relative,io.BytesIO(payload),hashlib.sha256(payload).hexdigest(),
                        len(payload),content_type,mutable)

    def file(self, relative, path, expected=None, content_type=None):
        digest=sha(path)
        if expected:
            assert digest==expected, f'Local hash mismatch: {path}'
        with path.open('rb') as handle:
            return self.put(relative,handle,digest,path.stat().st_size,
                            content_type or mimetypes.guess_type(path.name)[0] or 'application/octet-stream')


def validate_output(work, step, cfg, row, plan_digest):
    rid=row['request_id'];path=work/'results'/f'cfg{cfg}'/f'{rid}.png'
    proof=read(path.with_suffix('.json'))
    assert proof['request_id']==rid and proof['plan_sha256']==plan_digest
    assert proof['step']==step and proof['weights']=='EMA' and proof['cfg']==cfg and proof['steps']==50
    assert proof['seed']==row['generation_seed'] and proof['finite_latents'] is True
    assert proof['image_sha256']==sha(path)
    with Image.open(path) as image:
        image.load()
        assert image.size==(row['output_width'],row['output_height'])
    return path,proof


def dataset_record(row):
    rid=row['request_id'];prefix=ARCHIVE+f'dataset/{rid}/'
    try:
        instructions=json.loads(row['instructions'])
    except (ValueError,TypeError):
        instructions=row['instructions']
    return dict(request_id=rid,sample_id=row['original_mds_sample_id'],ordinal=row['ordinal'],
        source=row['source'],source_index=row['source_index'],status=row['status'],
        instructions_raw=row['instructions'],instructions=instructions,num_inputs=row['num_inputs'],
        input_images=[dict(index=i,uri=prefix+f'reference-{i:02}.image',sha256=digest)
                      for i,digest in enumerate(row['reference_sha256'])],
        target_image=dict(uri=prefix+'target.image',sha256=row['target_sha256']),
        annotation_uri=prefix+'annotation.json',annotation_sha256=row['annotation_sha256'],
        model_prompts_uri=prefix+'model-prompts.json' if row['status']=='ready' else None,
        bbox_uri=prefix+'bbox.json',generation_seed=row['generation_seed'],
        rejection=row.get('drop_counters') if row['status']!='ready' else None)


def output_manifest(row, step, cfg, proof, plan):
    rid=row['request_id'];record=dataset_record(row)
    record.update(checkpoint_step=step,checkpoint_uri=plan['checkpoint_uri'],weights='EMA',cfg=cfg,
        sampling_steps=50,timeshift=plan['timeshift'],seed=row['generation_seed'],
        prediction_uri=ARCHIVE+f'step-{step}/results/cfg{cfg}/{rid}.png',
        prediction_sha256=proof['image_sha256'],
        receipt_uri=ARCHIVE+f'step-{step}/results/cfg{cfg}/{rid}.json',
        plan_uri=ARCHIVE+f'step-{step}/plan.json',plan_sha256=proof['plan_sha256'],
        width=row['output_width'],height=row['output_height'],variant='v7+v4-grouped',
        target_pixels_used=False,annotation_conditioned=True)
    return record


def sync_outputs(store, work, step, rows, progress):
    plan=read(work/'plan.json');contract=sha(work/'plan.json')
    assert plan['step']==step and plan['weights']=='EMA' and plan['steps']==50 and plan['cfg']==[2,3]
    store.file(f'step-{step}/plan.json',work/'plan.json',contract)
    store.file(f'step-{step}/training.json',work/'training.json',plan['config_sha256'])
    expected={r['request_id']:r for r in rows if r['status']=='ready'}
    receipts=[(cfg,p) for cfg in [2,3] for p in sorted((work/'results'/f'cfg{cfg}').glob('*.json'))]
    records=[]
    def upload(item):
        cfg,receipt=item;rid=receipt.stem
        assert rid in expected, 'Unexpected output ID'
        row=expected[rid];path=receipt.with_suffix('.png')
        stamp=(receipt.stat().st_mtime_ns,path.stat().st_mtime_ns,path.stat().st_size,contract)
        cache_key=(step,cfg,rid)
        if store.output_cache.get(cache_key,{}).get('stamp')==stamp:
            return store.output_cache[cache_key]['record']
        path,proof=validate_output(work,step,cfg,row,contract)
        prefix=f'step-{step}/results/cfg{cfg}/{rid}'
        store.file(prefix+'.png',path,proof['image_sha256'],'image/png')
        store.file(prefix+'.json',receipt)
        record=output_manifest(row,step,cfg,proof,plan)
        store.output_cache[cache_key]=dict(stamp=stamp,record=record)
        return record
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(upload,receipts):
            records.append(record)
            if len(records)%100==0:
                progress(step=step,uploaded=len(records),available=len(receipts))
    store.save()
    return records,plan


def sync_dataset(store, rows, progress):
    jobs=[]
    for row in rows:
        rid=row['request_id'];source=BASE/'inputs'/rid;prefix=f'dataset/{rid}/'
        images=[(f'reference-{i:02}.image',digest) for i,digest in enumerate(row['reference_sha256'])]
        images.append(('target.image',row['target_sha256']))
        for name,digest in images:
            with Image.open(source/name) as im:
                content_type=Image.MIME[im.format]
            jobs.append((prefix+name,source/name,digest,content_type))
        jobs.append((prefix+'annotation.json',source/'annotation.json',row['annotation_sha256'],'application/json'))
        if row['status']=='ready':
            prompts=read(source/'model-prompts.json')
            for branch,field in [('positive','positive_prompt_sha256'),('weak','weak_prompt_sha256')]:
                assert hashlib.sha256(prompts[branch].encode()).hexdigest()==row[field], 'Model prompt changed'
            jobs.append((prefix+'model-prompts.json',source/'model-prompts.json',None,'application/json'))
        jobs.append((prefix+'bbox.json',ROOT/'web/assets'/rid/'bbox.json',None,'application/json'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(lambda args:store.file(*args),jobs),1):
            if i%200==0:
                progress(dataset_uploaded=i,dataset_objects=len(jobs))
                store.save()
    manifest=jsonl([dataset_record(r) for r in rows])
    proof=store.data('dataset/manifest.jsonl',manifest,'application/x-ndjson')
    store.file('dataset/native-manifest.jsonl',BASE/'manifest.jsonl')
    for name in ['data-freeze.json','input-audit.json','audit.json']:
        if (BASE/name).exists():store.file('dataset/'+name,BASE/name)
    summary=dict(status='PASS',source_samples=1364,admitted_samples=1361,rejected_samples=3,
        manifest_sha256=proof['sha256'],manifest_uri=ARCHIVE+'dataset/manifest.jsonl',objects=len(jobs))
    store.data('dataset/COMPLETE.json',summary)
    atomic(ROOT/'state/s3-dataset.json',summary)
    store.save()
    return summary


def finalize_step(store, work, step, records, plan):
    records=sorted(records,key=lambda r:(r['cfg'],r['ordinal']))
    payload=jsonl(records)
    digest=hashlib.sha256(payload).hexdigest()
    snapshot=f'step-{step}/manifests/{digest}.jsonl'
    store.data(snapshot,payload,'application/x-ndjson')
    manifest=store.data(f'step-{step}/manifest.jsonl',payload,'application/x-ndjson',mutable=True)
    # Completion can arrive after this cycle's snapshot; finish the archive on the next cycle.
    complete=(work/'COMPLETE.json').exists() and len(records)==2722
    summary=dict(step=step,outputs=len(records),expected_outputs=2722,complete=False,
        manifest_uri=ARCHIVE+f'step-{step}/manifest.jsonl',manifest_sha256=manifest['sha256'])
    if complete:
        proof=read(work/'COMPLETE.json')
        assert proof['status']=='PASS' and proof['outputs']==2722 and len(records)==2722
        assert proof['plan_sha256']==sha(work/'plan.json')
        assert len({(r['cfg'],r['request_id']) for r in records})==2722
        store.file(f'step-{step}/native-COMPLETE.json',work/'COMPLETE.json')
        for path in sorted((work/'evidence').glob('*.json')):
            store.file(f'step-{step}/evidence/{path.name}',path)
        summary['complete']=True
        store.data(f'step-{step}/COMPLETE.json',dict(status='PASS',**summary,
            source_samples=1364,admitted_samples=1361,rejected_samples=3,
            dataset_manifest_uri=ARCHIVE+'dataset/manifest.jsonl',plan_sha256=proof['plan_sha256'],
            verification='Local receipts, PNG SHA/dimensions and S3 SHA256 checksum/length verified.'))
    # Keep a convenient current alias, while scoring indexes point to a permanent snapshot.
    summary['current_manifest_uri']=summary['manifest_uri']
    summary['manifest_uri']=ARCHIVE+snapshot
    return summary


def publish_state(stage, **fields):
    old=read(ROOT/'state/s3-status.json') if (ROOT/'state/s3-status.json').exists() else {}
    value=dict(old,stage=stage,archive_uri=ARCHIVE,updated_at=time.time(),**fields)
    atomic(ROOT/'state/s3-status.json',value)
    atomic(ROOT/'web/archive.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def pause(seconds):
    until=time.monotonic()+seconds
    while not STOP and time.monotonic()<until:
        time.sleep(min(1,max(0,until-time.monotonic())))


def halt(signum, frame):
    global STOP
    STOP=True


def main():
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    with (ROOT/'s3-archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[json.loads(line) for line in (BASE/'manifest.jsonl').read_text().splitlines()]
        assert len(rows)==1364
        store=Store(client());dataset_ready=False;provenance_ready=False
        while not STOP:
            try:
                publish_state('syncing_outputs',error=None)
                if not provenance_ready:
                    store.data('evaluation.json',dict(schema=1,evaluation=OWNER,run='aqucja55',dataset_uri=DATASET,
                        source_samples=1364,admitted_samples=1361,rejected_samples=3,
                        weights='EMA',cfg=[2,3],steps=50,variant='v7+v4-grouped',text_cap=11776,
                        annotation_conditioned=True,target_pixels_used=False,
                        reference_order='input_images[index] matches frame_<index>',
                        completion='Use step-N/COMPLETE.json to select fully archived checkpoints.'))
                    for path in sorted(HANDOFF.rglob('*')):
                        if path.is_file() and path.suffix in ('.py','.json'):
                            store.file('provenance/handoff/'+str(path.relative_to(HANDOFF)),path)
                    store.file('provenance/rolling_runtime.py',ROOT/'source/rolling_runtime.py')
                    store.file('README.md',ROOT/'source/archive-format.md')
                    provenance_ready=True
                works=sorted([p for p in (ROOT/'runs').glob('step-*') if (p/'plan.json').exists()],
                             key=lambda p:int(p.name.split('-')[1]),reverse=True)+[BASE]
                synced=[]
                for work in works:
                    step=22000 if work==BASE else int(work.name.split('-')[1])
                    progress=lambda **fields:publish_state('syncing_outputs',**fields,error=None)
                    records,plan=sync_outputs(store,work,step,rows,progress)
                    synced.append((work,step,records,plan))
                    publish_state('syncing_outputs',step=step,uploaded=len(records),available=len(records),error=None)
                if not dataset_ready:
                    publish_state('syncing_dataset',error=None)
                    sync_dataset(store,rows,lambda **fields:publish_state('syncing_dataset',**fields,error=None))
                    dataset_ready=True
                summaries=[finalize_step(store,*entry) for entry in synced]
                index=dict(schema=1,evaluation=OWNER,dataset_manifest_uri=ARCHIVE+'dataset/manifest.jsonl',
                           checkpoints=summaries)
                store.data('index.json',index,mutable=True)
                store.save()
                dataset_objects=read(ROOT/'state/s3-dataset.json')['objects']
                publish_state('up_to_date',checkpoints=summaries,dataset_complete=True,
                              dataset_uploaded=dataset_objects,dataset_objects=dataset_objects,error=None)
                pause(30)
            except InterruptedError:
                break
            except Exception as exc:
                store.save()
                publish_state('retry_wait',error=str(exc),error_type=type(exc).__name__)
                if isinstance(exc,ClientError) and exc.response['Error']['Code'] in ('ExpiredToken','InvalidToken','ExpiredTokenException'):
                    try:
                        subprocess.run(['infra','auth','aws','--flavor','prod','--role','core-cn-readwrite'],
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45,check=True)
                        store.s3=client()
                    except (subprocess.SubprocessError,OSError):
                        pass
                pause(30)
        store.save()
        publish_state('stopped')


if __name__=='__main__':
    main()
