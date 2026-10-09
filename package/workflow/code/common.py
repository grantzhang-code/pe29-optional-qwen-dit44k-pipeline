"""Bounded S3 work queue on the user's held H100 nodes; never releases holders."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import base64
import errno
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import traceback
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/ephemeral/pcb-v2-9c4d-editor-runtime')
CFG=json.loads((ROOT/'campaign.json').read_bytes())
if CFG.get('run_configured') is not True:
    raise ValueError('Run configure_pipeline.py before inference')
from checkpoint_sources import cache_path, load_sources
CHECKPOINT=cache_path('dit44_ema')
if CFG['checkpoint_uri'] != load_sources()['dit44_ema']:
    raise ValueError('DiT checkpoint source differs from the run configuration')
if hashlib.sha256((ROOT/'checkpoint-sources.json').read_bytes()).hexdigest() != CFG['checkpoint_sources_sha256']:
    raise ValueError('Checkpoint sources changed after run configuration')
CONFIG_SHA=hashlib.sha256((ROOT/'campaign.json').read_bytes()).hexdigest()
OUT=urlsplit(CFG['publication_prefix'])
S3=boto3.Session(profile_name='core-cn-s3-readwrite',region_name='us-east-1').client('s3',config=Config(max_pool_connections=24,retries={'mode':'standard','max_attempts':8}))
COHORTS={c['arm']:c for c in CFG['cohorts']}
HOST=socket.gethostname()
RUNNER='/app/pcb_edit/run_single_protocol.py'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines() if x]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
# Caption dictionaries must retain schema order all the way to the encoder.
def enc(v):return (json.dumps(v,ensure_ascii=False)+'\n').encode()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_bytes(enc(v));t.replace(p)
def get(name):
    try:return json.loads(S3.get_object(Bucket=OUT.netloc,Key=OUT.path.lstrip('/')+name)['Body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] in ('NoSuchKey','404'):return None
        raise
def put(name,v,**kw):return S3.put_object(Bucket=OUT.netloc,Key=OUT.path.lstrip('/')+name,Body=enc(v),ContentType='application/json',**kw)
def upload(path,name):
    digest=sha(path);checksum=base64.b64encode(bytes.fromhex(digest)).decode()
    with Path(path).open('rb') as f:
        result=S3.put_object(Bucket=OUT.netloc,Key=OUT.path.lstrip('/')+name,Body=f,ChecksumSHA256=checksum,Metadata={'sha256':digest},ContentType='image/png' if str(path).endswith('.png') else 'application/json')
    assert result['ChecksumSHA256']==checksum
    return dict(uri=CFG['publication_prefix']+name,bytes=Path(path).stat().st_size,sha256=digest)
def download(uri,dst,expected):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and sha(dst)==expected:return
    u=urlsplit(uri);tmp=dst.with_suffix('.download')
    S3.download_file(u.netloc,u.path.lstrip('/'),str(tmp));assert sha(tmp)==expected,uri;tmp.replace(dst)
def state(stage,**kw):
    value=dict(stage=stage,host=HOST,config_sha256=CONFIG_SHA,updated_utc=now(),**kw)
    save(ROOT/'status.json',value);put('execution/nodes/'+HOST+'.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)

def alias(name,target):
    p=Path(name);target=Path(target)
    if p.is_symlink():
        if p.resolve()==target.resolve():return
        p.unlink()
    elif p.exists():
        backup=p.with_name(p.name+'.before-pcb44')
        if backup.exists():raise RuntimeError(f'Unexpected existing alias and backup: {p}')
        try:p.rename(backup)
        except OSError as e:
            if e.errno!=errno.EXDEV:raise
            shutil.copytree(p,backup,symlinks=True);shutil.rmtree(p)
    p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target,target_is_directory=True)

def bind(work,control):
    rootfs=BASE/'image-rootfs'
    # The frozen runner records paths relative to a physical /app directory.
    # Its resolved extension paths cannot use a symlink as the application root.
    app=Path('/app')
    if app.is_symlink():app.unlink()
    elif app.exists():
        if (app/'.pcb44-runtime').exists():shutil.rmtree(app)
        else:
            backup=Path('/app.before-pcb44')
            assert not backup.exists(), 'Unexpected application directory'
            try:app.rename(backup)
            except OSError as e:
                if e.errno!=errno.EXDEV:raise
                shutil.copytree(app,backup,symlinks=True);shutil.rmtree(app)
    shutil.copytree(ROOT/'runtime',app,symlinks=True)
    (app/'.pcb44-runtime').touch()
    overlay={
      'tayo_protocol/code/handoff/runtime_helpers.py':'opt/inference/runtime_helpers.py',
      'tayo_protocol/code/handoff/mariana_transformer_engine.py':'opt/pixelcraft/mariana/integrations/transformer_engine.py',
      'tayo_protocol/code/handoff/magi_calc_meta_h200.py':'opt/magi-attention-src/magi_attention/meta/collection/calc_meta.py',
      'training_reader/craft_v1_edit_mds.py':'opt/pixelcraft/tasks/pixelcraft/src/data/dataset/craft_v1_edit_mds.py',
      'training_reader/clean_caption.py':'opt/pixelcraft/tasks/pixelcraft/src/data/dataset/craft_v1_edit_clean_caption.py',
      'tayo_protocol/code/handoff/training-source/mariana/data/varlen_batch.py':'opt/pixelcraft/mariana/data/varlen_batch.py',
      'tayo_protocol/code/handoff/training-source/tasks/pixelcraft/src/models/layers/reference_kv.py':'opt/pixelcraft/tasks/pixelcraft/src/models/layers/reference_kv.py',
      'tayo_protocol/code/handoff/training-source/tasks/pixelcraft/src/native_task.py':'opt/pixelcraft/tasks/pixelcraft/src/native_task.py'}
    for src,dst in overlay.items():
        d=rootfs/dst;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'runtime'/src,d)
    sites=rootfs/'opt/runtime/lib/python3.12/site-packages'
    for n in ['_brotli.cpython-312-x86_64-linux-gnu.so','zstd.cpython-312-x86_64-linux-gnu.so']:shutil.copy2(ROOT/'native-binaries'/n,sites/n)
    shutil.copy2(ROOT/'native-binaries/cramjam.cpython-312-x86_64-linux-gnu.so',ROOT/'runtime/vendor/cramjam/cramjam.cpython-312-x86_64-linux-gnu.so')
    filtered=BASE/'host-site-filtered';filtered.mkdir(exist_ok=True)
    for p in Path('/usr/local/lib/python3.12/dist-packages').iterdir():
        if p.name.lower().replace('-','_').startswith('flash_attn') or p.suffix=='.pth':continue
        d=filtered/p.name
        if not d.exists() and not d.is_symlink():d.symlink_to(p,target_is_directory=p.is_dir())
    for n in ['tmp','magi','quack','triton','inductor','cuda','hf','torch']:(BASE/'cache'/n).mkdir(parents=True,exist_ok=True)
    (work/'inputs').mkdir(exist_ok=True);(ROOT/'source-cache').mkdir(exist_ok=True)
    for name,target in {'/control':control,'/pe-caption':control,'/work':work,'/inputs':work/'inputs','/source-cache':ROOT/'source-cache','/checkpoint':CHECKPOINT,'/models':BASE/'models','/cache':BASE/'cache','/opt/runtime':rootfs/'opt/runtime','/opt/pixelcraft':rootfs/'opt/pixelcraft','/opt/inference':rootfs/'opt/inference','/opt/magi-attention-src':rootfs/'opt/magi-attention-src'}.items():alias(name,target)

def native(args,log,gpu=None):
    shell='source '+shlex.quote(str(ROOT/'code/editor_env.sh'))+'\ncd /opt/pixelcraft\nexec /opt/runtime/bin/python -S -u '+shlex.join(args)
    env=dict(os.environ)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with Path(log).open('a') as stream:subprocess.run(['bash','-c',shell],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def prepare_sources(selected):
    cache=ROOT/'source-cache';cache.mkdir(exist_ok=True)
    candidates=list(Path('/mnt/ephemeral').glob('pcb-v2-9c4d-*/data/source'))+list(Path('/mnt/ephemeral').glob('pcb-*/workflow/source-cache'))
    unique={Path(urlsplit(r['source_s3_uri']).path).name:r for r in selected}
    def one(item):
        name,row=item;dst=cache/name;expected=row['selection']['source_sha256']
        if not dst.exists():
            for parent in candidates:
                old=parent/name
                if old.is_file() and sha(old)==expected:dst.symlink_to(old);break
        download(row['source_s3_uri'],dst,expected)
    with ThreadPoolExecutor(8) as pool:list(pool.map(one,unique.items()))



def gpu_groups(values,index,cost):
    groups=[[] for _ in range(8)]
    remaining=list(values)
    slow=set(CFG.get('slow_gpus',{}).get(str(index),[]))
    for gpu in sorted(slow):
        if remaining:
            row=min(remaining,key=cost);groups[gpu].append(row);remaining.remove(row)
    healthy=[g for g in range(8) if g not in slow]
    loads={g:0 for g in healthy}
    for row in sorted(remaining,key=cost,reverse=True):
        gpu=min(healthy,key=lambda g:(loads[g],len(groups[g]),g));groups[gpu].append(row);loads[gpu]+=max(1,cost(row))
    assert sum(map(len,groups))==len(values)
    return groups
