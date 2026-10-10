"""Independent, resumable PE checkpoint partition on one held H100 node."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
import argparse, base64, fcntl, hashlib, json, os, shlex, shutil, subprocess, sys, time, traceback
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package'; WORKFLOW=PACKAGE/'workflow'
CAMPAIGN=json.loads((ROOT/'campaign.json').read_text())
BASE=Path('/mnt/ephemeral/pcb-v2-9c4d-editor-runtime')
S3=boto3.Session(profile_name='core-cn-s3-readwrite').client('s3',config=Config(max_pool_connections=24,retries={'mode':'standard','max_attempts':8}))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def jsonl(p,values):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
def get(uri):
    u=urlsplit(uri)
    try:return json.loads(S3.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] in ('NoSuchKey','404'):return None
        raise
def download(uri,p,expected):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and sha(p)==expected:return
    u=urlsplit(uri);tmp=p.with_suffix('.download');S3.download_file(u.netloc,u.path.lstrip('/'),str(tmp))
    assert sha(tmp)==expected,uri;tmp.replace(p)
def upload(p,uri):
    p=Path(p);h=sha(p);u=urlsplit(uri);checksum=base64.b64encode(bytes.fromhex(h)).decode()
    with p.open('rb') as f:
        result=S3.put_object(Bucket=u.netloc,Key=u.path.lstrip('/'),Body=f,ChecksumSHA256=checksum,Metadata={'sha256':h})
    assert result['ChecksumSHA256']==checksum
    return dict(uri=uri,sha256=h,bytes=p.stat().st_size)
def wait_file(p):
    while not Path(p).exists():time.sleep(15)
def wait_json(uri):
    while True:
        value=get(uri)
        if value:return value
        time.sleep(15)
def command(args,log,env=None):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('a') as f:
        subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def stage_model(step,arm):
    cp=arm['checkpoint'];prefix=cp['model_uri'];model=ROOT/f'models/pe-{step}'
    model.mkdir(parents=True,exist_ok=True)
    download(prefix+'local_export_manifest.json',model/'local_export_manifest.json',cp['export_manifest_sha256'])
    manifest=read(model/'local_export_manifest.json')
    assert manifest['source_mcore_checkpoint']==dict(uri=cp['checkpoint_uri'],step=step,manifest_sha256=cp['checkpoint_manifest_sha256'])
    local=Path('/mnt/ephemeral/pe39c59-c3000-qwenpe-caption-20261009/models/pe-3000')
    def one(row):
        p=model/row['name'];old=local/row['name']
        if not p.exists() and old.is_file() and sha(old)==row['sha256']:p.symlink_to(old)
        download(prefix+row['name'],p,row['sha256'])
    with ThreadPoolExecutor(6) as pool:list(pool.map(one,manifest['files']))
    return model,{'export_manifest_sha256':cp['export_manifest_sha256']}

def stage_sources(selected):
    cache=WORKFLOW/'source-cache';cache.mkdir(exist_ok=True)
    candidates=list(Path('/mnt/ephemeral').glob('pe39c59-*/package/workflow/source-cache'))+list(Path('/mnt/ephemeral').glob('pe29-*/package/workflow/source-cache'))+list(Path('/mnt/ephemeral').glob('pcb-v2-9c4d-*/data/source'))+list(Path('/mnt/ephemeral').glob('pcb-*/workflow/source-cache'))
    def one(row):
        name=Path(urlsplit(row['source_s3_uri']).path).name;p=cache/name;expected=row['selection']['source_sha256']
        if not p.exists():
            old=next((d/name for d in candidates if (d/name).is_file() and sha(d/name)==expected),None)
            if old:p.symlink_to(old)
        download(row['source_s3_uri'],p,expected)
    with ThreadPoolExecutor(8) as pool:list(pool.map(one,{r["source_s3_uri"]:r for r in selected}.values()))

def caption(work,selected,model,ready,step,arm,canary=False):
    from adaptive_caption import run
    return run(sys.modules[__name__],work,selected,model,ready,step,arm,GPU_IDS)

def stage_dit(work):
    prefix=CAMPAIGN['shared_assets_prefix']+'models/'
    ready=wait_json(prefix+'DIT_READY.json')
    assert ready['checkpoint_sha256']==CAMPAIGN['dit_checkpoint_sha256']
    cached=Path(CAMPAIGN['dit_cache_root']);assert cached.is_dir()
    assert sha(cached/'CHECKPOINT.json')==ready['checkpoint_sha256']
    ckpt=ROOT/'models/dit-ema';ckpt.parent.mkdir(parents=True,exist_ok=True)
    if not ckpt.exists():ckpt.symlink_to(cached,target_is_directory=True)
    assert ckpt.resolve()==cached.resolve(), 'Unexpected DiT cache target'
    proof=read(ckpt/'CHECKPOINT.json');assert proof['keys']==463 and proof['step']==6000 and proof['weights']=='EMA' and proof['source_uri']==CAMPAIGN['arms']['3000']['checkpoint']['dit_checkpoint_uri']
    def one(row):assert sha(ckpt/row['name'])==row['sha256'],row['name']
    with ThreadPoolExecutor(12) as pool:list(pool.map(one,proof['files']))
    contract=get(prefix+'expected-resume-contract.json');assert contract
    save(work/'expected-resume-contract.json',contract)
    (work/'checkpoints').mkdir(exist_ok=True)
    p=work/'checkpoints/step6000'
    if not p.exists():p.symlink_to(ckpt,target_is_directory=True)
    shutil.copy2(ROOT/'runner/model-inventory.json',work/'model-inventory.json')

def bind_runtime():
    rootfs=BASE/'image-rootfs'
    for name in ['runtime','magi-attention-src']:
        target=rootfs/'opt'/name;p=Path('/opt')/name
        if not p.exists():p.symlink_to(target,target_is_directory=True)
    filtered=BASE/'host-site-filtered';filtered.mkdir(exist_ok=True)
    for p in Path('/usr/local/lib/python3.12/dist-packages').iterdir():
        if p.name.lower().replace('-','_').startswith('flash_attn') or p.suffix=='.pth':continue
        d=filtered/p.name
        if not d.exists() and not d.is_symlink():d.symlink_to(p,target_is_directory=p.is_dir())
    for name in ['tmp','magi','quack','triton','inductor','cuda','hf','torch']:(BASE/'cache'/name).mkdir(parents=True,exist_ok=True)
    # Retain the verified H100 FA4 metadata adapter and native extensions.
    shutil.copy2(WORKFLOW/'runtime/tayo_protocol/code/handoff/magi_calc_meta_h200.py',rootfs/'opt/magi-attention-src/magi_attention/meta/collection/calc_meta.py')
    sites=rootfs/'opt/runtime/lib/python3.12/site-packages'
    for name in ['_brotli.cpython-312-x86_64-linux-gnu.so','zstd.cpython-312-x86_64-linux-gnu.so']:shutil.copy2(WORKFLOW/'native-binaries'/name,sites/name)
    shutil.copy2(WORKFLOW/'native-binaries/cramjam.cpython-312-x86_64-linux-gnu.so',WORKFLOW/'runtime/vendor/cramjam/cramjam.cpython-312-x86_64-linux-gnu.so')

def native(work,mode,gpu=0,shards=1,ids=None,script='native_runner.py',device=None):
    args=[str(ROOT/'runner'/script),mode,'--root',str(work),'--models',str(BASE/'models'),'--step','6000','--shard',str(gpu),'--shards',str(shards)]
    if ids:args+=['--ids',','.join(ids)]
    extra=[ROOT/'runner',ROOT/'native',ROOT/'native/tasks/pixelcraft',WORKFLOW/'runtime/vendor']
    shell='source '+shlex.quote(str(WORKFLOW/'code/editor_env.sh'))+'\nexport PYTHONPATH='+shlex.quote(':'.join(map(str,extra)))+':"$PYTHONPATH"\nexec /opt/runtime/bin/python -S -u '+shlex.join(args)
    command(['bash','-c',shell],work/'logs'/f'{mode}-{gpu}.log',{**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu if device is None else device)})

def publish(work,prefix):
    records=[]
    for r in rows(work/'inputs/manifest.jsonl'):
        p=work/'runs/step6000/images'/f"{r['request_id']}.png"
        if not p.exists():continue
        asset=upload(p,prefix+'editor/images/'+p.name)
        receipt=upload(p.with_suffix('.json'),prefix+'editor/receipts/'+p.with_suffix('.json').name)
        records.append({**r,'output_s3_uri':asset['uri'],'output_sha256':asset['sha256'],'receipt_s3_uri':receipt['uri']})
    jsonl(work/'score-index.jsonl',records)
    for rel in ['caption/predictions.jsonl','caption/greedy-predictions.jsonl','caption/retry-predictions.jsonl','caption/excluded.jsonl','inputs/manifest.jsonl','audit.json','score-index.jsonl','runs/step6000/COMPLETE.json','expected-resume-contract.json']:
        p=work/rel
        if p.exists():upload(p,prefix+rel)
    for folder in ['audits','evidence','runs/step6000/evidence']:
        for p in (work/folder).glob('*.json'):upload(p,prefix+folder+'/'+p.name)
    proof=dict(status='PASS',images=len(records),caption_count=len(rows(work/'caption/predictions.jsonl')),excluded=len(rows(work/'caption/excluded.jsonl')),native_rejected=read(work/'audit.json')['rejected'],index_sha256=sha(work/'score-index.jsonl'))
    save(work/'COMPLETE.json',proof);upload(work/'COMPLETE.json',prefix+'COMPLETE.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',type=int,required=True);ap.add_argument('--canary',action='store_true');ap.add_argument('--caption-only',action='store_true');a=ap.parse_args()
    step,arm=next((int(s),v) for s,v in CAMPAIGN['arms'].items() if a.node in v['pod_indices'])
    global GPU_IDS, NODE
    NODE=a.node
    GPU_IDS=CAMPAIGN['gpu_ids'][str(a.node)]
    rank=arm['pod_indices'].index(a.node)
    work=ROOT/('canary' if a.canary else f'work-{step}');work.mkdir(exist_ok=True)
    prefix=arm['checkpoint']['publication_prefix']+('canary/' if a.canary else f'nodes/{rank:02d}/')
    def state(stage,**kwargs):
        v=dict(stage=stage,node=a.node,arm=step,rank=rank,time=time.time(),**kwargs);save(work/'status.json',v);upload(work/'status.json',prefix+'status.json');print(json.dumps(v),flush=True)
    try:
        state('staging_caption')
        selected=rows(WORKFLOW/'inputs.jsonl')[rank::arm['node_count']]
        if a.canary:
            canary_ids={'red:349','g2:in_image_text_translation_000045','g2:in_image_text_translation_000013'}
            selected=selected[:1]+[r for r in rows(WORKFLOW/'inputs.jsonl') if r['sample_id'] in canary_ids]
            assert len(selected)==len({r['sample_id'] for r in selected})==4
        stage_sources(selected);model,ready=stage_model(step,arm)
        locks=[]
        for name in ['/mnt/ephemeral/pcb-gpu-pipeline.lock',ROOT/'gpu-owner.lock']:
            f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already occupied'
        state('caption',cases=len(selected));values=caption(work,selected,model,ready,step,arm,a.canary)
        for name in ['predictions.jsonl','greedy-predictions.jsonl','retry-predictions.jsonl']:
            upload(work/'caption'/name,prefix+'caption/'+name)
        for summary in (work/'caption').glob('gpu-*/summary.json'):
            upload(summary,prefix+str(summary.relative_to(work)))
        from repetition_rules import validate_selected
        for value in values:validate_selected(value,CAMPAIGN['sampling'])
        state('caption_complete',cases=len(values))
        if a.caption_only:return
        if not a.canary:wait_file(ROOT/'CANARY_APPROVED.json')
        state('staging_editor');wait_file(ROOT/'EDITOR_ENV_READY.json');stage_dit(work);bind_runtime()
        native(work,'prepare',script='prepare.py')
        state('native_audit');native(work,'audit')
        admitted=len(rows(work/'inputs/manifest.jsonl'))-len(read(work/'audit.json')['rejected']);assert admitted>0
        state('encoding',admitted=admitted)
        with ThreadPoolExecutor(8) as pool:list(pool.map(lambda slot:native(work,'encode',slot,len(GPU_IDS),device=GPU_IDS[slot]),range(len(GPU_IDS))))
        state('generating',admitted=admitted)
        with ThreadPoolExecutor(8) as pool:list(pool.map(lambda slot:native(work,'infer',slot,len(GPU_IDS),device=GPU_IDS[slot]),range(len(GPU_IDS))))
        native(work,'verify');state('publishing');publish(work,prefix);state('complete')
    except BaseException:
        state('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
