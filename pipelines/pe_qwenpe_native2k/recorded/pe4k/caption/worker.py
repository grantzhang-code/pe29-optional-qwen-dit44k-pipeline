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
    local=Path('/mnt/ephemeral/pe39c59-c4000-export-20261009/model/hf')
    def one(row):
        p=model/row['name'];old=local/row['name']
        if not p.exists() and old.is_file() and sha(old)==row['sha256']:p.symlink_to(old)
        download(prefix+row['name'],p,row['sha256'])
    with ThreadPoolExecutor(6) as pool:list(pool.map(one,manifest['files']))
    return model,{'export_manifest_sha256':cp['export_manifest_sha256']}

def stage_sources(selected):
    cache=WORKFLOW/'source-cache';cache.mkdir(exist_ok=True)
    candidates=list(Path('/mnt/ephemeral').glob('pe29-*/package/workflow/source-cache'))+list(Path('/mnt/ephemeral').glob('pcb-v2-9c4d-*/data/source'))+list(Path('/mnt/ephemeral').glob('pcb-*/workflow/source-cache'))
    def one(row):
        name=Path(urlsplit(row['source_s3_uri']).path).name;p=cache/name;expected=row['selection']['source_sha256']
        if not p.exists():
            old=next((d/name for d in candidates if (d/name).is_file() and sha(d/name)==expected),None)
            if old:p.symlink_to(old)
        download(row['source_s3_uri'],p,expected)
    with ThreadPoolExecutor(8) as pool:list(pool.map(one,{r["source_s3_uri"]:r for r in selected}.values()))

def caption(work,selected,model,ready,step,arm,canary=False):
    sys.path.insert(0,str(WORKFLOW/'code'))
    import caption_support
    def one(slot):
        gpu=GPU_IDS[slot]
        group=selected[slot::len(GPU_IDS)]
        if not group:return []
        d=work/'caption'/f'gpu-{gpu}';d.mkdir(parents=True,exist_ok=True)
        inputs=d/'inputs.jsonl';jsonl(inputs,group)
        pred=d/'predictions.jsonl';summary=d/'summary.json'
        if not summary.exists():
            cmd=[str(PACKAGE/'assets/env/caption/bin/python'),'-u',str(WORKFLOW/'code/infer_caption.py'),
                 '--model',str(model),'--artifact-manifest-sha256',ready['export_manifest_sha256'],
                 '--checkpoint-step',str(step),'--checkpoint-manifest-sha256',arm['checkpoint']['checkpoint_manifest_sha256'],
                 '--input-jsonl',str(inputs),'--input-sha256',sha(inputs),'--expected-count',str(len(group)),
                 '--output-jsonl',str(pred),'--summary-json',str(summary),'--node-index','0','--node-count','1',
                 '--batch-size','4','--request-window-size','16','--download-workers','8',
                 '--max-model-len','32768','--max-new-tokens','27000','--max-pixels','2359296',
                 '--gpu-memory-utilization','0.90','--attention-backend','FLASHINFER','--num-speculative-tokens','1','--seed','42']
            env=caption_support.environment(gpu,gpu);env['AWS_PROFILE']='core-cn-s3-readwrite'
            command(cmd,d/'run.log',env)
        summary=read(summary);values=rows(pred)
        assert summary['complete'] and summary['results']['inference_errors']==0
        assert summary['checkpoint']==dict(step=step,manifest_sha256=arm['checkpoint']['checkpoint_manifest_sha256'])
        assert summary['input']['sha256']==sha(inputs)
        assert summary['results']['predictions_sha256']==sha(pred)
        assert len(values)==len(group)
        by={r['input_index']:r for r in group}
        for v in values:
            src=by[v['input_index']]
            assert all(v[k]==src[k] for k in ['sample_id','instruction','source_s3_uri','target_canvas_size','selection'])
            assert v['original_instruction']==src['original_instruction'] and not v.get('inference_error')
        from input_policy import validate_instruction
        config=read(WORKFLOW/'campaign.json')
        for value in values:validate_instruction(value,config)
        return values
    with ThreadPoolExecutor(8) as pool:values=[v for part in pool.map(one,range(len(GPU_IDS))) for v in part]
    values.sort(key=lambda v:v['input_index']);jsonl(work/'caption/predictions.jsonl',values)
    return values

def publish_caption(work,prefix,selected,values):
    valid=[];rejected=[]
    for v in values:
        reasons=[]
        if v.get('finish_reason')=='length':reasons.append('truncated')
        if v.get('json_error'):reasons.append('json_error')
        if v.get('schema_error'):reasons.append('schema_error')
        if not isinstance(v.get('prediction'),dict):reasons.append('missing_json_object')
        if reasons:rejected.append(dict(sample_id=v['sample_id'],input_index=v['input_index'],reasons=reasons,json_error=v.get('json_error'),schema_error=v.get('schema_error')))
        else:valid.append(v)
    jsonl(work/'caption/valid.jsonl',valid);jsonl(work/'caption/excluded.jsonl',rejected)
    artifacts={}
    for p in (work/'caption').rglob('*'):
        if p.is_file() and p.name in ['inputs.jsonl','predictions.jsonl','summary.json','run.log','valid.jsonl','excluded.jsonl']:
            name=str(p.relative_to(work));artifacts[name]=upload(p,prefix+name)
    proof=dict(status='PASS',scope='caption_only',caption_count=len(values),valid=len(valid),excluded=len(rejected),
        inference_errors=0,use_qwen_pe=True,checkpoint_step=4000,training_prompt_sha256=CAMPAIGN['training_prompt_sha256'],
        input_sha256=sha(WORKFLOW/'inputs.jsonl'),artifacts=artifacts)
    save(work/'COMPLETE.json',proof);upload(work/'COMPLETE.json',prefix+'COMPLETE.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',type=int,required=True);ap.add_argument('--canary',action='store_true');a=ap.parse_args()
    assert CAMPAIGN['scope']=='caption_only' and CAMPAIGN['use_qwen_pe'] is True
    global GPU_IDS,NODE
    NODE=a.node;GPU_IDS=CAMPAIGN['gpu_ids'][str(NODE)]
    step=4000;arm=CAMPAIGN['arms'][str(step)];rank=arm['pod_indices'].index(NODE)
    work=ROOT/('canary' if a.canary else 'work-4000');work.mkdir(exist_ok=True)
    prefix=CAMPAIGN['publication_prefix']+('canary/' if a.canary else f'nodes/{rank:02d}/')
    def state(stage,**kwargs):
        v=dict(stage=stage,node=NODE,rank=rank,time=time.time(),**kwargs);save(work/'status.json',v);upload(work/'status.json',prefix+'status.json');print(json.dumps(v),flush=True)
    try:
        assert sha(WORKFLOW/'inputs.jsonl')==CAMPAIGN['input_sha256']
        selected=rows(WORKFLOW/'inputs.jsonl')[rank::arm['node_count']]
        if a.canary:selected=selected[:4]
        state('staging_caption',cases=len(selected));stage_sources(selected);model,ready=stage_model(step,arm);wait_file(ROOT/'CAPTION_ENV_READY.json')
        locks=[]
        for name in ['/mnt/ephemeral/pcb-gpu-pipeline.lock',ROOT/'gpu-owner.lock']:
            f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already occupied'
        state('caption',cases=len(selected));values=caption(work,selected,model,ready,step,arm,a.canary)
        state('publishing',cases=len(values));publish_caption(work,prefix,selected,values);state('complete',cases=len(values))
    except BaseException:
        state('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
