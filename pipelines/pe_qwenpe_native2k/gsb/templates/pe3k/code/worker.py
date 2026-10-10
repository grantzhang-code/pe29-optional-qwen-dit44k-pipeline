"""UTP double-order GSB: one worker, 256 concurrent calls, released rubric unchanged."""
import hashlib
import importlib.metadata
import io
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import google.auth.aws as google_aws
from google.auth.transport.requests import Request
from google import genai
from google.genai import types
import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_BYTES = (ROOT/'protocol.json').read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)
SHA = lambda b: hashlib.sha256(b).hexdigest()
PROTOCOL_SHA = SHA(PROTOCOL_BYTES)
ENCODE = lambda v: (json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n').encode()
RANK = int(os.environ.get('SCORE_WORKER_RANK',os.environ.get('JOB_COMPLETION_INDEX','0')))
RECOVERY = os.environ.get('RECOVERY_ALL_REMAINING') == '1'
WORKERS = PROTOCOL['workers']
CONCURRENCY = PROTOCOL['concurrency_per_worker']
OUT = urlsplit(PROTOCOL['output_prefix'])
S3 = boto3.Session(profile_name='core-cn-s3-readwrite').client('s3',region_name='us-east-1',config=Config(max_pool_connections=300,retries={'max_attempts':6,'mode':'standard'}))
DURABLE = Path(os.environ['CANVA_CLUSTER_STORAGE_PATH'])
DURABLE.mkdir(parents=True,exist_ok=True)
LOCK = threading.Lock()
STATS = {'rank':RANK,'concurrency':CONCURRENCY,'completed':0,'failed':0,'requests':0,'retries':0,'active_requests':0,'peak_active_requests':0,'stage':'starting','started_utc':datetime.now(timezone.utc).isoformat()}
STOP = threading.Event()
ABORT = threading.Event()
CACHE_LOCKS = [threading.Lock() for _ in range(128)]
IMAGE_CACHE = {}
ATTEMPT_RUN = str(time.time_ns())

def utc(): return datetime.now(timezone.utc).isoformat()
def get(uri):
    import ssl
    from urllib3.exceptions import HTTPError
    from botocore.exceptions import ResponseStreamingError, IncompleteReadError, ConnectionClosedError, EndpointConnectionError
    p=urlsplit(uri)
    for attempt in range(6):
        body=None
        try:
            body=S3.get_object(Bucket=p.netloc,Key=p.path.lstrip('/'))['Body']
            return body.read()
        except (ssl.SSLError, HTTPError, ResponseStreamingError, IncompleteReadError, ConnectionClosedError, EndpointConnectionError) as e:
            if attempt==5: raise
            log('s3_stream_retry',attempt=attempt+1,error_type=type(e).__name__)
            time.sleep(min(8,0.25*2**attempt))
        finally:
            if body is not None: body.close()
def local_write(key,data):
    path=DURABLE/key
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        temp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.partial')
        try:
            temp.write_bytes(data);temp.replace(path);return
        except FileNotFoundError:
            if attempt == 3: raise
            path.parent.mkdir(parents=True,exist_ok=True)
            time.sleep(0.1 * (attempt + 1))

def put(key,data):
    local_write(key,data)
    S3.put_object(Bucket=OUT.netloc,Key=OUT.path.lstrip('/')+key,Body=data,ContentType='application/json',Metadata={'sha256':SHA(data)})

def read_output(key):
    try:
        data=get(PROTOCOL['output_prefix']+key)
        local_write(key,data)
        return json.loads(data)
    except ClientError as error:
        if str(error.response['Error']['Code']) in ('NoSuchKey','404','NotFound'):return None
        raise

def durable_json(key,value):put(key,ENCODE(value))

def log(event,**kw): print(json.dumps({'utc':utc(),'event':event,'rank':RANK,**kw}),flush=True)
def progress():
    with LOCK: value=dict(STATS,updated_utc=utc(),protocol_sha256=PROTOCOL_SHA)
    put(f'progress/worker-{RANK}.json',ENCODE(value))
    log('progress',**{k:v for k,v in value.items() if k!='rank'})
def heartbeat():
    while not STOP.wait(15):
        try: progress()
        except Exception as e: log('heartbeat_error',error=type(e).__name__)

class Supplier(google_aws.AwsSecurityCredentialsSupplier):
    def __init__(self): self.session=boto3.Session(profile_name='core-cn-s3-readwrite',region_name='us-east-1')
    def get_aws_security_credentials(self,context,request):
        c=self.session.get_credentials().get_frozen_credentials()
        return google_aws.AwsSecurityCredentials(access_key_id=c.access_key,secret_access_key=c.secret_key,session_token=c.token)
    def get_aws_region(self,context,request): return 'us-east-1'

def make_client():
    sa=os.environ['GOOGLE_WIF_SERVICE_ACCOUNT']
    credentials=google_aws.Credentials(audience=os.environ['GOOGLE_WIF_AUDIENCE'],subject_token_type='urn:ietf:params:aws:token-type:aws4_request',token_url='https://sts.googleapis.com/v1/token',service_account_impersonation_url=f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateAccessToken',aws_security_credentials_supplier=Supplier(),scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials.refresh(Request())
    return genai.Client(vertexai=True,project=os.environ['GOOGLE_CLOUD_PROJECT'],location='global',credentials=credentials,http_options=types.HttpOptions(api_version='v1',timeout=300000,client_args={'limits':httpx.Limits(max_connections=300,max_keepalive_connections=300)}))

RUBRIC=json.loads((ROOT/'inputs/gsb_rubric.json').read_bytes())
DIMS=PROTOCOL['dimensions']

def image_for(item):
    digest=item['sha256']
    with CACHE_LOCKS[int(digest[:8],16)%len(CACHE_LOCKS)]:
        if digest in IMAGE_CACHE: return IMAGE_CACHE[digest]
        raw=get(item['uri'])
        assert SHA(raw)==digest, 'image SHA mismatch'
        with Image.open(io.BytesIO(raw)) as source:
            assert list(source.size)==item['size'], 'image size mismatch'
            im=source.convert('RGB')
            w,h=im.size
            if w*h>PROTOCOL['pixel_budget']:
                scale=(PROTOCOL['pixel_budget']/(w*h))**0.5
                im=im.resize((max(1,int(w*scale)),max(1,int(h*scale))),Image.Resampling.LANCZOS)
            buf=io.BytesIO(); im.save(buf,format='WEBP',quality=90)
            data=buf.getvalue()
            value=(data,{'original_sha256':digest,'normalized_sha256':SHA(data),'original_size':item['size'],'judge_size':list(im.size),'bytes':len(data),'mime_type':'image/webp'})
        IMAGE_CACHE[digest]=value
        return value

def persist_first(key,row):
    data=ENCODE(row)
    try:
        S3.put_object(Bucket=OUT.netloc,Key=OUT.path.lstrip('/')+key,Body=data,ContentType='application/json',Metadata={'sha256':SHA(data)},IfNoneMatch='*')
    except ClientError as error:
        if str(error.response['Error']['Code']) not in ('PreconditionFailed','412'):raise
        previous=read_output(key)
        assert previous['protocol_sha256']==PROTOCOL_SHA and previous['task']==row['task'] and previous['status']=='complete'
        return previous
    local_write(key,data)
    return row

def compact(row):
    return {k:row[k] for k in ('task_id','pair_id','sample_id','input_index','group','pe_arm','order','first','second','status','winners','reasons','model_version','record_uri','protocol_sha256','attempts_count','completed_utc')}

def score(task,client):
    if ABORT.is_set(): raise RuntimeError('Evaluation stopped after a fatal API error')
    tid=task['task_id']; key=f'results/{tid}.json'
    previous=read_output(key)
    if previous:
        assert previous['protocol_sha256']==PROTOCOL_SHA and previous['task']==task and previous['status']=='complete'
        return compact(previous)
    source=image_for(task['source'])
    first=image_for(task['candidates'][task['first']])
    second=image_for(task['candidates'][task['second']])
    checkpoints='\n\nCONSISTENCY CHECKPOINTS (edit fidelity only):\n'+'\n'.join(f'{i}. {line}' for i,line in enumerate(task['checkpoints'],1)) if task['checkpoints'] else ''
    prompt=f"ORIGINAL EDIT INSTRUCTION:\n{task['instruction_en']}{checkpoints}"
    contents=[prompt,'SOURCE IMAGE:',types.Part.from_bytes(data=source[0],mime_type='image/webp'),
              'CANDIDATE 1:',types.Part.from_bytes(data=first[0],mime_type='image/webp'),
              'CANDIDATE 2:',types.Part.from_bytes(data=second[0],mime_type='image/webp')]
    config=types.GenerateContentConfig(temperature=PROTOCOL['temperature'],max_output_tokens=PROTOCOL['max_output_tokens'],response_mime_type='application/json',response_schema=RUBRIC['SCHEMA'],thinking_config=types.ThinkingConfig(thinking_level=PROTOCOL['thinking_level']),system_instruction=RUBRIC['SYSTEM'])
    attempts=[]; parsed=None
    for attempt in range(6):
        start=time.monotonic(); entry={'attempt':attempt+1,'started_utc':utc()}
        try:
            with LOCK:
                STATS['requests']+=1; STATS['active_requests']+=1
                STATS['peak_active_requests']=max(STATS['peak_active_requests'],STATS['active_requests'])
            try: response=client.models.generate_content(model=PROTOCOL['model'],contents=contents,config=config)
            finally:
                with LOCK: STATS['active_requests']-=1
            entry['raw_response']=response.model_dump(mode='json',exclude_none=True)
            answer=response.text or ''; entry['response_text']=answer
            obj=json.loads(answer)
            assert set(obj)=={f'{d}_{k}' for d in DIMS for k in ('winner','reason')}
            assert all(obj[f'{d}_winner'] in ('1','2','tie') and isinstance(obj[f'{d}_reason'],str) and obj[f'{d}_reason'].strip() for d in DIMS)
            parsed=obj
        except Exception as e:
            entry.update(error_type=type(e).__name__,error=str(e)[:3000],error_code=getattr(e,'code',None))
            if entry['error_code'] in (400,401,403,404):
                entry['fatal']=True; ABORT.set()
        entry['seconds']=round(time.monotonic()-start,3); attempts.append(entry)
        put(f'attempts/{tid}/{ATTEMPT_RUN}-{attempt+1:02d}.json',ENCODE(entry))
        if parsed is not None or entry.get('fatal'): break
        if attempt<5:
            with LOCK: STATS['retries']+=1
            time.sleep(min(60,2**attempt)+random.random()*2)
    mapping={'1':task['first'],'2':task['second'],'tie':'tie'}
    row={k:task[k] for k in ('task_id','pair_id','sample_id','input_index','group','pe_arm','order','first','second')}
    row.update(task=task,protocol_sha256=PROTOCOL_SHA,status='complete' if parsed else 'failed',
               winners={d:mapping[parsed[f'{d}_winner']] for d in DIMS} if parsed else {},
               reasons={d:parsed[f'{d}_reason'] for d in DIMS} if parsed else {},
               model_requested=PROTOCOL['model'],model_version=attempts[-1].get('raw_response',{}).get('model_version'),
               thinking_level=PROTOCOL['thinking_level'],request_config=config.model_dump(mode='json',exclude_none=True),
               user_prompt=prompt,prompt_sha256=SHA(prompt.encode()),judge_source=source[1],judge_first=first[1],judge_second=second[1],
               attempts=attempts,attempts_count=len(attempts),worker=RANK,completed_utc=utc(),record_uri=PROTOCOL['output_prefix']+key)
    if parsed: row=persist_first(key,row)
    else: durable_json(f'failures/{tid}.json',row)
    return compact(row)

def combine(a,b):
    assert a['order']=='ab' and b['order']=='ba' and a['pair_id']==b['pair_id']
    assert a['pe_arm']==b['pe_arm'] and a['first']==b['second']==a['pe_arm'] and a['second']==b['first']==PROTOCOL['control_arm']
    row={k:a[k] for k in ('pair_id','input_index','sample_id','group','pe_arm','protocol_sha256')}
    row.update(status='complete',dimensions={},orders=[a,b],completed_utc=utc())
    for d in DIMS:
        agrees=a['winners'][d]==b['winners'][d]
        winner=a['winners'][d] if agrees else 'tie'
        row['dimensions'][d]={'winner':winner,'outcome':PROTOCOL['outcome_mapping'][winner],
                              'order_agreement':agrees,'position_disagreement':not agrees,
                              'both_tie':a['winners'][d]==b['winners'][d]=='tie'}
    return row

def metrics(rows):
    result={}
    for d in DIMS:
        c=Counter(r['dimensions'][d]['outcome'] for r in rows); n=len(rows); decisive=c['good']+c['bad']
        result[d]={'n':n,**{k:c[k] for k in ('good','similar','bad')},
                   'net_win_rate':(c['good']-c['bad'])/n if n else None,
                   'win_rate_excluding_similar':c['good']/decisive if decisive else None,
                   'position_disagreements':sum(r['dimensions'][d]['position_disagreement'] for r in rows),
                   'both_tie':sum(r['dimensions'][d]['both_tie'] for r in rows)}
    return result

def publish_summary(pairs,final=False):
    rows=sorted(pairs.values(),key=lambda r:(r['pe_arm'],r['input_index']))
    summary={'status':'complete' if final else 'running','complete':final,'updated_utc':utc(),
             'protocol_sha256':PROTOCOL_SHA,'model':PROTOCOL['model'],'thinking_level':PROTOCOL['thinking_level'],
             'protocol':PROTOCOL,'expected_pairs':PROTOCOL['pairs_count'],'completed_pairs':len(rows),'expected_unique_cases':PROTOCOL['unique_cases'],'completed_unique_cases':len({r['sample_id'] for r in rows}),
             'expected_calls':PROTOCOL['tasks_count'],'completed_calls':STATS['completed'],
             'failed_calls':STATS['failed'],
             'comparisons':{arm:{'expected_pairs':PROTOCOL['cohort']['comparisons'][arm]['count'],'completed_pairs':sum(r['pe_arm']==arm for r in rows),'metrics':metrics([r for r in rows if r['pe_arm']==arm]),'by_task':{g:metrics([r for r in rows if r['pe_arm']==arm and r['group']==g]) for g in sorted({r['group'] for r in rows if r['pe_arm']==arm})}} for arm in PROTOCOL['pe_arms']},
             'all_models_common':{'expected_pairs_per_comparison':PROTOCOL['cohort']['all_models_common_count'],'comparisons':{arm:metrics([r for r in rows if r['pe_arm']==arm and r['sample_id'] in set(PROTOCOL['cohort']['all_models_common_sample_ids'])]) for arm in PROTOCOL['pe_arms']}},
             'model_versions':dict(Counter(o['model_version'] for r in rows for o in r['orders']))}
    durable_json('summary.json',summary)
    return summary

def main():
    for name,digest in PROTOCOL['input_hashes'].items(): assert SHA((ROOT/'inputs'/name).read_bytes())==digest,name
    for name,digest in PROTOCOL['code_hashes'].items(): assert SHA((ROOT/name).read_bytes())==digest,name
    assert SHA(RUBRIC['SYSTEM'].encode())==PROTOCOL['system_sha256']
    tasks=[json.loads(l) for l in (ROOT/'inputs/tasks.jsonl').read_bytes().splitlines()]
    assert len(tasks)==PROTOCOL['tasks_count'] and WORKERS==1 and CONCURRENCY==256
    log('aws_identity',arn=boto3.Session(profile_name='core-cn-s3-readwrite').client('sts',region_name='us-east-1').get_caller_identity()['Arn'])
    client=make_client(); STATS['stage']='canary'; progress()
    threading.Thread(target=heartbeat,daemon=True).start()
    orders=defaultdict(dict); pairs={}; results={}; last_publish=0
    def accept(row):
        nonlocal last_publish
        results[row['task_id']]=row
        STATS['completed' if row['status']=='complete' else 'failed']+=1
        if row['status']=='complete':
            orders[row['pair_id']][row['order']]=row
            if len(orders[row['pair_id']])==2:
                p=combine(orders[row['pair_id']]['ab'],orders[row['pair_id']]['ba'])
                pairs[p['pair_id']]=p
        STATS['completed_pairs']=len(pairs)
        if time.time()-last_publish>15:
            publish_summary(pairs); last_publish=time.time()
    canary_ids={min((t for t in tasks if t['pe_arm']==arm),key=lambda r:r['input_index'])['pair_id'] for arm in PROTOCOL['pe_arms']}
    with ThreadPoolExecutor(max_workers=min(CONCURRENCY,2*len(PROTOCOL['pe_arms']))) as pool:
        for row in pool.map(lambda t:score(t,client),[t for t in tasks if t['pair_id'] in canary_ids]):
            accept(row)
            assert row['status']=='complete','GSB canary failed'
    assert all(set(orders[pid])=={'ab','ba'} for pid in canary_ids)
    durable_json('CANARY_PASS.json',{'protocol_sha256':PROTOCOL_SHA,'pair_ids':sorted(canary_ids),'judgments':len(results),'completed_utc':utc()})
    if os.environ.get('GSB_CANARY_ONLY')=='1':
        STOP.set();STATS['stage']='canary_passed';progress();client.close();return
    STATS['stage']='scoring'; progress()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures={pool.submit(score,t,client):t for t in tasks if t['task_id'] not in results}
        for future in as_completed(futures):
            row=future.result(); accept(row)
    assert not STATS['failed'],f"{STATS['failed']} failed judgments remain"
    assert len(results)==len(tasks) and len(pairs)==PROTOCOL['pairs_count']
    STATS['stage']='aggregating'; progress()
    # Raw judgments are already durable. Bound independent pair-object writes.
    with ThreadPoolExecutor(max_workers=PROTOCOL['pair_storage_concurrency']) as storage:
        pending=[storage.submit(durable_json,f"pairs/{p['pair_id']}.json",p) for p in pairs.values()]
        for n,future in enumerate(as_completed(pending),1):
            future.result()
            with LOCK: STATS['published_pairs']=n
    ordered=sorted(pairs.values(),key=lambda r:(r['pe_arm'],r['input_index']))
    data=b''.join(ENCODE(r) for r in ordered); put('combined.jsonl',data)
    put('verdicts.jsonl',b''.join(ENCODE(r) for r in sorted(results.values(),key=lambda r:(r['pe_arm'],r['input_index'],r['order']))))
    summary=publish_summary(pairs,final=True)
    durable_json('ALL_COMPLETE.json',{'status':'complete','protocol_sha256':PROTOCOL_SHA,'pairs':len(pairs),'judgments':len(results),
                  'combined_sha256':SHA(data),'summary_sha256':SHA(ENCODE(summary)),'completed_utc':utc()})
    STOP.set(); STATS['stage']='complete'; progress(); client.close()

if __name__=='__main__':
    import fcntl
    lock=(DURABLE/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: main()
    except BaseException as error:
        STOP.set(); ABORT.set(); STATS['stage']='failed'; STATS['error']=repr(error)
        progress(); raise
