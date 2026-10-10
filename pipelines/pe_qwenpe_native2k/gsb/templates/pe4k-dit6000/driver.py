"""Run the frozen GSB worker and publish independently verified summaries."""
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
import base64,fcntl,hashlib,importlib.metadata,json,os,subprocess,sys
import boto3
ROOT=Path(__file__).resolve().parent
PROTOCOL=json.loads((ROOT/'protocol.json').read_bytes());BASE=PROTOCOL['publication_prefix']
S3=boto3.Session(profile_name='core-cn-s3-readwrite',region_name='us-east-1').client('s3')
def publish(name,data):
    u=urlsplit(BASE+name);h=hashlib.sha256(data).hexdigest();cs=base64.b64encode(bytes.fromhex(h)).decode()
    r=S3.put_object(Bucket=u.netloc,Key=u.path.lstrip('/'),Body=data,ChecksumSHA256=cs,Metadata={'sha256':h});assert r['ChecksumSHA256']==cs

def state(stage,**extra):
    row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pod=os.environ.get('HOSTNAME'),concurrency=256,**extra)
    data=(json.dumps(row,ensure_ascii=False)+'\n').encode();(ROOT/'execution-status.json').write_bytes(data);publish('execution/status.json',data);print(data.decode(),flush=True)
def main():
    versions={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        name,expected=line.split('==');versions[name]=importlib.metadata.version(name);assert versions[name]==expected,(name,versions[name],expected)
    publish('execution/runtime-versions.json',(json.dumps(versions)+'\n').encode())
    state('scoring',pairs=PROTOCOL['pairs_count'],order_judgments=PROTOCOL['tasks_count'])
    subprocess.run([sys.executable,'-u',str(ROOT/'code/worker.py')],check=True)
    state('verifying');subprocess.run([sys.executable,'-u',str(ROOT/'finalize.py')],check=True)
    for name in ['scores.jsonl','summary.json','metrics.csv','VERIFIED.json']:publish(name,(ROOT/name).read_bytes())
    state('complete',verified_uri=BASE+'VERIFIED.json',summary_uri=BASE+'summary.json')
if __name__=='__main__':
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise
