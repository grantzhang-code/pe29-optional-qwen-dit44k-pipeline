"""Download and verify the byte-preserving EMA extraction on a held node."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from urllib.parse import urlsplit
from checkpoint_sources import cache_path, load_sources
DEST=cache_path('dit44_ema')
ROOT=DEST.parent
SOURCE=urlsplit(load_sources()['dit44_ema'])
BUCKET=SOURCE.netloc
PREFIX=SOURCE.path.lstrip('/').rstrip('/')+'/'
S3=boto3.Session(profile_name='core-cn-s3-readwrite',region_name='us-east-1').client('s3',config=Config(max_pool_connections=24,retries={'mode':'standard','max_attempts':8}))

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(S3.get_object(Bucket=BUCKET,Key=PREFIX+'DOWNLOAD.json')['Body'].read())
    identity=json.loads((Path(__file__).resolve().parents[1]/'checkpoint-identity.json').read_text())
    if receipt['extraction_sha256'] != identity['extraction_sha256']:
        raise ValueError('DiT EMA extraction identity changed')
    def one(item):
        dst=DEST/item['name']
        if not dst.resolve().is_relative_to(DEST.resolve()) or not item['key'].startswith(PREFIX):
            raise ValueError('Invalid checkpoint manifest path')
        if not (dst.exists() and dst.stat().st_size==item['bytes'] and sha(dst)==item['sha256']):
            tmp=dst.with_suffix('.download')
            S3.download_file(BUCKET,item['key'],str(tmp),Config=TransferConfig(max_concurrency=2))
            assert tmp.stat().st_size==item['bytes'] and sha(tmp)==item['sha256']
            tmp.replace(dst)
        return item['bytes']
    with ThreadPoolExecutor(8) as pool:
        total=0
        for n,size in enumerate(pool.map(one,receipt['objects']),1):
            total+=size
            if n%16==0:print('CHECKPOINT_FILES',n,len(receipt['objects']),total,flush=True)
    proof=S3.get_object(Bucket=BUCKET,Key=PREFIX+'EXTRACTION_VERIFIED.json')['Body'].read()
    assert hashlib.sha256(proof).hexdigest()==receipt['extraction_sha256']
    assert json.loads(proof)['tensor_bytes_unchanged'] is True
    (DEST/'EXTRACTION_VERIFIED.json').write_bytes(proof)
    (DEST/'DOWNLOAD.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
    (ROOT/'checkpoint-node-ready.json').write_text(json.dumps({'status':'PASS','files':len(receipt['objects']),'bytes':total,'extraction_sha256':receipt['extraction_sha256'],'time':time.time()})+'\n')
    print('CHECKPOINT_READY',flush=True)

if __name__=='__main__':main()
