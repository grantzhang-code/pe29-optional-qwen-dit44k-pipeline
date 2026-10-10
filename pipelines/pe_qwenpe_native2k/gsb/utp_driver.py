"""Execute one GSB shard; rank zero reconciles and audits all four shards."""
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os, subprocess, sys, time
import boto3

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_bytes())
RANK = int(os.environ['JOB_COMPLETION_INDEX'])
os.environ['SCORE_WORKER_RANK'] = str(RANK)
BASE = PROTOCOL['publication_prefix']
S3 = boto3.client('s3', region_name='us-east-1')
DATA = Path(os.environ['CANVA_CLUSTER_STORAGE_PATH'])

def publish(name, data):
    u = urlsplit(BASE + name)
    digest = hashlib.sha256(data).hexdigest()
    S3.put_object(Bucket=u.netloc, Key=u.path.lstrip('/'), Body=data, Metadata={'sha256': digest})
    assert hashlib.sha256(S3.get_object(Bucket=u.netloc, Key=u.path.lstrip('/'))['Body'].read()).hexdigest() == digest

def state(stage, **extra):
    value = dict(stage=stage, rank=RANK, workers=4, concurrency_per_worker=256,
                 total_concurrency=1024, utc=datetime.now(timezone.utc).isoformat(),
                 pod=os.environ.get('HOSTNAME'), **extra)
    raw = (json.dumps(value) + '\n').encode()
    publish(f'execution/workers/{RANK}/status.json', raw)
    if RANK == 0:
        publish('execution/status.json', raw)
    print(raw.decode(), flush=True)

def main():
    assert PROTOCOL['workers'] == 4 and PROTOCOL['concurrency_per_worker'] == 256
    assert PROTOCOL['total_concurrency'] == 1024 and 0 <= RANK < 4
    versions = {}
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        name, expected = line.split('==')
        versions[name] = importlib.metadata.version(name)
        assert versions[name] == expected, (name, versions[name], expected)
    publish(f'execution/workers/{RANK}/runtime-versions.json', (json.dumps(versions) + '\n').encode())
    state('scoring')
    subprocess.run([sys.executable, '-u', str(ROOT / 'code/worker.py')], check=True)
    state('shard_complete')
    if RANK != 0:
        return
    state('waiting_for_shards')
    deadline = time.monotonic() + 3600
    while True:
        ready = []
        for rank in range(4):
            marker = DATA / f'workers/{rank}/ALL_COMPLETE.json'
            progress = DATA / f'progress/worker-{rank}.json'
            ready.append(marker.exists() and progress.exists() and json.loads(progress.read_text())['stage'] == 'complete')
        if all(ready):
            break
        assert time.monotonic() < deadline, 'Timed out waiting for the four completed shards'
        time.sleep(10)
    state('verifying')
    subprocess.run([sys.executable, '-u', str(ROOT / 'merge_workers.py')], check=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'finalize.py')], check=True)
    for name in ['scores.jsonl', 'summary.json', 'metrics.csv', 'VERIFIED.json']:
        publish(name, (ROOT / name).read_bytes())
    state('complete', verified_uri=BASE + 'VERIFIED.json', summary_uri=BASE + 'summary.json')

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=repr(error))
        raise
