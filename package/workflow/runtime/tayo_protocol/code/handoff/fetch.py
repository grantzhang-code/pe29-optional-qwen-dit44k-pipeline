"""Host-side staging. Requires AWS CLI; W&B resolution additionally uses wandb."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
RUNTIME = json.loads((HERE / 'runtime.json').read_text())

def aws(*args):
    return subprocess.check_output(['aws', *args], text=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def resolve(a):
    import wandb
    path = urlparse(a.wandb).path.strip('/').split('/')
    if len(path) != 4 or path[2] != 'runs':
        raise ValueError('Expected https://wandb.ai/<entity>/<project>/runs/<id>')
    run = wandb.Api().run('/'.join([path[0], path[1], path[3]]))
    root = run.config.get('artifact_s3_root')
    if not isinstance(root, str) or not root.startswith('s3://'):
        raise ValueError('W&B has no artifact_s3_root; obtain the checkpoint S3 URI from the run owner')
    print(root.rstrip('/') + f'/checkpoints/step={a.step}.ckpt/')

def checkpoint(a):
    if not a.uri.startswith('s3://'):
        raise ValueError('Expected S3 checkpoint URI')
    bucket, prefix = a.uri[5:].rstrip('/').split('/', 1)
    prefix += '/'
    def snapshot():
        data = json.loads(aws('s3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix, '--output', 'json'))
        return sorted([{'name': o['Key'][len(prefix):], 'key': o['Key'], 'bytes': o['Size'], 'etag': o['ETag']}
                       for o in data.get('Contents', [])
                       if re.fullmatch(r'__\d+_\d+\.distcp|\.metadata|meta\.pt', o['Key'][len(prefix):])], key=lambda o: o['name'])
    objects = snapshot()
    names = {o['name'] for o in objects}
    if not {'.metadata', 'meta.pt'} <= names or len(names) < 3 or any(o['bytes'] <= 0 for o in objects):
        raise ValueError('Checkpoint is missing control files or nonempty DCP shards')
    a.out.mkdir(parents=True, exist_ok=True)
    def get(obj):
        dest = a.out / obj['name']
        receipt = a.out / (obj['name'] + '.receipt.json')
        head = json.loads(aws('s3api', 'head-object', '--bucket', bucket, '--key', obj['key'], '--if-match', obj['etag']))
        identity = dict(obj, version=head.get('VersionId'))
        if receipt.exists() and dest.exists():
            old = json.loads(receipt.read_text())
            if all(old.get(k) == v for k, v in identity.items()) and sha(dest) == old['sha256']:
                return old
        args = ['s3api', 'get-object', '--bucket', bucket, '--key', obj['key'], '--if-match', obj['etag']]
        if identity['version']:
            args += ['--version-id', identity['version']]
        temp = dest.with_suffix(dest.suffix + '.partial')
        aws(*args, str(temp))
        if temp.stat().st_size != obj['bytes']:
            raise ValueError('Download size mismatch: ' + obj['name'])
        identity['sha256'] = sha(temp)
        temp.replace(dest)
        atomic(receipt, identity)
        print('STAGED', obj['name'], flush=True)
        return identity
    with ThreadPoolExecutor(max_workers=16) as pool:
        receipts = list(pool.map(get, objects))
    if snapshot() != objects:
        raise ValueError('Checkpoint changed during download; retry after training has finished writing it')
    atomic(a.out / 'DOWNLOAD.json', dict(uri=a.uri, objects=receipts, status='SNAPSHOT_DOWNLOADED'))
    print('Download complete. prepare will validate selected namespace, step, shapes/config and hashes.')

def inputs(a):
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ('shared-inputs.tar', 'inventory.json', 'done.json'):
        aws('s3', 'cp', RUNTIME['shared_inputs'] + name, str(a.out / name), '--only-show-errors')
    if sha(a.out / 'shared-inputs.tar') != RUNTIME['shared_inputs_tar_sha256']:
        raise ValueError('Shared-inputs archive differs from the fixed 1099 release')
    with tarfile.open(a.out / 'shared-inputs.tar') as archive:
        for member in archive.getmembers():
            dest = (a.out / member.name).resolve()
            if not dest.is_relative_to(a.out.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: ' + member.name)
        archive.extractall(a.out)
    inventory = json.loads((a.out / 'INVENTORY.json').read_text())
    for item in inventory['files']:
        path = (a.out / item['path']).resolve()
        if not path.is_relative_to(a.out.resolve()) or sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('Input archive mismatch: ' + item['path'])
    print('INPUTS_VERIFIED', len(inventory['files']))

def models(a):
    a.out.mkdir(parents=True, exist_ok=True)
    for name, key in [('Qwen3.8-27B', 'qwen'), ('FLUX.2-klein-4B-vae', 'vae')]:
        aws('s3', 'sync', RUNTIME[key], str(a.out / name), '--only-show-errors')
    # GPU work stays offline; all assets must already exist locally.
    print('MODELS_DOWNLOADED', a.out)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('resolve')
    cmd.add_argument('--wandb', required=True)
    cmd.add_argument('--step', type=int, required=True)
    cmd = sub.add_parser('checkpoint')
    cmd.add_argument('--uri', required=True)
    cmd.add_argument('--out', type=Path, required=True)
    for name in ('inputs', 'models'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    globals()[a.command](a)

if __name__ == '__main__':
    main()
