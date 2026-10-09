"""Restore PE/Qwen weights from S3 only; verify locked manifests and every file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/code'))
from checkpoint_sources import cache_path, load_sources, require_s3
from input_policy import QWEN_MANIFEST, QWEN_REVISION


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(name, client):
    sources = load_sources()
    uri = sources[name].rstrip('/') + '/'
    destination = cache_path(name)
    destination.mkdir(parents=True, exist_ok=True)

    def get(address):
        obj = urlsplit(require_s3(address))
        return client.get_object(Bucket=obj.netloc, Key=obj.path.lstrip('/'))['Body'].read()

    if name == 'pe29_hf':
        lock = json.loads((ROOT / 'workflow/caption-model.lock.json').read_text())
        if lock['model_uri'] != uri:
            raise ValueError('PE source differs from the frozen model lock')
        manifest_name = 'local_export_manifest.json'
        raw = get(uri + manifest_name)
        expected = lock['trained_artifact_manifest_sha256']
        ready = json.loads(get(uri.rsplit('model-29000/', 1)[0] + 'MODEL_READY.json'))
        if ready['status'] != 'PASS' or ready['manifest_sha256'] != expected:
            raise ValueError('PE model publication is incomplete')
        file_key = 'name'
    else:
        manifest_name = 'verified-manifest.json'
        raw = get(uri.rsplit('files/', 1)[0] + 'manifest.json')
        expected = QWEN_MANIFEST
        file_key = 'file'
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{name} manifest changed')
    manifest = json.loads(raw)
    if name == 'pe29_hf':
        if (manifest['source_mcore_checkpoint']['step'] != 29000
                or manifest['source_mcore_checkpoint']['manifest_sha256'] != lock['checkpoint_manifest_sha256']
                or manifest['tensor_count'] != 738 or manifest['mtp_tensor_count'] != 15):
            raise ValueError('PE-29k trained MTP identity changed')
    elif manifest['revision'] != QWEN_REVISION:
        raise ValueError('Qwen model revision changed')

    def one(row):
        name = row[file_key]
        path = destination / name
        if not path.resolve().is_relative_to(destination.resolve()) or Path(name).is_absolute():
            raise ValueError('Unsafe manifest path')
        if path.is_file() and path.stat().st_size == row['bytes'] and file_sha(path) == row['sha256']:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.download')
        obj = urlsplit(uri + name)
        client.download_file(obj.netloc, obj.path.lstrip('/'), str(temp))
        if temp.stat().st_size != row['bytes'] or file_sha(temp) != row['sha256']:
            raise ValueError(f'Model file verification failed: {name}')
        temp.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, manifest['files']))
    (destination / manifest_name).write_bytes(raw)
    print('MODEL_VERIFIED', name, uri, len(manifest['files']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pe29_hf', 'qwen_pe'], default='pe29_hf')
    parser.add_argument('--profile', default='core-cn-s3-readwrite')
    args = parser.parse_args()
    # Deliberately lazy: disabled Qwen does not download or import any Qwen model runtime.
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name=args.profile).client(
        's3', region_name='us-east-1', config=Config(retries={'mode': 'standard', 'max_attempts': 8}))
    restore(args.model, client)


if __name__ == '__main__':
    main()
