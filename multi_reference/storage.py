"""S3 downloads and immutable, readback-verified publication."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlsplit

from contracts import LOCK, digest, encode, file_sha, require_s3


def client(profile=None):
    import boto3
    from botocore.config import Config
    return boto3.Session(profile_name=profile).client('s3', region_name='us-east-1',
        config=Config(max_pool_connections=16, retries={'mode': 'standard', 'max_attempts': 8}))


def get(s3, uri):
    u = urlsplit(require_s3(uri))
    return s3.get_object(Bucket=u.netloc, Key=u.path.lstrip('/'))['Body'].read()


def download(s3, uri, path, expected, size=None):
    require_s3(uri)
    path = Path(path)
    if path.is_file() and (size is None or path.stat().st_size == size) and file_sha(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.download')
    u = urlsplit(uri)
    s3.download_file(u.netloc, u.path.lstrip('/'), str(temp))
    if (size is not None and temp.stat().st_size != size) or file_sha(temp) != expected:
        raise ValueError('Downloaded bytes differ from frozen manifest: ' + uri)
    temp.replace(path)


def save(path, value):
    save_bytes(path, encode(value))


def save_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_bytes(raw)
    temp.replace(path)


def publish(s3, uri, raw):
    from botocore.exceptions import ClientError
    u = urlsplit(require_s3(uri))
    sha = digest(raw)
    try:
        s3.put_object(Bucket=u.netloc, Key=u.path.lstrip('/'), Body=raw, IfNoneMatch='*',
            Metadata={'sha256': sha}, ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode())
    except ClientError as exc:
        if exc.response['Error']['Code'] not in ['PreconditionFailed', '412']:
            raise
    if digest(get(s3, uri)) != sha:
        raise ValueError('Existing S3 object differs; use a new run prefix: ' + uri)
    return {'uri': uri, 'bytes': len(raw), 'sha256': sha}


def stage_model(work, s3):
    uri = LOCK['model_uri']
    ready = json.loads(get(s3, uri + '_SUCCESS'))
    expected_checkpoint = {'uri': LOCK['checkpoint_uri'], 'step': 8000,
                           'manifest_sha256': LOCK['checkpoint_manifest_sha256']}
    if (ready.get('status') != 'complete' or ready.get('checkpoint') != expected_checkpoint
            or ready.get('export_manifest_sha256') != LOCK['model_export_manifest_sha256']):
        raise ValueError('HF export publication identity changed')
    model = Path(work) / 'model/hf'
    download(s3, uri + 'local_export_manifest.json', model / 'local_export_manifest.json',
             LOCK['model_export_manifest_sha256'])
    manifest = json.loads((model / 'local_export_manifest.json').read_text())
    if (manifest.get('kind') != 'local-mcore-hf-trained-mtp-export-v1'
            or manifest['source_mcore_checkpoint'] != expected_checkpoint
            or manifest['tensor_count'] != 738 or manifest['mtp_tensor_count'] != 15
            or len(manifest['mtp_changed_tensors']) != 15
            or manifest['fp32_state_preservation']['count'] != 48
            or manifest['fp32_state_preservation']['status'] != 'PASS'):
        raise ValueError('Trained MTP or FP32 preservation contract changed')

    def fetch(row):
        name = row['name']
        if Path(name).name != name or name in ['.', '..']:
            raise ValueError('Unexpected model manifest path')
        download(s3, uri + name, model / name, row['sha256'], row['bytes'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest['files']))
    save(Path(work) / 'MODEL_READY.json', dict(status='PASS', model_uri=uri,
         model_export_manifest_sha256=LOCK['model_export_manifest_sha256'],
         checkpoint=expected_checkpoint, files=len(manifest['files'])))
    return manifest


def verify_local_model(work):
    model = Path(work) / 'model/hf'
    if file_sha(model / 'local_export_manifest.json') != LOCK['model_export_manifest_sha256']:
        raise ValueError('Local model manifest changed')
    marker = json.loads((Path(work) / 'MODEL_READY.json').read_text())
    if marker.get('status') != 'PASS' or marker.get('model_uri') != LOCK['model_uri']:
        raise ValueError('Run stage-model before inference')
    manifest = json.loads((model / 'local_export_manifest.json').read_text())
    for row in manifest['files']:
        if Path(row['name']).name != row['name']:
            raise ValueError('Unsafe model file name')
        path = model / row['name']
        if not path.is_file() or path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
            raise ValueError('Model file changed: ' + row['name'])
    return model
