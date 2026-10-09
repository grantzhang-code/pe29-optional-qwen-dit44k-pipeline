"""Publish the prepared handover immutably, then verify every S3 object by readback."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='prod_core-cn-readwrite')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'package.manifest.json').read_text())
    target = urlsplit(manifest['prefix'])
    client = boto3.Session(profile_name=args.profile).client(
        's3', region_name='us-east-1',
        config=Config(connect_timeout=10, read_timeout=120, retries={'max_attempts': 3}))

    def publish(name, raw, content_type):
        digest = hashlib.sha256(raw).hexdigest()
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        key = target.path.lstrip('/') + name
        try:
            client.put_object(Bucket=target.netloc, Key=key, Body=raw,
                              ContentType=content_type, IfNoneMatch='*',
                              ChecksumSHA256=checksum, Metadata={'sha256': digest})
        except ClientError as error:
            if error.response['Error']['Code'] not in ('PreconditionFailed', '412'):
                raise
        existing = client.get_object(Bucket=target.netloc, Key=key)
        body = existing['Body'].read()
        assert len(body) == len(raw), name
        assert hashlib.sha256(body).hexdigest() == digest, 'Existing S3 object differs: ' + name
        assert existing['Metadata'].get('sha256') == digest, name
        print('S3_READBACK_PASS', name, len(raw), digest, flush=True)
        return dict(key=name, uri=manifest['prefix'] + name, bytes=len(raw), sha256=digest)

    verified = []
    for row in manifest['files']:
        raw = (root / row['key']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        content_type = ('text/markdown; charset=utf-8' if row['key'].endswith('.md')
                        else 'application/gzip' if row['key'].endswith('.gz') else 'application/json')
        verified.append(publish(row['key'], raw, content_type))
    verified.append(publish('package.manifest.json', (root / 'package.manifest.json').read_bytes(), 'application/json'))
    receipt_path = root / 'VERIFIED.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['objects'] == verified
    else:
        receipt = dict(status='PASS', method='S3 GET readback; bytes + SHA256 + metadata',
                       verified_utc=datetime.now(timezone.utc).isoformat(), objects=verified,
                       gpu_inference_rerun=False)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    publish('VERIFIED.json', receipt_path.read_bytes(), 'application/json')
    print('HANDOVER_PUBLISHED', manifest['prefix'] + 'handover.md', flush=True)


if __name__ == '__main__':
    main()
