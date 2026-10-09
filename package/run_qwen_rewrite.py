"""Run one of 64 Qwen-PE logical ranks with S3 weights and an explicit output prefix."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/code'))
from checkpoint_sources import cache_path, require_s3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--output-s3', required=True)
    parser.add_argument('--profile', default='core-cn-s3-readwrite')
    args = parser.parse_args()
    if not 0 <= args.rank < 64:
        raise ValueError('rank must be in 0..63')
    output = require_s3(args.output_s3).rstrip('/') + '/qwen-rewrite/'
    if 'YOUR_' in output or 'qwen21-pe-i2i-edit-eagle64-20260928/' in output:
        raise ValueError('Use a new output prefix')
    code = ROOT / 'qwen_rewrite'
    sys.path.insert(0, str(code))
    import common as c
    import boto3
    c.EFS, c.LOCAL = ROOT / 'qwen-run', ROOT / 'qwen-cache'
    c.MODEL_CACHE = cache_path('qwen_pe')
    target = urlsplit(output)
    c.BUCKET, c.PREFIX = target.netloc, target.path.lstrip('/')
    c.S3 = boto3.Session(profile_name=args.profile).client('s3', region_name='us-east-1')
    (c.EFS / 'inputs').mkdir(parents=True, exist_ok=True)
    for src, dst in [(ROOT / 'workflow/inputs/original.jsonl', c.EFS / 'inputs/all.jsonl'),
                     (code / 'run.lock.json', c.EFS / 'run.lock.json')]:
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise ValueError('Qwen run binding changed')
        else:
            shutil.copy2(src, dst)
    # Verify cached files on every run; stage_models.py is the sole download entry.
    manifest = json.loads((c.MODEL_CACHE / 'verified-manifest.json').read_text())
    from input_policy import QWEN_MANIFEST, QWEN_REVISION
    if c.file_sha(c.MODEL_CACHE / 'verified-manifest.json') != QWEN_MANIFEST or manifest['revision'] != QWEN_REVISION:
        raise ValueError('Qwen model manifest changed')
    for row in manifest['files']:
        path = c.MODEL_CACHE / row['file']
        if (not path.resolve().is_relative_to(c.MODEL_CACHE.resolve())
                or path.stat().st_size != row['bytes'] or c.file_sha(path) != row['sha256']):
            raise ValueError('Qwen model file changed')
    sys.argv = [str(code / 'pe_worker.py'), '--rank', str(args.rank)]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
