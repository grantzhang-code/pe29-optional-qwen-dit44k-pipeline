"""Restore only the frozen caption Python environment; never run inference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from urllib.parse import urlsplit

import boto3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    root = args.assets.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'env').exists():
        raise SystemExit('Use a fresh assets/env directory, or reuse an already verified environment.')
    shared = json.loads((Path(__file__).parent / 'shared-assets.json').read_text())
    row = next(x for x in shared['archives'] if x['key'].endswith('python-environments.tar.gz'))
    archive = root / 'python-environments.tar.gz'
    uri = urlsplit(row['uri'])
    if not archive.exists():
        boto3.Session(profile_name=args.profile).client('s3').download_file(uri.netloc, uri.path.lstrip('/'), str(archive))
    with archive.open('rb') as stream:
        assert archive.stat().st_size == row['bytes']
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
    caption_prefix = None
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            parts = PurePosixPath(member.name).parts
            if len(parts) >= 3 and parts[0] == 'scratch' and parts[2] == 'venv' and caption_prefix is None:
                caption_prefix = '/'.join(parts[:3])

            def remap(value):
                if caption_prefix and (value == caption_prefix or value.startswith(caption_prefix + '/')):
                    return 'env/caption' + value[len(caption_prefix):]
                if value == 'environments/python312' or value.startswith('environments/python312/'):
                    return 'env/python' + value[len('environments/python312'):]
                return None

            name = remap(member.name)
            if name is None:
                continue
            item = copy.copy(member)
            item.name = name
            if item.issym() or item.islnk():
                mapped = remap(item.linkname.removeprefix('@@PCB_ROOT@@/'))
                if mapped:
                    item.linkname = os.path.relpath(root / mapped, (root / name).parent) if item.issym() else mapped
            stream.extract(item, root, filter='data')
    env = root / 'env/caption'
    cfg = env / 'pyvenv.cfg'
    lines = cfg.read_text().splitlines()
    cfg.write_text('\n'.join('home = ' + str(root / 'env/python/bin') if x.startswith('home = ') else x
                             for x in lines if not x.startswith(('command =', 'executable ='))) + '\n')
    for path in (env / 'bin').iterdir():
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            if raw.startswith(b'#!') and b'python' in raw.split(b'\n', 1)[0]:
                path.write_bytes(('#!' + str(env / 'bin/python') + '\n').encode() + raw.split(b'\n', 1)[1])
    subprocess.run([str(env / 'bin/python'), '-c', 'import torch,vllm,transformers;print(torch.__version__,vllm.__version__,transformers.__version__)'], check=True)
    (root / 'ENVIRONMENT_READY.json').write_text(json.dumps({'status': 'PASS', 'archive_sha256': row['sha256']}) + '\n')


if __name__ == '__main__':
    main()
