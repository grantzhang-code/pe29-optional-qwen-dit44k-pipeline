"""Offline package integrity check. No model calls, S3 writes, or GPU use."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'files.manifest.json').read_text())
for row in manifest['files']:
    path = root / row['path']
    assert path.resolve().is_relative_to(root), row['path']
    with path.open('rb') as stream:
        assert path.stat().st_size == row['bytes'], row['path']
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'], row['path']
print('PACKAGE_VERIFIED', len(manifest['files']))
