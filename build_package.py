"""Build and verify the v2 handover; publishing is a separate immutable S3 step."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
PREFIX = 's3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/handovers/pe29-optional-qwen-dit44k-fixed-20261009-v2/'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(PACKAGE / 'tests'), '-v'], check=True)
    work = PACKAGE / 'workflow'
    sources = json.loads((work / 'checkpoint-sources.json').read_text())
    if len(sources) != 8 or not all(v.startswith('s3://') for v in sources.values()):
        raise ValueError('All checkpoint sources must be S3 URIs')
    adaptation = json.loads((work / 'ADAPTATION_VERIFIED.json').read_text())
    for name, digest in adaptation['fixed_runtime_files'].items():
        if sha(work / 'runtime' / name) != digest:
            raise ValueError('Frozen DiT fix changed: ' + name)
    # Ship a template, never a runnable experiment with somebody else's publication prefix.
    cfg = json.loads((work / 'campaign.json').read_text())
    cfg['run_configured'] = False
    cfg['publication_prefix'] = 's3://core-cn-oss.canva.com/usr/YOUR_NAME/pixelcraftbench-v2/evaluations/YOUR_UNIQUE_PE29_RUN/'
    cfg['models_lock_sha256'] = sha(work / 'models.lock.json')
    save(work / 'campaign.json', cfg)
    hashes = json.loads((work / 'code-hashes.json').read_text())
    save(work / 'code-hashes.json', {name: sha(work / name) for name in hashes if (work / name).is_file()})
    python_count = 0
    for path in PACKAGE.rglob('*.py'):
        ast.parse(path.read_text())
        python_count += 1
    blocks = re.findall(r'```bash\n(.*?)```', (ROOT / 'handover.md').read_text(), re.S)
    for block in blocks:
        subprocess.run(['bash', '-n'], input=block, text=True, check=True)
        for code in re.findall(r"<<'PY'\n(.*?)\nPY", block, re.S):
            ast.parse(code)
    for cache in PACKAGE.rglob('__pycache__'):
        shutil.rmtree(cache)
    paths = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.name != 'files.manifest.json')
    files = [dict(path=str(p.relative_to(PACKAGE)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    save(PACKAGE / 'files.manifest.json', dict(version=2, files=files))
    subprocess.run([sys.executable, str(PACKAGE / 'verify_package.py')], check=True)
    with tarfile.open(ROOT / 'workflow-sources.tar.gz', 'w:gz') as archive:
        for path in sorted(PACKAGE.iterdir()):
            archive.add(path, arcname=path.name)
    save(ROOT / 'LOCAL_VERIFIED.json', dict(
        status='PASS', verified_utc=datetime.now(timezone.utc).isoformat(),
        regression_tests=7, cases_per_mode=1464, modes=['original', 'qwen_rewrite'],
        real_pe_normalizer=True, python_files_parsed=python_count, bash_blocks_parsed=len(blocks),
        package_files=len(files), fixed_dit_runtime_files_unchanged=len(adaptation['fixed_runtime_files']),
        all_checkpoint_sources_s3=True, gpu_inference_rerun=False,
        full_model_download_rerun=False, full_environment_restore_rerun=False))
    keys = ['handover.md', 'workflow-sources.tar.gz', 'LOCAL_VERIFIED.json', 'S3_SOURCES_VERIFIED.json']
    save(ROOT / 'package.manifest.json', dict(version=2, prefix=PREFIX, files=[
        dict(key=key, uri=PREFIX + key, bytes=(ROOT / key).stat().st_size, sha256=sha(ROOT / key)) for key in keys]))
    print('PACKAGE_READY', PREFIX)


if __name__ == '__main__':
    main()
