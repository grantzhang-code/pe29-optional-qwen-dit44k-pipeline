from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config.json').read_text())
BASE = Path(CONFIG['baseline'])
ENC = ROOT / 'encoding'
HANDOFF = ROOT / 'source/handoff'
MODELS = Path(CONFIG['models'])
DOCKER_ENV = dict(os.environ, DOCKER_HOST=CONFIG['docker_host'])

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def status(stage, **fields):
    previous = read(ROOT / 'state/status.json') if (ROOT / 'state/status.json').exists() else {}
    value = dict(previous, stage=stage, updated_at=time.time(), **fields)
    atomic(ROOT / 'state/status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)

def docker_name(stage, step, rank):
    return f'large15-test1364-{stage}-{step}-{rank}'

def worker_command(stage, step, rank=0, count=1, ids=None):
    args = ['python3', str(ROOT / 'source/worker.py'), stage, '--step', str(step),
            '--rank', str(rank), '--count', str(count)]
    if ids: args += ['--ids', ','.join(ids)]
    return args
