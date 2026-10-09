"""Run the handoff CLI in the existing, pinned RTX image (no pip installs)."""
import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu', help='One GPU index; omit for CPU prepare/verify')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / 'cache').mkdir(exist_ok=True)
    image = json.loads((HERE / 'runtime.json').read_text())['image']
    cmd = ['docker', 'run', '--rm', '--network', 'none', '--log-driver', 'none', '--read-only', '--shm-size', '32g',
           '--tmpfs', '/tmp:rw,size=2g,mode=1777',
           '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', '/opt/pixelcraft',
           '--entrypoint', '/opt/runtime/bin/python']
    if a.gpu is not None:
        cmd += ['--gpus', f'device={a.gpu}']
    for host, guest, writable in [(HERE, '/handoff', False), (a.work, '/work', True),
                                  (a.inputs, '/inputs', False), (a.models, '/models', False),
                                  (a.checkpoint, '/checkpoint', False),
                                  (a.work / 'cache', '/cache', True)]:
        if not host.is_dir():
            p.error(f'Missing directory: {host}')
        cmd += ['-v', f'{host.resolve()}:{guest}' + ('' if writable else ':ro')]
    # Runtime overlays only make the documented FP8 recipe explicit on H200.
    for name, guest in [('runtime_helpers.py', '/opt/inference/runtime_helpers.py'),
                        ('mariana_transformer_engine.py', '/opt/pixelcraft/mariana/integrations/transformer_engine.py'),
                        ('magi_calc_meta_h200.py', '/opt/magi-attention-src/magi_attention/meta/collection/calc_meta.py')]:
        cmd += ['-v', f'{(HERE / name).resolve()}:{guest}:ro']
    for source in sorted((HERE / 'training-source').rglob('*.py')):
        target = '/opt/pixelcraft/' + str(source.relative_to(HERE / 'training-source'))
        cmd += ['-v', f'{source.resolve()}:{target}:ro']
    env = {'USER': 'coder', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': '/handoff:/opt/inference:/opt/pixelcraft:/opt/pixelcraft/tasks/pixelcraft',
           'XDG_CACHE_HOME': '/cache', 'HF_HOME': '/cache/hf', 'TORCH_HOME': '/cache/torch',
           'TRITON_CACHE_DIR': '/cache/triton', 'TORCHINDUCTOR_CACHE_DIR': '/cache/inductor',
           'CUDA_CACHE_PATH': '/cache/cuda', 'TMPDIR': '/cache',
           'MAGI_ATTENTION_WORKSPACE_BASE': '/cache/magi', 'QUACK_CACHE_DIR': '/cache/quack'}
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    tail = a.args[1:] if a.args[:1] == ['--'] else a.args
    if not tail:
        p.error('Provide a command after --, e.g. prepare --step 20000 --weights EMA')
    env_host = os.environ.copy()
    env_host['DOCKER_HOST'] = 'unix:///tmp/edit1k1099-docker-20260923/docker.sock'
    raise SystemExit(subprocess.call(cmd + [image, '/handoff/run.py', *tail], env=env_host))

if __name__ == '__main__':
    main()
