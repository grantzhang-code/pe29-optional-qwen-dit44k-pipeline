"""Launch only this evaluation's named containers with frozen offline dependencies."""
import argparse
import os
from pathlib import Path
from common import ROOT, BASE, ENC, HANDOFF, MODELS, CONFIG, DOCKER_ENV, docker_name, read

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'encode', 'infer', 'verify'])
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--ids')
    a = p.parse_args()
    work = ROOT / 'runs' / f'step-{a.step}'
    checkpoint = ROOT / 'checkpoints' / f'step-{a.step}'
    # Encoding is checkpoint independent and needs only the frozen input config.
    if a.stage == 'encode': work, checkpoint = ENC, BASE / 'checkpoint'
    cache = ROOT / 'runtime-cache' / f'gpu-{a.rank}'
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cmd = ['docker', 'run', '--rm', '--name', docker_name(a.stage,a.step,a.rank),
           '--label', 'pixelcraft-eval=large15-test1364-rolling-20260927',
           '--network', 'none', '--log-driver', 'none', '--read-only', '--shm-size', '32g',
           '--tmpfs', '/tmp:rw,size=2g,mode=1777', '--user', f'{os.getuid()}:{os.getgid()}',
           '--workdir', '/opt/pixelcraft', '--entrypoint', '/opt/runtime/bin/python']
    if a.stage in ('encode','infer'): cmd += ['--gpus', f'device={a.rank}']
    mounts = [(HANDOFF,'/handoff',False),(ROOT/'source/rolling_runtime.py','/rolling_runtime.py',False),
              (work,'/work',True),(ENC,'/encoding',True),(BASE,'/inputs',False),
              (BASE/'native-batches','/work/native-batches',False),
              (BASE/'native-batches','/encoding/native-batches',False),
              (MODELS,'/models',False),(checkpoint,'/checkpoint',False),(cache,'/cache',True)]
    for name, target in [('runtime_helpers.py','/opt/inference/runtime_helpers.py'),
                         ('mariana_transformer_engine.py','/opt/pixelcraft/mariana/integrations/transformer_engine.py'),
                         ('magi_calc_meta_h200.py','/opt/magi-attention-src/magi_attention/meta/collection/calc_meta.py')]:
        mounts.append((HANDOFF/name,target,False))
    for source in sorted((HANDOFF/'training-source').rglob('*.py')):
        mounts.append((source,'/opt/pixelcraft/'+str(source.relative_to(HANDOFF/'training-source')),False))
    for host, target, writable in mounts:
        assert host.exists(), host
        cmd += ['-v', f'{host.resolve()}:{target}'+('' if writable else ':ro')]
    env = {'USER':'coder','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
           'PYTHONPATH':'/handoff:/opt/inference:/opt/pixelcraft:/opt/pixelcraft/tasks/pixelcraft',
           'XDG_CACHE_HOME':'/cache','HF_HOME':'/cache/hf','TORCH_HOME':'/cache/torch',
           'TRITON_CACHE_DIR':'/cache/triton','TORCHINDUCTOR_CACHE_DIR':'/cache/inductor',
           'CUDA_CACHE_PATH':'/cache/cuda','TMPDIR':'/cache','MAGI_ATTENTION_WORKSPACE_BASE':'/cache/magi',
           'QUACK_CACHE_DIR':'/cache/quack'}
    for k,v in env.items(): cmd += ['-e',f'{k}={v}']
    tail = [a.stage,'--step',str(a.step),'--shard-index',str(a.rank),'--shard-count',str(a.count)]
    if a.ids: tail += ['--ids',a.ids]
    cmd += [read(HANDOFF/'runtime.json')['image'],'/rolling_runtime.py',*tail]
    os.execvpe('docker',cmd,DOCKER_ENV)

if __name__ == '__main__': main()
