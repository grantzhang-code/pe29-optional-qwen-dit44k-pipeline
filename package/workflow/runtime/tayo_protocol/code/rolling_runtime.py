"""Reuse native inference with an explicit, immutable shared-encoding contract."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0,'/handoff')
import run as native

ENC = Path('/encoding')
ORIGINAL_PLAN = native.plan
ORIGINAL_IDENTITY = native.script_identity

def identity():
    return ORIGINAL_IDENTITY() | {'rolling_runtime.py':native.sha(__file__)}

def encoding_plan():
    p = native.read(ENC/'plan.json')
    native.require(p['identity']==identity(),'Shared encoder runtime changed')
    native.require(p['manifest_sha256']==native.sha(ENC/'manifest.jsonl'),'Shared inputs changed')
    native.require(p['config_sha256']==native.sha(ENC/'training.json'),'Shared training config changed')
    # Full native model/dataset config, including grouped variant, ROI and frame codes.
    native.require(native.read(native.WORK/'training.json')==native.read(ENC/'training.json'),
                   'Checkpoint input/model configuration differs; rebuild native inputs and shared features')
    for f in p['model_files']:
        path = native.MODELS/f['name']
        native.require(path.stat().st_size==f['bytes'] and path.stat().st_mtime_ns==f['mtime_ns'],
                       'Frozen Qwen/VAE asset changed')
    return p

def bound_plan():
    p = ORIGINAL_PLAN()
    encoding_plan()
    return p

def check_shared_bundle(row, ignored_checkpoint_contract):
    path = ENC/'bundles'/f"{row['request_id']}.pt"
    proof = native.read(path.with_suffix('.json'))
    native.require(proof['plan_sha256']==native.sha(ENC/'plan.json') and proof['request_id']==row['request_id'],
                   'Shared bundle contract mismatch')
    native.require(native.sha(path)==proof['sha256'],'Shared bundle SHA mismatch')
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','encode','infer','verify'])
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--ids')
    a = p.parse_args()
    a.weights,a.cfg='EMA',[2,3]
    a.checkpoint_uri=f's3://core-cn-training.canva.com/pixelcraft/usr/tayo/runs/edit1k-97src-15n-0925-0340/checkpoints/step={a.step}.ckpt/'
    if a.command=='encode':
        native.WORK=ENC
        native.plan=encoding_plan
    else:
        native.script_identity=lambda:identity() | {'encoding/plan.json':native.sha(ENC/'plan.json')}
        native.plan=bound_plan
        native.bundle_paths=lambda rid:(ENC/'bundles'/f'{rid}.pt',ENC/'bundles'/f'{rid}.json')
        native.check_bundle=check_shared_bundle
    encoding_plan()
    lock = native.WORK/f'{a.command}-{a.shard_index}.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare' and (native.WORK/'plan.json').exists():
            prepared = bound_plan()
            native.require(prepared['step']==a.step and prepared['weights']=='EMA','Wrong checkpoint plan')
            print('PREPARE_ALREADY_COMPLETE',a.step,flush=True)
        else: getattr(native,a.command)(a)

if __name__ == '__main__': main()
