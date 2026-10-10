"""Preserve prepared annotations and actual model prompts after node verification."""
from pathlib import Path
import argparse,tarfile,time
import worker as w

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',type=int,required=True);a=ap.parse_args()
    step,arm=next((int(s),v) for s,v in w.CAMPAIGN['arms'].items() if a.node in v['pod_indices'])
    rank=arm['pod_indices'].index(a.node);work=w.ROOT/f'work-{step}'
    while not (work/'COMPLETE.json').exists():time.sleep(20)
    assert w.read(work/'COMPLETE.json')['status']=='PASS'
    paths=sorted(set(work.glob('inputs/cases/*/*.json')) | set(work.glob('audits/*.json')) | set(work.glob('bundles/*.json')) | set(work.glob('caption/gpu-*/summary.json')) | set(work.glob('caption/gpu-*/**/*.json')) | set(work.glob('caption/gpu-*/**/*.jsonl')) | set(work.glob('caption/gpu-*/**/*.log')))
    archive=work/'prepared-inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in paths:tar.add(path,arcname=str(path.relative_to(work)),recursive=False)
    proof=w.upload(archive,arm['checkpoint']['publication_prefix']+f'nodes/{rank:02d}/reproduction/prepared-inputs.tar.gz')
    proof['files']=len(paths);w.save(work/'prepared-inputs.json',proof)
    w.upload(work/'prepared-inputs.json',arm['checkpoint']['publication_prefix']+f'nodes/{rank:02d}/reproduction/COMPLETE.json')
    print(proof,flush=True)
if __name__=='__main__':main()
