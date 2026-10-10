"""Reuse conditioning only when inputs, runtime and training contracts match exactly."""
from pathlib import Path
from types import SimpleNamespace
import argparse, json, shutil
import native_runner as N

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--step',type=int,required=True)
    a,_=ap.parse_known_args();a.ids=None;a.shard=0;a.shards=1
    campaign=N.read(Path(__file__).resolve().parents[1]/'campaign.json')
    old=Path(campaign['bundle_reuse_root'])/a.root.name
    assert N.read(old/'COMPLETE.json')['status']=='PASS'
    assert N.sha(old/'caption/predictions.jsonl')==N.sha(a.root/'caption/predictions.jsonl')
    assert N.read(old/'expected-resume-contract.json')==N.read(a.root/'expected-resume-contract.json')
    assert N.sha(old/'inputs/manifest.jsonl')==N.sha(a.root/'inputs/manifest.jsonl')
    assert N.sha(old/'model-inventory.json')==N.sha(a.root/'model-inventory.json')
    identity=N.encoding_identity(a)
    assert N.read(old/'audit.json')==N.read(a.root/'audit.json')
    assert N.read(old/'audit.json')['encoding_identity']==identity
    rejected={r['request_id'] for r in N.read(a.root/'audit.json')['rejected']}
    accepted=[r for r in N.rows(a) if r['request_id'] not in rejected]
    assert len(accepted)==N.read(old/'COMPLETE.json')['images']
    (a.root/'bundles').mkdir(exist_ok=True)
    sources=[]
    old_args=SimpleNamespace(**vars(a));old_args.root=old
    for row in accepted:
        source=N.verify_bundle(old_args,row)
        meta=N.read(source.with_suffix('.json'));assert meta['encoding_identity']==identity
        dest=N.bundle_path(a,row)
        if not dest.exists():dest.symlink_to(source)
        shutil.copy2(source.with_suffix('.json'),dest.with_suffix('.json'))
        assert N.verify_bundle(a,row)==dest
        sources.append(dict(request_id=row['request_id'],sha256=meta['sha256'],source=str(source)))
    for source in (old/'evidence').glob('encoder-*.json'):
        shutil.copy2(source,a.root/'evidence'/source.name)
    N.save(a.root/'evidence/bundle-reuse.json',dict(status='PASS',bundles=len(sources),source_root=str(old),
        encoding_identity=identity,manifest_sha256=N.sha(a.root/'inputs/manifest.jsonl'),
        resume_contract_sha256=N.sha(a.root/'expected-resume-contract.json'),sources=sources))
    print(json.dumps(dict(status='PASS',bundles=len(sources),encoding_identity=identity)),flush=True)

if __name__=='__main__':main()
