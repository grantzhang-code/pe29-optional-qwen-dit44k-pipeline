"""Verify this source handover without credentials, network, GPU use or submission."""
from pathlib import Path
import ast, hashlib, json

ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'files.sha256.json').read_text())
    for name,digest in manifest.items():
        p=ROOT/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
    for name,step in [('pe3k',3000),('pe4k',4000)]:
        p=json.loads((ROOT/'profiles'/f'{name}.json').read_text())
        assert p['pe_step']==p['caption_checkpoint']['checkpoint_step']==step
        assert p['pe_run']=='39c59fafbc35' and p['input_count']==1464
        assert p['inference']['dit_step']==4000 and p['inference']['ema_keys']==463
        assert p['inference']['cfg']==2.5 and p['inference']['sampling_steps']==50
        assert p['gsb']['workers']==4 and p['gsb']['concurrency_per_worker']==256
        assert all(r['uri'].startswith('s3://') and len(r['sha256'])==64 for r in p['bundles'].values())
    print(json.dumps(dict(status='PASS',files=len(manifest),profiles=['pe3k','pe4k'])))
if __name__=='__main__':main()
