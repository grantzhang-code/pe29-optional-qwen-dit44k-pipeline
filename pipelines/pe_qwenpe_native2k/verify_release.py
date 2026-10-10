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
    adaptive=ROOT/'adaptive_pe3k'
    config=json.loads((adaptive/'profile.json').read_text())
    source_manifest=json.loads((adaptive/'recorded/SOURCE_MANIFEST.json').read_text())
    for path in (adaptive/'recorded').rglob('*'):
        if not path.is_file():continue
        name=str(path.relative_to(adaptive/'recorded'))
        if name in source_manifest:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source_manifest[name],name
    summary=json.loads((adaptive/'evidence/gsb-summary.json').read_text())
    proof=json.loads((adaptive/'evidence/gsb-VERIFIED.json').read_text())
    assert hashlib.sha256((adaptive/'evidence/gsb-summary.json').read_bytes()).hexdigest()==proof['summary_sha256']
    assert proof['status']=='PASS' and proof['pairs']==summary['pairs']==1275
    assert proof['judgments']==summary['judgments']==2550
    assert summary['workers']==4 and summary['concurrency_per_worker']==256
    comparison=json.loads((adaptive/'evidence/dit4k-comparison.json').read_text())
    assert comparison['final_scores_sha256']==proof['scores_sha256']
    assert comparison['common_cases']==len(set(comparison['common_sample_ids']))==1256
    assert config['sampling']['strategy']=='greedy_then_rule_triggered_official_retry_v1'
    assert config['sampling']['max_retry_attempts']==1 and config['inference']['dit_step']==6000
    for value in comparison['metrics'].values():
        for name in ['dit4k','dit6k_adaptive']:
            m=value[name];assert m['good']+m['similar']+m['bad']==m['n']==1256
            assert abs(m['net_win_rate']-(m['good']-m['bad'])/m['n'])<1e-12
        assert abs(value['net_win_change_percentage_points']-100*(value['dit6k_adaptive']['net_win_rate']-value['dit4k']['net_win_rate']))<1e-12
    print(json.dumps(dict(status='PASS',files=len(manifest),profiles=['pe3k','pe4k','adaptive_pe3k'])))
if __name__=='__main__':main()
