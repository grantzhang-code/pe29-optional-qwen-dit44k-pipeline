"""Check disjoint complete worker shards before the unchanged judgment audit."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import worker as w

read = lambda p: json.loads(p.read_bytes())
rows = lambda p: [json.loads(line) for line in p.read_bytes().splitlines() if line]
all_pairs = {}
all_verdicts = {}
progress = []
assigned = read(ROOT / 'inputs/worker-shards.json')
for rank in range(4):
    prefix = w.DURABLE / f'workers/{rank}'
    marker = read(prefix / 'ALL_COMPLETE.json')
    p = read(w.DURABLE / f'progress/worker-{rank}.json')
    assert marker['status'] == p['stage'] == 'complete'
    assert marker['protocol_sha256'] == p['protocol_sha256'] == w.PROTOCOL_SHA
    assert p['rank'] == rank and p['concurrency'] == 256 and p['failed'] == 0
    assert 0 < p['peak_active_requests'] <= 256
    expected = set(assigned[str(rank)])
    pairs = rows(prefix / 'combined.jsonl')
    verdicts = rows(prefix / 'verdicts.jsonl')
    assert w.SHA((prefix / 'combined.jsonl').read_bytes()) == marker['combined_sha256']
    assert w.SHA((prefix / 'summary.json').read_bytes()) == marker['summary_sha256']
    assert marker['pairs'] == len(pairs) == len(expected)
    assert marker['judgments'] == len(verdicts) == 2 * len(expected) == p['completed']
    assert {v['pair_id'] for v in pairs} == expected
    assert {v['pair_id'] for v in verdicts} == expected
    for pair in pairs:
        assert pair['pair_id'] not in all_pairs
        assert pair['protocol_sha256'] == w.PROTOCOL_SHA
        all_pairs[pair['pair_id']] = pair
    for verdict in verdicts:
        assert verdict['task_id'] not in all_verdicts
        assert verdict['status'] == 'complete' and verdict['protocol_sha256'] == w.PROTOCOL_SHA
        all_verdicts[verdict['task_id']] = verdict
    progress.append(p)
expected_tasks = rows(ROOT / 'inputs/tasks.jsonl')
assert set(all_verdicts) == {t['task_id'] for t in expected_tasks}
assert set(all_pairs) == {t['pair_id'] for t in expected_tasks}
assert len(all_pairs) == w.PROTOCOL['pairs_count']
assert len(all_verdicts) == w.PROTOCOL['tasks_count']
w.STATS.update(completed=len(all_verdicts), failed=0,
               requests=sum(p['requests'] for p in progress), retries=sum(p['retries'] for p in progress))
combined = b''.join(w.ENCODE(p) for p in sorted(all_pairs.values(), key=lambda p: (p['pe_arm'], p['input_index'])))
w.put('combined.jsonl', combined)
w.put('verdicts.jsonl', b''.join(w.ENCODE(p) for p in sorted(all_verdicts.values(), key=lambda p: (p['pe_arm'], p['input_index'], p['order']))))
summary = w.publish_summary(all_pairs, final=True, aggregate=True)
w.durable_json('ALL_COMPLETE.json', dict(status='complete', workers=4, protocol_sha256=w.PROTOCOL_SHA,
    pairs=len(all_pairs), judgments=len(all_verdicts), combined_sha256=w.SHA(combined),
    summary_sha256=w.SHA(w.ENCODE(summary)), completed_utc=w.utc()))
print(json.dumps(dict(status='PASS', workers=4, pairs=len(all_pairs), judgments=len(all_verdicts))), flush=True)
