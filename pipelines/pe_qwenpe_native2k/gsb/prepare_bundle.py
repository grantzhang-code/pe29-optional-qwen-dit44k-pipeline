"""Adapt only GSB execution, pin four disjoint shards, and build an audited bundle."""
from pathlib import Path
import ast, hashlib, json, shutil, subprocess, sys, tarfile

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
ENC = lambda value: (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()
read = lambda p: json.loads(p.read_bytes())

def replace_once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)

def prepare(step, run=None, cap=None):
    run = Path(run) if run is not None else W / f'pe39c59_c{step}_qwenpe_dit4k_cfg25_euc1_20261009'
    cap = Path(cap) if cap is not None else W / f'pe39c59_c{step}_qwenpe_caption_euc1_20261009'
    g = run / 'gsb_vs_qwen21'
    receipt = g / 'utp-bundle.json'
    if receipt.exists():
        value = read(receipt)
        assert SHA((g / 'bundle.tar.gz').read_bytes()) == value['bundle_sha256']
        return value
    assert read(run / 'results/PUBLICATION_VERIFIED.json')['status'] == 'PASS'
    assert read(run / 'results/encoder-order-audit.json')['status'] == 'PASS'
    for name in ['COMPLETE.json', 'PUBLICATION_VERIFIED.json']:
        shutil.copy2(run / 'results' / name, g / 'inputs' / name)
    shutil.copy2(run / 'campaign-execution.json', g / 'inputs/generation-campaign.json')
    shutil.copy2(cap / 'results/COMPLETE.json', g / 'inputs/caption-COMPLETE.json')
    shutil.copy2(cap / 'results/PUBLICATION_VERIFIED.json', g / 'inputs/caption-PUBLICATION_VERIFIED.json')
    for name in ['editor/score-index.jsonl', 'editor/images.manifest.jsonl', 'caption/excluded.jsonl']:
        p = g / f'inputs/generation/{step}' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / f'results/{step}' / name, p)
    ids = sorted(json.loads(line)['sample_id'] for line in (run / f'results/{step}/editor/score-index.jsonl').read_text().splitlines())
    (g / 'inputs/common-output-sample-ids.json').write_bytes(ENC(ids))
    assert not (g / 'protocol.json').exists(), 'Do not repin an already frozen scoring protocol'
    subprocess.run([sys.executable, str(g / 'prepare.py')], check=True)
    protocol = read(g / 'protocol.json')
    (g / 'single-worker-protocol.reference.json').write_bytes((g / 'protocol.json').read_bytes())
    worker = g / 'code/worker.py'
    original = worker.read_text()
    assert SHA(worker.read_bytes()) == '6ef61a420620655538b499fbba12fee13f478041a4451b047e022a00a55ea8af'
    source = original.replace("boto3.Session(profile_name='core-cn-s3-readwrite',", 'boto3.Session(').replace("boto3.Session(profile_name='core-cn-s3-readwrite')", 'boto3.Session()')
    source = replace_once(source, 'def publish_summary(pairs,final=False):', 'def publish_summary(pairs,final=False,aggregate=False):')
    source = replace_once(source, "durable_json('summary.json',summary)", "durable_json('summary.json' if aggregate else f'workers/{RANK}/summary.json',summary)")
    source = replace_once(source, "assert len(tasks)==PROTOCOL['tasks_count'] and WORKERS==1 and CONCURRENCY==256", "assert len(tasks)==PROTOCOL['tasks_count'] and WORKERS==4 and CONCURRENCY==256\n    assigned=json.loads((ROOT/'inputs/worker-shards.json').read_bytes())\n    local_ids=set(assigned[str(RANK)])\n    tasks=[task for task in tasks if task['pair_id'] in local_ids]\n    assert len(tasks)==2*len(local_ids) and local_ids\n    STATS.update(assigned_pairs=len(local_ids),assigned_judgments=len(tasks))")
    for name in ['CANARY_PASS.json', 'ALL_COMPLETE.json']:
        source = replace_once(source, f"durable_json('{name}',", f"durable_json(f'workers/{{RANK}}/{name}',")
    source = replace_once(source, "len(pairs)==PROTOCOL['pairs_count']", 'len(pairs)==len(local_ids)')
    for name in ['combined.jsonl', 'verdicts.jsonl']:
        source = replace_once(source, f"put('{name}',", f"put(f'workers/{{RANK}}/{name}',")
    source = replace_once(source, "(DURABLE/'worker.lock')", "(DURABLE/f'worker-{RANK}.lock')")
    worker.write_text(source)
    before = {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(original).body if isinstance(x, ast.FunctionDef)}
    after = {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(source).body if isinstance(x, ast.FunctionDef)}
    for name in ['score', 'image_for', 'combine', 'metrics', 'make_client', 'persist_first', 'get', 'put']:
        assert before[name] == after[name], name
    shutil.copy2(ROOT / 'utp_driver.py', g / 'utp_driver.py')
    shutil.copy2(ROOT / 'merge_workers.py', g / 'merge_workers.py')
    finalizer = g / 'finalize.py'
    source = finalizer.read_text()
    old = "    progress =read (DATA /'progress/worker-0.json')\n    assert progress ['concurrency']==protocol ['total_concurrency']==256 \n    assert 0 <progress ['peak_active_requests']<=256 "
    new = "    progress =[read(DATA /f'progress/worker-{rank}.json') for rank in range(4)]\n    assert protocol['workers']==4 and protocol['total_concurrency']==1024\n    assert all(p['concurrency']==256 and p['stage']=='complete' and p['failed']==0 and 0<p['peak_active_requests']<=256 for p in progress)\n    assert sum(p['completed'] for p in progress)==J"
    source = replace_once(source, old, new)
    source = replace_once(source, "total_concurrency =256 ,peak_active_requests =progress ['peak_active_requests']", "total_concurrency =1024 ,workers =4 ,concurrency_per_worker =256 ,sum_worker_peak_active_requests =sum(p['peak_active_requests'] for p in progress)")
    finalizer.write_text(source)
    pairs = [json.loads(line) for line in (g / 'inputs/pairs.jsonl').read_text().splitlines()]
    tasks = [json.loads(line) for line in (g / 'inputs/tasks.jsonl').read_text().splitlines()]
    shards = {str(rank): [] for rank in range(4)}
    for index, pair in enumerate(sorted(pairs, key=lambda p: p['pair_id'])):
        shards[str(index % 4)].append(pair['pair_id'])
    flattened = [pid for values in shards.values() for pid in values]
    assert len(flattened) == len(set(flattened)) == len(pairs)
    for rank, values in shards.items():
        subset = [t for t in tasks if t['pair_id'] in set(values)]
        assert len(subset) == 2 * len(values)
        for pid in values:
            assert {t['order'] for t in subset if t['pair_id'] == pid} == {'ab', 'ba'}
    (g / 'inputs/worker-shards.json').write_bytes(ENC(shards))
    plan = read(g / 'execution-plan.json')
    plan.update(workers=4, concurrency_per_worker=256, total_concurrency=1024,
                concurrency=1024, execution='Arnold UTP JobSet, four indexed CPU workers, stable disjoint pair shards')
    (g / 'execution-plan.json').write_bytes(ENC(plan))
    protocol.update(version=protocol['version'] + '-utp4x256', workers=4, total_concurrency=1024,
                    execution_policy='UTP 4 indexed CPU workers, 256 requests each, pod AWS identity via Vertex WIF',
                    worker_changes='Only shard ownership, per-worker output/lock paths, and AWS workload credential selection. score, image_for, combine, metrics and make_client ASTs unchanged.')
    protocol['input_hashes'] = {str(p.relative_to(g / 'inputs')): SHA(p.read_bytes()) for p in sorted((g / 'inputs').rglob('*')) if p.is_file()}
    names = list(protocol['code_hashes']) + ['utp_driver.py', 'merge_workers.py']
    protocol['code_hashes'] = {name: SHA((g / name).read_bytes()) for name in names}
    for name in names:
        if name.endswith('.py'):
            ast.parse((g / name).read_text(), filename=name)
    (g / 'protocol.json').write_bytes(ENC(protocol))
    with tarfile.open(g / 'bundle.tar.gz', 'w:gz') as tar:
        for p in [g / 'protocol.json', *[g / name for name in names], *sorted((g / 'inputs').rglob('*'))]:
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(g)), recursive=False)
    value = dict(status='PASS', checkpoint_step=step, pairs=len(pairs), judgments=len(tasks), workers=4,
                 concurrency_per_worker=256, total_concurrency=1024, pairs_per_worker={rank: len(values) for rank, values in shards.items()},
                 judge_functions_unchanged=True, original_worker_sha256=SHA(original.encode()),
                 worker_sha256=SHA(worker.read_bytes()), protocol_sha256=SHA((g / 'protocol.json').read_bytes()),
                 bundle_sha256=SHA((g / 'bundle.tar.gz').read_bytes()), publication_prefix=protocol['publication_prefix'])
    receipt.write_bytes(ENC(value))
    return value

if __name__ == '__main__':
    print(json.dumps(prepare(int(sys.argv[1]))), flush=True)
