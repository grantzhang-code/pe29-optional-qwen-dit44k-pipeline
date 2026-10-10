"""Audit frozen GSB judgments and exclude the user's cancelled incomplete pair.

No judge calls are made. The original protocol, code, and raw judgments remain
unchanged. PASS applies only to the explicitly reduced, complete AB/BA cohort.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
import argparse, ast, csv, hashlib, io, json, tarfile
import boto3
from botocore.config import Config

ENC = lambda value: (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda path: json.loads(path.read_bytes())
rows = lambda path: [json.loads(line) for line in path.read_bytes().splitlines() if line]
DIMS = ('consistency', 'quality')
utc = lambda: datetime.now(timezone.utc).isoformat()


def main(root, authorization):
    protocol_raw = (root / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    digest = SHA(protocol_raw)
    scope = read(authorization)
    assert scope['request'] == '丢掉最后一条请求'
    cutoff = datetime.fromisoformat(scope['utc'])
    assert protocol['pairs_count'] == 1258 and protocol['tasks_count'] == 2516
    assert protocol['workers'] == 4 and protocol['concurrency_per_worker'] == 256
    for name, expected in protocol['input_hashes'].items():
        assert SHA((root / 'inputs' / name).read_bytes()) == expected, name
    for name, expected in protocol['code_hashes'].items():
        assert SHA((root / name).read_bytes()) == expected, name
    tasks = {r['task_id']: r for r in rows(root / 'inputs/tasks.jsonl')}
    pairs = {r['pair_id']: r for r in rows(root / 'inputs/pairs.jsonl')}
    assert len(tasks) == 2516 and len(pairs) == 1258
    assert all(p['checkpoint_step'] == 3000 and p['editor_checkpoint_step'] == 6000 for p in pairs.values())

    s3 = boto3.Session(profile_name='core-cn-s3-readwrite').client('s3', config=Config(max_pool_connections=48))
    base = urlsplit(protocol['publication_prefix'])
    prefix = base.path.lstrip('/')
    result_prefix = prefix + 'judgments/results/'
    objects = [o for page in s3.get_paginator('list_objects_v2').paginate(Bucket=base.netloc, Prefix=result_prefix)
               for o in page.get('Contents', [])]
    local = root / 'partial-results'
    (local / 'raw').mkdir(parents=True, exist_ok=True)

    def fetch(obj):
        tid = Path(obj['Key']).stem
        assert tid in tasks
        response = s3.get_object(Bucket=base.netloc, Key=obj['Key'])
        raw = response['Body'].read()
        assert SHA(raw) == response['Metadata']['sha256']
        (local / 'raw' / (tid + '.json')).write_bytes(raw)
        return tid, json.loads(raw)

    with ThreadPoolExecutor(max_workers=32) as executor:
        records = dict(executor.map(fetch, objects))
    assert len(records) == len(objects)
    accepted = {tid: r for tid, r in records.items() if datetime.fromisoformat(r['completed_utc']) <= cutoff}
    missing = set(tasks) - set(accepted)
    assert len(missing) == 1 and len(accepted) == 2515, (len(missing), len(accepted))
    missing_tid = next(iter(missing))
    excluded_pid = tasks[missing_tid]['pair_id']
    excluded_pair = pairs[excluded_pid]
    included_pairs = {pid: p for pid, p in pairs.items() if pid != excluded_pid}
    used = {tid: r for tid, r in accepted.items() if r['pair_id'] != excluded_pid}
    assert len(included_pairs) == 1257 and len(used) == 2514
    rubric = read(root / 'inputs/gsb_rubric.json')
    normalized = {}
    orders = {}
    # Audit all saved judgments, including the orphan retained outside scoring.
    for tid, r in records.items():
        task = tasks[tid]
        assert r['status'] == 'complete' and r['task'] == task and r['protocol_sha256'] == digest
        cfg = r['request_config']
        assert cfg['system_instruction'] == rubric['SYSTEM']
        assert cfg['temperature'] == 0 and cfg['response_mime_type'] == 'application/json'
        assert cfg['thinking_config']['thinking_level'] == 'LOW'
        assert cfg['max_output_tokens'] == protocol['max_output_tokens']
        assert r['model_requested'] == protocol['model']
        answer = json.loads(r['attempts'][-1]['response_text'])
        mapping = {'1': task['first'], '2': task['second'], 'tie': 'tie'}
        assert r['winners'] == {d: mapping[answer[d + '_winner']] for d in DIMS}
        assert r['reasons'] == {d: answer[d + '_reason'] for d in DIMS}
        for key, image in [('source', task['source']), ('first', task['candidates'][task['first']]),
                           ('second', task['candidates'][task['second']])]:
            m = r['judge_' + key]
            assert m['original_sha256'] == image['sha256'] and m['original_size'] == image['size']
            assert m['judge_size'][0] * m['judge_size'][1] <= protocol['pixel_budget']
            assert m['mime_type'] == 'image/webp'
            identity = (m['normalized_sha256'], tuple(m['judge_size']))
            previous = normalized.setdefault(m['original_sha256'], identity)
            assert previous == identity
        if tid in used:
            orders.setdefault(r['pair_id'], {})[r['order']] = r
    assert set(orders) == set(included_pairs)
    assert all(set(v) == {'ab', 'ba'} for v in orders.values())

    # Execute the exact combine and metrics functions from the frozen source.
    namespace = {'PROTOCOL': protocol, 'DIMS': DIMS, 'utc': utc, 'Counter': Counter}
    for filename, function in [('code/worker.py', 'combine'), ('finalize.py', 'metrics')]:
        tree = ast.parse((root / filename).read_text())
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == function)
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    scores = []
    for pid, pair_orders in orders.items():
        pair = included_pairs[pid]
        result = namespace['combine'](pair_orders['ab'], pair_orders['ba'])
        assert all(result[k] == pair[k] for k in ('input_index', 'sample_id', 'group', 'pe_arm'))
        for d in DIMS:
            a, b = (pair_orders[o]['winners'][d] for o in ['ab', 'ba'])
            expected = 'similar' if a != b or a == 'tie' else ('good' if a == pair['pe_arm'] else 'bad')
            assert result['dimensions'][d]['outcome'] == expected
        scores.append(dict(input_index=pair['input_index'], sample_id=pair['sample_id'], group=pair['group'],
            cfg=pair['pe_cfg'], pe_arm=pair['pe_arm'], instruction_en=pair['instruction_en'],
            checkpoint_step=pair['checkpoint_step'], caption_run=pair['caption_run'],
            candidate_uri=pair['candidates'][pair['pe_arm']]['uri'],
            control_uri=pair['candidates'][protocol['control_arm']]['uri'], source_uri=pair['source']['uri'],
            dimensions=result['dimensions'], order_reasons={order: dict(first=r['first'], second=r['second'],
                reasons=r['reasons'], record_uri=r['record_uri']) for order, r in pair_orders.items()}))
    scores.sort(key=lambda r: (r['pe_arm'], r['input_index']))
    metrics = namespace['metrics']
    comparisons = {}
    for arm in protocol['pe_arms']:
        cohort = [r for r in scores if r['pe_arm'] == arm]
        comparisons[arm] = dict(checkpoint_step=3000, caption_run='39c59fafbc35', label=protocol['arm_labels'][arm],
            cases=len(cohort), metrics={d: metrics(cohort, d) for d in DIMS},
            by_task={g: {d: metrics([r for r in cohort if r['group'] == g], d) for d in DIMS}
                     for g in sorted({r['group'] for r in cohort})})
    exclusion = dict(status='user_requested_exclusion', authorization=scope, original_protocol_sha256=digest,
        expected_pairs=1258, expected_judgments=2516, scored_pairs=1257, scored_judgments=2514,
        saved_judgments=len(records), completed_before_stop=len(accepted), omitted_orphan_judgments=1,
        discarded_request=missing_tid, pair_id=excluded_pid, sample_id=excluded_pair['sample_id'],
        input_index=excluded_pair['input_index'], missing_order=tasks[missing_tid]['order'],
        reason='User requested dropping the final outstanding request; exclude its entire incomplete AB/BA pair.',
        late_judgments_ignored=sorted(set(records) - set(accepted)), finalizer_sha256=SHA(Path(__file__).read_bytes()))
    versions = dict(Counter(r['model_version'] for r in used.values()))
    summary = dict(status='complete_with_exclusion', full_run_complete=False, pairs=1257, judgments=2514,
        requested_pairs=1258, requested_judgments=2516, completed_judgments=len(accepted),
        excluded_scoring_cases=1, omitted_orphan_judgments=1, cancelled_judgments=1,
        protocol_sha256=digest, workers=4, concurrency_per_worker=256, total_concurrency=1024,
        judge=protocol['model'], thinking_level='LOW', temperature=0, model_versions=versions,
        comparisons=comparisons, all_models_common=comparisons, common_cases=1257,
        completed_utc=utc(), publication_prefix=protocol['publication_prefix'],
        control='Frozen Qwen-Image 2.1 CFG1, 40 steps, seed=42+input_index',
        generation='PE39c59 checkpoint3000 + Qwen-PE -> DiT step6000 EMA463, CFG2.5, 50 steps, native2K, timeshift4.6',
        normalization=protocol['normalization'], tie_rule=protocol['tie_rule'],
        comparison_meaning='Complete AB/BA pairs only versus frozen Qwen controls; one incomplete pair excluded by user request.',
        original_job_status='cancelled_by_user_request', exclusion_uri=protocol['publication_prefix']+'EXCLUSIONS.json')
    artifacts = local / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    (artifacts / 'scores.jsonl').write_bytes(b''.join(ENC(r) for r in scores))
    (artifacts / 'summary.json').write_bytes(ENC(summary))
    (artifacts / 'EXCLUSIONS.json').write_bytes(ENC(exclusion))
    with (artifacts / 'metrics.csv').open('w') as output:
        fields = ['scope', 'arm', 'checkpoint_step', 'dimension', 'n', 'good', 'similar', 'bad',
                  'net_win_rate', 'win_rate_excluding_similar', 'position_disagreements']
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for arm, value in comparisons.items():
            for dimension, m in value['metrics'].items():
                writer.writerow(dict(scope='user_authorized_complete_pairs', arm=arm, checkpoint_step=3000, dimension=dimension, **m))
    verified = dict(status='PASS', scope='user_authorized_complete_pairs_only', full_run_complete=False,
        pairs=1257, judgments=2514, original_expected_pairs=1258, original_expected_judgments=2516,
        raw_judgments_audited=len(records), excluded_scoring_cases=1, orphan_judgments_excluded=1,
        protocol_sha256=digest, model_versions=versions, normalized_images_verified=len(normalized),
        scores_sha256=SHA((artifacts/'scores.jsonl').read_bytes()), summary_sha256=SHA((artifacts/'summary.json').read_bytes()),
        metrics_sha256=SHA((artifacts/'metrics.csv').read_bytes()), exclusions_sha256=SHA((artifacts/'EXCLUSIONS.json').read_bytes()),
        finalizer_sha256=exclusion['finalizer_sha256'], completed_utc=summary['completed_utc'])
    (artifacts / 'VERIFIED.json').write_bytes(ENC(verified))
    (artifacts / 'protocol.json').write_bytes(protocol_raw)

    def publish(name, raw):
        value = SHA(raw)
        s3.put_object(Bucket=base.netloc, Key=prefix+name, Body=raw, Metadata={'sha256': value})
        returned = s3.get_object(Bucket=base.netloc, Key=prefix+name)
        assert SHA(returned['Body'].read()) == value == returned['Metadata']['sha256']
        return dict(sha256=value, bytes=len(raw))

    publication = {p.name: publish(p.name, p.read_bytes()) for p in artifacts.iterdir() if p.is_file()}
    code_name = 'execution/partial-finalize-' + exclusion['finalizer_sha256'] + '.py'
    publish(code_name, Path(__file__).read_bytes())
    publication_proof = dict(status='PASS', scope=verified['scope'], full_run_complete=False, artifacts=publication)
    (artifacts / 'PUBLICATION_VERIFIED.json').write_bytes(ENC(publication_proof))
    publish('PUBLICATION_VERIFIED.json', ENC(publication_proof))
    publish('execution/status.json', ENC(dict(stage='complete_with_exclusion', original_job_status='cancelled_by_user_request',
        utc=utc(), pairs=1257, scored_judgments=2514, saved_judgments=len(records), excluded_scoring_cases=1)))
    archive = local / 'artifacts.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(artifacts.iterdir()):
            tar.add(p, arcname=p.name, recursive=False)
    raw = archive.read_bytes()
    print(json.dumps(dict(status='PASS', scope=verified['scope'], pairs=1257, judgments=2514,
        comparisons={k:v['metrics'] for k,v in comparisons.items()},
        archive=dict(path=str(archive), bytes=len(raw), sha256=SHA(raw)))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.authorization)
