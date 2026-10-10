"""Fresh greedy generation followed by one rule-triggered sampling retry."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import sys


def run(w, work, selected, model, ready, step, arm, gpu_ids):
    sys.path.insert(0, str(w.WORKFLOW/'code'))
    import caption_support
    from input_policy import validate_instruction
    from repetition_rules import inspect, POLICY
    plan = w.CAMPAIGN['sampling']

    def one(slot):
        gpu = gpu_ids[slot]
        group = selected[slot::len(gpu_ids)]
        if not group:
            return []
        d = work/'caption'/f'gpu-{gpu}'
        d.mkdir(parents=True, exist_ok=True)
        full_inputs = d/'inputs.jsonl'
        w.jsonl(full_inputs, group)

        def attempt(mode, inputs):
            directory = d/mode
            directory.mkdir(exist_ok=True)
            source = directory/'inputs.jsonl'
            w.jsonl(source, inputs)
            predictions = directory/'predictions.jsonl'
            summary_path = directory/'summary.json'
            if not summary_path.exists():
                cmd = [str(w.PACKAGE/'assets/env/caption/bin/python'), '-u', str(w.WORKFLOW/'code/infer_caption.py'),
                       '--model', str(model), '--artifact-manifest-sha256', ready['export_manifest_sha256'],
                       '--checkpoint-step', str(step), '--checkpoint-manifest-sha256', arm['checkpoint']['checkpoint_manifest_sha256'],
                       '--input-jsonl', str(source), '--input-sha256', w.sha(source), '--expected-count', str(len(inputs)),
                       '--output-jsonl', str(predictions), '--summary-json', str(summary_path), '--node-index', '0', '--node-count', '1',
                       '--batch-size', '4', '--request-window-size', '16', '--download-workers', '8', '--max-model-len', '32768',
                       '--max-new-tokens', '27000', '--max-pixels', '2359296', '--gpu-memory-utilization', '0.90',
                       '--attention-backend', 'FLASHINFER', '--num-speculative-tokens', '1', '--seed', '42']
                env = caption_support.environment(gpu, gpu)
                env.update(AWS_PROFILE='core-cn-s3-readwrite', PE_SAMPLING_MODE=mode)
                w.command(cmd, directory/'run.log', env)
            summary = w.read(summary_path)
            values = w.rows(predictions)
            assert summary['complete'] and summary['results']['inference_errors'] == 0
            assert summary['input']['sha256'] == w.sha(source)
            assert summary['results']['predictions_sha256'] == w.sha(predictions)
            assert summary['checkpoint'] == dict(step=step, manifest_sha256=arm['checkpoint']['checkpoint_manifest_sha256'])
            assert len(values) == len(inputs)
            by = {r['sample_id']: r for r in inputs}
            for value in values:
                src = by[value['sample_id']]
                assert all(value[k] == src[k] for k in ['input_index', 'instruction', 'source_s3_uri', 'target_canvas_size', 'selection', 'original_instruction'])
                assert all(value['sampling'][k] == v for k, v in plan[mode].items())
                assert value['sampling']['sampling_mode'] == mode
                assert value['repetition_audit'] == inspect(value['response'])
                validate_instruction(value, w.CAMPAIGN)
            return values, summary

        greedy, greedy_summary = attempt('greedy', group)
        hits = {r['sample_id'] for r in greedy if r['repetition_audit']['hit']}
        retries, retry_summary = attempt('retry', [r for r in group if r['sample_id'] in hits]) if hits else ([], None)
        assert {r['sample_id'] for r in retries} == hits
        retried = {r['sample_id']: r for r in retries}
        final = []
        for first in greedy:
            second = retried.get(first['sample_id'])
            value = copy.deepcopy(second if second is not None else first)
            attempts = [first] + ([second] if second is not None else [])
            value['generation_policy'] = plan['strategy']
            value['selected_attempt'] = len(attempts)-1
            value['generation_attempts'] = [{
                'mode': r['sampling']['sampling_mode'], 'sampling': r['sampling'],
                'response_sha256': hashlib.sha256(r['response'].encode()).hexdigest(),
                'output_tokens': r['output_tokens'], 'finish_reason': r['finish_reason'],
                'json_valid': r['json_valid'], 'schema_valid': r['schema_valid'],
                'repetition_audit': r['repetition_audit'],
                'artifact': f"gpu-{gpu}/{r['sampling']['sampling_mode']}/predictions.jsonl",
            } for r in attempts]
            value['repetition_rejected'] = bool(value['repetition_audit']['hit'])
            final.append(value)
        w.jsonl(d/'predictions.jsonl', final)
        summary = copy.deepcopy(greedy_summary)
        summary.update(input={'path': str(full_inputs), 'sha256': w.sha(full_inputs), 'rows': len(group)},
                       sampling=plan, repetition_rules=POLICY,
                       stages={'greedy': greedy_summary, 'retry': retry_summary})
        summary['results'] = dict(rows=len(final), inference_errors=0,
                                 json_valid=sum(r['json_valid'] for r in final), schema_valid=sum(r['schema_valid'] for r in final),
                                 greedy_loop_hits=len(hits), retried=len(retries),
                                 unresolved_loop_cases=sum(r['repetition_rejected'] for r in final),
                                 predictions_sha256=w.sha(d/'predictions.jsonl'), predictions_bytes=(d/'predictions.jsonl').stat().st_size)
        w.save(d/'summary.json', summary)
        return final

    with ThreadPoolExecutor(len(gpu_ids)) as pool:
        values = [row for part in pool.map(one, range(len(gpu_ids))) for row in part]
    values.sort(key=lambda r: r['input_index'])
    w.jsonl(work/'caption/predictions.jsonl', values)
    for mode in ['greedy','retry']:
        all_rows=[r for path in sorted((work/'caption').glob(f'gpu-*/{mode}/predictions.jsonl')) for r in w.rows(path)]
        w.jsonl(work/'caption'/f'{mode}-predictions.jsonl',sorted(all_rows,key=lambda r:r['input_index']))
    return values
