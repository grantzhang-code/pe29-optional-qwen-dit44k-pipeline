#!/usr/bin/env python3
"""Portable e6d88249d08d checkpoint-8000 multi-reference PE pipeline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from contracts import LOCK, ROOT, digest, encode, file_sha, normalize_rows, require_s3, rows_for_rank, validate_prediction
from storage import client, get, publish, save, save_bytes, stage_model


def implementation_sha():
    return digest(encode({p.name: file_sha(p) for p in sorted(ROOT.glob('*.py'))}))


def load_run(work):
    work = Path(work)
    config = json.loads((work / 'run.json').read_text())
    content = dict(config)
    fingerprint = content.pop('run_sha256')
    if digest(encode(content)) != fingerprint:
        raise ValueError('Run configuration changed')
    if config['model_lock_sha256'] != file_sha(ROOT / 'model.lock.json') or config['implementation_sha256'] != implementation_sha():
        raise ValueError('Code or model lock changed; create a new run')
    if file_sha(work / 'inputs.jsonl') != config['input_sha256']:
        raise ValueError('Prepared input snapshot changed')
    rows = normalize_rows([json.loads(line) for line in (work / 'inputs.jsonl').read_bytes().splitlines()])
    if len(rows) != config['cases']:
        raise ValueError('Prepared input count changed')
    return config, rows


def freeze_run(work, rows, output, world_size, sampling, input_kind):
    output = require_s3(output).rstrip('/') + '/'
    if 'YOUR_' in output or output.startswith(LOCK['historical_output_uri']) or world_size < 1:
        raise ValueError('Use a new S3 output prefix and a positive world size')
    work = Path(work)
    if (work / 'ranks').exists() or (work / 'delivery').exists():
        raise ValueError('Inference already started; use a new work directory to change configuration')
    raw = b''.join(encode(row) for row in normalize_rows(rows))
    config = dict(version=1, run_id=LOCK['run_id'], checkpoint_step=8000,
        model_lock_sha256=file_sha(ROOT / 'model.lock.json'), implementation_sha256=implementation_sha(),
        input_sha256=digest(raw), input_kind=input_kind, cases=len(rows), world_size=world_size,
        sampling_temperatures=[0.0] if sampling == 'greedy' else LOCK['sampling_temperatures'],
        instruction_policy='original_user_instruction', reference_policy='all_ordered_reference_bytes',
        output_s3=output)
    config['run_sha256'] = digest(encode(config))
    if (work / 'run.json').exists() and json.loads((work / 'run.json').read_text()) != config:
        raise ValueError('Work directory is already bound to a different run')
    save_bytes(work / 'inputs.jsonl', raw)
    save(work / 'run.json', config)
    return config


def validate_rank(raw, marker, config, inputs, rank):
    expected = rows_for_rank(inputs, rank, config['world_size'])
    if (marker.get('run_sha256') != config['run_sha256'] or marker.get('rank') != rank
            or marker.get('world_size') != config['world_size'] or marker.get('rows') != len(expected)
            or marker.get('predictions_sha256') != digest(raw)):
        raise ValueError('Rank receipt identity/hash/coverage mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != len(expected):
        raise ValueError('Incomplete rank output')
    for record, source in zip(rows, expected, strict=True):
        validate_prediction(record, source, config)
        if not record['schema_valid'] and record['attempt'] != len(config['sampling_temperatures']) - 1:
            raise ValueError('Invalid row stopped before completing its retry budget')
    if marker['schema_valid'] != sum(r['schema_valid'] for r in rows):
        raise ValueError('Rank schema count mismatch')
    return rows


def aggregate(work, config, inputs, s3=None):
    rows = []
    for rank in range(config['world_size']):
        if s3 is None:
            out = Path(work) / f'ranks/rank-{rank:04d}'
            raw = (out / 'predictions.jsonl').read_bytes()
            marker = json.loads((out / 'COMPLETE.json').read_text())
        else:
            prefix = config['output_s3'] + f'caption/rank-{rank:04d}/'
            marker = json.loads(get(s3, prefix + 'COMPLETE.json'))
            raw = get(s3, prefix + 'predictions.jsonl')
        rows.extend(validate_rank(raw, marker, config, inputs, rank))
    rows.sort(key=lambda r: r['input_index'])
    if len(rows) != len(inputs) or len({r['sample_id'] for r in rows}) != len(inputs):
        raise ValueError('Merged coverage mismatch')
    out = Path(work) / 'delivery'
    for name, selected in [('predictions.jsonl', rows), ('valid-predictions.jsonl', [r for r in rows if r['schema_valid']]),
                           ('invalid-predictions.jsonl', [r for r in rows if not r['schema_valid']])]:
        save_bytes(out / name, b''.join(encode(r) for r in selected))
    valid = sum(r['schema_valid'] for r in rows)
    complete = dict(status='complete' if valid == len(rows) else 'completed_with_invalid_captions',
        run_sha256=config['run_sha256'], run_id=LOCK['run_id'], checkpoint_step=8000,
        checkpoint_uri=LOCK['checkpoint_uri'], model_uri=LOCK['model_uri'],
        model_export_manifest_sha256=LOCK['model_export_manifest_sha256'], prompt_sha256=LOCK['prompt_sha256'],
        cases=len(rows), json_valid=sum(r['json_valid'] for r in rows), schema_valid=valid,
        inference_errors=sum(bool(r['inference_error']) for r in rows), bbox_order='xyxy_1000',
        input_sha256=config['input_sha256'], world_size=config['world_size'],
        predictions_sha256=file_sha(out / 'predictions.jsonl'),
        files={name: file_sha(out / name) for name in ['predictions.jsonl', 'valid-predictions.jsonl', 'invalid-predictions.jsonl']},
        structural_repairs_applied=False)
    save(out / 'COMPLETE.json', complete)
    return complete


def publish_delivery(work, config, inputs, s3):
    out = Path(work) / 'delivery'
    complete = json.loads((out / 'COMPLETE.json').read_text())
    if complete['run_sha256'] != config['run_sha256'] or complete['cases'] != len(inputs):
        raise ValueError('Delivery binding changed')
    rows = [json.loads(line) for line in (out / 'predictions.jsonl').read_bytes().splitlines()]
    expected = sorted(inputs, key=lambda r: r['input_index'])
    if len(rows) != len(expected):
        raise ValueError('Delivery count changed')
    for row, source in zip(rows, expected, strict=True):
        validate_prediction(row, source, config)
    objects = []
    for name, sha in complete['files'].items():
        if Path(name).name != name or file_sha(out / name) != sha:
            raise ValueError('Delivery content changed')
        objects.append(publish(s3, config['output_s3'] + 'caption/' + name, (out / name).read_bytes()))
    for name in ['run.json', 'inputs.jsonl']:
        objects.append(publish(s3, config['output_s3'] + name, (Path(work) / name).read_bytes()))
    objects.append(publish(s3, config['output_s3'] + 'COMPLETE.json', (out / 'COMPLETE.json').read_bytes()))
    receipt = dict(status='PASS', method='S3 GET readback SHA256', objects=objects)
    publish(s3, config['output_s3'] + 'VERIFIED.json', encode(receipt))
    save(out / 'VERIFIED.json', receipt)
    return receipt


def run_node(args, config):
    gpus = args.gpus_per_node
    if gpus < 1 or config['world_size'] % gpus or not 0 <= args.node_index < config['world_size'] // gpus:
        raise ValueError('node index/GPU count do not match world size')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(gpus)]
    if len(devices) < gpus or any(not d or d == '-1' for d in devices):
        raise ValueError('Fewer assigned visible devices than requested GPU workers')
    processes, logs = [], []
    try:
        for gpu in range(gpus):
            rank = args.node_index * gpus + gpu
            log_path = args.work_root / f'logs/rank-{rank:04d}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open('a')
            logs.append(log)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'infer', '--work-root', str(args.work_root), '--rank', str(rank)]
            if args.profile:
                command.extend(['--profile', args.profile])
            if args.publish:
                command.append('--publish')
            processes.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': devices[gpu]},
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None, 0] for p in processes):
                raise RuntimeError('GPU worker failed; inspect rank logs')
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('GPU worker failed; inspect rank logs')
    finally:
        for process in processes:
            # Include each worker's vLLM descendants, even when its Python parent already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['prepare', 'stage-model', 'infer', 'run-node', 'aggregate', 'publish']:
        command = sub.add_parser(name)
        command.add_argument('--work-root', type=Path, required=True)
        command.add_argument('--profile', help='AWS profile; omitted uses the workload/default credential chain')
        if name == 'prepare':
            command.add_argument('--output-s3', required=True)
            command.add_argument('--world-size', type=int, default=1)
            command.add_argument('--input-jsonl', type=Path, help='Omit to prepare the locked 419-row MDS evaluation')
            command.add_argument('--sampling', choices=['bounded', 'greedy'], default='bounded')
        if name in ['infer', 'run-node']:
            command.add_argument('--publish', action='store_true', help='Publish completed ranks to the configured S3 prefix')
        if name == 'infer':
            command.add_argument('--rank', type=int, default=0)
        if name == 'run-node':
            command.add_argument('--node-index', type=int, required=True)
            command.add_argument('--gpus-per-node', type=int, default=8)
        if name == 'aggregate':
            command.add_argument('--from-s3', action='store_true')
    args = parser.parse_args()
    work = args.work_root.resolve()
    args.work_root = work
    if work == ROOT or work.is_relative_to(ROOT):
        raise ValueError('Use a work directory outside the source tree')
    if args.command == 'stage-model':
        stage_model(work, client(args.profile))
        print('MODEL_READY', LOCK['model_uri'])
    elif args.command == 'prepare':
        if (work / 'ranks').exists() or (work / 'delivery').exists():
            raise ValueError('Inference already started; use a new work directory')
        output = require_s3(args.output_s3).rstrip('/') + '/'
        if output.startswith(LOCK['historical_output_uri']) or 'YOUR_' in output or args.world_size < 1:
            raise ValueError('Use a new output prefix and a positive world size')
        from inputs import prepare_custom, prepare_mds
        s3 = client(args.profile)
        rows = prepare_custom(work, s3, args.input_jsonl) if args.input_jsonl else prepare_mds(work, s3)
        print(json.dumps(freeze_run(work, rows, output, args.world_size, args.sampling,
                                   'custom_jsonl' if args.input_jsonl else 'frozen_mds'), indent=2))
    else:
        config, rows = load_run(work)
        if args.command == 'infer':
            from inference import infer
            print(json.dumps(infer(work, config, rows, args.rank, client(args.profile) if args.publish else None), indent=2))
        elif args.command == 'run-node':
            run_node(args, config)
        elif args.command == 'aggregate':
            print(json.dumps(aggregate(work, config, rows, client(args.profile) if args.from_s3 else None), indent=2))
        elif args.command == 'publish':
            print(json.dumps(publish_delivery(work, config, rows, client(args.profile)), indent=2))


if __name__ == '__main__':
    main()
