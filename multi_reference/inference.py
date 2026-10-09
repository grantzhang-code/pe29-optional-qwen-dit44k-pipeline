"""One vLLM engine per assigned GPU; bounded, audited schema retries."""
import fcntl
import json
import os
from pathlib import Path

from contracts import LOCK, build_request, encode, file_sha, parse_response, rows_for_rank, sample_seed, validate_prediction
from storage import publish, save, save_bytes, verify_local_model


def infer(work, config, sources, rank, s3=None):
    work = Path(work)
    rows = rows_for_rank(sources, rank, config['world_size'])
    out = work / f'ranks/rank-{rank:04d}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'writer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = {}
        by_id = {row['sample_id']: row for row in rows}
        path = out / 'predictions.jsonl'
        if path.exists():
            for line in path.read_bytes().splitlines():
                record = json.loads(line)
                sid = record['sample_id']
                if sid not in by_id or sid in selected:
                    raise ValueError('Duplicate or unrelated prediction during resume')
                validate_prediction(record, by_id[sid], config)
                selected[sid] = record
        if (out / 'COMPLETE.json').exists():
            complete = json.loads((out / 'COMPLETE.json').read_text())
            if (complete['run_sha256'] != config['run_sha256'] or complete['rows'] != len(rows)
                    or complete['predictions_sha256'] != file_sha(path) or len(selected) != len(rows)):
                raise ValueError('Completed rank changed')
            if s3 is not None:
                publish_rank(out, config, rank, s3)
            return complete
        # Completed invalid rows remain invalid after the configured final attempt.
        pending = [r for r in rows if r['sample_id'] not in selected or
                   (not selected[r['sample_id']]['schema_valid'] and
                    selected[r['sample_id']]['attempt'] + 1 < len(config['sampling_temperatures']))]
        if pending:
            # Each process independently validates actual model bytes, including trained MTP/FP32 shards.
            model = verify_local_model(work)
            configure_environment(work, rank)
            from vllm import LLM, SamplingParams
            kwargs = dict(model=str(model), tokenizer=str(model), trust_remote_code=True,
                dtype='bfloat16', tensor_parallel_size=1, max_model_len=LOCK['max_model_len'],
                max_num_seqs=4, max_num_batched_tokens=32768, gpu_memory_utilization=0.82,
                limit_mm_per_prompt={'image': max(len(r['reference_images']) for r in rows), 'video': 0},
                mm_processor_kwargs={'max_pixels': LOCK['max_pixels']}, skip_mm_profiling=True,
                attention_backend='FLASHINFER', seed=42, enforce_eager=True,
                speculative_config={'method': 'mtp', 'num_speculative_tokens': 1})
            # Durable provenance uses S3; kwargs use the verified runtime cache only.
            save(out / 'engine.json', {**kwargs, 'model': LOCK['model_uri'], 'tokenizer': LOCK['model_uri'],
                                      'run_sha256': config['run_sha256']})
            engine = LLM(**kwargs)
            tokenizer = engine.get_tokenizer()
            spec = engine.llm_engine.vllm_config.speculative_config
            if spec is None or spec.method != 'mtp':
                raise ValueError('Trained MTP was not enabled')
            with (out / 'attempts.jsonl').open('ab', buffering=0) as attempts:
                for attempt, temperature in enumerate(config['sampling_temperatures']):
                    batch_rows = [r for r in rows if r['sample_id'] not in selected or
                                  (not selected[r['sample_id']]['schema_valid'] and selected[r['sample_id']]['attempt'] < attempt)]
                    for start in range(0, len(batch_rows), 8):
                        batch = batch_rows[start:start + 8]
                        requests = [build_request(row, tokenizer, work) for row in batch]
                        params = [SamplingParams(temperature=temperature, top_p=1.0 if attempt == 0 else 0.95,
                                  top_k=-1, max_tokens=LOCK['max_new_tokens'], seed=sample_seed(r['input_index'], attempt)) for r in batch]
                        try:
                            outputs = engine.generate(requests, params, use_tqdm=False)
                        except Exception:
                            outputs = []
                            for request, param in zip(requests, params, strict=True):
                                try:
                                    outputs.append(engine.generate([request], [param], use_tqdm=False)[0])
                                except Exception as exc:
                                    outputs.append(exc)
                        for row, result in zip(batch, outputs, strict=True):
                            if isinstance(result, Exception):
                                text, finish, tokens, inference_error = '', None, 0, repr(result)
                            else:
                                candidate = result.outputs[0]
                                text, finish, tokens = candidate.text, candidate.finish_reason, len(candidate.token_ids)
                                inference_error = None
                            parsed, error, schema_error = parse_response(text, len(row['reference_images']))
                            if finish == 'length':
                                schema_error = 'Generation reached token/context limit'
                            record = {**row, 'run_sha256': config['run_sha256'], 'caption_run': LOCK['run_id'],
                                'checkpoint_step': 8000, 'checkpoint_manifest_sha256': LOCK['checkpoint_manifest_sha256'],
                                'model_export_manifest_sha256': LOCK['model_export_manifest_sha256'],
                                'prompt_sha256': LOCK['prompt_sha256'], 'bbox_order': 'xyxy_1000',
                                'attempt': attempt, 'sampling_temperature': temperature,
                                'sampling_seed': sample_seed(row['input_index'], attempt),
                                'response': text, 'json_caption': parsed, 'json_valid': error is None,
                                'schema_valid': error is None and schema_error is None and inference_error is None,
                                'json_error': error, 'schema_error': schema_error, 'inference_error': inference_error,
                                'output_tokens': tokens, 'finish_reason': finish}
                            validate_prediction(record, row, config)
                            attempts.write(encode(record))
                            selected[row['sample_id']] = record
                        save_bytes(path, b''.join(encode(selected[r['sample_id']]) for r in rows if r['sample_id'] in selected))
                        print('RANK_PROGRESS', rank, 'attempt', attempt, 'written', len(selected), flush=True)
        if len(selected) != len(rows):
            raise ValueError('Rank coverage is incomplete')
        save_bytes(path, b''.join(encode(selected[r['sample_id']]) for r in rows))
        complete = dict(run_sha256=config['run_sha256'], rank=rank, world_size=config['world_size'], rows=len(rows),
                        schema_valid=sum(r['schema_valid'] for r in selected.values()), predictions_sha256=file_sha(path))
        save(out / 'COMPLETE.json', complete)
        if s3 is not None:
            publish_rank(out, config, rank, s3)
        return complete


def publish_rank(out, config, rank, s3):
    prefix = config['output_s3'] + f'caption/rank-{rank:04d}/'
    for name in ['predictions.jsonl', 'attempts.jsonl', 'engine.json', 'COMPLETE.json']:
        path = out / name
        if path.exists():
            publish(s3, prefix + name, path.read_bytes())


def configure_environment(work, rank):
    import sys
    cuda = Path(sys.prefix) / 'lib/python3.12/site-packages/nvidia/cu13'
    if not cuda.is_dir():
        raise ValueError('Use the restored caption Python environment containing nvidia/cu13')
    for link, target in [(cuda / 'lib64', 'lib'), (cuda / 'lib/stubs', '/usr/local/cuda/lib64/stubs')]:
        if not link.exists():
            try:
                link.symlink_to(target)
            except FileExistsError:
                pass  # Another local GPU worker restored the same compatibility alias.
    cache = Path(work) / f'cache/rank-{rank:04d}'
    cache.mkdir(parents=True, exist_ok=True)
    from contracts import digest
    tmp = Path('/tmp') / ('mref-' + digest(str(Path(work).resolve()).encode())[:10] + f'-r{rank}')
    tmp.mkdir(parents=True, exist_ok=True)
    settings = dict(CUDA_HOME=str(cuda), PATH=str(cuda / 'bin') + os.pathsep + os.environ.get('PATH', ''),
        TOKENIZERS_PARALLELISM='false', VLLM_USE_DEEP_GEMM='0', VLLM_MOE_USE_DEEP_GEMM='0',
        VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_NO_USAGE_STATS='1', VLLM_ENABLE_CUDA_COMPATIBILITY='0',
        TORCH_CUDA_ARCH_LIST='9.0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        VLLM_CACHE_ROOT=str(cache), XDG_CACHE_HOME=str(cache),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'), TMPDIR=str(tmp))
    os.environ.update(settings)
