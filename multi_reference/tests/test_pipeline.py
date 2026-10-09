import copy
import gzip
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contracts as c
import inference
import inputs
import pipeline
import storage


class MultiReferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in (ROOT / 'tests/fixtures/historical-inputs.jsonl').read_bytes().splitlines()]
        with gzip.open(ROOT / 'tests/fixtures/historical-responses.jsonl.gz', 'rt') as stream:
            cls.responses = [json.loads(line) for line in stream]
        cls.valid = next(row for row in cls.responses if row['schema_valid'] and row['num_references'] == 2)

    def work(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def small_rows(self, work):
        from PIL import Image
        refs = []
        for i, color in enumerate(['red', 'green']):
            stream = io.BytesIO()
            Image.new('RGB', (16 + i, 20), color).save(stream, format='PNG')
            raw = stream.getvalue()
            ref = dict(reference_id=f'frame_{i}', source_uri=f's3://test/ref-{i}.png', sha256=c.digest(raw), width=16 + i, height=20)
            storage.save_bytes(c.reference_path(work, ref), raw)
            refs.append(ref)
        return [dict(input_index=i, sample_id=f'case-{i}', instruction='Combine image 1 and image 2; literal <image> is text.',
                     reference_images=copy.deepcopy(refs), target_canvas_size=dict(width=1024, height=768)) for i in range(2)]

    def prediction(self, source, config, response, valid):
        parsed, error, schema_error = c.parse_response(response, len(source['reference_images']))
        attempt = 0 if valid else 5
        return {**source, 'run_sha256': config['run_sha256'], 'caption_run': c.LOCK['run_id'], 'checkpoint_step': 8000,
            'checkpoint_manifest_sha256': c.LOCK['checkpoint_manifest_sha256'],
            'model_export_manifest_sha256': c.LOCK['model_export_manifest_sha256'],
            'prompt_sha256': c.LOCK['prompt_sha256'], 'bbox_order': 'xyxy_1000',
            'attempt': attempt, 'sampling_temperature': config['sampling_temperatures'][attempt],
            'sampling_seed': c.sample_seed(source['input_index'], attempt), 'response': response,
            'json_caption': parsed, 'json_valid': error is None, 'schema_valid': error is None and schema_error is None,
            'json_error': error, 'schema_error': schema_error, 'inference_error': None, 'finish_reason': 'stop', 'output_tokens': 1}

    def test_all_419_inputs_keep_reference_order_and_identity(self):
        rows = c.normalize_rows(self.rows)
        self.assertEqual(len(rows), 419)
        self.assertEqual({len(r['reference_images']) for r in rows}, set(range(1, 13)))
        for row in rows:
            self.assertEqual([r['reference_id'] for r in row['reference_images']], [f'frame_{i}' for i in range(len(row['reference_images']))])
            prompt = c.PROMPT.build_pe_prompt(row['instruction'], len(row['reference_images']),
                                            (row['target_canvas_size']['width'], row['target_canvas_size']['height']))
            self.assertEqual(prompt.count('<image>'), len(row['reference_images']))

    def test_all_419_historical_responses_reproduce_417_valid(self):
        total = 0
        for row in self.responses:
            _, error, schema_error = c.parse_response(row['response'], row['num_references'])
            self.assertEqual(error is None, row['json_valid'], row['input_index'])
            self.assertEqual(error is None and schema_error is None, row['schema_valid'], row['input_index'])
            total += error is None and schema_error is None
        self.assertEqual(total, 417)

    def test_target_leakage_and_frame_reordering_are_rejected(self):
        for key in ['output_image', 'target_image', 'target_s3_uri', 'json_caption', 'instructions_detailed']:
            row = copy.deepcopy(self.rows[0]); row[key] = 'forbidden'
            with self.subTest(key=key), self.assertRaises(ValueError): c.normalize_rows([row])
        row = copy.deepcopy(self.rows[0]); row['reference_images'].reverse()
        with self.assertRaises(ValueError): c.normalize_rows([row])
        with self.assertRaises(ValueError): c.normalize_rows([self.rows[0], self.rows[0]])

    def test_s3_sources_only(self):
        for uri in ['/home/coder/model', '/tmp/ckpt', 'file:///tmp/model', 's3://bucket', 's3://bucket/../model']:
            with self.subTest(uri=uri), self.assertRaises(ValueError): c.require_s3(uri)

    def test_multiimage_request_preserves_pixels_and_excludes_target(self):
        work = self.work(); row = self.small_rows(work)[0]
        captured = {}
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured.update(messages=messages, kwargs=kwargs)
                return 'training-template<think>\n\n</think>\n\n'
        request = c.build_request(row, Tokenizer(), work)
        images = request['multi_modal_data']['image']
        self.assertEqual([im.size for im in images], [(16, 20), (17, 20)])
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(images[1].getpixel((0, 0)), (0, 128, 0))
        content = captured['messages'][1]['content']
        self.assertEqual(sum(part['type'] == 'image' for part in content), 2)
        self.assertFalse(captured['kwargs']['enable_thinking'])
        texts = ''.join(part.get('text', '') for part in content)
        self.assertIn('\\u003cimage\\u003e', texts)
        self.assertIn('1024x768', texts)
        c.reference_path(work, row['reference_images'][0]).write_bytes(b'corrupt')
        with self.assertRaises(ValueError): c.build_request(row, Tokenizer(), work)

    def test_frozen_sharding_covers_each_sample_once(self):
        for size in [1, 8, 64, 512]:
            groups = [c.rows_for_rank(self.rows, rank, size) for rank in range(size)]
            flattened = [r['input_index'] for group in groups for r in group]
            self.assertEqual(sorted(flattened), sorted(r['input_index'] for r in self.rows))
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_aggregate_all_419_across_64_shards(self):
        work = self.work()
        config = pipeline.freeze_run(work, self.rows, 's3://test/aggregate', 64, 'bounded', 'fixture')
        responses = {r['input_index']: r for r in self.responses}
        for rank in range(64):
            records = [self.prediction(row, config, responses[row['input_index']]['response'], responses[row['input_index']]['schema_valid'])
                       for row in c.rows_for_rank(self.rows, rank, 64)]
            raw = b''.join(c.encode(r) for r in records)
            out = work / f'ranks/rank-{rank:04d}'
            storage.save_bytes(out / 'predictions.jsonl', raw)
            storage.save(out / 'COMPLETE.json', dict(rank=rank, world_size=64, run_sha256=config['run_sha256'],
                rows=len(records), schema_valid=sum(r['schema_valid'] for r in records), predictions_sha256=c.digest(raw)))
        complete = pipeline.aggregate(work, config, self.rows)
        self.assertEqual((complete['cases'], complete['schema_valid']), (419, 417))
        self.assertEqual(len((work / 'delivery/invalid-predictions.jsonl').read_bytes().splitlines()), 2)
        out = work / 'ranks/rank-0000'
        marker = json.loads((out / 'COMPLETE.json').read_text()); marker['run_sha256'] = 'wrong'
        storage.save(out / 'COMPLETE.json', marker)
        with self.assertRaises(ValueError): pipeline.aggregate(work, config, self.rows)

    def test_resume_rejects_changed_input_or_configuration(self):
        work = self.work(); rows = self.small_rows(work)
        pipeline.freeze_run(work, rows, 's3://test/resume', 1, 'greedy', 'fixture')
        pipeline.load_run(work)
        altered = copy.deepcopy(rows); altered[0]['instruction'] = 'different'
        storage.save_bytes(work / 'inputs.jsonl', b''.join(c.encode(r) for r in altered))
        with self.assertRaises(ValueError): pipeline.load_run(work)
        with self.assertRaises(ValueError): pipeline.freeze_run(work, rows, 's3://test/different', 1, 'greedy', 'fixture')

    def test_cpu_mocked_inference_retries_and_resumes_without_new_model_calls(self):
        work = self.work(); rows = self.small_rows(work)
        config = pipeline.freeze_run(work, rows, 's3://test/infer', 1, 'bounded', 'fixture')
        valid_response = self.valid['response']
        calls = []
        class Engine:
            def __init__(self, **kwargs):
                self.llm_engine = types.SimpleNamespace(vllm_config=types.SimpleNamespace(speculative_config=types.SimpleNamespace(method='mtp')))
            def get_tokenizer(self):
                return types.SimpleNamespace(apply_chat_template=lambda *a, **k: '<think>\n\n</think>\n\n')
            def generate(self, requests, params, **kwargs):
                calls.append(params)
                text = 'invalid JSON' if len(calls) == 1 else valid_response
                candidate = types.SimpleNamespace(text=text, finish_reason='stop', token_ids=[1])
                return [types.SimpleNamespace(outputs=[candidate]) for _ in requests]
        fake = types.SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw)
        with patch.dict(sys.modules, {'vllm': fake}), patch.object(inference, 'verify_local_model', return_value=work / 'model'), patch.object(inference, 'configure_environment'):
            complete = inference.infer(work, config, rows, 0)
            self.assertEqual(complete['schema_valid'], 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0]['temperature'], 0.0)
            self.assertEqual(calls[1][0]['temperature'], 0.2)
            repeated = inference.infer(work, config, rows, 0)
            self.assertEqual(repeated, complete)
            self.assertEqual(len(calls), 2)

    def test_empty_rank_needs_no_model(self):
        work = self.work(); rows = self.small_rows(work)
        config = pipeline.freeze_run(work, rows, 's3://test/empty', 8, 'greedy', 'fixture')
        with patch.object(inference, 'verify_local_model', side_effect=AssertionError('No model needed')):
            complete = inference.infer(work, config, rows, 7)
        self.assertEqual(complete['rows'], 0)

    def test_restricted_mds_unpickler_rejects_classes(self):
        with self.assertRaises(ValueError):
            inputs.RestrictedUnpickler(io.BytesIO(pickle.dumps(Path('/tmp/not-executed')))).load()

    def test_custom_s3_inputs_validate_downloaded_bytes(self):
        work = self.work(); rows = self.small_rows(work)
        payloads = {ref['source_uri'].split('s3://test/')[1]: c.reference_path(work, ref).read_bytes() for ref in rows[0]['reference_images']}
        for ref in rows[0]['reference_images']: c.reference_path(work, ref).unlink()
        path = work / 'source.jsonl'; storage.save_bytes(path, b''.join(c.encode(r) for r in rows))
        class S3:
            def download_file(self, bucket, key, dest): Path(dest).write_bytes(payloads[key])
        self.assertEqual(inputs.prepare_custom(work, S3(), path), rows)
        payloads['ref-0.png'] = b'wrong bytes'
        c.reference_path(work, rows[0]['reference_images'][0]).unlink()
        with self.assertRaises(ValueError): inputs.prepare_custom(work, S3(), path)


if __name__ == '__main__':
    unittest.main()
