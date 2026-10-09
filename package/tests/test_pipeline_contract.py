"""CPU-only regression tests using the complete frozen 1,464-case benchmark."""
import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'workflow/code')]
import configure_pipeline as configure
import infer_caption
from checkpoint_sources import load_sources, require_s3
from input_policy import ORIGINAL, QWEN, build_inputs, encode, sha, validate_instruction


class PipelineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = [json.loads(line) for line in (ROOT / 'workflow/inputs/original.jsonl').read_bytes().splitlines()]
        cls.qwen_raw = (ROOT / 'workflow/inputs/qwen-pe-predictions.jsonl').read_bytes()
        cls.predictions = [json.loads(line) for line in cls.qwen_raw.splitlines()]
        cls.uri = configure.BASE + 'inputs/qwen-pe-predictions.jsonl'

    def configure_fixture(self, enabled):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        work = root / 'workflow'
        work.mkdir()
        shutil.copytree(ROOT / 'workflow/inputs', work / 'inputs')
        for name in ['campaign.json', 'caption-source-contract.json', 'checkpoint-sources.json']:
            shutil.copy2(ROOT / 'workflow' / name, work / name)
        with patch.object(configure, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
            configure.main(['--output-s3', 's3://test/new-run', '--node-count', '3'] + (['--use-qwen-pe'] if enabled else []))
        config = json.loads((work / 'campaign.json').read_text())
        rows = [json.loads(line) for line in (work / 'inputs.jsonl').read_bytes().splitlines()]
        return work, config, rows

    def test_default_original_full_benchmark_reaches_real_pe_normalizer(self):
        work, config, rows = self.configure_fixture(False)
        self.assertFalse(config['use_qwen_pe'])
        self.assertIsNone(config['qwen_pe_predictions_sha256'])
        self.assertEqual(len(rows), 1464)
        with patch.object(infer_caption, 'TASK', work / 'code'):
            for source, row in zip(self.originals, rows):
                record = infer_caption.normalize_record(row, line_number=row['input_index'] + 1)
                self.assertEqual(record.instruction.encode(), source['instruction'].encode())
                self.assertEqual(record.selection['instruction_policy'], ORIGINAL)
                self.assertFalse(any(key.startswith('qwen_pe_') for key in record.selection))
        self.assertEqual(config['input_sha256'], sha((work / 'inputs.jsonl').read_bytes()))
        self.assertEqual([r['input_index'] for r in rows], [r['input_index'] for r in self.originals])

    def test_qwen_full_benchmark_reaches_real_pe_normalizer(self):
        work, config, rows = self.configure_fixture(True)
        by_index = {r['input_index']: r for r in self.predictions}
        self.assertTrue(config['use_qwen_pe'])
        with patch.object(infer_caption, 'TASK', work / 'code'):
            for source, row in zip(self.originals, rows):
                record = infer_caption.normalize_record(row, line_number=row['input_index'] + 1)
                self.assertEqual(record.instruction.encode(), by_index[row['input_index']]['rewritten_prompt'].encode())
                self.assertEqual(row['original_instruction'], source['instruction'])
                self.assertEqual(row['target_canvas_size'], source['target_canvas_size'])
                self.assertEqual(record.source_s3_uri, source['source_s3_uri'])
                self.assertEqual(record.selection['instruction_policy'], QWEN)

    def test_disabled_never_reads_qwen_predictions(self):
        original_read = Path.read_bytes
        def guarded(path):
            if path.name == 'qwen-pe-predictions.jsonl':
                raise AssertionError('Disabled Qwen must not read predictions')
            return original_read(path)
        with patch.object(Path, 'read_bytes', guarded):
            self.configure_fixture(False)

    def test_invalid_qwen_outputs_fail_instead_of_falling_back(self):
        valid = copy.deepcopy(self.predictions[:2])
        variants = [valid[:1], [valid[0], valid[0]], valid + [self.predictions[2]]]
        for key, value in [('rewritten_prompt', ''), ('rewritten_prompt', ' \n'), ('status', 'failed'),
                           ('sample_id', 'wrong'), ('instruction_en', 'wrong'), ('source_sha256', 'wrong'),
                           ('model_revision', 'wrong'), ('model_manifest_sha256', 'wrong'),
                           ('pe_target_canvas_size', {'width': 1, 'height': 1}),
                           ('attempts', [{'parse_ok': True, 'finish_reason': 'length'}])]:
            records = copy.deepcopy(valid)
            records[0][key] = value
            variants.append(records)
        for records in variants:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                build_inputs(self.originals[:2], use_qwen_pe=True, predictions=records,
                             predictions_sha=sha(self.qwen_raw), predictions_uri=self.uri)

    def test_new_frozen_qwen_sha_supported_and_mode_mismatch_rejected(self):
        predictions = copy.deepcopy(self.predictions[:1])
        predictions[0]['rewritten_prompt'] = '  Keep exact whitespace.\n'
        digest = sha(b''.join(encode(r) for r in predictions))
        rows = build_inputs(self.originals[:1], use_qwen_pe=True, predictions=predictions,
                            predictions_sha=digest, predictions_uri='s3://test/new-predictions.jsonl')
        config = dict(use_qwen_pe=True, instruction_policy=QWEN, qwen_pe_predictions_sha256=digest,
                      qwen_pe_predictions_uri='s3://test/new-predictions.jsonl')
        validate_instruction(rows[0], config)
        self.assertEqual(rows[0]['instruction'], '  Keep exact whitespace.\n')
        with self.assertRaises(ValueError):
            validate_instruction(rows[0], dict(use_qwen_pe=False, instruction_policy=ORIGINAL))
        rows[0]['instruction'] += 'tampered'
        with self.assertRaises(ValueError):
            validate_instruction(rows[0], config)

    def test_checkpoint_sources_reject_local_paths(self):
        self.assertEqual(len(load_sources()), 8)
        for value in ['/home/coder/model', '/mnt/ephemeral/model', './checkpoint', 'file:///tmp/ckpt',
                      'Qwen/Qwen-Image-2.1-PE-I2I', 'https://example.com/model', 's3://bucket',
                      's3://bucket/a/../model', 's3://bucket/model?version=1', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_s3(value)

    def test_cannot_switch_mode_in_existing_inference_work(self):
        work, _, _ = self.configure_fixture(False)
        (work / 'caption').mkdir()
        before = (work / 'inputs.jsonl').read_bytes()
        with patch.object(configure, 'ROOT', work.parent), self.assertRaises(ValueError):
            configure.main(['--output-s3', 's3://test/new-run', '--use-qwen-pe'])
        self.assertEqual(before, (work / 'inputs.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
