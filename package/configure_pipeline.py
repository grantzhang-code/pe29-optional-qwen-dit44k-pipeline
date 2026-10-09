"""Freeze one input mode before starting PE-29k or native editor work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/code'))
from checkpoint_sources import load_sources, require_s3
from input_policy import ORIGINAL, QWEN, build_inputs, encode, sha, validate_instruction

BASE = 's3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/9c4d29-qwenrewrite-dit44k-fixed-20261007/'
FROZEN_QWEN_SHA = '528ee56b1090a2c4841a661057cf863778618e18f10a2a7b6a20393828dd56c2'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-qwen-pe', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--qwen-predictions', type=Path)
    parser.add_argument('--qwen-predictions-uri')
    parser.add_argument('--output-s3', required=True)
    parser.add_argument('--node-count', type=int, default=1)
    args = parser.parse_args(argv)
    work = ROOT / 'workflow'
    for name in ['work', 'caption', 'admission', 'partition.json']:
        if (work / name).exists():
            raise ValueError('Existing inference state: extract the package into a fresh run directory')
    output = require_s3(args.output_s3).rstrip('/') + '/'
    if 'YOUR_' in output or output.startswith(BASE) or args.node_count < 1:
        raise ValueError('Use a new output prefix and a positive node count')
    sources = load_sources(work / 'checkpoint-sources.json')
    original_raw = (work / 'inputs/original.jsonl').read_bytes()
    if sha(original_raw) != 'dcadb3cde9b3d37d45008cc89aa6b5057201ab051fcdc1804a756181cdab8f03':
        raise ValueError('Frozen benchmark input changed; adapt the dataset contract explicitly')
    originals = [json.loads(line) for line in original_raw.splitlines() if line]
    predictions, digest, uri = None, None, None
    if args.use_qwen_pe:
        path = args.qwen_predictions or work / 'inputs/qwen-pe-predictions.jsonl'
        raw = path.read_bytes()
        digest = sha(raw)
        uri = args.qwen_predictions_uri
        if uri is None:
            if digest != FROZEN_QWEN_SHA:
                raise ValueError('New Qwen outputs require --qwen-predictions-uri pointing at the frozen S3 snapshot')
            uri = BASE + 'inputs/qwen-pe-predictions.jsonl'
        require_s3(uri)
        predictions = [json.loads(line) for line in raw.splitlines() if line]
    elif args.qwen_predictions is not None or args.qwen_predictions_uri is not None:
        raise ValueError('Qwen options require --use-qwen-pe')
    rows = build_inputs(originals, use_qwen_pe=args.use_qwen_pe, predictions=predictions,
                        predictions_sha=digest, predictions_uri=uri)
    raw = b''.join(encode(row) for row in rows)
    config = json.loads((work / 'campaign.json').read_text())
    config.update(version=2, run_configured=True, campaign='pe29-optional-qwen-dit44k-fixed', use_qwen_pe=args.use_qwen_pe,
                  instruction_policy=QWEN if args.use_qwen_pe else ORIGINAL, input_sha256=sha(raw),
                  original_input_sha256=sha(original_raw), qwen_pe_predictions_sha256=digest,
                  qwen_pe_predictions_uri=uri, checkpoint_uri=sources['dit44_ema'],
                  source_checkpoint_uri=sources['dit44_original'], publication_prefix=output,
                  units_per_cohort=args.node_count, max_workers=args.node_count,
                  requested_gpus=args.node_count * 8, caption_retry_policy='first_pass_greedy_only',
                  partition_policy='ordered_inputs[node_index::node_count]; original indices retained',
                  checkpoint_sources_sha256=sha((work / 'checkpoint-sources.json').read_bytes()), slow_gpus={})
    for row in rows:
        validate_instruction(row, config)
    contract = json.loads((work / 'caption-source-contract.json').read_text())
    contract.update(version=2, use_qwen_pe=args.use_qwen_pe,
                    instruction_policy=config['instruction_policy'], pe_input_sha256=sha(raw),
                    original_input_sha256=sha(original_raw), qwen_pe_predictions_sha256=digest,
                    qwen_pe_predictions_uri=uri, caption_retry_policy='first_pass_greedy_only',
                    admitted_cases=None, caption_schema_valid=None, native_rejections=None)
    receipt = dict(status='PASS', cases=len(rows), use_qwen_pe=args.use_qwen_pe,
                   instruction_policy=config['instruction_policy'], input_sha256=sha(raw),
                   original_input_sha256=sha(original_raw), qwen_pe_predictions_sha256=digest,
                   qwen_pe_predictions_uri=uri, checkpoint_sources=sources,
                   gpu_inference_rerun=False)
    # All validation precedes mutation. Inference must begin only after this command succeeds.
    for path, data in [(work / 'inputs.jsonl', raw), (work / 'campaign.json', encode(config)),
                       (work / 'caption-source-contract.json', encode(contract)),
                       (work / 'INPUT_VERIFIED.json', encode(receipt))]:
        temp = path.with_suffix(path.suffix + '.partial')
        temp.write_bytes(data)
        temp.replace(path)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
