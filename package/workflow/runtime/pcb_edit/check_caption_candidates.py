"""Check replacement captions through the complete frozen native MDS reader."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from pcb_edit.prepare_pcb_native import adapt, initialize, native


def rejection(source, reason):
    index=source['input_index']
    rid=f'{index:04d}-'+hashlib.sha256(('pcbv2:'+source['sample_id']).encode()).hexdigest()[:16]
    return dict(request_id=rid,ordinal=index,source='pcbv2',source_index=index,original_mds_sample_id=source['sample_id'],generation_seed=42+index,num_inputs=1,status='invalid_pe_caption',reason=reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    sources = {row['input_index']: row for row in map(json.loads, Path('/control/inputs/all.jsonl').read_text().splitlines())}
    payload = args.predictions.read_bytes()
    candidates = [json.loads(line) for line in payload.splitlines() if line.strip()]
    assert len({row['input_index'] for row in candidates}) == len(candidates)
    args.work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile('/work/training.json', args.work/'training.json')
    (args.work/'mds/pcbv2').mkdir(parents=True, exist_ok=True)
    bindings, annotations, jobs, rejected = [], [], [], []
    for row in candidates:
        index = row['input_index']
        source = sources[index]
        assert all(row.get(key) == source[key] for key in ('sample_id', 'instruction', 'source_s3_uri', 'target_canvas_size', 'selection'))
        bindings.append(source)
        if row.get('schema_valid') is not True:
            rejected.append(rejection(source, row.get('schema_error') or row.get('json_error')))
            continue
        try:
            annotation = adapt(row['prediction'])
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(rejection(source, str(error)))
            continue
        annotations.append(dict(input_index=index, annotation=annotation))
        jobs.append((index, 'pcbv2', index))
    bp, ap = args.work/'source-bindings.json', args.work/'pe-native-annotations.json'
    bp.write_text(json.dumps(bindings, ensure_ascii=False)+'\n')
    ap.write_text(json.dumps(annotations, ensure_ascii=False)+'\n')
    with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(str(args.work), str(bp), str(ap))) as pool:
        results = list(pool.map(native.prepare, jobs))
    results = sorted(results+rejected, key=lambda row: row['ordinal'])
    assert len(results) == len(candidates)
    (args.work/'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in results))
    audit = dict(candidate_sha256=hashlib.sha256(payload).hexdigest(), counts=dict(Counter(row['status'] for row in results)),
                 accepted_indices=[row['ordinal'] for row in results if row['status']=='ready'],
                 rejected=[row for row in results if row['status']!='ready'])
    native.atomic(args.work/'input-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
