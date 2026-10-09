"""Freeze test inputs using the exact training image's native MDS processor."""
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from streaming import LocalDataset
from transformers import AutoTokenizer
from torchvision.transforms.functional import to_tensor

from mariana.data.processors.base import Keys
from src.data.dataset.craft_v1_edit_mds import CraftV1EditMdsDataset
from src.data.dataset.craft_v1_edit_media import decode_reference_images
from src.data.dataset.craft_v1_edit_clean_caption import CAPTION_PREPARATION_VERSION

ROOT = Path('/data')
OP = Path('/operation')
OVERRIDES = dict(shuffle=False, sample_holdout_mod=None, meta_filters=None,
                 json_post_encoder_dropout=False, target_description_drop_prob=0.0,
                 corr_code_null_sample_ratio=0.0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item()) + '\n')
    temp.replace(path)


def initialize():
    global PROCESSOR, DATASETS, CONFIG
    torch.set_num_threads(1)
    CONFIG = json.loads((OP / 'training.json').read_text())
    args = copy.deepcopy(CONFIG['data']['train_dataset']['init_args'])
    args.pop('streams'); args.pop('streaming_kwargs', None); args.update(OVERRIDES)
    ds = CraftV1EditMdsDataset(streams=None, **args)
    ds.configure_from_denoiser(SimpleNamespace(**CONFIG['model']['denoiser']['init_args']))
    tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.8-27B', local_files_only=True)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = CONFIG['model']['conditioner']['init_args']['max_length']
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    conditioner = SimpleNamespace(**CONFIG['model']['conditioner']['init_args'], tokenizer=tokenizer,
                                  reference_region_visual_stride=32)
    ds.configure_text_tokenizer_from_conditioner(conditioner)
    PROCESSOR = ds.build_varlen_processor(**{k: args[k] for k in
        ('pack_target_T', 'pack_buffer_size', 'pack_buffer_bytes', 'pack_max_samples_per_pack')})
    DATASETS = {}


def prepare(entry):
    ordinal, source, index = entry
    if source not in DATASETS:
        DATASETS[source] = LocalDataset(local=str(ROOT / 'mds' / source))
    row = dict(DATASETS[source][index])
    sample_id = str(row['sample_id'])
    rid = f'{ordinal:04}-' + hashlib.sha256((source + ':' + sample_id).encode()).hexdigest()[:16]
    dest = ROOT / 'inputs' / rid
    dest.mkdir(parents=True, exist_ok=True)
    references = decode_reference_images(row, max_images=12)
    assert len(references) == int(row['num_inputs'])
    encoded = [bytes(x) for x in references]
    for i, raw in enumerate(encoded):
        (dest / f'reference-{i:02}.image').write_bytes(raw)
    (dest / 'target.image').write_bytes(bytes(row['output_image']))
    annotation = row['image_edit_v6_annotation']
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    atomic(dest / 'annotation.json', annotation)
    safe_row = {k: v for k, v in row.items() if k not in ('input_images', 'input_masks', 'output_image', 'output_mask')}
    atomic(dest / 'record.json', safe_row)
    result = dict(request_id=rid, ordinal=ordinal, source=source, source_index=index,
                  original_mds_sample_id=sample_id, generation_seed=42 + ordinal,
                  instructions=row['instructions'], num_inputs=len(encoded),
                  reference_sha256=[hashlib.sha256(x).hexdigest() for x in encoded],
                  annotation_sha256=sha(dest / 'annotation.json'), target_sha256=sha(dest / 'target.image'))
    row[Keys.SAMPLE_ID] = ordinal
    before = dict(PROCESSOR._pack_skip)
    try:
        light = PROCESSOR._to_light_sample(row, epoch=0)
        if light is None:
            result.update(status='rejected_by_training_reader', drop_counters={
                k: v - before.get(k, 0) for k, v in PROCESSOR._pack_skip.items() if v != before.get(k, 0)})
            return result
        prepared = light.prepared
        assert len(prepared.references) == len(encoded)
        for plane, raw in zip([*prepared.references, prepared.target], [*encoded, bytes(row['output_image'])], strict=True):
            with Image.open(io.BytesIO(raw)) as im:
                expected = to_tensor(im.convert('RGB')).mul_(2).sub_(1)
            assert torch.equal(plane.tensor, expected), 'Stored image pixels changed'
        light.prepared = replace(prepared, target=replace(prepared.target, tensor=torch.zeros_like(prepared.target.tensor)))
        item = PROCESSOR._item_from(light)
        assert item.cost <= PROCESSOR.packer.target_T
        # The shared training assembler; its padding field records training capacity.
        batch = PROCESSOR._decode_pack([item], 0)[0]
        assert batch.img_seg_frame.tolist() == [*range(1, len(encoded) + 1), 0], batch.img_seg_frame.tolist()
        assert batch.img_seg_is_target.tolist() == [False] * len(encoded) + [True], batch.img_seg_is_target.tolist()
        target_tokens = int(batch.img_grids[-1].prod())
        assert not torch.count_nonzero(batch.img_patches[-target_tokens:]).item()
        meta = batch.metadata[0]
        pos, weak = int(batch.text_lens[0]), int(meta['cfg_null_qwen_text_len'])
        image_tokens = int(batch.img_grids.prod(dim=1).sum())
        assert image_tokens + max(pos, weak) <= PROCESSOR.packer.target_T
        assert batch.pad_tokens == PROCESSOR.packer.target_T - image_tokens - pos
        result.update(status='ready', caption_preparation_version=CAPTION_PREPARATION_VERSION,
            positive_text_tokens=pos, weak_text_tokens=weak,
            strict_pack_cost=image_tokens + max(pos, weak), image_tokens=image_tokens,
            output_width=prepared.target.final_width, output_height=prepared.target.final_height,
            source_sizes=[[x.final_width, x.final_height] for x in prepared.references],
            positive_prompt_sha256=hashlib.sha256(batch.captions[0].encode()).hexdigest(),
            weak_prompt_sha256=hashlib.sha256(str(meta['cfg_null_caption']).encode()).hexdigest(),
            target_pixels_used=False, reference_order_preserved=True, decoded_pixels_equal_stored=True,
            training_padding_tokens=int(batch.pad_tokens), variant=CONFIG['data']['train_dataset']['init_args']['variant'],
            roi_token_counts=[int(x) for x in meta.get('reference_region_token_counts', [])])
        atomic(dest / 'model-prompts.json', {'positive': batch.captions[0], 'weak': str(meta['cfg_null_caption'])})
        path = ROOT / 'native-batches' / (rid + '.pt')
        path.parent.mkdir(exist_ok=True)
        torch.save({'batch': batch, 'case': result}, path)
        result['native_batch_sha256'] = sha(path)
    except Exception as exc:
        result.update(status='preparation_error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    atomic(dest / 'input-audit.json', result)
    return result


def main():
    jobs, indexes, shards = [], [], []
    for path in sorted((ROOT / 'mds').glob('*/index.json')):
        index = json.loads(path.read_text())
        count = sum(s['samples'] for s in index['shards'])
        for shard in index['shards']:
            raw = shard['raw_data']; f = path.parent / raw['basename']
            assert f.resolve().is_relative_to((ROOT / 'mds').resolve())
            assert f.stat().st_size == raw['bytes']
            payload = f.read_bytes()
            for algorithm, expected in raw['hashes'].items():
                assert hashlib.new(algorithm, payload).hexdigest() == expected
            shards.append({'path': str(f.relative_to(ROOT / 'mds')), 'bytes': len(payload),
                           'sha256': hashlib.sha256(payload).hexdigest()})
        indexes.append({'source': path.parent.name, 'samples': count, 'index_sha256': sha(path)})
        offset = len(jobs)
        jobs.extend((offset + i, path.parent.name, i) for i in range(count))
    assert len(jobs) == 1364, len(jobs)
    assert [x[0] for x in jobs] == list(range(1364))
    atomic(ROOT / 'data-freeze.json', {'samples': len(jobs), 'indexes': indexes, 'shards': shards, 'eval_overrides': OVERRIDES})
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
        rows = []
        for result in pool.map(prepare, jobs):
            rows.append(result)
            if len(rows) % 50 == 0:
                print('PREPARED', len(rows), flush=True)
    (ROOT / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    from collections import Counter
    good = [r for r in rows if r['status'] == 'ready']
    report = {'samples': len(rows), 'counts': dict(Counter(r['status'] for r in rows)),
              'reference_counts': dict(Counter(r['num_inputs'] for r in rows)),
              'max_positive': max((r['positive_text_tokens'] for r in good), default=0),
              'max_pack': max((r['strict_pack_cost'] for r in good), default=0),
              'feature_bytes_estimate': sum((r['positive_text_tokens'] + r['weak_text_tokens']) * 16 * 5120 * 2 for r in good),
              'rejections': [r for r in rows if r['status'] != 'ready']}
    atomic(ROOT / 'input-audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'rejections'}), flush=True)


if __name__ == '__main__':
    main()
