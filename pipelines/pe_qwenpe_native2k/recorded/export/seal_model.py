"""Seal the historical HF initializer layout without assuming a dedicated MTP shard."""
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file

from common import ROOT, CODE, CFG, save, sha, publish
sys.path.insert(0, str(CODE / 'repo'))
from scripts.checkpoint.hf_artifact_s3 import validate_local_artifact
from scripts.checkpoint.mcore_s3 import validate_local_mcore_checkpoint
from scripts.mtp_eval.hf_artifact import read_weight_layout, mtp_tensor_hashes


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def restore_fp32_state(model, checkpoint):
    marker = model / 'fp32-state-preservation.json'
    if marker.exists():
        return json.loads(marker.read_text())
    directory = checkpoint / f"iter_{CFG['checkpoint_step']:07d}"
    metadata = pickle.loads((directory / '.metadata').read_bytes())
    names = sorted(k for k in metadata.state_dict_metadata
                   if re.fullmatch(r'language_model.decoder.layers.\d+.self_attention.(A_log|dt_bias)', k))
    assert len(names) == 48
    state = {k: torch.empty(metadata.state_dict_metadata[k].size, dtype=torch.float32) for k in names}
    assert all(metadata.state_dict_metadata[k].properties.dtype == torch.float32 for k in names)
    dcp.load(state, checkpoint_id=str(directory), planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
    index_path = model / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    replacements = {}
    for name, value in state.items():
        layer, parameter = re.fullmatch(r'language_model.decoder.layers.(\d+).self_attention.(A_log|dt_bias)', name).groups()
        hf_name = f'model.language_model.layers.{layer}.linear_attn.{parameter}'
        assert hf_name in index['weight_map'] and torch.isfinite(value).all()
        replacements[hf_name] = (name, value)
    audit = []
    added_bytes = 0
    for shard in sorted({index['weight_map'][k] for k in replacements}):
        path = model / shard
        tensors = load_file(path)
        for name, (source_name, value) in replacements.items():
            if index['weight_map'][name] != shard:
                continue
            previous = tensors[name]
            assert previous.shape == value.shape
            assert torch.equal(previous, value.to(previous.dtype)), (name, 'exported value does not match source')
            audit.append({'hf_name': name, 'mcore_name': source_name,
                          'before_dtype': str(previous.dtype), 'after_dtype': str(value.dtype),
                          'source_tensor_sha256': tensor_sha(value)})
            added_bytes += value.numel() * (value.element_size() - previous.element_size())
            tensors[name] = value
        temp = path.with_suffix('.fp32.tmp')
        save_file(tensors, temp, metadata={'format': 'pt'})
        temp.replace(path)
        del tensors
    index['metadata']['total_size'] += added_bytes
    save(index_path, index)
    result = {'status': 'PASS', 'count': len(audit), 'source_checkpoint_manifest_sha256': CFG['checkpoint_manifest_sha256'],
              'policy': 'Preserve exact FP32 A_log/dt_bias from committed MCore checkpoint; all other tensors untouched.',
              'tensors': audit}
    save(marker, result)
    return result


def main():
    initial = ROOT / 'model/initial-hf'
    model = ROOT / 'model/hf'
    checkpoint = ROOT / f"model/checkpoint-{CFG['checkpoint_step']}"
    validate_local_artifact(initial)
    assert sha(initial / 'conversion_manifest.json') == CFG['initial_artifact_manifest_sha256']
    assert validate_local_mcore_checkpoint(checkpoint, expected_step=CFG['checkpoint_step']) == CFG['checkpoint_step']
    receipt = json.loads((checkpoint / '.s3_restore.json').read_text())
    assert receipt['manifest_sha256'] == CFG['checkpoint_manifest_sha256']
    precision = restore_fp32_state(model, checkpoint)
    initial_layout, trained_layout = read_weight_layout(initial), read_weight_layout(model)
    assert len(initial_layout.weight_map) == len(trained_layout.weight_map) == 738
    assert set(initial_layout.weight_map) == set(trained_layout.weight_map)
    old, new = mtp_tensor_hashes(initial_layout), mtp_tensor_hashes(trained_layout)
    assert len(old) == len(new) == 15 and set(old) == set(new)
    changed = sorted(name for name in old if old[name] != new[name])
    assert len(changed) == 15
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    assert 1 in [text.get('mtp_num_hidden_layers'), text.get('num_nextn_predict_layers')]
    manifest = {'kind': 'local-mcore-hf-trained-mtp-export-v1',
        'source_mcore_checkpoint': {'uri': CFG['checkpoint_uri'], 'step': CFG['checkpoint_step'],
                                   'manifest_sha256': CFG['checkpoint_manifest_sha256']},
        'initial_artifact': {'uri': CFG['initial_artifact_uri'], 'manifest_sha256': CFG['initial_artifact_manifest_sha256']},
        'tensor_count': 738, 'mtp_tensor_count': 15, 'mtp_changed_tensors': changed,
        'fp32_state_preservation': precision,
        'files': [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(model.iterdir())
                  if f.is_file() and f.name != 'local_export_manifest.json']}
    save(model / 'local_export_manifest.json', manifest)
    ready = {'status': 'complete', 'checkpoint': manifest['source_mcore_checkpoint'],
             'export_manifest_sha256': sha(model / 'local_export_manifest.json'),
             'tensor_count': 738, 'mtp_tensor_count': 15, 'mtp_changed_tensors': len(changed),
             'fp32_state_parameters_preserved': precision['count'], 'model_dir': str(model),
             'publication_protocol': manifest['kind']}
    for f in sorted(model.iterdir()):
        if f.is_file():
            publish(f, 'model/hf/' + f.name)
    save(ROOT / 'model/conversion_provenance.json', ready)
    publish(ROOT / 'model/conversion_provenance.json', 'model/hf/_SUCCESS')
    publish(ROOT / 'model/conversion_provenance.json', 'model/conversion_provenance.json')
    save(ROOT / 'LOCAL_MODEL_READY.json', ready)
    print('MODEL_SEALED', json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
