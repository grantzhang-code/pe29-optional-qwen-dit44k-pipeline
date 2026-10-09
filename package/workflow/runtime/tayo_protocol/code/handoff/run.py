"""Native training inputs -> Qwen/VAE encode -> EMA inference -> verification.

Run through docker_run.py; native model/sampler/helpers come from runtime.json's
immutable image, not the user's checkout or a developer's absolute paths.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

WORK = Path('/work')
INPUTS = Path('/inputs')
CHECKPOINT = Path('/checkpoint')
MODELS = Path('/models')
HERE = Path(__file__).resolve().parent
TEXT_CAP = int(json.loads((WORK / 'training.json').read_text())['model']['conditioner']['init_args']['max_length'])

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def inside(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f'Path escapes root: {name}')
    return path

def verified(root, name, expected):
    p = inside(root, name)
    require(sha(p) == expected, f'Changed file: {name}')
    return p

def policy(config):
    mode = config['model']['denoiser']['init_args'].get('conditioning_policy', 'full')
    require(mode in ('full', 'reference_only'),
            f'Pinned image does not support {mode!r}; do not substitute another attention policy')
    return mode

def script_identity():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*.py'))} | {'runtime.json': sha(HERE / 'runtime.json')}

def plan():
    p = read(WORK / 'plan.json')
    require(p['scripts'] == script_identity(), 'Runner changed: use a new work directory')
    require(p['manifest_sha256'] == sha(WORK / 'manifest.jsonl'), 'Manifest changed')
    require(p['config_sha256'] == sha(WORK / 'training.json'), 'Training config changed')
    require(p['control_sha256'] == sha(CHECKPOINT / 'meta.pt'), 'Checkpoint control changed')
    require(p['metadata_sha256'] == sha(CHECKPOINT / '.metadata'), 'Checkpoint metadata changed')
    return p

def all_rows():
    return [r for line in (WORK / 'manifest.jsonl').read_text().splitlines() if (r := json.loads(line))['status'] == 'ready']

def selected(args):
    rows = all_rows()
    if args.ids:
        wanted = set(args.ids.split(','))
        require(wanted <= {r['request_id'] for r in rows}, 'Unknown request ID')
        rows = [r for r in rows if r['request_id'] in wanted]
    require(0 <= args.shard_index < args.shard_count, 'Invalid shard index/count')
    return [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]

def prepare(args):
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    require(not (WORK / 'plan.json').exists(), 'Work directory already frozen; choose a new directory')
    control = torch.load(CHECKPOINT / 'meta.pt', map_location='cpu', weights_only=False)
    require(int(control['global_step']) == args.step, 'Requested checkpoint step mismatch')
    config = control['mariana_resume_contract']['effective_config']
    mode = policy(config)
    model = config['model']
    require(str(model['conditioner']['init_args']['weight_path']).rstrip('/').endswith('/Qwen3.8-27B'), 'Unsupported Qwen asset: do not substitute weights')
    require(str(model['vae']['init_args']['weight_path']).rstrip('/').endswith('/FLUX.2-klein-4B-vae'), 'Unsupported VAE asset: do not substitute weights')
    require(model['diffusion_trainer']['init_args']['cfg_dropout_mode'] == 'multimodal_text_null',
            'Only native multimodal_text_null checkpoints are supported')
    if args.weights == 'EMA':
        ema = control.get('native_ema_state_dict', {})
        require(int(ema.get('previous_step', -1)) == args.step and int(ema.get('update_count', 0)) > 0,
                'EMA state is not complete for the requested step')
    prefix = 'state_dict.' + ('ema_denoiser.' if args.weights == 'EMA' else 'denoiser.')
    metadata = FileSystemReader(CHECKPOINT).read_metadata()
    keys = [key for key in metadata.state_dict_metadata if key.startswith(prefix)]
    require(bool(keys), f'Checkpoint has no {prefix} tensors')
    files = sorted({v.relative_path for k, v in metadata.storage_data.items() if k.fqn.startswith(prefix)})
    def inventory_entry(name):
        path = inside(CHECKPOINT, name)
        require(path.is_file() and path.stat().st_size > 0, f'Missing DCP shard: {name}')
        return {'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        inventory = list(pool.map(inventory_entry, files))
    require((CHECKPOINT / 'DOWNLOAD.json').exists(), 'Checkpoint download has not completed')
    source_rows = [json.loads(line) for line in (WORK / 'manifest.jsonl').read_text().splitlines()]
    require(len(source_rows) == 1364 and len({r['request_id'] for r in source_rows}) == 1364,
            'Expected exact 1364 unique source samples')
    require(all(r['status'] in ('ready', 'rejected_by_training_reader') for r in source_rows),
            'Input preparation errors remain')
    rows = all_rows()
    for r in rows:
        require('/' not in r['request_id'] and '..' not in r['request_id'], 'Invalid request ID')
        verified(WORK / 'native-batches', r['request_id'] + '.pt', r['native_batch_sha256'])
    require(config == read(WORK / 'training.json'), 'Frozen training config differs from checkpoint')
    p = dict(schema=1, step=args.step, weights=args.weights, namespace=prefix,
             checkpoint_uri=args.checkpoint_uri, conditioning_policy=mode,
             use_reference_kv_cache=mode == 'reference_only', text_cap=TEXT_CAP,
             steps=50, cfg=args.cfg, timeshift=4.6, last_step=0.02, noise_scale=1.0,
             roi_policy='pixel_aspect_fixed', expected_samples=len(rows), source_samples=1364,
             admission_rejections=[r for r in source_rows if r['status'] != 'ready'],
             runtime=read(HERE / 'runtime.json'), scripts=script_identity(),
             manifest_sha256=sha(WORK / 'manifest.jsonl'), config_sha256=sha(WORK / 'training.json'),
             control_sha256=sha(CHECKPOINT / 'meta.pt'), metadata_sha256=sha(CHECKPOINT / '.metadata'),
             checkpoint_files=inventory, checkpoint_keys=len(keys))
    atomic(WORK / 'plan.json', p)
    print(json.dumps({'status': 'PREPARED', 'policy': mode, 'text_cap': TEXT_CAP, 'keys': len(keys)}), flush=True)

def native_prepare(context, row, inputsroot, pack_target=None):
    import torch
    from src.inference.craft_v1_edit import _segment_layout
    from src.data.dataset.craft_v1_edit_clean_caption import require_caption_preparation_version
    require_caption_preparation_version(row)
    path = verified(WORK / 'native-batches', row['request_id'] + '.pt', row['native_batch_sha256'])
    value = torch.load(path, map_location='cpu', weights_only=False)
    batch, case = value['batch'], value['case']
    require(all(row[k] == v for k, v in case.items()), 'Native batch case differs from manifest')
    targets, lengths = _segment_layout(batch)
    require(not torch.count_nonzero(batch.img_patches[targets]), 'Nonzero target pixels')
    require(batch.img_seg_frame.tolist() == [*range(1, row['num_inputs'] + 1), 0], 'Reference order changed')
    meta = batch.metadata[0]
    require('post_encoder_text_keep_mask' not in meta, 'Training random text pruning is enabled')
    require(0 < min(case['positive_text_tokens'], case['weak_text_tokens']) and
            max(case['positive_text_tokens'], case['weak_text_tokens']) < TEXT_CAP,
            'Text reached cap: investigate native admission/truncation')
    require(case['strict_pack_cost'] <= batch.pack_target == 22528, 'Training admission budget changed')
    require(hashlib.sha256(batch.captions[0].encode()).hexdigest() == case['positive_prompt_sha256'], 'Positive caption changed')
    require(hashlib.sha256(str(meta['cfg_null_caption']).encode()).hexdigest() == case['weak_prompt_sha256'], 'Weak caption changed')
    require(lengths == [case['image_tokens']], 'Image token accounting changed')
    case.update(text_cap=TEXT_CAP, truncated=False, model_padding_tokens=0,
                input_pipeline='exact-training-image-native-MDS-processor')
    return batch, case


def audit(args):
    import torch
    frozen = (WORK / 'plan.json').exists()
    if frozen:
        plan()
    torch.set_num_threads(2)
    cases = []
    for row in all_rows():
        _, case = native_prepare(None, row, None)
        cases.append(case)
    summary = dict(plan_sha256=sha(WORK / 'plan.json') if frozen else None, samples=len(cases), truncated=False,
                   max_positive=max(c['positive_text_tokens'] for c in cases),
                   max_weak=max(c['weak_text_tokens'] for c in cases),
                   max_total_pack=max(c['strict_pack_cost'] for c in cases),
                   canary_ids=list(dict.fromkeys([max(cases, key=lambda c:c['strict_pack_cost'])['request_id'],
                       max(cases, key=lambda c:c['num_inputs'])['request_id'], cases[0]['request_id']])), cases=cases)
    atomic(WORK / 'audit.json', summary)
    print(json.dumps({k: v for k,v in summary.items() if k != 'cases'}), flush=True)


def bundle_paths(rid):
    return WORK / 'bundles' / (rid + '.pt'), WORK / 'bundles' / (rid + '.json')

def check_bundle(row, contract):
    path, receipt = bundle_paths(row['request_id'])
    proof = read(receipt)
    require(proof['plan_sha256'] == contract and proof['request_id'] == row['request_id'], 'Bundle identity mismatch')
    require(sha(path) == proof['sha256'], 'Bundle SHA mismatch')
    return path

def encode(args):
    import torch
    import runtime_helpers as H
    p = plan()
    contract = sha(WORK / 'plan.json')
    require(read(WORK / 'audit.json')['plan_sha256'] == contract, 'Run the full native audit first')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    H.prepare_case = native_prepare
    context = H.load_conditioner(read(WORK / 'training.json'), MODELS, 'cuda:0', text_cap=TEXT_CAP)
    atomic(WORK / 'evidence' / f'encoder-{args.shard_index}.json', context.evidence)
    for row in selected(args):
        path, receipt = bundle_paths(row['request_id'])
        if receipt.exists():
            check_bundle(row, contract)
            continue
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            bundle, case = H.build_case(context, row, INPUTS / 'native')
        import shutil
        needed = sum(v.numel() * v.element_size() for v in bundle.values() if isinstance(v, torch.Tensor))
        require(shutil.disk_usage(WORK).free > needed + (80 << 30), 'Scratch reserve below 80 GiB')
        # Preserve the native positive and actual multimodal-null feature branches.
        bundle = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in bundle.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.partial')
        torch.save({'bundle': bundle, 'case': case}, tmp)
        tmp.replace(path)
        atomic(receipt, dict(request_id=row['request_id'], plan_sha256=contract, sha256=sha(path), case=case))
        print('ENCODED', row['request_id'], case['strict_pack_cost'], flush=True)
        del bundle
        gc.collect()
    # Exiting this process releases Qwen before a separate DiT process is started.

def restore(model, p):
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    def verify_shard(f):
        return verified(CHECKPOINT, f['name'], f['sha256'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_shard, p['checkpoint_files']))
    reader = FileSystemReader(CHECKPOINT)
    metadata = reader.read_metadata()
    prefix = p['namespace']
    entries = {k[len(prefix):]: v for k, v in metadata.state_dict_metadata.items() if k.startswith(prefix)}
    state = model.state_dict()
    require(set(entries) == set(state), 'Strict checkpoint key mismatch')
    require(all(tuple(entries[k].size) == tuple(v.shape) for k, v in state.items()), 'Strict tensor shape mismatch')
    namespace = prefix.removeprefix('state_dict.')
    selected_state = {namespace + k: v for k, v in state.items()}
    load_state_dict({'state_dict': selected_state}, storage_reader=reader, no_dist=True)
    model.load_state_dict({k.removeprefix(namespace): v for k, v in selected_state.items()}, strict=True)
    return dict(step=p['step'], weights=p['weights'], namespace=prefix, keys=len(state),
                numel=sum(t.numel() for t in state.values()), optimizer_loaded=False)

def result_paths(cfg, rid):
    path = WORK / 'results' / f'cfg{cfg}' / f'{rid}.png'
    return path, path.with_suffix('.json')

def check_result(row, cfg, contract):
    from PIL import Image
    path, receipt = result_paths(cfg, row['request_id'])
    r = read(receipt)
    require(r['plan_sha256'] == contract and r['request_id'] == row['request_id'], 'Output identity mismatch')
    require(r['cfg'] == cfg and r['seed'] == row['generation_seed'] and r['steps'] == 50, 'Output sampling identity mismatch')
    require(r['finite_latents'] is True and sha(path) == r['image_sha256'], 'Invalid output hash/latents')
    with Image.open(path) as im:
        im.load()
        require(im.size == (row['output_width'], row['output_height']), 'Output dimensions mismatch')
    return r

def infer(args):
    import torch
    import runtime_helpers as H
    from src.diffusion.flow_matching.sampling import EulerSamplerJiT
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    from src.models.layers import attention_op
    from mariana.integrations.flash_attention_4 import fa4_direct_enabled
    p = plan()
    contract = sha(WORK / 'plan.json')
    rows = selected(args)
    for row in rows:
        check_bundle(row, contract)
    require(attention_op.ACTIVE_BACKEND.name == 'fa4' and fa4_direct_enabled(), 'Expected direct FA4 runtime')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    config = read(WORK / 'training.json')
    spec = config['model']['denoiser']
    with torch.device('cpu'), H._default_dtype(torch.bfloat16):
        model = H._class(spec['class_path'])(**copy.deepcopy(spec['init_args']))
    model.requires_grad_(False).eval()
    proof = restore(model, p)
    require(getattr(model, 'conditioning_policy', 'full') == p['conditioning_policy'], 'Model attention policy mismatch')
    model.strict_pack_target = False
    model.pack_target_T = None
    model.to('cuda')
    context = H.load_vae_only(config, MODELS, 'cuda:0')
    atomic(WORK / 'evidence' / f'restore-{args.shard_index}.json', proof)
    host = {'img_grids', 'img_seg_sample_id', 'img_seg_frame', 'img_seg_is_target'}
    for row in rows:
        path = check_bundle(row, contract)
        data = torch.load(path, map_location='cpu', weights_only=False)
        case = data['case']
        bundle = {k: v.to('cuda') if isinstance(v, torch.Tensor) and k not in host else v for k, v in data['bundle'].items()}
        for cfg in p['cfg']:
            output, receipt = result_paths(cfg, row['request_id'])
            if receipt.exists():
                check_result(row, cfg, contract)
                continue
            sampler = EulerSamplerJiT(num_steps=50, timeshift=4.6, prediction_type='v',
                                      guidance=float(cfg), scheduler=LinearScheduler())
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                latents = sampler.sample_varlen_edit(model, **bundle, guidance=float(cfg), noise_scale=1,
                                                     seed=row['generation_seed'],
                                                     use_reference_kv_cache=p['use_reference_kv_cache'])
                require(all(torch.isfinite(t).all().item() for t in latents), 'Nonfinite latents')
                im = H.decode(context, latents, case)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix('.partial')
            im.save(tmp, format='PNG')
            tmp.replace(output)
            atomic(receipt, dict(request_id=row['request_id'], plan_sha256=contract, cfg=cfg, steps=50,
                                  seed=row['generation_seed'], step=p['step'], weights=p['weights'],
                                  cache_enabled=p['use_reference_kv_cache'], image_sha256=sha(output),
                                  finite_latents=True, bundle_sha256=sha(path), case=case))
            print('GENERATED', cfg, row['request_id'], flush=True)
            del latents
        del bundle, data
        gc.collect()

def verify(args):
    p = plan()
    contract = sha(WORK / 'plan.json')
    rows = all_rows()
    receipts = [check_result(r, cfg, contract) for cfg in p['cfg'] for r in rows]
    expected = {(cfg, r['request_id']) for cfg in p['cfg'] for r in rows}
    actual = {(cfg, path.stem) for cfg in p['cfg'] for path in (WORK / 'results' / f'cfg{cfg}').glob('*.json')}
    require(expected == actual, 'Unexpected or missing output IDs')
    (WORK / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in receipts))
    atomic(WORK / 'COMPLETE.json', dict(plan_sha256=contract, samples=p['expected_samples'], source_samples=1364, admission_rejections=len(p['admission_rejections']), outputs=len(receipts), status='PASS'))
    print('COMPLETE', len(receipts))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--step', type=int, required=True)
    prep.add_argument('--weights', choices=['EMA', 'LIVE'], required=True)
    prep.add_argument('--checkpoint-uri', required=True)
    prep.add_argument('--cfg', nargs='+', type=int, choices=[2, 3], default=[2, 3])
    for name in ('encode', 'infer'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--shard-index', type=int, default=0)
        cmd.add_argument('--shard-count', type=int, default=1)
        cmd.add_argument('--ids', help='Comma-separated canary IDs; omit for the full cohort')
    sub.add_parser('audit')
    sub.add_parser('verify')
    a = p.parse_args()
    # A single stage/shard cannot have two live writers. Different shards are independent.
    WORK.mkdir(parents=True, exist_ok=True)
    lock = WORK / f'{a.command}-{getattr(a, "shard_index", 0)}.lock'
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[a.command](a)

if __name__ == '__main__':
    main()
