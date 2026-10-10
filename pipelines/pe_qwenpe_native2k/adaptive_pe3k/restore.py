"""Restore the adaptive PE3k campaign into a fresh output namespace."""
from pathlib import Path
import argparse
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from materialize import read, restore_bundle, safe_destination, save, sha


def destination(value):
    value = safe_destination(value)
    config = read(HERE / 'profile.json')
    for protected in [config['generation_source'], config['reference_dit4k_source']]:
        if value.startswith(protected) or protected.startswith(value):
            raise ValueError('Use a new prefix; recorded evaluations are immutable')
    return value


def configure(root, output, runtime_from=None, dit_cache_from=None):
    config = read(HERE / 'profile.json')
    campaign = read(root / 'campaign.json')
    assert campaign['input_sha256'] == config['input_sha256']
    assert campaign['qwen_pe_predictions_sha256'] == config['qwen_predictions_sha256']
    assert campaign['training_prompt_sha256'] == config['prompt_sha256']
    assert campaign['sampling'] == config['sampling']
    assert campaign['arms']['3000']['checkpoint'] == config['caption_checkpoint']
    assert campaign['dit_step'] == 6000
    campaign['publication_prefix'] = output
    campaign['arms']['3000']['checkpoint']['publication_prefix'] = output
    campaign['gsb_enabled'] = False  # GSB is a separate, explicitly submitted stage.
    if dit_cache_from:
        cache = Path(dit_cache_from).resolve()
        assert sha(cache / 'CHECKPOINT.json') == config['inference']['dit_checkpoint_sha256']
        campaign['dit_cache_root'] = str(cache)
    save(root / 'campaign.json', campaign)
    workflow_path = root / 'package/workflow/campaign.json'
    workflow = read(workflow_path)
    # These three fields were descriptive leftovers in the historical bundle.
    # Effective runtime sampling remains pinned by campaign.json and PE_SAMPLING_MODE.
    workflow.update(campaign=config['version'], output_s3=output, publication_prefix=output,
                    caption_retry_policy=config['sampling']['strategy'])
    save(workflow_path, workflow)
    if runtime_from:
        runtime_from = Path(runtime_from).resolve()
        env = root / 'package/assets/env'
        assert not env.exists() and not env.is_symlink()
        assert (runtime_from / 'package/assets/env/caption/bin/python').is_file()
        env.symlink_to(runtime_from / 'package/assets/env', target_is_directory=True)
        for name in ['CAPTION_ENV_READY.json', 'EDITOR_ENV_READY.json']:
            shutil.copy2(runtime_from / name, root / name)
    save(root / 'CONFIGURED.json', dict(
        status='PASS', stage='adaptive_pe3k_dit6000', pe_step=3000,
        campaign_sha256=sha(root / 'campaign.json'),
        workflow_campaign_sha256=sha(workflow_path),
        bundle_sha256=config['bundles']['generation']['sha256'], publication_prefix=output,
        source_manifest='SOURCE_MANIFEST.json records the original bundle before path configuration.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-s3', required=True)
    parser.add_argument('--runtime-from', type=Path)
    parser.add_argument('--dit-cache-from', type=Path)
    parser.add_argument('--bundle-file', type=Path)
    parser.add_argument('--aws-profile', default='core-cn-s3-readwrite')
    args = parser.parse_args()
    output = destination(args.output_s3)
    config = read(HERE / 'profile.json')
    root = restore_bundle(config['bundles']['generation'], args.workspace.resolve() / 'run',
                          args.aws_profile, args.bundle_file)
    configure(root, output, args.runtime_from, args.dit_cache_from)
    print(json.dumps(dict(status='PASS', root=str(root), publication_prefix=output)))


if __name__ == '__main__':
    main()
