"""Build a new adaptive PE3k GSB bundle and Arnold 4 x 256 manifest."""
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import json
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from materialize import read, restore_bundle, save, sha
from restore import destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--reference-bundle', type=Path)
    parser.add_argument('--aws-profile', default='core-cn-s3-readwrite')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?', args.job_name):
        parser.error('Use a unique lowercase JobSet name of at most 50 characters')
    workspace = args.workspace.resolve()
    run = workspace / 'run'
    campaign = read(run / 'campaign.json')
    configured = read(run / 'CONFIGURED.json')
    assert sha(run / 'campaign.json') == configured['campaign_sha256']
    destination(campaign['publication_prefix'])
    assert campaign['dit_step'] == 6000
    assert campaign['sampling'] == read(HERE / 'profile.json')['sampling']
    for name in ['COMPLETE.json', 'PUBLICATION_VERIFIED.json', 'encoder-order-audit.json']:
        assert read(run / 'final' / name)['status'] == 'PASS'
    g = run / 'gsb_vs_qwen21'
    assert not g.exists(), 'Use a fresh scoring directory; never overwrite frozen judgments'
    reference = restore_bundle(read(HERE / 'profile.json')['bundles']['gsb'],
                               workspace / 'gsb-reference', args.aws_profile, args.reference_bundle)
    protocol = read(reference / 'protocol.json')
    for name, digest in protocol['input_hashes'].items():
        assert sha(reference / 'inputs' / name) == digest, name
    for name, digest in protocol['code_hashes'].items():
        assert sha(reference / name) == digest, name
    shutil.copytree(HERE / 'recorded/gsb', g)
    inputs = g / 'inputs'
    inputs.mkdir()
    for name in ['control-pairs.jsonl', 'control-protocol.json', 'reference-protocol.json', 'gsb_rubric.json']:
        shutil.copy2(reference / 'inputs' / name, inputs / name)
    for name in ['COMPLETE.json', 'PUBLICATION_VERIFIED.json', 'encoder-order-audit.json']:
        shutil.copy2(run / 'final' / name, inputs / name)
    shutil.copy2(run / 'campaign.json', inputs / 'generation-campaign.json')
    for name in ['editor/score-index.jsonl', 'editor/images.manifest.jsonl', 'caption/excluded.jsonl']:
        path = inputs / 'generation/3000' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / 'final/3000' / name, path)
    ids = sorted(json.loads(line)['sample_id'] for line in
                 (inputs / 'generation/3000/editor/score-index.jsonl').read_text().splitlines())
    save(inputs / 'common-output-sample-ids.json', ids)
    subprocess.run([sys.executable, str(g / 'prepare.py')], check=True)
    protocol = read(g / 'protocol.json')
    digest = sha(g / 'bundle.tar.gz')
    uri = protocol['publication_prefix'] + 'code/bundle-' + digest + '.tar.gz'
    if not args.no_upload:
        import boto3
        import hashlib
        s3 = boto3.Session(profile_name=args.aws_profile).client('s3')
        for local, remote in [(g / 'bundle.tar.gz', uri),
                              (g / 'protocol.json', protocol['publication_prefix'] + 'protocol.json')]:
            target = urlsplit(remote)
            raw = local.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            s3.put_object(Bucket=target.netloc, Key=target.path.lstrip('/'), Body=raw,
                          Metadata={'sha256': checksum})
            assert hashlib.sha256(s3.get_object(Bucket=target.netloc, Key=target.path.lstrip('/'))
                                  ['Body'].read()).hexdigest() == checksum
    sys.path.insert(0, str(HERE.parent / 'gsb'))
    from build_job import build
    out = workspace / 'utp'
    plan = build(3000, uri, digest, job_name=args.job_name, output_dir=out)
    save(out / 'plan.json', dict(plan, bundle_uploaded=not args.no_upload))
    print(json.dumps(dict(status='PASS', pairs=len(ids), judgments=2 * len(ids),
                          workers=4, concurrency_per_worker=256,
                          manifest=plan['manifest'], bundle_uploaded=not args.no_upload)))


if __name__ == '__main__':
    main()
