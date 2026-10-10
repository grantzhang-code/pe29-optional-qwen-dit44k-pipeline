"""Run the frozen adaptive worker with canary, repetition and encoder-order gates."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from materialize import read, save, sha
from run_node import audit_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--node', type=int, default=0)
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    configured = read(root / 'CONFIGURED.json')
    assert configured['stage'] == 'adaptive_pe3k_dit6000'
    assert sha(root / 'campaign.json') == configured['campaign_sha256']
    assert sha(root / 'package/workflow/campaign.json') == configured['workflow_campaign_sha256']
    sys.path.insert(0, str(root))
    import worker as w
    sys.path.insert(0, str(w.WORKFLOW / 'code'))
    from repetition_rules import validate_selected
    assert 0 <= args.node < 16
    prefix = w.CAMPAIGN['publication_prefix']
    if args.command == 'finalize':
        assert args.node == 0 and not args.canary
        for script in ['finalize.py', 'verify_publication.py']:
            subprocess.run([sys.executable, '-u', str(root / script)], check=True)
        audits = [w.wait_json(prefix + f'nodes/{i:02d}/evidence/encoder-order-audit.json')
                  for i in range(16)]
        assert all(a['status'] == 'PASS' for a in audits)
        ids = [sid for a in audits for sid in a['sample_ids']]
        expected = read(root / 'final/COMPLETE.json')['arms']['3000']['images']
        assert len(ids) == len(set(ids)) == expected
        proof = dict(status='PASS', images_verified=expected, nodes=audits,
                     root_schema_order_preserved=True, retained_layout_order_preserved=True,
                     actual_encoder_prompt_hashes_match_image_receipts=True)
        path = root / 'final/encoder-order-audit.json'
        save(path, proof)
        w.upload(path, prefix + 'execution/encoder-order-audit.json')
        print(json.dumps(dict(status='PASS', images=expected, publication_prefix=prefix)))
        return
    if args.canary:
        assert args.node == 0
    else:
        approved = w.wait_json(prefix + 'CANARY_PASS.json')
        assert approved['status'] == 'PASS'
        assert approved['campaign_sha256'] == configured['campaign_sha256']
        save(root / 'CANARY_APPROVED.json', approved)
    command = [sys.executable, '-u', str(root / 'start.py'), '--node', str(args.node)]
    if args.canary:
        command.append('--canary')
    subprocess.run(command, check=True)
    work = root / ('canary' if args.canary else 'work-3000')
    values = w.rows(work / 'caption/predictions.jsonl')
    for row in values:
        validate_selected(row, w.CAMPAIGN['sampling'], recompute=True)
    complete = read(work / 'COMPLETE.json')
    assert complete['status'] == 'PASS'
    audit = audit_order(work, 6000)
    path = work / 'evidence/encoder-order-audit.json'
    save(path, audit)
    w.upload(path, prefix + ('canary/' if args.canary else f'nodes/{args.node:02d}/')
             + 'evidence/encoder-order-audit.json')
    if args.canary:
        assert len(values) == 4 and complete['images'] >= 1
        reasons = {reason for row in values
                   for reason in row['generation_attempts'][0]['repetition_audit']['reasons']}
        assert reasons == {'exact_region_sequence_loop', 'long_string_loop'}
        assert any(row['selected_attempt'] == 0 for row in values)
        for path in (work / 'runs/step6000/images').glob('*.json'):
            receipt = read(path)
            assert receipt['step'] == 6000 and receipt['weights'] == 'EMA'
            assert receipt['cfg'] == 2.5 and receipt['steps'] == 50 and receipt['finite_latents']
        restores = list((work / 'runs/step6000/evidence').glob('restore-*.json'))
        assert restores
        for path in restores:
            proof = read(path)
            assert proof['keys'] == 463 and proof['strict'] and not proof['optimizer_loaded']
        proof = dict(complete, campaign_sha256=configured['campaign_sha256'],
                     pe_step=3000, dit_step=6000, both_rules_exercised=True,
                     retry_only_on_rule_hits=True)
        path = root / 'CANARY_PASS.json'
        save(path, proof)
        w.upload(path, prefix + 'CANARY_PASS.json')
    print(json.dumps(dict(status='PASS', node=args.node, canary=args.canary)))


if __name__ == '__main__':
    main()
