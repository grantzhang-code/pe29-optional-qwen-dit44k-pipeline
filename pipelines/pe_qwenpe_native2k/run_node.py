"""Run a restored stage on an authorized idle holder, with canary and order gates."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
from materialize import read, save, sha

def audit_order(work,dit_step):
    checked=[]
    for row in map(json.loads,(work/'score-index.jsonl').read_text().splitlines()):
        rid=row['request_id'];d=work/'inputs/cases'/rid
        prompts=read(d/'model-prompts.json');p=json.loads(prompts['positive']);pe=read(d/'pe.json');ann=read(d/'annotation.json')
        audit=read(work/'audits'/f'{rid}.json');receipt=read(work/f'runs/step{dit_step}/images/{rid}.json')
        assert list(p)[:3]==['normalized_edit_instruction','high_level_description','style_description']
        assert list(ann)==['normalized_edit_instruction','high_level_description','style_description','compositional_deconstruction']
        for name in ['positive','weak']:
            digest=hashlib.sha256(prompts[name].encode()).hexdigest()
            assert digest==audit[name+'_prompt_sha256']==receipt['case'][name+'_prompt_sha256']
        assert receipt['input_order']=='trainorder-v2' and row['caption_preparation_version']=='trainorder-v2-native2k-v1'
        positions={x['label']:i for i,x in enumerate(pe['elements']['layout'])}
        kept=[positions[x['label']] for x in p.get('compositional_deconstruction',{}).get('elements',[]) if x.get('label') in positions]
        assert kept==sorted(kept) and len(set(kept))==len(kept)
        assert [x['label'] for x in ann['compositional_deconstruction']['elements']]==[x['label'] for x in pe['elements']['layout']]
        checked.append(row['sample_id'])
    assert len(set(checked))==len(checked)
    return dict(status='PASS',images_verified=len(checked),sample_ids=checked,
                root_schema_order_preserved=True,retained_layout_order_preserved=True,
                actual_encoder_prompt_hashes_match_image_receipts=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','finalize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--node',type=int,default=0)
    p.add_argument('--canary',action='store_true')
    a=p.parse_args();root=a.root.resolve();record=read(root/'CONFIGURED.json')
    assert record['campaign_sha256']==sha(root/'campaign.json'),'Configured campaign changed'
    sys.path.insert(0,str(root));import worker as w
    stage=record['stage'];caption=stage=='caption';step=record['pe_step'];dit=int(stage.removeprefix('dit')) if not caption else None
    assert 0<=a.node<16 and set(w.CAMPAIGN['gpu_ids'])==set(map(str,range(16)))
    prefix=w.CAMPAIGN['publication_prefix']
    if a.command=='finalize':
        assert a.node==0 and not a.canary
        subprocess.run([sys.executable,'-u',str(root/'finalize.py')],check=True)
        if not caption:
            subprocess.run([sys.executable,'-u',str(root/'verify_publication.py')],check=True)
            proofs=[w.wait_json(prefix+f'nodes/{rank:02d}/evidence/encoder-order-audit.json') for rank in range(16)]
            assert all(v['status']=='PASS' for v in proofs)
            ids=[s for v in proofs for s in v['sample_ids']]
            expected=read(root/'final/COMPLETE.json')['arms'][str(step)]['images']
            assert len(ids)==len(set(ids))==expected
            proof=dict(status='PASS',images_verified=expected,root_schema_order_preserved=True,
                       retained_layout_order_preserved=True,actual_encoder_prompt_hashes_match_image_receipts=True,nodes=proofs)
            out=root/'final/encoder-order-audit.json';save(out,proof);w.upload(out,prefix+'execution/encoder-order-audit.json')
        # Existing GSB preparation reads the verified results/ directory.
        alias=root/'results'
        if not alias.exists():alias.symlink_to(root/'final',target_is_directory=True)
        print(json.dumps(dict(status='PASS',stage=stage,publication_prefix=prefix)))
        return
    if a.canary:assert a.node==0,'The recorded canary uses node zero'
    else:
        approved=w.wait_json(prefix+'CANARY_PASS.json')
        assert approved['status']=='PASS' and approved['campaign_sha256']==record['campaign_sha256']
        if not caption:save(root/'CANARY_APPROVED.json',approved)
    command=[sys.executable,'-u',str(root/'start.py'),'--node',str(a.node)]
    if a.canary:command.append('--canary')
    subprocess.run(command,check=True)
    work=root/('canary' if a.canary else f'work-{step}')
    complete=read(work/'COMPLETE.json');assert complete['status']=='PASS'
    if not caption:
        audit=audit_order(work,dit);out=work/'evidence/encoder-order-audit.json';save(out,audit)
        suffix='canary/' if a.canary else f'nodes/{a.node:02d}/'
        w.upload(out,prefix+suffix+'evidence/encoder-order-audit.json')
    if a.canary:
        assert complete['caption_count']==4
        assert complete['valid']>=1 if caption else complete['images']>=1
        if not caption:
            receipts=list((work/f'runs/step{dit}/images').glob('*.json'));assert receipts
            for f in receipts:
                r=read(f);assert r['step']==dit and r['weights']=='EMA' and r['cfg']==2.5 and r['steps']==50 and r['finite_latents']
            restores=list((work/f'runs/step{dit}/evidence').glob('restore-*.json'));assert restores
            for f in restores:
                r=read(f);assert r['keys']==463 and r['strict'] and not r['optimizer_loaded']
        proof=dict(complete,campaign_sha256=record['campaign_sha256'],pe_step=step,dit_step=dit)
        out=root/'CANARY_PASS.json';save(out,proof);w.upload(out,prefix+'CANARY_PASS.json')
    print(json.dumps(dict(status='PASS',stage=stage,node=a.node,canary=a.canary)))
if __name__=='__main__':main()
