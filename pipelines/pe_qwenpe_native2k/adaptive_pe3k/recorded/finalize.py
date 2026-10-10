"""Reconcile all input identities before committing a completed campaign."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib,json,time
import worker as w
import sys
sys.path.insert(0,str(w.WORKFLOW/"code"))
from input_policy import validate_instruction
from repetition_rules import validate_selected

def read_jsonl(uri):
    u=urlsplit(uri)
    return [json.loads(v) for v in w.S3.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read().splitlines()]

def main():
    root=w.ROOT/'final';root.mkdir(exist_ok=True)
    inputs=w.rows(w.WORKFLOW/'inputs.jsonl');expected={r['sample_id'] for r in inputs};assert len(expected)==1464
    arm_results={}
    for step,arm in w.CAMPAIGN['arms'].items():
        prefix=arm['checkpoint']['publication_prefix'];predictions=[];excluded=[];index=[];proofs=[];native=[];greedy=[];retry=[]
        for rank in range(arm['node_count']):
            node=prefix+f'nodes/{rank:02d}/';proof=w.wait_json(node+'COMPLETE.json');assert proof['status']=='PASS'
            preds=read_jsonl(node+'caption/predictions.jsonl');out=read_jsonl(node+'score-index.jsonl')
            ex=read_jsonl(node+'caption/excluded.jsonl');manifest=read_jsonl(node+'inputs/manifest.jsonl');by={r['request_id']:r for r in manifest}
            for rejected in proof['native_rejected']:
                row=by[rejected['request_id']];ex.append(dict(request_id=row['request_id'],sample_id=row['sample_id'],input_index=row['input_index'],error=rejected['error'],stage='native_admission'))
            assert len(out)==proof['images'] and len(out)+len(ex)==len(preds)
            assert len({r['sample_id'] for r in out+ex})==len(preds)
            predictions+=preds;excluded+=ex;index+=out;native+=manifest;proofs.append(proof)
            greedy+=read_jsonl(node+'caption/greedy-predictions.jsonl');retry+=read_jsonl(node+'caption/retry-predictions.jsonl')
        assert len(greedy)==1464 and len({r['sample_id'] for r in greedy})==1464
        assert {r['sample_id'] for r in retry}=={r['sample_id'] for r in greedy if r['repetition_audit']['hit']}
        greedy_by={r['sample_id']:r for r in greedy};retry_by={r['sample_id']:r for r in retry}
        assert len(predictions)==1464 and {r['sample_id'] for r in predictions}==expected
        assert len(index)+len(excluded)==1464 and {r['sample_id'] for r in index+excluded}==expected
        for row in predictions:
            assert row['prompt_contract_sha256']==w.CAMPAIGN['training_prompt_sha256']
            validate_instruction(row,w.CAMPAIGN)
            validate_selected(row,w.CAMPAIGN['sampling'])
            first=greedy_by[row['sample_id']]
            assert hashlib.sha256(first['response'].encode()).hexdigest()==row['generation_attempts'][0]['response_sha256']
            if row['selected_attempt']==1:assert row['response']==retry_by[row['sample_id']]['response']
            else:assert row['response']==first['response']
        directory=root/step;directory.mkdir(exist_ok=True)
        artifacts={}
        datasets={'caption/predictions.jsonl':predictions,'caption/greedy-predictions.jsonl':greedy,'caption/retry-predictions.jsonl':retry,'caption/excluded.jsonl':excluded,'editor/score-index.jsonl':index,'editor/native-manifest.jsonl':native,
                  'editor/images.manifest.jsonl':[{'sample_id':r['sample_id'],'request_id':r['request_id'],'uri':r['output_s3_uri'],'sha256':r['output_sha256']} for r in index]}
        admitted_ids={r['sample_id'] for r in index}
        assert all(not r['repetition_rejected'] for r in predictions if r['sample_id'] in admitted_ids)
        datasets['caption/admitted.jsonl']=[r for r in predictions if r['sample_id'] in admitted_ids]
        for name,values in datasets.items():
            path=directory/name;w.jsonl(path,sorted(values,key=lambda r:r.get('input_index',r['sample_id'])));artifacts[name]=w.upload(path,prefix+name)
        w.save(directory/'node-proofs.json',proofs);w.upload(directory/'node-proofs.json',prefix+'node-proofs.json')
        result=dict(status='PASS',checkpoint_step=int(step),input_count=1464,caption_count=len(predictions),images=len(index),excluded=len(excluded),
                    pe_checkpoint=arm['checkpoint']['checkpoint_uri'],dit_checkpoint=arm['checkpoint']['dit_checkpoint_uri'],weights='EMA',denoiser_keys=463,
                    cfg=2.5,steps=50,timeshift=4.6,max_pixels=4194304,input_order='trainorder-v2',encoder_recipe='Float8CurrentScaling',use_qwen_pe=True,sampling=w.CAMPAIGN['sampling'],seed=42,fresh_pe_captions=True,fresh_encoding=True,greedy_loop_hits=sum(len(r['generation_attempts'])==2 for r in predictions),retried=sum(len(r['generation_attempts'])==2 for r in predictions),unresolved_loop_cases=sum(r['repetition_rejected'] for r in predictions),artifacts=artifacts)
        w.save(directory/'COMPLETE.json',result);w.upload(directory/'COMPLETE.json',prefix+'ARM_COMPLETE.json');arm_results[step]=result
        print('ARM_COMPLETE',step,len(index),len(excluded),flush=True)
    result={'status':'PASS','inputs':w.CAMPAIGN['total_inputs'],'gpus':w.CAMPAIGN['gpus'],'arms':arm_results,'completed_at':time.time()}
    w.save(root/'COMPLETE.json',result);w.upload(root/'COMPLETE.json',w.CAMPAIGN['publication_prefix']+'COMPLETE.json')
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
