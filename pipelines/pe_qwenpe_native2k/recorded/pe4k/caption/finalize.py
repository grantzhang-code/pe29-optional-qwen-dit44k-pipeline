"""Reconcile caption coverage and fully read back published files."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib,json,sys,time
import worker as w
sys.path.insert(0,str(w.WORKFLOW/'code'))
from input_policy import validate_instruction

def fetch(asset):
    u=urlsplit(asset['uri']);raw=w.S3.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read()
    assert len(raw)==asset['bytes'] and hashlib.sha256(raw).hexdigest()==asset['sha256'],asset
    return raw

def main():
    root=w.ROOT/'final';root.mkdir(exist_ok=True)
    inputs=w.rows(w.WORKFLOW/'inputs.jsonl');by={r['sample_id']:r for r in inputs};assert len(by)==1464
    config=w.read(w.WORKFLOW/'campaign.json');all_values=[];valid=[];excluded=[];proofs=[]
    for rank in range(16):
        prefix=w.CAMPAIGN['publication_prefix']+f'nodes/{rank:02d}/'
        proof=w.wait_json(prefix+'COMPLETE.json');assert proof['status']=='PASS' and proof['scope']=='caption_only'
        assert proof['input_sha256']==w.CAMPAIGN['input_sha256'] and proof['inference_errors']==0
        artifacts=proof['artifacts'];contents={name:fetch(asset) for name,asset in artifacts.items()}
        values=[json.loads(l) for l in contents['caption/predictions.jsonl'].splitlines()]
        assert len(values)==proof['caption_count']
        for value in values:
            src=by[value['sample_id']]
            assert all(value[k]==src[k] for k in ['input_index','instruction','original_instruction','selection','source_s3_uri','target_canvas_size'])
            assert value['prompt_contract_sha256']==w.CAMPAIGN['training_prompt_sha256'] and not value.get('inference_error')
            validate_instruction(value,config)
        summaries=[json.loads(raw) for name,raw in contents.items() if name.endswith('/summary.json')]
        assert len(summaries)==len(w.CAMPAIGN['gpu_ids'][str(rank)])
        for summary in summaries:
            assert summary['complete'] and summary['results']['inference_errors']==0
            assert summary['checkpoint']==dict(step=4000,manifest_sha256=w.CAMPAIGN['arms']['4000']['checkpoint']['checkpoint_manifest_sha256'])
        all_values+=values;proofs.append(proof)
        valid += [json.loads(l) for l in contents['caption/valid.jsonl'].splitlines()]
        excluded += [json.loads(l) for l in contents['caption/excluded.jsonl'].splitlines()]
    assert len(all_values)==1464 and {v['sample_id'] for v in all_values}==set(by)
    assert len(valid)+len(excluded)==1464 and {v['sample_id'] for v in valid+excluded}==set(by)
    artifacts={}
    for name,values in [('caption/predictions.jsonl',all_values),('caption/valid.jsonl',valid),('caption/excluded.jsonl',excluded)]:
        w.jsonl(root/name,sorted(values,key=lambda v:v['input_index']));artifacts[name]=w.upload(root/name,w.CAMPAIGN['publication_prefix']+name)
    for name,path in [('inputs/pe-inputs.jsonl',w.WORKFLOW/'inputs.jsonl'),('inputs/qwen-pe-predictions.jsonl',w.WORKFLOW/'inputs/qwen-pe-predictions.jsonl'),('inputs/original.jsonl',w.WORKFLOW/'inputs/original.jsonl'),('campaign.json',w.ROOT/'campaign.json')]:
        artifacts[name]=w.upload(path,w.CAMPAIGN['publication_prefix']+name)
    for asset in artifacts.values():fetch(asset)
    w.save(root/'node-proofs.json',proofs)
    result=dict(status='PASS',scope='caption_only',inputs=1464,captions=len(all_values),valid=len(valid),excluded=len(excluded),inference_errors=0,
        pe_run='39c59fafbc35',checkpoint_step=4000,use_qwen_pe=True,qwen_rewrites_reused=1464,pe_captions_reused=0,
        newly_inferred_captions=1464,training_prompt_sha256=w.CAMPAIGN['training_prompt_sha256'],gpus=127,allocated_gpus=128,
        artifacts=artifacts,publication_prefix=w.CAMPAIGN['publication_prefix'],completed_at=time.time())
    w.save(root/'COMPLETE.json',result);complete=w.upload(root/'COMPLETE.json',w.CAMPAIGN['publication_prefix']+'COMPLETE.json');fetch(complete)
    verified=dict(status='PASS',caption_cases=1464,all_published_files_read_back_sha256=True,artifacts=artifacts,complete=complete)
    w.save(root/'PUBLICATION_VERIFIED.json',verified);w.upload(root/'PUBLICATION_VERIFIED.json',w.CAMPAIGN['publication_prefix']+'PUBLICATION_VERIFIED.json')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
