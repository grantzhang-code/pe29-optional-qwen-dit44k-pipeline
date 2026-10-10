"""Verify published model identity, output object checksums, and complete coverage."""
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
import base64,json,time
import worker as w

def read_jsonl(uri):
    u=urlsplit(uri)
    return [json.loads(line) for line in w.S3.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read().splitlines()]
def main():
    result=w.wait_json(w.CAMPAIGN['publication_prefix']+'COMPLETE.json')
    assert result['status']=='PASS' and result['inputs']==w.CAMPAIGN['total_inputs']
    evidence={};checkpoint_sha=w.wait_json(w.CAMPAIGN['shared_assets_prefix']+'models/DIT_READY.json')['checkpoint_sha256']
    for step,arm in w.CAMPAIGN['arms'].items():
        prefix=arm['checkpoint']['publication_prefix'];index=read_jsonl(prefix+'editor/score-index.jsonl');models=[];archives=[]
        for rank in range(arm['node_count']):
            node=prefix+f'nodes/{rank:02d}/'
            archives.append(w.wait_json(node+'reproduction/COMPLETE.json'))
            for gpu in range(len(w.CAMPAIGN['gpu_ids'][str(arm['pod_indices'][rank])])):
                proof=w.get(node+f'runs/step6000/evidence/restore-{gpu}.json')
                assert proof['step']==6000 and proof['weights']=='EMA' and proof['strict'] and proof['keys']==463
                assert proof['checkpoint_sha256']==checkpoint_sha and not proof['optimizer_loaded']
                models.append(proof)
        def verify(row):
            u=urlsplit(row['output_s3_uri'])
            obj=w.S3.head_object(Bucket=u.netloc,Key=u.path.lstrip('/'),ChecksumMode='ENABLED')
            assert obj['Metadata']['sha256']==row['output_sha256']
            assert obj['ChecksumSHA256']==base64.b64encode(bytes.fromhex(row['output_sha256'])).decode()
            assert obj['ContentLength']>0
            receipt=w.get(row['receipt_s3_uri'])
            assert receipt['image_sha256']==row['output_sha256'] and receipt['checkpoint_sha256']==checkpoint_sha
            assert receipt['step']==6000 and receipt['weights']=='EMA' and receipt['steps']==50 and receipt['cfg']==2.5 and receipt['finite_latents']
            return obj['ContentLength']
        with ThreadPoolExecutor(16) as pool:sizes=list(pool.map(verify,index))
        assert len(index)==result['arms'][step]['images']
        evidence[step]={'status':'PASS','images':len(index),'image_bytes':sum(sizes),'restore_proofs':len(models),'prepared_input_archives':archives,'all_s3_sha256_match':True,'all_receipts_match':True}
        print('VERIFIED',step,len(index),len(models),flush=True)
    proof={'status':'PASS','verified_at':time.time(),'dit_checkpoint_proof_sha256':checkpoint_sha,'arms':evidence}
    w.save(w.ROOT/'final/PUBLICATION_VERIFIED.json',proof)
    print(w.upload(w.ROOT/'final/PUBLICATION_VERIFIED.json',w.CAMPAIGN['publication_prefix']+'PUBLICATION_VERIFIED.json'),flush=True)
if __name__=='__main__':main()
