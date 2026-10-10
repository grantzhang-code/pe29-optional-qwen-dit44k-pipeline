"""Prepare the unchanged double-order judge and an Arnold 4x256 JobSet."""
from pathlib import Path
from urllib.parse import urlsplit
import argparse, json, re, shutil, sys
from materialize import HERE, profile, restore_bundle, read, sha, save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['pe3k','pe4k'],required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--job-name',required=True)
    p.add_argument('--aws-profile',default='core-cn-s3-readwrite')
    p.add_argument('--reference-bundle',type=Path)
    p.add_argument('--no-upload',action='store_true',help='Build and verify locally without publishing a bundle')
    a=p.parse_args();config=profile(a.profile);workspace=a.workspace.resolve()
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?',a.job_name):
        p.error('Use a unique lowercase JobSet name of at most 50 characters')
    cap=workspace/'caption';run=workspace/'dit4000';g=run/'gsb_vs_qwen21';out=workspace/'utp'
    assert read(run/'campaign.json')['dit_step']==4000
    assert read(run/'campaign.json')['arms'][str(config['pe_step'])]['checkpoint']['checkpoint_step']==config['pe_step']
    assert not (g/'protocol.json').exists(),'Use a fresh scoring directory; never overwrite a frozen protocol'
    restored=restore_bundle(config['bundles']['gsb'],workspace/'gsb-reference',a.aws_profile,a.reference_bundle)
    original=read(restored/'protocol.json')
    for name,digest in original['input_hashes'].items():assert sha(restored/'inputs'/name)==digest,name
    shutil.copytree(HERE/'gsb/templates'/a.profile,g,dirs_exist_ok=True)
    (g/'inputs').mkdir(exist_ok=True)
    for name in ['control-pairs.jsonl','control-protocol.json','reference-protocol.json','gsb_rubric.json']:
        shutil.copy2(restored/'inputs'/name,g/'inputs'/name)
    shutil.copy2(run/'campaign.json',run/'campaign-execution.json')
    sys.path.insert(0,str(HERE/'gsb'))
    from prepare_bundle import prepare
    from build_job import build
    proof=prepare(config['pe_step'],run=run,cap=cap)
    uri=proof['publication_prefix']+'code/bundle-'+proof['bundle_sha256']+'.tar.gz'
    if not a.no_upload:
        import boto3
        s=boto3.Session(profile_name=a.aws_profile).client('s3');u=urlsplit(uri);raw=(g/'bundle.tar.gz').read_bytes()
        s.put_object(Bucket=u.netloc,Key=u.path.lstrip('/'),Body=raw,Metadata={'sha256':proof['bundle_sha256']})
        import hashlib
        assert hashlib.sha256(s.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read()).hexdigest()==proof['bundle_sha256']
        u=urlsplit(proof['publication_prefix']+'protocol.json');raw=(g/'protocol.json').read_bytes()
        s.put_object(Bucket=u.netloc,Key=u.path.lstrip('/'),Body=raw,Metadata={'sha256':proof['protocol_sha256']})
    plan=build(config['pe_step'],uri,proof['bundle_sha256'],job_name=a.job_name,output_dir=out)
    save(out/'plan.json',dict(plan,bundle_uploaded=not a.no_upload))
    print(json.dumps(dict(status='PASS',pairs=proof['pairs'],judgments=proof['judgments'],workers=4,
                         concurrency_per_worker=256,bundle_uploaded=not a.no_upload,manifest=plan['manifest'])))
if __name__=='__main__':main()
