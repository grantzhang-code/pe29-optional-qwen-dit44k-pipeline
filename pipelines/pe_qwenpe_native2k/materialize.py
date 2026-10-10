"""Restore a verified execution bundle into a fresh, independently published run."""
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import argparse, hashlib, json, os, shutil, tarfile

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def profile(name):return read(HERE/'profiles'/f'{name}.json')
def s3_prefix(value):
    u=urlsplit(value)
    if u.scheme!='s3' or not u.netloc or not u.path.strip('/') or u.query or u.fragment:
        raise ValueError('Use an S3 bucket and nonempty run-specific key prefix')
    return value.rstrip('/')+'/'
def safe_destination(value):
    value=s3_prefix(value)
    for p in (HERE/'profiles').glob('*.json'):
        old=read(p)
        protected_prefixes=[old['caption_source'],old['generation_source']]
        protected_prefixes += [v['uri'].split('reproduction/',1)[0] for v in old['bundles'].values() if 'reproduction/' in v['uri']]
        for protected in protected_prefixes:
            if value.startswith(protected) or protected.startswith(value):
                raise ValueError('Choose a new run prefix; published evaluation results are immutable')
    return value
def restore_bundle(receipt,target,profile_name,bundle_file=None):
    target=Path(target)
    if target.exists() and any(target.iterdir()):raise ValueError('Restore requires an empty directory')
    target.mkdir(parents=True,exist_ok=True)
    archive=Path(bundle_file) if bundle_file else target.parent/(target.name+'.source.tar.gz')
    if not bundle_file:
        import boto3
        u=urlsplit(receipt['uri'])
        boto3.Session(profile_name=profile_name).client('s3').download_file(u.netloc,u.path.lstrip('/'),str(archive))
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256'],'Bundle checksum mismatch'
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p=PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
                raise ValueError('Unexpected link or unsafe archive path')
        tar.extractall(target,filter='data')
    manifest=target/'SOURCE_MANIFEST.json'
    if manifest.exists():
        for name,digest in read(manifest).items():assert sha(target/name)==digest,name
    return target
def configure(root,config,stage,output,workspace,runtime_from=None,dit_cache_from=None):
    campaign=read(root/'campaign.json');step=str(config['pe_step'])
    assert campaign['arms'][step]['checkpoint']['checkpoint_manifest_sha256']==config['caption_checkpoint']['checkpoint_manifest_sha256']
    assert campaign['input_sha256']==config['input_sha256']
    assert campaign['training_prompt_sha256']==config['prompt_sha256']
    assert campaign['qwen_pe_predictions_sha256']==config['qwen_predictions_sha256']
    destination=output+stage+'/'
    campaign['publication_prefix']=destination
    campaign['arms'][step]['checkpoint']['publication_prefix']=destination
    campaign['gsb_enabled']=False
    if dit_cache_from:
        if stage!='dit6000' or 'dit_cache_root' not in campaign:
            raise ValueError('This source bundle does not use a read-only DiT6000 cache')
        campaign['dit_cache_root']=str(Path(dit_cache_from).resolve())
    wf=root/'package/workflow'
    if stage!='caption':
        campaign['caption_source_prefix']=output+'caption/'
        campaign['reuse_root']=str(workspace/'caption')
        if stage=='dit6000':campaign['bundle_reuse_root']=str(workspace/'dit4000')
        # Model source URIs stay pinned. Avoid writing through another run's cache symlink.
        p=root/'worker.py';text=p.read_text()
        old="if not ckpt.exists():ckpt.symlink_to('/mnt/ephemeral/pe29-dit4k-cfg25-20261009/models/dit-ema',target_is_directory=True)"
        text=text.replace(old,'ckpt.mkdir(exist_ok=True)');p.write_text(text)
    save(root/'campaign.json',campaign)
    workflow=read(wf/'campaign.json');workflow.update(publication_prefix=destination,output_s3=destination)
    save(wf/'campaign.json',workflow)
    if runtime_from:
        runtime_from=Path(runtime_from).resolve()
        env=root/'package/assets/env';assert not env.exists()
        assert (runtime_from/'package/assets/env/caption/bin/python').exists()
        env.symlink_to(runtime_from/'package/assets/env',target_is_directory=True)
        names=['CAPTION_ENV_READY.json'] if stage=='caption' else ['CAPTION_ENV_READY.json','EDITOR_ENV_READY.json']
        for name in names:
            assert (runtime_from/name).is_file(),name
            shutil.copy2(runtime_from/name,root/name)
    save(root/'CONFIGURED.json',dict(status='PASS',pe_step=config['pe_step'],stage=stage,
         bundle_sha256=config['bundles'][stage]['sha256'],campaign_sha256=sha(root/'campaign.json'),
         source_manifest='SOURCE_MANIFEST.json records the original bundle before output-path configuration.',
         publication_prefix=destination,workspace=str(workspace)))
    return destination
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['pe3k','pe4k'],required=True)
    p.add_argument('--stage',choices=['caption','dit4000','dit6000'],required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output-s3',required=True)
    p.add_argument('--aws-profile',default='core-cn-s3-readwrite')
    p.add_argument('--bundle-file',type=Path)
    p.add_argument('--runtime-from',type=Path)
    p.add_argument('--dit-cache-from',type=Path,help='Verified read-only DiT6000 DCP cache, for PE-3k')
    a=p.parse_args();config=profile(a.profile)
    if a.stage not in config['bundles']:p.error('This stage has no verified source bundle for the selected PE profile')
    output=safe_destination(a.output_s3);workspace=a.workspace.resolve()
    root=restore_bundle(config['bundles'][a.stage],workspace/a.stage,a.aws_profile,a.bundle_file)
    destination=configure(root,config,a.stage,output,workspace,a.runtime_from,a.dit_cache_from)
    print(json.dumps(dict(status='PASS',root=str(root),publication_prefix=destination,pe_step=config['pe_step'],stage=a.stage)))
if __name__=='__main__':main()
