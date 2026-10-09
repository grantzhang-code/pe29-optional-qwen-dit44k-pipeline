"""Restore the already validated H100 native runtime on an existing holder."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from checkpoint_sources import load_sources

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/ephemeral/pcb-v2-9c4d-editor-runtime')
ARCHIVE='s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/handovers/18k-caption-48k-ema-gsb-20261001-v1/archives/docker-native-editor.tar.gz'
SHA='ed7c5c4d9e79f5377a938cc3c63c9ccc1fa0f4f92b1f290d267794347be310dd'

def run(*cmd):
    print('RUN',*cmd,flush=True);subprocess.run(cmd,check=True)

def status(stage):
    p=ROOT/'runtime-stage.json';tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps({'stage':stage,'time':time.time()})+'\n');tmp.replace(p)

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    rootfs=BASE/'image-rootfs';archive=BASE/'docker-native-editor.tar.gz';extracted=BASE/'image-archive'
    if not (rootfs/'IMAGE_READY.json').exists():
        status('downloading_runtime')
        if not (BASE/'ARCHIVE_VERIFIED.json').exists():
            if not archive.exists() or archive.stat().st_size!=11532540962:
                run('aws','s3','cp',ARCHIVE,str(archive),'--profile','core-cn-s3-readwrite','--only-show-errors')
            with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
            (BASE/'ARCHIVE_VERIFIED.json').write_text(json.dumps({'sha256':SHA})+'\n')
        status('unpacking_runtime')
        if not (extracted/'manifest.json').exists():
            extracted.mkdir(exist_ok=True);run('tar','-xzf',str(archive),'-C',str(extracted))
        run(sys.executable,str(ROOT/'code/unpack_editor_image.py'),'--archive-dir',str(extracted),'--rootfs',str(rootfs))
    if not (BASE/'models/MODELS_VERIFIED.json').exists():
        status('downloading_models')
        for name,uri in [('Qwen3.8-27B',load_sources()['text_encoder']),('FLUX.2-klein-4B-vae',load_sources()['vae'])]:
            dst=BASE/'models'/name;dst.mkdir(parents=True,exist_ok=True)
            run('aws','s3','sync',uri,str(dst),'--profile','core-cn-s3-readwrite','--only-show-errors')
        run(sys.executable,str(ROOT/'code/verify_editor_models.py'),'--lock',str(ROOT/'models.lock.json'),'--models',str(BASE/'models'))
    status('ready')
    print('RUNTIME_READY',flush=True)

if __name__=='__main__':main()
