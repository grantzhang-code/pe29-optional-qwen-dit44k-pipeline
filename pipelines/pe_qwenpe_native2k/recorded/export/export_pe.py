from common import ROOT,CODE,CFG,save
import fcntl,os,subprocess
locks=[]
for p in ['/mnt/ephemeral/pcb-gpu-pipeline.lock',str(ROOT/'gpu-owner.lock')]:
    f=open(p,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cmd=['/opt/mcore-venv/bin/python',str(CODE/'repo/scripts/convert_mcore_checkpoint_for_inference.py'),
 '--checkpoint-s3',CFG['checkpoint_uri'],'--checkpoint-step',str(CFG['checkpoint_step']),
 '--checkpoint-manifest-sha256',CFG['checkpoint_manifest_sha256'],
 '--initial-artifact-s3',CFG['initial_artifact_uri'],'--initial-artifact-manifest-sha256',CFG['initial_artifact_manifest_sha256'],
 '--work-root',str(ROOT/'model'),'--output-dir',str(ROOT/'model/hf'),
 '--mcore-download-workers','8','--s3-transfer-max-concurrency','4','--megatron-bin','/opt/mcore-venv/bin/megatron',
 '--expected-tensors','738','--expected-mtp-tensors','15']
save(ROOT/'EXPORT_STARTED.json',{'pid':os.getpid(),'command':cmd})
subprocess.run(cmd,check=True,env={**os.environ,'AWS_PROFILE':'core-cn-s3-readwrite'})
subprocess.run(['/opt/mcore-venv/bin/python','-u',str(CODE/'seal_model.py')],check=True)
