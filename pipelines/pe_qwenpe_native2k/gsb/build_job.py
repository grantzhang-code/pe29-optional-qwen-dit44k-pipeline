"""Four persistent CPU workers in the Core CN UTP queue."""
from pathlib import Path
import json, yaml

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
CLUSTER = 'p-iad-eagle'

def build(step, bundle_uri, bundle_sha, *, job_name, output_dir):
    reference = Path(__file__).with_name('jobset.template.yaml')
    manifest = yaml.safe_load(reference.read_text())
    manifest['metadata']['annotations'] = {
        'core-cn.canva.com/gsb-bundle-sha256': bundle_sha,
        'core-cn.canva.com/gsb-workers': '4',
        'core-cn.canva.com/gsb-concurrency-per-worker': '256',
    }
    replica = manifest['spec']['replicatedJobs'][0]
    assert replica['replicas'] == 1
    job = replica['template']['spec']
    job.update(completions=4, parallelism=4, activeDeadlineSeconds=7200)
    pod = job['template']['spec']
    worker = pod['containers'][0]
    assert worker['resources']['requests'] == {'cpu': '80', 'memory': '180Gi'}
    env = {item['name']: item for item in worker['env']}
    env.pop('SCORE_WORKER_RANK', None)
    env['JOB_COMPLETION_INDEX'] = dict(name='JOB_COMPLETION_INDEX', valueFrom=dict(fieldRef=dict(fieldPath="metadata.annotations['batch.kubernetes.io/job-completion-index']")))
    worker['env'] = list(env.values())
    code = f'''import boto3,hashlib,io,tarfile,os,pathlib,sys,subprocess
from urllib.parse import urlsplit
u=urlsplit({bundle_uri!r})
data=boto3.client('s3',region_name='us-east-1').get_object(Bucket=u.netloc,Key=u.path.lstrip('/'))['Body'].read()
assert hashlib.sha256(data).hexdigest()=={bundle_sha!r}
root=pathlib.Path('/tmp/pe39-gsb-score');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:archive.extractall(root,filter='data')
venv=root/'venv'
subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
python=venv/'bin/python'
subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(root/'requirements.txt')],check=True,env={{**os.environ,'PIP_CONSTRAINT':''}})
os.execv(str(python),[str(python),'-u',str(root/'utp_driver.py')])
'''
    worker['command'] = ['python3', '-u', '-c', code]
    assert not any('nvidia' in k for k in worker['resources']['requests'])
    assert pod['serviceAccountName'] == '${arnold_application_id}-model-trainer'
    assert pod['affinity']['podAntiAffinity']['requiredDuringSchedulingIgnoredDuringExecution'][0]['namespaceSelector'] == {}
    folder = Path(output_dir)
    folder.mkdir(exist_ok=True)
    path = folder / 'jobset.yaml'
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return dict(job_name=job_name, cluster=CLUSTER, namespace='b-core-cn-prod', manifest=str(path),
                workers=4, concurrency_per_worker=256, total_concurrency=1024,
                requested_cpu=320, requested_memory_gib=720, gpus=0,
                image=worker['image'], bundle_uri=bundle_uri, bundle_sha256=bundle_sha)
