"""One independent caption/admission/editor partition per held 8-H100 node."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,time,traceback
import common as w
import caption_support as c

ROOT=w.ROOT
BINDINGS=['input_index','sample_id','instruction','source_s3_uri','target_canvas_size','selection']

def sample(group,gpu,tag,settings=None):
    directory=ROOT/'caption'/tag/f'gpu-{gpu}';directory.mkdir(parents=True,exist_ok=True)
    source=directory/'inputs.jsonl';raw=b''.join(w.enc(v) for v in group)
    if source.exists():assert source.read_bytes()==raw
    else:source.write_bytes(raw)
    lock=w.read(ROOT/'caption-model.lock.json');proof_path=directory/'VERIFIED.json'
    if not proof_path.exists():
        command=c.make_command(gpu,source,directory,lock,node_count=8)
        env=c.environment(gpu,gpu);env.update(AWS_PROFILE='core-cn-s3-readwrite')
        if settings:
            command[command.index('--seed')+1]=str(settings['seed'])
            command[command.index('--request-window-size')+1]='4'
            env.update(PE_RETRY_TEMPERATURE=str(settings['temperature']),PE_RETRY_TOP_P=str(settings['top_p']),PE_RETRY_REPETITION_PENALTY=str(settings['repetition_penalty']),PE_RETRY_STRATEGY='bounded_schema_and_native_admission_retry_v1')
        with (directory/'run.log').open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=1800)
            except BaseException:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        assert code==0,(tag,gpu,code)
        summary=w.read(directory/'summary.json');values=w.rows(directory/'predictions.jsonl')
        assert summary['complete'] and summary['results']['inference_errors']==0
        assert summary['input']['sha256']==hashlib.sha256(raw).hexdigest()
        assert summary['checkpoint']==dict(step=29000,manifest_sha256=lock['checkpoint_manifest_sha256'])
        assert summary['results']['predictions_sha256']==w.sha(directory/'predictions.jsonl')
        assert len(values)==len(group) and {r['input_index'] for r in values}=={r['input_index'] for r in group}
        by={r['input_index']:r for r in group}
        for row in values:
            src=by[row['input_index']];assert all(row[k]==src[k] for k in BINDINGS)
            assert row['original_instruction']==src['original_instruction'] and not row.get('inference_error')
        w.save(proof_path,dict(status='PASS',cases=len(values),predictions_sha256=w.sha(directory/'predictions.jsonl'),input_sha256=hashlib.sha256(raw).hexdigest(),settings=settings,complete_utc=w.now()))
    proof=w.read(proof_path);assert proof['predictions_sha256']==w.sha(directory/'predictions.jsonl')
    return w.rows(directory/'predictions.jsonl')

def infer(groups,tag,settings=None):
    with ThreadPoolExecutor(8) as pool:
        futures=[pool.submit(sample,group,gpu,tag,settings) for gpu,group in enumerate(groups) if group]
        values=[row for future in futures for row in future.result()]
    values.sort(key=lambda r:r['input_index']);path=ROOT/'caption'/tag/'predictions.jsonl';path.write_bytes(b''.join(w.enc(v) for v in values));return path

def native_check(predictions,tag):
    dest=ROOT/'admission'/tag
    if not (dest/'input-audit.json').exists():
        dest.mkdir(parents=True,exist_ok=True)
        w.native(['/app/pcb_edit/check_caption_candidates.py','--predictions',str(predictions),'--work',str(dest)],dest/'check.log')
    audit=w.read(dest/'input-audit.json');assert audit['candidate_sha256']==w.sha(predictions)
    assert sum(audit['counts'].values())==len(w.rows(predictions));return dest,audit

# v2 exposes only first-pass sample/infer/native_check.
# Historical retrying worker.main and publication schedulers live in the v1 S3 archive.
