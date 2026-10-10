"""CPU-only restore of the committed, pinned checkpoint while prior GPUs run."""
import fcntl
import json
import shutil
import sys
import traceback
from common import ROOT, CODE, CFG, save

sys.path.insert(0, str(CODE / 'repo'))
from scripts.checkpoint.common import Boto3ObjectStore
from scripts.checkpoint.mcore_s3 import inspect_committed_checkpoint, restore_checkpoint

lock = (ROOT / 'cpu-stage.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    save(ROOT / 'CPU_STAGE_STATUS.json', {'stage': 'restoring'})
    store = Boto3ObjectStore(profile='core-cn-s3-readwrite', max_concurrency=4)
    inspection = inspect_committed_checkpoint(store, CFG['checkpoint_uri'], verify_objects=True)
    assert inspection['manifest_sha256'] == CFG['checkpoint_manifest_sha256']
    assert inspection['step'] == CFG['checkpoint_step'] == 4000
    model = ROOT / 'model'
    model.mkdir(exist_ok=True)
    initial = model / 'initial-hf'
    if not initial.exists():
        shutil.copytree('/mnt/ephemeral/pe39c59-c1000-export-20261009/model/initial-hf', initial)
    result = restore_checkpoint(store, CFG['checkpoint_uri'], model / 'checkpoint-4000',
                                download_workers=8, replace_existing=False)
    assert result['manifest_sha256'] == CFG['checkpoint_manifest_sha256']
    save(ROOT / 'CPU_STAGE_STATUS.json', {'stage': 'complete', 'inspection': inspection, 'restore': result})
    print('CPU_STAGE_COMPLETE', json.dumps(result), flush=True)
except BaseException:
    save(ROOT / 'CPU_STAGE_STATUS.json', {'stage': 'failed', 'error': traceback.format_exc()})
    raise
