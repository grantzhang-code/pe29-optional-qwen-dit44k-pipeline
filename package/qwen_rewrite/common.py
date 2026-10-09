"""Frozen inputs, output receipts, and S3 persistence for the Qwen baseline."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
import boto3
from botocore.exceptions import ClientError
from PIL import Image

# Bound explicitly by run_qwen_rewrite.py; never inherit an old devbox/EFS run.
EFS = None
LOCAL = None
MODEL_CACHE = None
BUCKET = None
PREFIX = None
INPUT_SHA = 'dcadb3cde9b3d37d45008cc89aa6b5057201ab051fcdc1804a756181cdab8f03'
S3 = None


def client():
    global S3
    if S3 is None:
        S3 = boto3.client('s3', region_name='us-east-1')
    return S3


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.partial-{os.getpid()}')
    temp.write_bytes(raw)
    temp.replace(path)


def publish(key, raw, immutable=True, content_type='application/json'):
    args = {'Bucket': BUCKET, 'Key': PREFIX + key, 'Body': raw,
            'ContentType': content_type, 'Metadata': {'sha256': sha(raw)}}
    if immutable:
        args['IfNoneMatch'] = '*'
    try:
        client().put_object(**args)
    except ClientError as error:
        if not immutable or error.response['Error']['Code'] not in ('PreconditionFailed', '412'):
            raise
        existing = client().get_object(Bucket=BUCKET, Key=PREFIX + key)['Body'].read()
        if sha(existing) != sha(raw):
            raise ValueError(f'Immutable result changed: {key}')
    return f's3://{BUCKET}/{PREFIX}{key}'


def load_lock():
    return json.loads((EFS / 'run.lock.json').read_bytes())


def rows(rank=None):
    raw = (EFS / 'inputs/all.jsonl').read_bytes()
    if sha(raw) != INPUT_SHA:
        raise ValueError('Frozen GPT-size input snapshot changed')
    values = [json.loads(line) for line in raw.splitlines() if line]
    if len(values) != 1464 or len({r['input_index'] for r in values}) != 1464:
        raise ValueError('Benchmark coverage changed')
    selected = []
    for position, row in enumerate(values):
        if rank is not None and position % 64 != rank:
            continue
        size, selection = row['target_canvas_size'], row['selection']
        if (size['width'], size['height']) != (selection['gpt_output_width'], selection['gpt_output_height']):
            raise ValueError('PE target dimensions differ from GPT output')
        row = dict(row, rank=position % 64)
        selected.append(row)
    return selected


def source(row):
    expected = row['selection']
    path = LOCAL / 'sources' / (expected['source_sha256'] + '.png')
    if not path.exists():
        uri = urlsplit(row['source_s3_uri'])
        raw = client().get_object(Bucket=uri.netloc, Key=uri.path.lstrip('/'))['Body'].read()
        if len(raw) != expected['source_bytes'] or sha(raw) != expected['source_sha256']:
            raise ValueError('Source image bytes changed')
        atomic(path, raw)
    raw = path.read_bytes()
    if sha(raw) != expected['source_sha256']:
        raise ValueError('Cached source image changed')
    image = Image.open(BytesIO(raw))
    image.load()
    if image.size != (expected['source_width'], expected['source_height']):
        raise ValueError('Source image dimensions changed')
    return image.convert('RGB')


def output_size(row):
    width, height = row['target_canvas_size']['width'], row['target_canvas_size']['height']
    scale = min(1.0, math.sqrt(1048576 / (width * height)))
    w, h = max(32, max(1, round(width * scale)) // 32 * 32), max(32, max(1, round(height * scale)) // 32 * 32)
    while w * h > 1048576:
        if w >= h and w > 32:
            w -= 32
        elif h > 32:
            h -= 32
        else:
            raise ValueError('Cannot satisfy output area constraint')
    return w, h


def case_path(stage, row):
    return EFS / 'results' / stage / f"{row['input_index']:04d}.json"


def binding(row):
    return {'input_index': row['input_index'], 'sample_id': row['sample_id'], 'rank': row['rank'],
            'case_id': row['selection']['case_id'], 'instruction_en': row['instruction'],
            'source_s3_uri': row['source_s3_uri'], 'source_sha256': row['selection']['source_sha256'],
            'gpt_output_sha256': row['selection']['gpt_output_sha256'],
            'pe_target_canvas_size': row['target_canvas_size'], 'seed': 42 + row['input_index'],
            'run_lock_sha256': file_sha(EFS / 'run.lock.json')}


def validate_binding(record, row):
    if any(record.get(key) != value for key, value in binding(row).items()):
        raise ValueError(f"Resume binding changed: {row['input_index']}")


def save_record(stage, row, record):
    validate_binding(record, row)
    raw = encoded(record)
    uri = publish(f"results/{stage}/{row['input_index']:04d}.json", raw)
    atomic(case_path(stage, row), raw)
    return uri


def progress(rank, stage, completed=0, **extra):
    data = {'rank': rank, 'node': rank // 8, 'gpu': rank % 8, 'stage': stage,
            'completed': completed, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    raw = encoded(data)
    atomic(EFS / f'status/rank-{rank:02d}.json', raw)
    publish(f'status/rank-{rank:02d}.json', raw, immutable=False)
    print('QWEN_PROGRESS', json.dumps(data), flush=True)
