"""Frozen multi-reference PE prompt, portable inputs, and output verification."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    # Preserve caption field and reference/list order, including when publishing.
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require_s3(uri):
    if not isinstance(uri, str):
        raise ValueError('Expected s3://bucket/key')
    parsed = urlsplit(uri)
    if (parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/')
            or parsed.query or parsed.fragment or '..' in parsed.path.split('/')
            or any(c.isspace() for c in uri)):
        raise ValueError(f'S3 source required: {uri!r}')
    return uri


def load_lock():
    lock = json.loads((ROOT / 'model.lock.json').read_text())
    for key in ['checkpoint_uri', 'initial_artifact_uri', 'model_uri', 'dataset_uri', 'historical_output_uri']:
        require_s3(lock[key])
    require_s3(lock['environment_archive']['uri'])
    if lock['run_id'] != 'e6d88249d08d' or lock['checkpoint_step'] != 8000:
        raise ValueError('This pipeline is locked to e6d88249d08d checkpoint-8000')
    return lock


LOCK = load_lock()
os.environ['DLMS_MULTIREFERENCE_LIBRARY_ONLY'] = '1'
prompt_path = ROOT / 'frozen/multireference_editing_caption.py'
if file_sha(prompt_path) != LOCK['frozen_prompt_file_sha256']:
    raise ValueError('Frozen training prompt module changed')
spec = importlib.util.spec_from_file_location('frozen_multireference_prompt', prompt_path)
PROMPT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROMPT)
if PROMPT.pe_prompt_sha256() != LOCK['prompt_sha256']:
    raise ValueError('Training PE prompt contract changed')


def normalize_rows(rows):
    """Reject target leakage, ambiguous frame numbering, and nonportable inputs."""
    indices, ids, output = set(), set(), []
    for row in rows:
        required = {'input_index', 'sample_id', 'instruction', 'reference_images', 'target_canvas_size'}
        if not isinstance(row, dict) or not required <= set(row) <= required | {'mds_shard', 'mds_row'}:
            raise ValueError('Input fields must contain only source references, instruction, canvas and identity')
        index, sample_id = row['input_index'], row['sample_id']
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError('Duplicate or invalid input_index')
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError('Duplicate or invalid sample_id')
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('Empty instruction')
        size = row['target_canvas_size']
        if (not isinstance(size, dict) or set(size) != {'width', 'height'}
                or any(type(size[k]) is not int or size[k] <= 0 for k in size)):
            raise ValueError('Canvas must contain positive integer width/height')
        refs = row['reference_images']
        if not isinstance(refs, list) or not 1 <= len(refs) <= LOCK['max_references']:
            raise ValueError('Expected 1–12 ordered reference images')
        for position, ref in enumerate(refs):
            fields = {'reference_id', 'source_uri', 'sha256', 'width', 'height'}
            if not isinstance(ref, dict) or not fields <= set(ref) <= fields | {'role', 'bytes'}:
                raise ValueError('References require S3 URI, hash, dimensions, and frame ID; local paths are not inputs')
            if ref['reference_id'] != f'frame_{position}':
                raise ValueError('Reference order must exactly match frame_0, frame_1, ...')
            require_s3(ref['source_uri'])
            if not isinstance(ref['sha256'], str) or re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) is None:
                raise ValueError('Reference SHA256 is required')
            if any(type(ref[k]) is not int or ref[k] <= 0 for k in ['width', 'height']):
                raise ValueError('Invalid reference dimensions')
            if 'bytes' in ref and (type(ref['bytes']) is not int or ref['bytes'] <= 0):
                raise ValueError('Invalid reference byte count')
        output.append(row)
        indices.add(index)
        ids.add(sample_id)
    if not output:
        raise ValueError('Empty input dataset')
    return output


def parse_response(text, num_references):
    text = re.sub(r'^\s*<think>.*?</think>\s*', '', text, count=1, flags=re.S)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, count=1)
    text = re.sub(r'\s*```\s*$', '', text, count=1).strip()
    try:
        value = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, RecursionError) as exc:
        return None, str(exc), None
    try:
        if not isinstance(value, dict) or set(value) != {'normalized_edit_instruction', 'high_level_description', 'style_description', 'elements'}:
            raise ValueError('Unexpected caption root fields')
        elements = value['elements']
        if not isinstance(elements, dict) or set(elements) != {'background', 'layout', 'reference_edits'}:
            raise ValueError('Unexpected caption elements fields')
        ordered = PROMPT.pe_ordered_target({
            'normalized_edit_instruction': value['normalized_edit_instruction'],
            'high_level_description': value['high_level_description'],
            'style_description': value['style_description'],
            'compositional_deconstruction': {'background': elements['background'],
                'elements': elements['layout'], 'reference_edits': elements['reference_edits']}}, num_references)
        return ordered, None, None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, RecursionError) as exc:
        return value, None, f'{type(exc).__name__}: {exc}'


def sample_seed(index, attempt):
    return 42 + index + attempt * 10000 if attempt < 3 else 100042 + index + (attempt - 3) * 10000


def rows_for_rank(rows, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid rank/world size')
    return [r for r in rows if r['input_index'] % world_size == rank]


def reference_path(work, ref):
    return Path(work) / 'references' / (ref['sha256'] + '.image')


def build_request(row, tokenizer, work):
    from PIL import Image
    refs, size = row['reference_images'], row['target_canvas_size']
    user = PROMPT.build_pe_prompt(row['instruction'], len(refs), (size['width'], size['height']))
    parts = user.split('<image>')
    if len(parts) != len(refs) + 1:
        raise ValueError('Image placeholder count changed')
    content = [{'type': 'text', 'text': parts[0]}] if parts[0] else []
    for part in parts[1:]:
        content.extend([{'type': 'image', 'image': '<image>'}, {'type': 'text', 'text': part}])
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': PROMPT.PE_SYSTEM_PROMPT}, {'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith('<think>\n\n</think>\n\n'):
        raise ValueError('Non-thinking training assistant prefix changed')
    images = []
    for ref in refs:
        path = reference_path(work, ref)
        if file_sha(path) != ref['sha256'] or ('bytes' in ref and path.stat().st_size != ref['bytes']):
            raise ValueError('Reference image hash/size changed')
        with Image.open(path) as image:
            image.load()
            if image.size != (ref['width'], ref['height']):
                raise ValueError('Reference image dimensions changed')
            images.append(image.convert('RGB'))
    return {'prompt': prompt, 'multi_modal_data': {'image': images}}


def validate_prediction(record, source, config):
    if any(record.get(k) != value for k, value in source.items()):
        raise ValueError('Prediction source binding changed')
    expected = dict(run_sha256=config['run_sha256'], caption_run=LOCK['run_id'], checkpoint_step=8000,
                    checkpoint_manifest_sha256=LOCK['checkpoint_manifest_sha256'],
                    model_export_manifest_sha256=LOCK['model_export_manifest_sha256'],
                    prompt_sha256=LOCK['prompt_sha256'], bbox_order='xyxy_1000')
    if any(record.get(k) != value for k, value in expected.items()):
        raise ValueError('Prediction checkpoint/prompt/run identity changed')
    attempt = record.get('attempt')
    if type(attempt) is not int or not 0 <= attempt < len(config['sampling_temperatures']):
        raise ValueError('Invalid prediction attempt')
    if (record.get('sampling_temperature') != config['sampling_temperatures'][attempt]
            or record.get('sampling_seed') != sample_seed(source['input_index'], attempt)):
        raise ValueError('Prediction sampling changed')
    parsed, error, schema_error = parse_response(record['response'], len(source['reference_images']))
    valid = error is None and schema_error is None and record['finish_reason'] != 'length' and not record['inference_error']
    if (record['json_valid'] != (error is None) or record['schema_valid'] != valid
            or record['json_caption'] != parsed):
        raise ValueError('Prediction schema flags/content disagree with its raw response')
