"""Prepare ordered references from frozen MDS bytes or explicit S3 JSONL inputs."""
import io
import json
from pathlib import Path
import pickle
import struct

from contracts import LOCK, ROOT, digest, file_sha, normalize_rows, reference_path
from storage import download, save_bytes


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Unexpected pickle class: {module}.{name}')


def mds_rows(raw, desc):
    count = struct.unpack_from('<I', raw)[0]
    if count != desc['samples']:
        raise ValueError('MDS sample count changed')
    offsets = struct.unpack_from(f'<{count + 1}I', raw, 4)
    if offsets[-1] != len(raw) or offsets[0] <= 4 * (count + 2):
        raise ValueError('Invalid MDS offset table')
    variable = sum(size is None for size in desc['column_sizes'])
    for start, end in zip(offsets, offsets[1:]):
        if not 0 <= start < end <= len(raw) or start + 4 * variable > end:
            raise ValueError('Invalid MDS row bounds')
        sizes = iter(struct.unpack_from(f'<{variable}I', raw, start))
        cursor, row = start + 4 * variable, {}
        for name, encoding, fixed in zip(desc['column_names'], desc['column_encodings'], desc['column_sizes'], strict=True):
            size = next(sizes) if fixed is None else fixed
            if cursor + size > end:
                raise ValueError('MDS column exceeds row bounds')
            value = raw[cursor:cursor + size]
            cursor += size
            if encoding == 'str':
                value = value.decode()
            elif encoding == 'pkl':
                value = RestrictedUnpickler(io.BytesIO(value)).load()
            elif encoding == 'int64':
                value = struct.unpack('<q', value)[0]
            elif encoding == 'float64':
                value = struct.unpack('<d', value)[0]
            elif encoding != 'bytes':
                raise ValueError('Unsupported MDS encoding: ' + encoding)
            row[name] = value
        if cursor != end:
            raise ValueError('MDS row has trailing bytes')
        yield row


def validate_image(raw, width, height):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError('Image dimensions differ from metadata')


def prepare_mds(work, s3):
    index_path = ROOT / 'frozen/mds-index.json'
    if file_sha(index_path) != LOCK['dataset_index_sha256']:
        raise ValueError('Frozen MDS index changed')
    rows = []
    for shard in json.loads(index_path.read_text())['shards']:
        data = shard['raw_data']
        relative = Path(data['basename'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unexpected MDS shard name')
        path = Path(work) / 'mds' / relative
        download(s3, LOCK['dataset_uri'] + data['basename'], path, data['hashes']['sha256'], data['bytes'])
        for shard_row, source in enumerate(mds_rows(path.read_bytes(), shard)):
            payloads = source['input_images']
            if not isinstance(payloads, list) or len(payloads) != source['num_inputs']:
                raise ValueError('MDS reference count changed')
            instructions = json.loads(source['instructions'])
            if not isinstance(instructions, list) or len(instructions) != 1:
                raise ValueError('Expected exactly one original instruction')
            widths, heights = json.loads(source['input_widths']), json.loads(source['input_heights'])
            uris, roles = json.loads(source['input_s3_paths']), json.loads(source['input_roles'])
            refs = []
            for i, (raw, width, height, uri, role) in enumerate(zip(payloads, widths, heights, uris, roles, strict=True)):
                if not isinstance(raw, bytes) or not raw:
                    raise ValueError('Missing embedded reference bytes')
                validate_image(raw, width, height)
                ref = dict(reference_id=f'frame_{i}', source_uri=uri, role=role,
                           sha256=digest(raw), width=width, height=height)
                save_bytes(reference_path(work, ref), raw)
                refs.append(ref)
            # Target pixels are read only to audit dimensions, never saved to the model input/cache.
            validate_image(source['output_image'], source['output_width'], source['output_height'])
            rows.append(dict(input_index=len(rows), sample_id=source['sample_id'], instruction=instructions[0],
                             reference_images=refs, target_canvas_size=dict(width=source['output_width'], height=source['output_height']),
                             mds_shard=data['basename'], mds_row=shard_row))
    if len(rows) != LOCK['expected_rows']:
        raise ValueError('Frozen benchmark must contain 419 rows')
    return normalize_rows(rows)


def prepare_custom(work, s3, path):
    rows = normalize_rows([json.loads(line) for line in Path(path).read_bytes().splitlines() if line])
    for row in rows:
        for ref in row['reference_images']:
            path = reference_path(work, ref)
            download(s3, ref['source_uri'], path, ref['sha256'], ref.get('bytes'))
            validate_image(path.read_bytes(), ref['width'], ref['height'])
    return rows
