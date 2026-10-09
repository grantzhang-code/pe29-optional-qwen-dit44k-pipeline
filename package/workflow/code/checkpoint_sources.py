"""S3 is the only model source; filesystem paths here are disposable runtime caches."""
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / 'assets'
NAMES = {'qwen_pe', 'pe29_mcore', 'pe29_hf', 'pe_initializer', 'dit44_original',
         'dit44_ema', 'text_encoder', 'vae'}


def require_s3(value):
    if not isinstance(value, str):
        raise ValueError(f'Expected an S3 URI, got {value!r}')
    parsed = urlsplit(value)
    if (parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/')
            or parsed.query or parsed.fragment or any(c.isspace() for c in value)
            or '..' in parsed.path.split('/')):
        raise ValueError(f'Checkpoint/source must be an s3:// URI: {value!r}')
    return value


def load_sources(path=None):
    values = json.loads(Path(path or ROOT / 'checkpoint-sources.json').read_text())
    if set(values) != NAMES:
        raise ValueError('Checkpoint source set changed')
    for value in values.values():
        require_s3(value)
    return values


def cache_path(name):
    # These locations are derived, never accepted as checkpoint source configuration.
    return CACHE / {'qwen_pe': 'qwen-pe', 'pe29_hf': 'model-29000',
                    'dit44_ema': 'checkpoint-ema-compact'}[name]
