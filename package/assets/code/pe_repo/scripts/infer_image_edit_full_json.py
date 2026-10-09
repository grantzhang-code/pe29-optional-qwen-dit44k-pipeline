#!/usr/bin/env python3
"""Infer complete inline image-edit JSON from two ordered images and an instruction.

The model-visible contract is byte-equivalent to the current SFT contract:

    reference image (frame_0) + Target image + instruction
        -> complete target_json.with_inline_edit JSON

The script consumes either one request from CLI flags or prepared JSONL rows,
loads local/HTTP/S3 images as ordered PIL objects, and runs a converted Hugging
Face checkpoint with vLLM.  Generation starts directly after the Qwen assistant
header; no synthetic ``<think></think>`` prefix is inserted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_inference_contract import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    SUPPORTED_TARGET_SCHEMAS,
    TARGET_SCHEMA,
    V4_TARGET_SCHEMA,
    V6_TARGET_SCHEMA,
    InlineTargetValidationError,
    build_user_prompt,
    prompt_contract_sha256,
    require_target_schema,
    validate_inline_target,
)
from dlms.datasets.image_edit_v6_hld_postprocess import (  # noqa: E402
    HLDPostprocessError,
    postprocess_v6_hld_labels,
)


DEFAULT_MAX_PIXELS = 1536 * 1536
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_NEW_TOKENS = 16384
CUDA_GRAPH_WORKER_VERIFICATION = "vllm_worker_extension_collective_rpc_v2"
VLLM_WORKER_EXTENSION_CLS = (
    "scripts.infer_image_edit_full_json.SidecarCompilationConfigWorkerExtension"
)
RUNTIME_STACK_VERIFICATION = "importlib_metadata_torch_public_and_cuda_exact_v3"
_RUNTIME_STACK_ARGUMENTS = (
    ("expected_torch_version", "torch"),
    ("expected_transformers_version", "transformers"),
    ("expected_vllm_version", "vllm"),
    ("expected_flashinfer_version", "flashinfer-python"),
    ("expected_flashinfer_cubin_version", "flashinfer-cubin"),
    ("expected_flashinfer_jit_cache_version", "flashinfer-jit-cache"),
    ("expected_triton_version", "triton"),
    ("expected_cuda_version", "torch_cuda"),
)
_REQUIRED_RUNTIME_STACK_PACKAGES = {
    "torch",
    "transformers",
    "vllm",
    "flashinfer-python",
    "triton",
    "torch_cuda",
}
_VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
_ASSISTANT_HEADER = "<|im_start|>assistant\n"
_PROMPT_TEMPLATE_VALIDATION_INSTRUCTION = "__INFERENCE_PROMPT_TEMPLATE_INSTRUCTION_SLOT__"
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_MEDIA_URI_FIELDS = ("s3_uri", "uri", "gcs_uri", "path")
_DESIGN_GENERATION_BUCKET = "design-generation-core-oss.canva.com"
_DESIGN_GENERATION_GETTY_PREFIX = "datasets/getty-images/"
_DESIGN_GENERATION_ACCOUNT_ID = "051687089423"
_S3_THREAD_LOCAL = threading.local()
_MEDIA_LOAD_MAX_ATTEMPTS = 3
_MEDIA_LOAD_RETRY_DELAYS_SECONDS = (0.25, 1.0)
QWEN_PROCESSOR_COMPATIBILITY_VERSION = (
    "qwen3vl_extreme_aspect_ratio_short_edge_expand_v1"
)
_QWEN_PROCESSOR_MAX_ASPECT_RATIO = 200.0
QWEN_INSTRUCTION_COMPATIBILITY_VERSION = (
    "runaway_of_the_repeat_collapse_v1"
)
_RUNAWAY_OF_THE_RE = re.compile(r"(?:\s+of the){16,}", re.IGNORECASE)
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_S3_ERROR_CODES = frozenset(
    {
        "ExpiredToken",
        "InternalError",
        "InternalFailure",
        "InvalidToken",
        "RequestExpired",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
    }
)


def parse_s3_media_uri(uri: str) -> tuple[str, str]:
    """Split an S3 media URI without interpreting object-key delimiters.

    S3 object keys may legally contain literal ``#`` and ``?`` characters.
    Generic URL parsers treat those characters as fragment/query delimiters
    and silently truncate the key, so media reads must split only at the
    bucket/key boundary.  Percent escapes are deliberately left untouched:
    an existing key containing the literal bytes ``%23`` must keep that
    identity rather than being changed into ``#``.
    """

    if not isinstance(uri, str) or not uri or uri != uri.strip() or not uri.startswith("s3://"):
        raise ValueError(f"invalid S3 media URI: {uri!r}")
    bucket, separator, key = uri[5:].partition("/")
    if (
        not bucket
        or not separator
        or not key
        or "@" in bucket
        or ":" in bucket
        or "?" in bucket
        or "#" in bucket
    ):
        raise ValueError(f"invalid S3 media URI: {uri!r}")
    return bucket, key


@dataclass(frozen=True)
class InferenceRecord:
    """One normalized inference request with fixed reference/Target ordering."""

    input_index: int
    sample_id: str
    instruction: str
    reference_image: str
    target_image: str
    evaluation_metadata: Mapping[str, Any] | None = None
    dataset_source: Mapping[str, Any] | None = None
    reference_image_identity: str | None = None
    target_image_identity: str | None = None


@dataclass(frozen=True)
class InferenceRuntime:
    """One initialized vLLM engine reusable across independent record streams."""

    llm: Any
    tokenizer: Any
    sampling_params: Any
    runtime_stack_validation: dict[str, Any] = field(default_factory=dict)
    cuda_graph_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceExecution:
    """Counters and the single-request convenience prediction from one stream."""

    counts: dict[str, int]
    single_prediction: dict[str, Any] | None
    telemetry: dict[str, Any] = field(default_factory=dict)


class SidecarCompilationConfigWorkerExtension:
    """Expose serializable worker graph evidence through vLLM's safe string RPC.

    vLLM 0.24 rejects callable RPC payloads unless insecure pickle
    serialization is explicitly enabled.  ``worker_extension_cls`` installs
    this method on every worker, allowing ``collective_rpc`` to address it by
    name while keeping the default msgspec-safe transport.
    """

    def sidecar_compilation_config_snapshot(self) -> dict[str, Any]:
        return _compilation_config_snapshot(self.vllm_config.compilation_config)


@dataclass
class _RecordImageLoad:
    """One record's decoded images or its isolated load failure."""

    images: tuple[Any, Any] | None = None
    error: BaseException | None = None
    traceback_text: str | None = None

    def close(self) -> None:
        """Close successfully decoded images exactly once."""

        images, self.images = self.images, None
        if images is None:
            return
        for image in images:
            try:
                image.close()
            except Exception:
                pass


@dataclass(frozen=True)
class _WindowExecution:
    """Rows and counters produced by one bounded vLLM request window."""

    rows: list[dict[str, Any]]
    generated_requests: int
    prompt_tokens: int
    output_tokens: int
    generation_seconds: float
    postprocess_seconds: float


_EVALUATION_METADATA_FIELDS = (
    "eval_task",
    "eval_task_version",
    "eval_task_matched_rule",
    "eval_selection_index_within_task",
    "eval_seed",
    "ground_truth_s3_uri",
    "source_dataset_s3",
)


def _evaluation_metadata(row: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {key: row[key] for key in _EVALUATION_METADATA_FIELDS if key in row}
    return metadata or None


def _dataset_source(row: Mapping[str, Any], *, input_index: int) -> dict[str, Any] | None:
    value = row.get("dataset_source")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"row {input_index}: dataset_source must be an object")
    return dict(value)


def _media_uri(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        uri = value.strip()
    elif isinstance(value, Mapping):
        uri = next(
            (
                str(value[key]).strip()
                for key in _MEDIA_URI_FIELDS
                if isinstance(value.get(key), str) and str(value[key]).strip()
            ),
            "",
        )
    else:
        uri = ""
    if not uri:
        raise ValueError(f"{field_name} does not contain a media URI/path")
    return uri


def _media_identity(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value.strip()


def _output_media(record: InferenceRecord) -> tuple[str, str]:
    """Return stable output identities while retaining runtime load paths internally."""

    return (
        record.reference_image_identity or record.reference_image,
        record.target_image_identity or record.target_image,
    )


def normalize_input_row(row: Mapping[str, Any], input_index: int) -> InferenceRecord:
    """Normalize a compact or prepared training row without reading its target label."""

    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"row {input_index}: instruction must be a non-empty string")

    reference_value: Any
    references = row.get("reference_images")
    if references is not None:
        if not isinstance(references, list) or len(references) != 1:
            raise ValueError(f"row {input_index}: reference_images must contain exactly frame_0")
        reference_value = references[0]
        if not isinstance(reference_value, Mapping) or reference_value.get("reference_id") != "frame_0":
            raise ValueError(f"row {input_index}: the only reference_id must be frame_0")
    elif "reference_image" in row:
        reference_value = row["reference_image"]
    else:
        raise ValueError(f"row {input_index}: missing reference_image/reference_images")

    target_value = row.get("target_gcs_uri") or row.get("target_image")
    if target_value is None:
        raise ValueError(f"row {input_index}: missing target_gcs_uri/target_image")
    sample_id = row.get("sample_id", input_index)
    return InferenceRecord(
        input_index=input_index,
        sample_id=str(sample_id),
        instruction=instruction.strip(),
        reference_image=_media_uri(reference_value, field_name=f"row {input_index} reference_image"),
        target_image=_media_uri(target_value, field_name=f"row {input_index} target_image"),
        reference_image_identity=_media_identity(
            row.get("reference_image_identity"),
            field_name=f"row {input_index} reference_image_identity",
        ),
        target_image_identity=_media_identity(
            row.get("target_image_identity"),
            field_name=f"row {input_index} target_image_identity",
        ),
        evaluation_metadata=_evaluation_metadata(row),
        dataset_source=_dataset_source(row, input_index=input_index),
    )


def iter_input_jsonl(
    path: Path,
    *,
    max_samples: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[InferenceRecord]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: rank={rank} world_size={world_size}")
    input_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if max_samples and input_index >= max_samples:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            if input_index % world_size == rank:
                yield normalize_input_row(row, input_index)
            input_index += 1


def _user_content(instruction: str) -> list[dict[str, str]]:
    """Convert the two literal image markers to ordered Qwen multimodal parts."""

    template_prompt = build_user_prompt(_PROMPT_TEMPLATE_VALIDATION_INSTRUCTION)
    if template_prompt.count("<image>") != 2:
        raise RuntimeError("the training user prompt must contain exactly two <image> markers")
    user_prompt = build_user_prompt(instruction)
    parts = user_prompt.split("<image>", 2)
    if len(parts) != 3:
        raise RuntimeError("the training user prompt no longer contains its two fixed <image> markers")
    return [
        {"type": "text", "text": parts[0]},
        {"type": "image", "image": "<image>"},
        {"type": "text", "text": parts[1]},
        {"type": "image", "image": "<image>"},
        {"type": "text", "text": parts[2]},
    ]


def expected_direct_json_prompt(instruction: str) -> str:
    """Return the exact unexpanded Qwen sequence used by direct-JSON SFT."""

    user_prompt = build_user_prompt(instruction).replace("<image>", _VISION_MARKER, 2)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{_ASSISTANT_HEADER}"
    )


def render_direct_json_prompt(tokenizer: Any, instruction: str) -> str:
    """Render through the checkpoint tokenizer and remove only its empty think prefix."""

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(instruction)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return a string")
    assistant_offset = rendered.rfind(_ASSISTANT_HEADER)
    if assistant_offset < 0:
        raise RuntimeError("Qwen chat template did not emit an assistant header")
    prefix_end = assistant_offset + len(_ASSISTANT_HEADER)
    assistant_prefix = rendered[prefix_end:]
    if assistant_prefix and not re.fullmatch(r"<think>\s*</think>\s*", assistant_prefix):
        raise RuntimeError(
            f"Qwen chat template inserted an unexpected generation prefix: {assistant_prefix!r}"
        )
    rendered = rendered[:prefix_end]

    expected = expected_direct_json_prompt(instruction)
    if rendered != expected:
        raise RuntimeError(
            "checkpoint tokenizer chat template does not match the pinned image-edit SFT prompt"
        )
    if rendered.count("<|image_pad|>") != 2 or not rendered.endswith(_ASSISTANT_HEADER):
        raise RuntimeError("rendered prompt does not preserve two images and direct JSON generation")
    return rendered


def parse_model_response(response: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse strict JSON and independently validate the current inline-edit schema."""

    text = _THINK_RE.sub("", response, count=1)
    text = _FENCE_START_RE.sub("", text, count=1)
    text = _FENCE_END_RE.sub("", text, count=1).strip()

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant: {token}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}", None

    try:
        validate_inline_target(value)
    except InlineTargetValidationError as exc:
        return value, None, f"{type(exc).__name__}: {exc}"
    return value, None, None


def _read_uri_bytes(
    uri: str,
    *,
    s3_client: Any | None = None,
    gcs_client: Any | None = None,
) -> bytes:
    if uri.startswith("parquet+s3://"):
        # Reuse the exact ETag-pinned Parquet-cell resolver used during SFT.
        # It performs column-pruned range reads and validates row/column/index.
        from dlms.patches.s3_image_load import _read_parquet_s3_bytes

        return _read_parquet_s3_bytes(uri)
    if uri.startswith("s3://"):
        bucket, key = parse_s3_media_uri(uri)
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    parsed = urlparse(uri)
    if parsed.scheme == "gs":
        from dlms.gcp_wif import download_gcs_bytes

        return download_gcs_bytes(uri, client=gcs_client)
    if parsed.scheme in {"http", "https"}:
        with urlopen(uri, timeout=120) as response:  # noqa: S310 - user explicitly supplies media URI
            return response.read()
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(f"unsupported media URI scheme {parsed.scheme!r}: {uri}")
    else:
        path = Path(uri).expanduser()
    return path.read_bytes()


def qwen_processor_compatible_rgb_image(image: Any) -> Any:
    """Expand only an impossible short edge so Qwen3-VL can process the row.

    Transformers rejects images whose absolute aspect ratio exceeds 200.  A
    one-pixel-wide source can therefore make vLLM fail an otherwise healthy
    request window.  Stretching only the short axis is the minimal deterministic
    repair: it keeps every row, leaves ordinary images byte-for-byte untouched,
    and preserves normalized bbox coordinates under the axis-aligned scale.
    """

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    if max(width, height) / min(width, height) <= _QWEN_PROCESSOR_MAX_ASPECT_RATIO:
        return image

    if width < height:
        resized = (math.ceil(height / _QWEN_PROCESSOR_MAX_ASPECT_RATIO), height)
    else:
        resized = (width, math.ceil(width / _QWEN_PROCESSOR_MAX_ASPECT_RATIO))
    from PIL import Image

    return image.resize(resized, resample=Image.Resampling.BICUBIC)


def qwen_context_compatible_instruction(
    instruction: str,
) -> tuple[str, dict[str, Any] | None]:
    """Collapse only a pathological generated phrase loop.

    One frozen source row contains more than 129k characters because ``of the``
    repeats thousands of times.  It exceeds the pinned 32k context and causes
    vLLM to reject the entire request window.  The surrounding prefix and
    suffix contain the intended edit instruction, so collapsing only a run of
    at least 16 identical phrases preserves them while leaving normal prompts
    byte-for-byte unchanged.
    """

    effective, replacements = _RUNAWAY_OF_THE_RE.subn(" of the", instruction)
    if replacements == 0:
        return instruction, None
    return effective, {
        "version": QWEN_INSTRUCTION_COMPATIBILITY_VERSION,
        "original_characters": len(instruction),
        "effective_characters": len(effective),
        "collapsed_runs": replacements,
    }


def load_rgb_image(uri: str, *, s3_client: Any | None = None, gcs_client: Any | None = None) -> Any:
    from PIL import Image

    payload = _read_uri_bytes(uri, s3_client=s3_client, gcs_client=gcs_client)
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = image.convert("RGB")
    try:
        compatible = qwen_processor_compatible_rgb_image(rgb)
    except BaseException:
        rgb.close()
        raise
    if compatible is not rgb:
        original_size = list(rgb.size)
        rgb.close()
        print(
            json.dumps(
                {
                    "event": "media_processor_compatibility_repair",
                    "policy": QWEN_PROCESSOR_COMPATIBILITY_VERSION,
                    "uri": uri,
                    "original_size": original_size,
                    "processor_size": list(compatible.size),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return compatible


@lru_cache(maxsize=8)
def _cached_s3_client(download_workers: int) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=max(download_workers, 2),
        ),
    )


def _s3_client_config(*, max_pool_connections: int) -> Any:
    from botocore.config import Config

    return Config(
        retries={"max_attempts": 5, "mode": "adaptive"},
        max_pool_connections=max(max_pool_connections, 2),
    )


def _access_grant_target(uri: str) -> tuple[str, str] | None:
    """Return the known Access Grants account/target required by *uri*."""

    try:
        bucket, key = parse_s3_media_uri(uri)
    except ValueError:
        return None
    if bucket == _DESIGN_GENERATION_BUCKET and key.startswith(_DESIGN_GENERATION_GETTY_PREFIX):
        return (
            _DESIGN_GENERATION_ACCOUNT_ID,
            f"s3://{_DESIGN_GENERATION_BUCKET}/{_DESIGN_GENERATION_GETTY_PREFIX}",
        )
    return None


def _refreshable_access_grant_s3_client(
    *,
    account_id: str,
    target: str,
    max_pool_connections: int,
) -> Any:
    """Create a long-lived S3 client backed by auto-refreshing Access Grants."""

    import boto3
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session
    from datetime import datetime, timedelta, timezone

    region = os.environ.get("AWS_REGION", "us-east-1")

    def fetch() -> dict[str, str]:
        response = (
            boto3.Session()
            .client("s3control", region_name=region)
            .get_data_access(
                AccountId=account_id,
                Target=target,
                Permission="READ",
                Privilege="Default",
            )
        )
        credentials = response["Credentials"]
        expiration = credentials.get("Expiration")
        if isinstance(expiration, datetime):
            expiry_time = expiration.astimezone(timezone.utc).isoformat()
        else:
            # The service currently returns an expiration, but keep a guarded
            # refresh window for older botocore response models.
            expiry_time = (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat()
        return {
            "access_key": credentials["AccessKeyId"],
            "secret_key": credentials["SecretAccessKey"],
            "token": credentials["SessionToken"],
            "expiry_time": expiry_time,
        }

    botocore_session = get_session()
    botocore_session._credentials = RefreshableCredentials.create_from_metadata(  # noqa: SLF001
        metadata=fetch(),
        refresh_using=fetch,
        method="s3-access-grants",
    )
    botocore_session.set_config_variable("region", region)
    return boto3.Session(botocore_session=botocore_session).client(
        "s3",
        region_name=region,
        config=_s3_client_config(max_pool_connections=max_pool_connections),
    )


def _thread_s3_client(uri: str, *, download_workers: int) -> Any:
    """Return a per-download-thread client, routing known cross-account media."""

    bucket, _key = parse_s3_media_uri(uri)
    grant = _access_grant_target(uri)
    identity = (bucket, grant)
    clients = getattr(_S3_THREAD_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _S3_THREAD_LOCAL.clients = clients
    if identity not in clients:
        if grant is None:
            import boto3

            clients[identity] = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                config=_s3_client_config(max_pool_connections=2),
            )
        else:
            account_id, target = grant
            clients[identity] = _refreshable_access_grant_s3_client(
                account_id=account_id,
                target=target,
                max_pool_connections=2,
            )
    return clients[identity]


def _discard_thread_s3_client(uri: str) -> None:
    """Drop only this thread's client after a retryable transport failure."""

    bucket, _key = parse_s3_media_uri(uri)
    identity = (bucket, _access_grant_target(uri))
    clients = getattr(_S3_THREAD_LOCAL, "clients", None)
    if not isinstance(clients, dict):
        return
    client = clients.pop(identity, None)
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _status_code(value: Any) -> int | None:
    """Normalize integer and enum-like HTTP status values."""

    if callable(value):
        value = value()
    value = getattr(value, "value", value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_retryable_media_read_error(error: BaseException) -> bool:
    """Return whether re-reading an unchanged remote object can safely help."""

    try:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            CredentialRetrievalError,
            NoCredentialsError,
            ParamValidationError,
            PartialCredentialsError,
        )
    except ImportError:  # pragma: no cover - boto3 is required in deployed jobs
        pass
    else:
        if isinstance(error, ClientError):
            response = error.response if isinstance(error.response, Mapping) else {}
            error_code = str(response.get("Error", {}).get("Code", ""))
            status = _status_code(response.get("ResponseMetadata", {}).get("HTTPStatusCode"))
            return error_code in _RETRYABLE_S3_ERROR_CODES or status in _RETRYABLE_HTTP_STATUS_CODES
        if isinstance(
            error,
            (CredentialRetrievalError, NoCredentialsError, ParamValidationError, PartialCredentialsError),
        ):
            return False
        if isinstance(error, BotoCoreError):
            # Includes response-stream truncation, connection closure, endpoint
            # connection failure, and read/connect timeouts. Botocore retries the
            # GetObject request itself, but cannot replay a failed Body.read().
            return True

    from http.client import IncompleteRead
    from urllib.error import HTTPError, URLError

    if isinstance(error, HTTPError):
        return _status_code(error.code) in _RETRYABLE_HTTP_STATUS_CODES
    if isinstance(error, URLError):
        return True
    if isinstance(error, (TimeoutError, ConnectionError, IncompleteRead)):
        return True

    if type(error).__module__.startswith("google.api_core.exceptions"):
        return _status_code(getattr(error, "code", None)) in _RETRYABLE_HTTP_STATUS_CODES
    return False


def _load_media_uri_with_retry(
    uri: str,
    *,
    download_workers: int,
    gcs_client: Any | None,
) -> Any:
    """Load one image with bounded remote-read retries and strict final failure."""

    for attempt in range(1, _MEDIA_LOAD_MAX_ATTEMPTS + 1):
        try:
            return load_rgb_image(
                uri,
                s3_client=(
                    _thread_s3_client(uri, download_workers=download_workers)
                    if uri.startswith("s3://")
                    else None
                ),
                gcs_client=gcs_client,
            )
        except BaseException as error:
            if attempt >= _MEDIA_LOAD_MAX_ATTEMPTS or not _is_retryable_media_read_error(error):
                raise
            if uri.startswith("s3://"):
                _discard_thread_s3_client(uri)
            delay = _MEDIA_LOAD_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                json.dumps(
                    {
                        "event": "media_load_retry",
                        "uri": uri,
                        "attempt": attempt,
                        "max_attempts": _MEDIA_LOAD_MAX_ATTEMPTS,
                        "delay_seconds": delay,
                        "error_type": type(error).__name__,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("bounded media retry loop did not return or raise")


def preflight_known_s3_access_grants() -> list[str]:
    """Acquire each cross-account grant once, without exposing credentials."""

    target = f"s3://{_DESIGN_GENERATION_BUCKET}/{_DESIGN_GENERATION_GETTY_PREFIX}"
    _refreshable_access_grant_s3_client(
        account_id=_DESIGN_GENERATION_ACCOUNT_ID,
        target=target,
        max_pool_connections=2,
    )
    return [target]


@lru_cache(maxsize=8)
def _download_executor(download_workers: int) -> ThreadPoolExecutor:
    """Keep download threads alive so their scoped S3 clients can be reused."""

    return ThreadPoolExecutor(
        max_workers=download_workers,
        thread_name_prefix=f"image-edit-media-{download_workers}",
    )


@lru_cache(maxsize=1)
def _cached_gcs_client() -> Any:
    from dlms.gcp_wif import make_gcs_client

    return make_gcs_client()


def _format_exception(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _load_record_image_results(
    records: Sequence[InferenceRecord], *, download_workers: int
) -> list[_RecordImageLoad]:
    """Load all media concurrently while isolating failures by record."""

    if download_workers <= 0:
        raise ValueError("download_workers must be positive")
    uris = [uri for record in records for uri in (record.reference_image, record.target_image)]
    gcs_client = None
    if any(uri.startswith("gs://") for uri in uris):
        gcs_client = _cached_gcs_client()
    executor = _download_executor(download_workers)

    def load_uri(uri: str) -> Any:
        return _load_media_uri_with_retry(
            uri,
            download_workers=download_workers,
            gcs_client=gcs_client,
        )

    futures: list[Any] = []
    loaded: list[Any | None] = [None] * len(uris)
    failures: list[tuple[BaseException, str] | None] = [None] * len(uris)
    try:
        for uri in uris:
            futures.append(executor.submit(load_uri, uri))
    except BaseException:
        # Submission itself can fail after earlier work has started.  Drain
        # every submitted future before returning control to the caller.
        for future in futures:
            future.cancel()
        for index, future in enumerate(futures):
            if future.cancelled():
                continue
            try:
                loaded[index] = future.result()
            except BaseException:
                pass
        for image in loaded:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
        raise

    # Always drain every media future.  This both avoids leaking decoded PIL
    # images and lets one bad URI affect only its own record in continue mode.
    for index, future in enumerate(futures):
        try:
            loaded[index] = future.result()
        except BaseException as exc:
            failures[index] = (exc, _format_exception(exc))

    results: list[_RecordImageLoad] = []
    try:
        for record_index in range(len(records)):
            reference_index = record_index * 2
            target_index = reference_index + 1
            failure = failures[reference_index] or failures[target_index]
            if failure is None:
                reference = loaded[reference_index]
                target = loaded[target_index]
                if reference is None or target is None:
                    raise AssertionError("image future completed without an image or failure")
                results.append(_RecordImageLoad(images=(reference, target)))
                loaded[reference_index] = None
                loaded[target_index] = None
                continue

            # Close the other image if only one URI in this record failed.
            for image_index in (reference_index, target_index):
                image = loaded[image_index]
                loaded[image_index] = None
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
            error, traceback_text = failure
            results.append(_RecordImageLoad(error=error, traceback_text=traceback_text))
    except BaseException:
        _close_image_loads(results)
        for image in loaded:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
        raise
    return results


def load_record_images(records: Sequence[InferenceRecord], *, download_workers: int) -> list[tuple[Any, Any]]:
    """Load ordered image pairs, preserving the original fail-fast API."""

    results = _load_record_image_results(records, download_workers=download_workers)
    first_failure = next((result for result in results if result.error is not None), None)
    if first_failure is not None:
        for result in results:
            result.close()
        assert first_failure.error is not None
        raise first_failure.error
    return [result.images for result in results if result.images is not None]


def expected_runtime_stack(args: argparse.Namespace) -> dict[str, str]:
    """Return the pinned runtime identity, with optional FlashInfer artifacts.

    FlashInfer 0.6.14 no longer publishes matching ``flashinfer-cubin`` or
    ``flashinfer-jit-cache`` distributions.  Those package pins remain
    supported for older baked runtimes, but the PR #36 runtime contract only
    requires ``flashinfer-python`` plus the core Torch/vLLM stack.
    """

    values = {
        contract_name: getattr(args, argument_name, None)
        for argument_name, contract_name in _RUNTIME_STACK_ARGUMENTS
    }
    configured = {name: str(value) for name, value in values.items() if value is not None}
    if configured and not _REQUIRED_RUNTIME_STACK_PACKAGES.issubset(configured):
        missing = sorted(_REQUIRED_RUNTIME_STACK_PACKAGES - set(configured))
        raise ValueError(f"incomplete expected runtime stack; missing: {missing}")
    return configured


def validate_runtime_stack(args: argparse.Namespace) -> dict[str, Any]:
    """Fail before engine creation unless every pinned runtime version matches.

    PyTorch CUDA wheels publish a PEP 440 local suffix (for example,
    ``2.11.0+cu130``).  The Torch public version is matched against the pinned
    version while ``torch.version.cuda`` remains a separate exact gate.  Every
    other package continues to require an exact metadata-version match.
    """

    expected = expected_runtime_stack(args)
    if not expected:
        return {"enabled": False}

    from importlib.metadata import PackageNotFoundError, version

    try:
        actual = {
            package: version(package)
            for package in expected
            if package not in {"torch", "torch_cuda"}
        }
        if "torch" in expected or "torch_cuda" in expected:
            import torch

            if "torch" in expected:
                actual["torch"] = version("torch")
            if "torch_cuda" in expected:
                actual["torch_cuda"] = str(torch.version.cuda)
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError("cannot inspect the pinned inference runtime stack") from exc
    from packaging.version import InvalidVersion, Version

    mismatches: dict[str, dict[str, str]] = {}
    for package, wanted in expected.items():
        observed = actual.get(package)
        matches = observed == wanted
        if package == "torch":
            try:
                matches = Version(str(observed)).public == Version(wanted).public
            except InvalidVersion:
                matches = False
        if not matches:
            mismatches[package] = {"expected": wanted, "actual": str(observed)}
    if mismatches:
        raise RuntimeError(
            "inference runtime stack mismatch: "
            f"mismatches={mismatches!r} expected={expected!r} actual={actual!r}"
        )
    return {
        "enabled": True,
        "verification": RUNTIME_STACK_VERIFICATION,
        "versions": actual,
    }


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 2, "video": 0},
        "mm_processor_kwargs": {"max_pixels": args.max_pixels},
        "disable_log_stats": not args.use_mtp,
        "seed": args.seed,
    }
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    mm_processor_cache_gb = getattr(args, "mm_processor_cache_gb", None)
    if mm_processor_cache_gb is not None:
        kwargs["mm_processor_cache_gb"] = mm_processor_cache_gb
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    gdn_prefill_backend = getattr(args, "gdn_prefill_backend", None)
    if gdn_prefill_backend:
        kwargs["gdn_prefill_backend"] = gdn_prefill_backend
    mm_encoder_attn_backend = getattr(args, "mm_encoder_attn_backend", None)
    if mm_encoder_attn_backend:
        kwargs["mm_encoder_attn_backend"] = mm_encoder_attn_backend
    cudagraph_mode = getattr(args, "cudagraph_mode", None)
    max_cudagraph_capture_size = getattr(args, "max_cudagraph_capture_size", None)
    cudagraph_mm_encoder = bool(getattr(args, "cudagraph_mm_encoder", False))
    cudagraph_specialize_lora = getattr(args, "cudagraph_specialize_lora", None)
    graph_requested = bool(
        cudagraph_mode
        or max_cudagraph_capture_size is not None
        or cudagraph_mm_encoder
        or cudagraph_specialize_lora is not None
    )
    if graph_requested:
        kwargs["worker_extension_cls"] = VLLM_WORKER_EXTENSION_CLS
    if args.use_mtp and cudagraph_mode in {"FULL", "FULL_AND_PIECEWISE"}:
        main_backend = str(args.attention_backend or "").upper()
        draft_backend = getattr(args, "speculative_attention_backend", None)
        draft_backend_upper = str(draft_backend).upper() if draft_backend is not None else None
        expected_vllm_version = getattr(args, "expected_vllm_version", None)
        if expected_vllm_version in {None, "0.24.0"}:
            if main_backend != "FLASH_ATTN":
                raise ValueError(
                    "vLLM 0.24 requires attention_backend=FLASH_ATTN for MTP with full CUDA graphs"
                )
            if draft_backend_upper not in {None, "AUTO", "FLASH_ATTN"}:
                raise ValueError(
                    "vLLM 0.24 requires speculative_attention_backend=FLASH_ATTN (or unset to "
                    "inherit it) for MTP with full CUDA graphs"
                )
        elif expected_vllm_version == "0.26.0":
            if main_backend not in {"FLASHINFER", "FLASH_ATTN"}:
                raise ValueError(
                    "vLLM 0.26 requires a validated FLASHINFER or FLASH_ATTN decoder backend "
                    "for MTP with full CUDA graphs"
                )
            if draft_backend_upper not in {None, "AUTO", main_backend}:
                raise ValueError(
                    "vLLM 0.26 requires the speculative attention backend to inherit or match "
                    "the decoder backend for MTP with full CUDA graphs"
                )
        else:
            raise ValueError(
                "MTP with full CUDA graphs is not validated for expected_vllm_version="
                f"{expected_vllm_version!r}"
            )
    if args.enforce_eager and graph_requested:
        raise ValueError("CUDA graph options cannot be combined with enforce_eager=true")
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    else:
        if cudagraph_mm_encoder:
            raise ValueError(
                "cudagraph_mm_encoder=true is forbidden for this vLLM 0.24 multimodal "
                "runtime because it can corrupt image semantics and duplicate responses"
            )
        if graph_requested:
            compilation_config: dict[str, Any] = {
                "cudagraph_mode": cudagraph_mode or "FULL_AND_PIECEWISE",
                "cudagraph_mm_encoder": False,
            }
            if max_cudagraph_capture_size is not None:
                compilation_config["max_cudagraph_capture_size"] = max_cudagraph_capture_size
            if cudagraph_specialize_lora is not None:
                compilation_config["cudagraph_specialize_lora"] = bool(cudagraph_specialize_lora)
            kwargs["compilation_config"] = compilation_config
    enable_tower_connector_lora = getattr(args, "enable_tower_connector_lora", None)
    if enable_tower_connector_lora is not None:
        kwargs["enable_tower_connector_lora"] = bool(enable_tower_connector_lora)
    enable_prefix_caching = getattr(args, "enable_prefix_caching", None)
    if enable_prefix_caching is not None:
        kwargs["enable_prefix_caching"] = bool(enable_prefix_caching)
    if args.use_mtp:
        speculative_attention_backend = getattr(args, "speculative_attention_backend", None)
        if speculative_attention_backend is None:
            # Preserve the established B300 behavior for callers that do not
            # select an independent draft backend.
            speculative_attention_backend = args.attention_backend
        elif speculative_attention_backend.lower() == "auto":
            speculative_attention_backend = None
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            **({"attention_backend": speculative_attention_backend} if speculative_attention_backend else {}),
        }
    return kwargs


def validate_resolved_speculative_config(llm: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Fail unless vLLM resolved exactly the requested MTP configuration."""

    try:
        config = llm.llm_engine.vllm_config.speculative_config
    except AttributeError as exc:
        raise RuntimeError("cannot inspect vLLM resolved speculative_config") from exc
    use_mtp = bool(args.use_mtp)
    if not use_mtp:
        if config is not None:
            raise RuntimeError("MTP is disabled but vLLM resolved a speculative_config")
        return {
            "enabled": False,
            "method": None,
            "num_speculative_tokens": 0,
            "draft_architectures": [],
        }
    if config is None:
        raise RuntimeError("MTP is enabled but vLLM did not resolve speculative_config")

    method = getattr(config, "method", None)
    tokens = getattr(config, "num_speculative_tokens", None)
    if method != "mtp" or tokens != args.num_speculative_tokens:
        raise RuntimeError(
            "resolved speculative_config differs from the requested MTP contract: "
            f"method={method!r} tokens={tokens!r} "
            f"expected_tokens={args.num_speculative_tokens!r}"
        )
    draft_config = getattr(config, "draft_model_config", None)
    hf_config = getattr(draft_config, "hf_config", None)
    raw_architectures = getattr(hf_config, "architectures", None)
    if isinstance(raw_architectures, str):
        architectures = [raw_architectures]
    else:
        architectures = list(raw_architectures or [])
    if architectures and "Qwen3_5MTP" not in architectures:
        raise RuntimeError(f"unexpected MTP draft architectures: {architectures!r}")
    return {
        "enabled": True,
        "method": method,
        "num_speculative_tokens": tokens,
        "draft_architectures": architectures,
    }


def validate_resolved_compilation_config(llm: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Fail if vLLM silently changed the explicitly requested CUDA graph plan."""

    expected_mode = getattr(args, "cudagraph_mode", None)
    expected_capture = getattr(args, "max_cudagraph_capture_size", None)
    expected_specialize = getattr(args, "cudagraph_specialize_lora", None)
    requested = bool(expected_mode or expected_capture is not None or expected_specialize is not None)
    if not requested:
        return {"enabled": False}
    try:
        config = llm.llm_engine.vllm_config.compilation_config
    except AttributeError as exc:
        raise RuntimeError("cannot inspect vLLM resolved compilation_config") from exc

    raw_mode = getattr(config, "cudagraph_mode", None)
    resolved_mode = getattr(raw_mode, "name", None) or str(raw_mode).rsplit(".", 1)[-1]
    resolved_capture = getattr(config, "max_cudagraph_capture_size", None)
    resolved_mm_encoder = getattr(config, "cudagraph_mm_encoder", None)
    resolved_specialize = getattr(config, "cudagraph_specialize_lora", None)
    if expected_mode and resolved_mode != expected_mode:
        raise RuntimeError(
            f"vLLM resolved a different cudagraph_mode: expected={expected_mode!r} actual={resolved_mode!r}"
        )
    if expected_capture is not None and resolved_capture != expected_capture:
        raise RuntimeError(
            "vLLM resolved a different max_cudagraph_capture_size: "
            f"expected={expected_capture!r} actual={resolved_capture!r}"
        )
    if resolved_mm_encoder is not False:
        raise RuntimeError(
            f"vLLM did not preserve cudagraph_mm_encoder=false: actual={resolved_mm_encoder!r}"
        )
    if expected_specialize is not None and resolved_specialize != bool(expected_specialize):
        raise RuntimeError(
            "vLLM resolved a different cudagraph_specialize_lora value: "
            f"expected={bool(expected_specialize)!r} actual={resolved_specialize!r}"
        )
    return {
        "enabled": True,
        "cudagraph_mode": resolved_mode,
        "max_cudagraph_capture_size": resolved_capture,
        "cudagraph_mm_encoder": resolved_mm_encoder,
        "cudagraph_specialize_lora": resolved_specialize,
    }


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def _compilation_config_snapshot(config: Any) -> dict[str, Any]:
    """Serialize only stable CUDA graph fields for cross-process RPC."""

    return {
        "compilation_mode": _enum_name(getattr(config, "mode", None)),
        "cudagraph_mode": _enum_name(getattr(config, "cudagraph_mode", None)),
        "max_cudagraph_capture_size": getattr(config, "max_cudagraph_capture_size", None),
        "cudagraph_mm_encoder": getattr(config, "cudagraph_mm_encoder", None),
        "cudagraph_specialize_lora": getattr(config, "cudagraph_specialize_lora", None),
    }


def validate_worker_compilation_config(llm: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Fail unless EngineCore workers retained the requested effective graph plan."""

    expected_mode = getattr(args, "cudagraph_mode", None)
    expected_capture = getattr(args, "max_cudagraph_capture_size", None)
    expected_specialize = getattr(args, "cudagraph_specialize_lora", None)
    requested = bool(expected_mode or expected_capture is not None or expected_specialize is not None)
    if not requested:
        return {"enabled": False}
    try:
        snapshots = llm.collective_rpc(
            "sidecar_compilation_config_snapshot",
            timeout=60,
        )
    except Exception as exc:
        raise RuntimeError("cannot inspect vLLM worker effective compilation_config") from exc
    if not isinstance(snapshots, list) or not snapshots:
        raise RuntimeError(f"vLLM worker compilation RPC returned no evidence: {snapshots!r}")
    for worker_index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise RuntimeError(
                f"vLLM worker {worker_index} returned invalid compilation evidence: {snapshot!r}"
            )
        if expected_mode and snapshot.get("cudagraph_mode") != expected_mode:
            raise RuntimeError(
                "vLLM worker resolved a different effective cudagraph_mode: "
                f"worker={worker_index} expected={expected_mode!r} "
                f"actual={snapshot.get('cudagraph_mode')!r}"
            )
        if expected_capture is not None and snapshot.get("max_cudagraph_capture_size") != expected_capture:
            raise RuntimeError(
                "vLLM worker resolved a different max_cudagraph_capture_size: "
                f"worker={worker_index} expected={expected_capture!r} "
                f"actual={snapshot.get('max_cudagraph_capture_size')!r}"
            )
        if snapshot.get("cudagraph_mm_encoder") is not False:
            raise RuntimeError(
                "vLLM worker did not preserve cudagraph_mm_encoder=false: "
                f"worker={worker_index} actual={snapshot.get('cudagraph_mm_encoder')!r}"
            )
        if expected_specialize is not None and snapshot.get("cudagraph_specialize_lora") != bool(
            expected_specialize
        ):
            raise RuntimeError(
                "vLLM worker resolved a different cudagraph_specialize_lora value: "
                f"worker={worker_index} expected={bool(expected_specialize)!r} "
                f"actual={snapshot.get('cudagraph_specialize_lora')!r}"
            )
    return {
        "enabled": True,
        "verification": CUDA_GRAPH_WORKER_VERIFICATION,
        "effective_mode": expected_mode,
        "workers": [dict(snapshot) for snapshot in snapshots],
    }


def _metric_samples(metrics: Iterable[Any]) -> Iterable[tuple[str, float]]:
    for family in metrics:
        samples = getattr(family, "samples", None)
        if samples is None:
            name = str(getattr(family, "name", ""))
            value = getattr(family, "value", None)
            if isinstance(value, (int, float)):
                yield name, float(value)
            continue
        for sample in samples:
            value = getattr(sample, "value", None)
            if isinstance(value, (int, float)):
                yield str(getattr(sample, "name", "")), float(value)


def speculative_metric_snapshot(metrics: Iterable[Any]) -> dict[str, float]:
    """Extract cumulative vLLM speculative counters without histogram gauges."""

    result = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    for raw_name, value in _metric_samples(metrics):
        name = raw_name.lower().removesuffix("_total")
        if "spec_decode" not in name and "speculative" not in name:
            continue
        if "accepted_tokens_per_pos" in name:
            continue
        if name.endswith("num_draft_tokens"):
            result["draft_tokens"] += value
        elif name.endswith("num_drafts"):
            result["drafts"] += value
        elif name.endswith("num_accepted_tokens"):
            result["accepted_tokens"] += value
    return result


def speculative_metric_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for key in ("drafts", "draft_tokens", "accepted_tokens"):
        delta = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        if not math.isfinite(delta) or delta < 0.0:
            raise RuntimeError(
                f"speculative metric counter reset or became invalid for {key}: "
                f"before={before.get(key, 0.0)} after={after.get(key, 0.0)}"
            )
        values[key] = delta
    draft_tokens = float(values["draft_tokens"] or 0.0)
    accepted_tokens = float(values["accepted_tokens"] or 0.0)
    values["acceptance_rate"] = accepted_tokens / draft_tokens if draft_tokens else None
    return values


def validate_speculative_metric_delta(
    values: Mapping[str, float | None],
    *,
    generated_requests: int,
    num_speculative_tokens: int,
) -> None:
    drafts = float(values.get("drafts") or 0.0)
    draft_tokens = float(values.get("draft_tokens") or 0.0)
    accepted_tokens = float(values.get("accepted_tokens") or 0.0)
    if generated_requests > 0 and (drafts <= 0.0 or draft_tokens <= 0.0):
        raise RuntimeError(
            f"MTP generated requests but reported no positive speculative draft counters: {dict(values)}"
        )
    if draft_tokens > drafts * num_speculative_tokens:
        raise RuntimeError(
            "MTP reported more draft tokens than its configured per-draft maximum: "
            f"drafts={drafts} draft_tokens={draft_tokens} "
            f"num_speculative_tokens={num_speculative_tokens}"
        )
    if accepted_tokens < 0.0 or accepted_tokens > draft_tokens:
        raise RuntimeError(
            f"MTP accepted-token counter is invalid: accepted={accepted_tokens} draft_tokens={draft_tokens}"
        )


def _batches(values: Iterable[InferenceRecord], size: int) -> Iterator[list[InferenceRecord]]:
    batch: list[InferenceRecord] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _candidate(request_output: Any) -> Any:
    candidates = getattr(request_output, "outputs", None)
    if not candidates:
        raise RuntimeError("vLLM returned no generation candidate")
    return candidates[0]


def _result_row(
    *,
    record: InferenceRecord,
    user_prompt: str,
    response: str,
    request_output: Any,
    candidate: Any,
    include_conversation: bool,
) -> dict[str, Any]:
    prediction, json_error, schema_error = parse_model_response(response)
    original_prediction: dict[str, Any] | None = None
    hld_label_cleanup: dict[str, Any] | None = None
    if (
        TARGET_SCHEMA == V6_TARGET_SCHEMA
        and prediction is not None
        and json_error is None
        and schema_error is None
    ):
        try:
            processed_prediction, hld_label_cleanup = postprocess_v6_hld_labels(
                prediction
            )
            if hld_label_cleanup["changed"]:
                original_prediction = prediction
            prediction = processed_prediction
            # The raw model JSON passed the V6 gate above. Re-run the same gate
            # on the final postprocessed projection before making it visible.
            validate_inline_target(prediction)
        except (HLDPostprocessError, InlineTargetValidationError) as exc:
            # Deterministic model-output defects are schema failures, not
            # infrastructure failures. Preserve the row and raw response so a
            # single pathological generation cannot strand an entire unit.
            schema_error = f"{type(exc).__name__}: {exc}"
    reference_image, target_image = _output_media(record)
    row: dict[str, Any] = {
        "input_index": record.input_index,
        "sample_id": record.sample_id,
        "instruction": record.instruction,
        "reference_image": reference_image,
        "target_image": target_image,
        **({"target_schema": TARGET_SCHEMA} if TARGET_SCHEMA != V4_TARGET_SCHEMA else {}),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "response": response,
        "prediction": prediction,
        "json_valid": json_error is None,
        "schema_valid": json_error is None and schema_error is None,
        "json_error": json_error,
        "schema_error": schema_error,
        "prompt_tokens": len(getattr(request_output, "prompt_token_ids", None) or []),
        "output_tokens": len(getattr(candidate, "token_ids", None) or []),
        "finish_reason": getattr(candidate, "finish_reason", None),
        "stop_reason": getattr(candidate, "stop_reason", None),
    }
    if hld_label_cleanup is not None:
        row["hld_label_cleanup"] = hld_label_cleanup
    if original_prediction is not None:
        row["original_prediction"] = original_prediction
    if record.evaluation_metadata:
        row.update(record.evaluation_metadata)
    if record.dataset_source is not None:
        row["dataset_source"] = dict(record.dataset_source)
    if include_conversation:
        row["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        row["images"] = [reference_image, target_image]
    return row


def _serialize_jsonl_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize a complete window before making any output write visible."""

    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in rows
    )


def _error_row(
    record: InferenceRecord,
    error: BaseException,
    *,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    reference_image, target_image = _output_media(record)
    row = {
        "input_index": record.input_index,
        "sample_id": record.sample_id,
        "instruction": record.instruction,
        "reference_image": reference_image,
        "target_image": target_image,
        **({"target_schema": TARGET_SCHEMA} if TARGET_SCHEMA != V4_TARGET_SCHEMA else {}),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "response": None,
        "prediction": None,
        "json_valid": False,
        "schema_valid": False,
        "json_error": None,
        "schema_error": None,
        "inference_error": f"{type(error).__name__}: {error}",
        "traceback": traceback_text or _format_exception(error),
    }
    if record.evaluation_metadata:
        row.update(record.evaluation_metadata)
    if record.dataset_source is not None:
        row["dataset_source"] = dict(record.dataset_source)
    return row


def validate_prompt_contract() -> None:
    """Fail before model initialization if the pinned training prompt drifted."""

    actual_prompt_digest = prompt_contract_sha256()
    if actual_prompt_digest != PROMPT_CONTRACT_SHA256:
        raise RuntimeError(
            "image-edit prompt source drifted from the pinned training contract: "
            f"expected={PROMPT_CONTRACT_SHA256} actual={actual_prompt_digest}"
        )


def configure_aws_environment(args: argparse.Namespace) -> None:
    """Apply the existing CLI AWS profile/region contract to this process."""

    if getattr(args, "aws_profile", None):
        os.environ["AWS_PROFILE"] = args.aws_profile
    os.environ.setdefault("AWS_REGION", args.aws_region)
    os.environ.setdefault("AWS_DEFAULT_REGION", args.aws_region)


def initialize_inference_runtime(args: argparse.Namespace) -> InferenceRuntime:
    """Initialize one vLLM engine and deterministic sampler for repeated use."""

    validate_prompt_contract()
    runtime_stack_validation = validate_runtime_stack(args)
    # Keep CUDA/vLLM imports inside the real entrypoint so multiprocessing spawn
    # can safely re-import this module.
    from vllm import LLM, SamplingParams

    llm = LLM(**build_llm_kwargs(args))
    validate_resolved_speculative_config(llm, args)
    validate_resolved_compilation_config(llm, args)
    cuda_graph_validation = validate_worker_compilation_config(llm, args)
    tokenizer = llm.get_tokenizer()
    return InferenceRuntime(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        runtime_stack_validation=runtime_stack_validation,
        cuda_graph_validation=cuda_graph_validation,
    )


def _request_window_size(args: argparse.Namespace) -> int:
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    value = getattr(args, "request_window_size", None)
    resolved = int(batch_size if value is None else value)
    if resolved <= 0:
        raise ValueError("request_window_size must be positive")
    if resolved < batch_size:
        raise ValueError("request_window_size must be at least batch_size")
    return resolved


def _prefetch_images(args: argparse.Namespace) -> bool:
    # Namespaces constructed by older callers retain their synchronous image
    # behavior; parsed CLI invocations explicitly default this option on.
    return bool(getattr(args, "prefetch_images", False))


def _execute_window(
    args: argparse.Namespace,
    runtime: InferenceRuntime,
    records: Sequence[InferenceRecord],
    image_loads: Sequence[_RecordImageLoad],
) -> _WindowExecution:
    """Generate one queued window with record-local preparation failures."""

    if len(image_loads) != len(records):
        raise RuntimeError(f"image load count {len(image_loads)} differs from record count {len(records)}")
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    rows: list[dict[str, Any] | None] = [None] * len(records)
    valid: list[
        tuple[int, InferenceRecord, str, str, dict[str, Any] | None]
    ] = []
    requests: list[dict[str, Any]] = []

    for index, (record, image_load) in enumerate(zip(records, image_loads, strict=True)):
        if image_load.error is not None:
            if not continue_on_error:
                raise image_load.error.with_traceback(image_load.error.__traceback__)
            rows[index] = _error_row(
                record,
                image_load.error,
                traceback_text=image_load.traceback_text,
            )
            continue
        if image_load.images is None:
            raise AssertionError("successful record image load has no images")
        try:
            effective_instruction, instruction_compatibility = (
                qwen_context_compatible_instruction(record.instruction)
            )
            prompt = render_direct_json_prompt(runtime.tokenizer, effective_instruction)
            reference, target = image_load.images
            request = {
                "prompt": prompt,
                "multi_modal_data": {"image": [reference, target]},
                "mm_processor_kwargs": {"max_pixels": args.max_pixels},
            }
        except BaseException as exc:
            if not continue_on_error:
                raise
            rows[index] = _error_row(record, exc, traceback_text=_format_exception(exc))
            continue
        valid.append(
            (
                index,
                record,
                prompt,
                effective_instruction,
                instruction_compatibility,
            )
        )
        requests.append(request)

    generated_requests = 0
    outputs: Sequence[Any] = []
    generation_error: BaseException | None = None
    generation_traceback: str | None = None
    generation_started = time.monotonic()
    if requests:
        try:
            outputs = runtime.llm.generate(
                requests,
                [runtime.sampling_params for _ in requests],
                use_tqdm=args.tqdm,
            )
            generated_requests = len(requests)
            if len(outputs) != len(requests):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} inputs")
        except BaseException as exc:
            if not continue_on_error:
                raise
            generation_error = exc
            generation_traceback = _format_exception(exc)
    generation_seconds = time.monotonic() - generation_started

    postprocess_started = time.monotonic()
    if generation_error is not None:
        for index, record, _prompt, _effective_instruction, _compatibility in valid:
            rows[index] = _error_row(
                record,
                generation_error,
                traceback_text=generation_traceback,
            )
    else:
        for (
            index,
            record,
            _prompt,
            effective_instruction,
            instruction_compatibility,
        ), request_output in zip(valid, outputs, strict=True):
            try:
                candidate = _candidate(request_output)
                response = str(candidate.text)
                rows[index] = _result_row(
                    record=record,
                    user_prompt=build_user_prompt(effective_instruction),
                    response=response,
                    request_output=request_output,
                    candidate=candidate,
                    include_conversation=args.include_conversation,
                )
                if instruction_compatibility is not None:
                    rows[index]["instruction_compatibility_repair"] = (
                        instruction_compatibility
                    )
            except BaseException as exc:
                if not continue_on_error:
                    raise
                rows[index] = _error_row(record, exc, traceback_text=_format_exception(exc))
    postprocess_seconds = time.monotonic() - postprocess_started

    completed_rows: list[dict[str, Any]] = []
    for row in rows:
        if row is None:
            raise AssertionError("window record did not produce exactly one result row")
        completed_rows.append(row)
    return _WindowExecution(
        rows=completed_rows,
        generated_requests=generated_requests,
        prompt_tokens=sum(int(row.get("prompt_tokens", 0)) for row in completed_rows),
        output_tokens=sum(int(row.get("output_tokens", 0)) for row in completed_rows),
        generation_seconds=generation_seconds,
        postprocess_seconds=postprocess_seconds,
    )


def _close_image_loads(image_loads: Sequence[_RecordImageLoad]) -> None:
    for image_load in image_loads:
        image_load.close()


def _prepare_prefetched_window(
    windows: Iterator[list[InferenceRecord]],
    *,
    download_workers: int,
) -> tuple[list[InferenceRecord], list[_RecordImageLoad]] | None:
    """Materialize the next record window and decode it in one producer thread."""

    records = next(windows, None)
    if records is None:
        return None
    return records, _load_record_image_results(records, download_workers=download_workers)


def _drain_prefetched_window(future: Any) -> None:
    """Wait for abandoned producer work and close every decoded image."""

    try:
        prepared = future.result()
    except BaseException:
        # _load_record_image_results drains and closes partial work before an
        # unexpected exception escapes.  Preserve the primary inference error.
        return
    if prepared is not None:
        _records, image_loads = prepared
        _close_image_loads(image_loads)


def infer_records(
    args: argparse.Namespace,
    runtime: InferenceRuntime,
    records: Iterable[InferenceRecord],
    output_handle: TextIO,
) -> InferenceExecution:
    """Infer one record stream with an already initialized reusable runtime."""

    started_at = time.monotonic()
    mtp_enabled = bool(getattr(args, "use_mtp", False))
    speculative_before = speculative_metric_snapshot(runtime.llm.get_metrics()) if mtp_enabled else None
    speculative_cursor = speculative_before
    speculative_totals = {
        "drafts": 0.0,
        "draft_tokens": 0.0,
        "accepted_tokens": 0.0,
    }
    counts = {"rows": 0, "json_valid": 0, "schema_valid": 0, "failed": 0}
    single_prediction: dict[str, Any] | None = None
    generated_requests = 0
    prompt_tokens = 0
    output_tokens = 0
    image_wait_seconds = 0.0
    generation_seconds = 0.0
    postprocess_seconds = 0.0
    write_seconds = 0.0
    window_size = _request_window_size(args)
    windows = iter(_batches(records, window_size))
    prefetch_executor: ThreadPoolExecutor | None = None
    pending_window_future: Any | None = None
    if _prefetch_images(args):
        prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="image-edit-window-prefetch",
        )
        pending_window_future = prefetch_executor.submit(
            _prepare_prefetched_window,
            windows,
            download_workers=args.download_workers,
        )
        current_records: list[InferenceRecord] | None = None
    else:
        current_records = next(windows, None)

    try:
        while True:
            image_wait_started = time.monotonic()
            if prefetch_executor is None:
                if current_records is None:
                    break
                image_loads = _load_record_image_results(
                    current_records,
                    download_workers=args.download_workers,
                )
            else:
                if pending_window_future is None:
                    break
                current_future = pending_window_future
                pending_window_future = None
                prepared = current_future.result()
                if prepared is None:
                    break
                current_records, image_loads = prepared
                # The producer now consumes source units and decodes the next
                # images while the current H200 window is generating.
                pending_window_future = prefetch_executor.submit(
                    _prepare_prefetched_window,
                    windows,
                    download_workers=args.download_workers,
                )
            image_wait_seconds += time.monotonic() - image_wait_started

            try:
                window = _execute_window(args, runtime, current_records, image_loads)
                if mtp_enabled:
                    assert speculative_cursor is not None
                    speculative_after = speculative_metric_snapshot(runtime.llm.get_metrics())
                    window_speculative = speculative_metric_delta(
                        speculative_cursor,
                        speculative_after,
                    )
                    validate_speculative_metric_delta(
                        window_speculative,
                        generated_requests=window.generated_requests,
                        num_speculative_tokens=args.num_speculative_tokens,
                    )
                    speculative_cursor = speculative_after
                    for key in speculative_totals:
                        speculative_totals[key] += float(window_speculative[key] or 0.0)
                generated_requests += window.generated_requests
                prompt_tokens += window.prompt_tokens
                output_tokens += window.output_tokens
                generation_seconds += window.generation_seconds
                postprocess_seconds += window.postprocess_seconds

                write_started = time.monotonic()
                payload = _serialize_jsonl_rows(window.rows)
                output_handle.write(payload)
                write_seconds += time.monotonic() - write_started

                for record, row in zip(current_records, window.rows, strict=True):
                    counts["rows"] += 1
                    counts["json_valid"] += int(row["json_valid"])
                    counts["schema_valid"] += int(row["schema_valid"])
                    counts["failed"] += int(not row["schema_valid"])
                    if (
                        record.input_index == 0
                        and row["schema_valid"]
                        and isinstance(row["prediction"], dict)
                    ):
                        single_prediction = row["prediction"]
                if counts["rows"] % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "inference_progress",
                                "rank": args.rank,
                                "world_size": args.world_size,
                                **counts,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if prefetch_executor is None:
                    current_records = next(windows, None)
            finally:
                _close_image_loads(image_loads)
    finally:
        if pending_window_future is not None:
            _drain_prefetched_window(pending_window_future)
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
    if mtp_enabled:
        draft_tokens = speculative_totals["draft_tokens"]
        accepted_tokens = speculative_totals["accepted_tokens"]
        speculative: dict[str, float | None] = {
            **speculative_totals,
            "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        }
    else:
        speculative = {
            "drafts": 0.0,
            "draft_tokens": 0.0,
            "accepted_tokens": 0.0,
            "acceptance_rate": None,
        }
    telemetry = {
        "generated_requests": generated_requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": time.monotonic() - started_at,
        "image_wait_seconds": image_wait_seconds,
        "generation_seconds": generation_seconds,
        "postprocess_seconds": postprocess_seconds,
        "write_seconds": write_seconds,
        "speculative": {"enabled": mtp_enabled, **speculative},
    }
    return InferenceExecution(
        counts=counts,
        single_prediction=single_prediction,
        telemetry=telemetry,
    )


def run(args: argparse.Namespace) -> dict[str, int]:
    configure_aws_environment(args)

    if args.input_jsonl:
        records: Iterable[InferenceRecord] = iter_input_jsonl(
            args.input_jsonl,
            max_samples=args.max_samples,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        records = [
            InferenceRecord(
                input_index=0,
                sample_id=args.sample_id,
                instruction=args.instruction.strip(),
                reference_image=_media_uri(args.reference_image, field_name="reference_image"),
                target_image=_media_uri(args.target_image, field_name="target_image"),
            )
        ]

    runtime = initialize_inference_runtime(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        execution = infer_records(args, runtime, records, output_handle)

    if args.prediction_output:
        if execution.single_prediction is None:
            raise RuntimeError(
                "cannot write --prediction-output because the response did not pass the exact schema"
            )
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        args.prediction_output.write_text(
            json.dumps(
                execution.single_prediction,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return execution.counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-schema",
        choices=SUPPORTED_TARGET_SCHEMAS,
        default=TARGET_SCHEMA,
        help="Must match IMAGE_EDIT_TARGET_SCHEMA selected before process import",
    )
    parser.add_argument("--model", required=True, help="Converted Hugging Face checkpoint directory")
    parser.add_argument("--tokenizer", help="Tokenizer directory; defaults to --model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path, help="Prepared or compact request JSONL")
    source.add_argument("--reference-image", help="Single-request frame_0 local/HTTP/S3 image")
    parser.add_argument("--target-image", help="Single-request Target local/HTTP/S3 image")
    parser.add_argument("--instruction", help="Single-request edit instruction")
    parser.add_argument("--sample-id", default="single-request")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--prediction-output",
        type=Path,
        help="Single-request convenience output containing only the parsed complete JSON object",
    )
    parser.add_argument("--include-conversation", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--request-window-size",
        type=int,
        help="Requests queued per blocking vLLM call; defaults to --batch-size",
    )
    parser.add_argument(
        "--prefetch-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load/decode the next request window while the current window generates",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mm-processor-cache-gb", type=float)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--gdn-prefill-backend")
    parser.add_argument("--mm-encoder-attn-backend")
    parser.add_argument("--cudagraph-mode")
    parser.add_argument("--max-cudagraph-capture-size", type=int)
    parser.add_argument(
        "--cudagraph-mm-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must remain disabled for the pinned vLLM 0.24 multimodal runtime",
    )
    parser.add_argument(
        "--cudagraph-specialize-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-tower-connector-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--speculative-attention-backend",
        help="Draft attention backend; use 'auto' to let vLLM select it independently",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--expected-torch-version")
    parser.add_argument("--expected-transformers-version")
    parser.add_argument("--expected-vllm-version")
    parser.add_argument("--expected-flashinfer-version")
    parser.add_argument("--expected-flashinfer-cubin-version")
    parser.add_argument("--expected-flashinfer-jit-cache-version")
    parser.add_argument("--expected-triton-version")
    parser.add_argument("--expected-cuda-version")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_target_schema(args.target_schema)
    except ValueError as exc:
        parser.error(str(exc))
    if args.input_jsonl:
        if args.target_image or args.instruction:
            parser.error("--target-image/--instruction cannot be combined with --input-jsonl")
        if args.prediction_output and args.max_samples != 1:
            parser.error("--prediction-output with --input-jsonl requires --max-samples 1")
        if args.input_jsonl.resolve() == args.output_jsonl.resolve():
            parser.error("--output-jsonl must not overwrite --input-jsonl")
        if args.prediction_output and args.input_jsonl.resolve() == args.prediction_output.resolve():
            parser.error("--prediction-output must not overwrite --input-jsonl")
    else:
        if not args.target_image or not args.instruction or not args.instruction.strip():
            parser.error("single-request mode requires --reference-image, --target-image, and --instruction")
        if args.max_samples:
            parser.error("--max-samples is only valid with --input-jsonl")
    for name in (
        "batch_size",
        "download_workers",
        "tensor_parallel_size",
        "max_model_len",
        "max_tokens",
        "max_pixels",
        "num_speculative_tokens",
        "world_size",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.max_cudagraph_capture_size is not None:
        if args.max_cudagraph_capture_size <= 0:
            parser.error("--max-cudagraph-capture-size must be positive")
        if args.max_cudagraph_capture_size > args.batch_size:
            parser.error("--max-cudagraph-capture-size must not exceed --batch-size")
    if args.cudagraph_mm_encoder:
        parser.error("--cudagraph-mm-encoder is unsafe with the pinned vLLM 0.24 runtime")
    if args.enforce_eager and (
        args.cudagraph_mode
        or args.max_cudagraph_capture_size is not None
        or args.cudagraph_specialize_lora is not None
    ):
        parser.error("CUDA graph options cannot be combined with --enforce-eager")
    if args.mm_processor_cache_gb is not None and args.mm_processor_cache_gb < 0:
        parser.error("--mm-processor-cache-gb must be non-negative")
    if args.request_window_size is None:
        args.request_window_size = args.batch_size
    if args.request_window_size <= 0:
        parser.error("--request-window-size must be positive")
    if args.request_window_size < args.batch_size:
        parser.error("--request-window-size must be at least --batch-size")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--rank must be in [0, --world-size)")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be smaller than --max-model-len")
    if args.prediction_output and args.prediction_output.resolve() == args.output_jsonl.resolve():
        parser.error("--prediction-output and --output-jsonl must be different paths")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    counts = run(args)
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **counts}, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
