#!/usr/bin/env python3
"""Build a deterministic 5M high-quality sample from the V6 full inference run.

The pipeline is deliberately split into restartable JobSet phases:

``prepare``
    Pins and validates both source manifests, builds exact SHA-256 exclusion
    indexes for the 7.2M V6 training rows, and shards the inference-unit work.
``scan-worker``
    Validates inference JSONL objects, joins compact metadata from the frozen
    source Parquet row, excludes every training ``sample_id``, applies the
    quality contract, counts edit-task strata, and retains a deterministic
    hash sample of candidates.
``plan``
    Uses Hamilton apportionment to assign exactly 5,000,000 proportional
    per-task quotas and proves that the retained candidate reservoir covers
    every quota.
``select-worker``
    Selects the smallest deterministic hashes within each task stratum and
    emits final-only, response-free JSONL shards.
``finalize``
    Reads every output row back, proves exact count/proportions, uniqueness,
    and zero V6-training ``sample_id`` overlap, then commits ``manifest.json``
    and finally ``_SUCCESS``.

The full inference and training manifests are immutable inputs.  No source
object is changed in place.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse


PIPELINE_VERSION = "image_edit_v6_inference_high_quality_proportional_5m_v1"
EXCLUSION_VERSION = "image_edit_v6_training_sample_id_sha256_exact_v1"
QUALITY_VERSION = "image_edit_v6_source_quality_and_prediction_contract_v1"
TASK_VERSION = "image_edit_operation_type_normalized_v1"
APPORTIONMENT_VERSION = "hamilton_largest_remainder_lexical_tiebreak_v1"

INFERENCE_MANIFEST_FORMAT = "image_edit_full_json_sidecar24_unit_inventory_manifest_v1"
TRAINING_MANIFEST_FORMAT = "image_edit_v6_training_manifest_v1"
EXPECTED_INFERENCE_ROWS = 115_802_213
EXPECTED_INFERENCE_UNITS = 372_215
EXPECTED_TRAINING_ROWS = 7_234_587
EXPECTED_TRAINING_SHARDS = 256
EXPECTED_PROMPT_VERSION = "image_edit_v6_direct_root_xyxy_v7_qwen35_v7"
EXPECTED_PROMPT_SHA256 = "7984173abd05fe72ea188ff4065db1305c49df1c41c01f18fc3a284046299f6f"
EXPECTED_POSTPROCESS_VERSION = "image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7"
EXPECTED_PACKAGING_VERSION = "image_edit_v6_prediction_xyxy_to_yxyx_v1"
EXPECTED_TARGET_SCHEMA = "image_edit_v6_direct_root_xyxy"
EXPECTED_COORDINATE_ORDER = "yxyx_1000"

DEFAULT_INFERENCE_MANIFEST_URI = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/craftv0/full/"
    "checkpoint-8000/1912rtx-128h200-pr36/20260815094200/results/manifest.json"
)
DEFAULT_INFERENCE_MANIFEST_SHA256 = (
    "714f33dc928df69cbcf90a60e33bc37bd9d73414418597c4f95e59a0220a98e6"
)
DEFAULT_INFERENCE_MANIFEST_VERSION_ID = "lnPkdqkuPJsUmafYho0In2Z2uhKzRbwk"
DEFAULT_TRAINING_MANIFEST_URI = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/prune-image-edit-v6/"
    "full-gemini31-lite-10x800k-20260813-v1/"
    "training-v6-7984173a-all10-256shards-v1/"
    "image_edit_v6_full_json.manifest.json"
)
DEFAULT_TRAINING_MANIFEST_SHA256 = (
    "692bf21ce252bba6f221524d628bfdc1679684935c36783d2ed8bc3f1d1548cc"
)
DEFAULT_TRAINING_MANIFEST_VERSION_ID = "5cwqU2tJqwjLv9_KUpNy8IYsBJXGGaaI"
DEFAULT_OUTPUT_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/craftv0/sampled/"
    "high-quality-5m-no-v6-train/20260818-v1"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
JSON_CONTENT_TYPE = "application/json"
JSONL_CONTENT_TYPE = "application/x-ndjson"
HASH_BYTES = 32
MIN_CANDIDATES_PER_TASK_PER_SCAN_WORKER = 128

SOURCE_COLUMNS = (
    "sample_id",
    "operation_type",
    "task_type",
    "task",
    "edit_category",
    "difficulty",
    "design_type",
    "source_type",
    "image_type",
    "data_source",
    "quality_label",
    "quality_reasons",
    "image_quality_score",
    "alignment_quality",
    "content_preservation",
    "aesthetic_score",
    "design_quality_score",
    "reversibility_score",
    "edit_magnitude",
    "meta_info",
)

SOURCE_PROJECTION_COLUMNS = (
    "operation_type",
    "task_type",
    "task",
    "edit_category",
    "difficulty",
    "design_type",
    "source_type",
    "image_type",
    "data_source",
    "quality_label",
    "quality_reasons",
    "image_quality_score",
    "alignment_quality",
    "content_preservation",
    "aesthetic_score",
    "design_quality_score",
    "reversibility_score",
    "edit_magnitude",
)

JUDGE_CONTAINER_KEYS = (
    "judge_scores",
    "gemini_judge",
    "openai_judge",
    "judge",
)
JUDGE_SCORE_KEYS = frozenset(
    {
        "edit_adherence",
        "instruction_following",
        "alignment_quality",
        "visual_quality",
        "image_quality",
        "content_preservation",
        "direction_correctness",
        "change_coverage",
        "visual_grounding",
        "trainability",
        "capability_consistency",
        "reversibility_score",
    }
)
TOP_LEVEL_QUALITY_KEYS = (
    "image_quality_score",
    "alignment_quality",
    "content_preservation",
    "reversibility_score",
)
PASS_MARKERS = frozenset(
    {
        "hard_pass",
        "judged_ge4",
        "all_pass_v1",
        "reversibility_ge4",
        "correct",
    }
)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def require_sha256(value: str, name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def parse_s3_uri(value: str) -> S3Location:
    parsed = urlparse(value)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid S3 URI: {value!r}")
    return S3Location(parsed.netloc, key)


def join_root(root: str, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{root.rstrip('/')}/{suffix}" if suffix else root.rstrip("/")


def error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return ""
    error = response.get("Error")
    return str(error.get("Code") or "") if isinstance(error, Mapping) else ""


def make_s3_client(
    *, profile: str | None, region: str, max_pool_connections: int = 64
) -> Any:
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 12, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


class OutputStore:
    """Small immutable local/S3 store used by every pipeline phase."""

    def __init__(self, root: str, client: Any | None):
        self.root = root.rstrip("/")
        self.is_s3 = self.root.startswith("s3://")
        self.client = client
        if self.is_s3:
            parse_s3_uri(self.root)
            if client is None:
                raise ValueError("S3 output requires a client")
        else:
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def absolute(self, relative: str) -> str:
        return join_root(self.root, relative)

    def _local(self, relative: str) -> Path:
        return Path(self.root) / relative

    def head(self, relative: str, *, missing_ok: bool = False) -> Mapping[str, Any] | None:
        if not self.is_s3:
            path = self._local(relative)
            if not path.is_file():
                if missing_ok:
                    return None
                raise FileNotFoundError(path)
            return {"ContentLength": path.stat().st_size, "Metadata": {}}
        location = parse_s3_uri(self.absolute(relative))
        try:
            return self.client.head_object(Bucket=location.bucket, Key=location.key)
        except Exception as exc:
            if missing_ok and error_code(exc) in NOT_FOUND_CODES:
                return None
            raise

    def read_bytes(self, relative: str, *, missing_ok: bool = False) -> bytes | None:
        if not self.is_s3:
            path = self._local(relative)
            if not path.is_file():
                if missing_ok:
                    return None
                raise FileNotFoundError(path)
            return path.read_bytes()
        location = parse_s3_uri(self.absolute(relative))
        try:
            response = self.client.get_object(Bucket=location.bucket, Key=location.key)
        except Exception as exc:
            if missing_ok and error_code(exc) in NOT_FOUND_CODES:
                return None
            raise
        return response["Body"].read()

    def read_json(self, relative: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        payload = self.read_bytes(relative, missing_ok=missing_ok)
        if payload is None:
            return None
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"{self.absolute(relative)} is not a JSON object")
        if self.is_s3:
            head = self.head(relative)
            metadata = head.get("Metadata") if isinstance(head, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("sha256") != sha256_bytes(payload):
                raise ValueError(f"{self.absolute(relative)} JSON SHA metadata mismatch")
        return value

    def put_json_once(self, relative: str, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = canonical_json_bytes(value, newline=True)
        digest = sha256_bytes(payload)
        existing = self.read_json(relative, missing_ok=True)
        if existing is not None:
            if existing != dict(value):
                raise ValueError(f"refusing to overwrite mismatched {self.absolute(relative)}")
            head = self.head(relative)
            return {
                "relative_path": relative,
                "uri": self.absolute(relative),
                "bytes": len(payload),
                "sha256": digest,
                "version_id": str(head.get("VersionId") or "local"),
            }
        if self.is_s3:
            from botocore.exceptions import ClientError

            destination = parse_s3_uri(self.absolute(relative))
            try:
                self.client.put_object(
                    Bucket=destination.bucket,
                    Key=destination.key,
                    Body=payload,
                    ContentType=JSON_CONTENT_TYPE,
                    Metadata={"sha256": digest},
                    IfNoneMatch="*",
                )
            except ClientError as exc:
                if error_code(exc) not in {"412", "PreconditionFailed"}:
                    raise
                raced = self.read_json(relative)
                if raced != dict(value):
                    raise ValueError(f"concurrent mismatched write: {self.absolute(relative)}")
        else:
            path = self._local(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                raced = self.read_json(relative)
                if raced != dict(value):
                    raise ValueError(f"concurrent mismatched write: {path}")
            else:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        committed = self.read_json(relative)
        if committed != dict(value):
            raise ValueError(f"JSON read-back mismatch: {self.absolute(relative)}")
        head = self.head(relative)
        return {
            "relative_path": relative,
            "uri": self.absolute(relative),
            "bytes": len(payload),
            "sha256": digest,
            "version_id": str(head.get("VersionId") or "local"),
        }

    def put_file_once(
        self,
        relative: str,
        path: Path,
        *,
        content_type: str,
        content_encoding: str | None = None,
    ) -> dict[str, Any]:
        size, digest = sha256_file(path)
        existing = self.head(relative, missing_ok=True)
        if existing is not None:
            metadata = existing.get("Metadata") if isinstance(existing, Mapping) else {}
            if int(existing.get("ContentLength", -1)) != size:
                raise ValueError(f"{self.absolute(relative)} size mismatch")
            if self.is_s3 and metadata.get("sha256") != digest:
                raise ValueError(f"{self.absolute(relative)} SHA metadata mismatch")
            if not self.is_s3 and sha256_file(self._local(relative))[1] != digest:
                raise ValueError(f"{self.absolute(relative)} SHA mismatch")
        elif self.is_s3:
            destination = parse_s3_uri(self.absolute(relative))
            extra: dict[str, Any] = {
                "ContentType": content_type,
                "Metadata": {"sha256": digest},
            }
            if content_encoding:
                extra["ContentEncoding"] = content_encoding
            self.client.upload_file(
                str(path), destination.bucket, destination.key, ExtraArgs=extra
            )
        else:
            destination = self._local(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            shutil.copyfile(path, temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            finally:
                temporary.unlink(missing_ok=True)
        head = self.head(relative)
        return {
            "relative_path": relative,
            "uri": self.absolute(relative),
            "bytes": size,
            "sha256": digest,
            "version_id": str(head.get("VersionId") or "local"),
        }

    def download(self, relative: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.is_s3:
            source = parse_s3_uri(self.absolute(relative))
            self.client.download_file(source.bucket, source.key, str(destination))
        else:
            shutil.copyfile(self._local(relative), destination)


def get_pinned_json(
    client: Any,
    *,
    uri: str,
    sha256: str,
    version_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha256(sha256, f"{uri} SHA-256")
    location = parse_s3_uri(uri)
    response = client.get_object(
        Bucket=location.bucket,
        Key=location.key,
        VersionId=version_id,
    )
    payload = response["Body"].read()
    actual = sha256_bytes(payload)
    if actual != sha256:
        raise ValueError(f"{uri}: SHA-256 {actual} != {sha256}")
    actual_version = str(response.get("VersionId") or "")
    if actual_version != version_id:
        raise ValueError(f"{uri}: VersionId {actual_version!r} != {version_id!r}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{uri}: pinned JSON is not an object")
    return value, {
        "uri": uri,
        "sha256": actual,
        "version_id": actual_version,
        "bytes": len(payload),
    }


def canonical_identity_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


def identity_digest(namespace: str, *values: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    for value in values:
        encoded = canonical_identity_text(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def training_identity_digests(row: Mapping[str, Any]) -> tuple[bytes, bytes]:
    sample_id = row.get("sample_id")
    instruction = row.get("instruction")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("training row sample_id must be a non-empty string")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"training row {sample_id!r} instruction is invalid")
    return (
        identity_digest("sample-id-v1", sample_id),
        identity_digest("sample-id-instruction-v1", sample_id, instruction),
    )


def selection_digest(row: Mapping[str, Any], *, seed: str) -> str:
    source = row.get("dataset_source")
    if not isinstance(source, Mapping):
        raise ValueError("inference row has no dataset_source")

    def first_present(*candidates: Any) -> str:
        for candidate in candidates:
            if candidate is not None:
                return str(candidate)
        return ""

    values = (
        seed,
        first_present(source.get("dataset_id")),
        first_present(source.get("global_input_index"), row.get("input_index")),
        first_present(source.get("parquet_s3_uri")),
        first_present(source.get("source_parquet_local_row")),
        first_present(row.get("sample_id")),
    )
    return identity_digest("selection-v1", *values).hex()


def candidate_selected(hash_hex: str, *, numerator: int, denominator: int) -> bool:
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("invalid candidate sampling fraction")
    value = int(hash_hex[:16], 16)
    threshold = ((1 << 64) * numerator) // denominator
    return value < threshold


def task_bucket(task_type: str, bucket_count: int) -> int:
    if bucket_count <= 0 or bucket_count > 65_536:
        raise ValueError("task bucket count must be within 1..65536")
    return int.from_bytes(identity_digest("task-bucket-v1", task_type)[:8], "big") % bucket_count


def parse_meta_info(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    candidates = [value.strip()]
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == "'":
        candidates.append(stripped[1:-1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (SyntaxError, ValueError):
            pass
    return {}


def _parse_task_sequence(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if str(item).strip()]
    return [stripped]


def _task_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[\s\-/&]+", "_", normalized)
    normalized = re.sub(r"[^\w.]+", "_", normalized, flags=re.UNICODE)
    return normalized.strip("_.")


def normalize_task_type(
    source_row: Mapping[str, Any], *, dataset_id: str
) -> tuple[str, Any, str]:
    meta = parse_meta_info(source_row.get("meta_info"))
    candidates: list[tuple[str, Any]] = [
        ("operation_type", source_row.get("operation_type")),
        ("task_type", source_row.get("task_type")),
        ("task", source_row.get("task")),
        ("meta_info.edit_type", meta.get("edit_type")),
        ("meta_info.task_type", meta.get("task_type")),
        ("meta_info.task", meta.get("task")),
        ("meta_info.operation_type", meta.get("operation_type")),
        ("edit_category", source_row.get("edit_category")),
        ("meta_info.edit_category", meta.get("edit_category")),
    ]
    for source, raw in candidates:
        pieces = sorted({_task_slug(item) for item in _parse_task_sequence(raw) if _task_slug(item)})
        if not pieces:
            continue
        task = pieces[0] if len(pieces) == 1 else "multi:" + "+".join(pieces)
        return task, raw, source
    fallback = "dataset:" + _task_slug(dataset_id)
    return fallback, None, "dataset_fallback"


def _score_passes(value: float) -> bool:
    if not math.isfinite(value):
        return False
    if 0.0 <= value <= 1.0:
        return value >= 0.8
    if 1.0 < value <= 5.0:
        return value >= 4.0
    if 5.0 < value <= 100.0:
        return value >= 80.0
    return False


def _walk_mapping(value: Any) -> Iterator[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        yield str(key), child
        if isinstance(child, Mapping):
            yield from _walk_mapping(child)


def source_quality_decision(source_row: Mapping[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    failures: list[str] = []
    explicit = False

    quality_label = source_row.get("quality_label")
    if type(quality_label) is bool:
        explicit = True
        evidence.append(f"quality_label={str(quality_label).lower()}")
        if not quality_label:
            failures.append("quality_label_false")

    for key in TOP_LEVEL_QUALITY_KEYS:
        value = source_row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            explicit = True
            number = float(value)
            evidence.append(f"{key}={number:g}")
            if not _score_passes(number):
                failures.append(f"{key}_below_threshold")

    meta = parse_meta_info(source_row.get("meta_info"))
    for container_name in JUDGE_CONTAINER_KEYS:
        container = meta.get(container_name)
        if not isinstance(container, Mapping) or not container:
            continue
        saw_container_evidence = False
        for key, value in _walk_mapping(container):
            lowered = key.lower()
            if lowered in JUDGE_SCORE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                explicit = True
                saw_container_evidence = True
                number = float(value)
                evidence.append(f"{container_name}.{lowered}={number:g}")
                if not _score_passes(number):
                    failures.append(f"{container_name}.{lowered}_below_threshold")
            if (lowered.endswith("_pass") or lowered in {"pass", "quality_pass"}) and type(value) is bool:
                explicit = True
                saw_container_evidence = True
                evidence.append(f"{container_name}.{lowered}={str(value).lower()}")
                if not value:
                    failures.append(f"{container_name}.{lowered}_false")
        if saw_container_evidence:
            evidence.append(f"judge_container={container_name}")

    marker = source_row.get("edit_magnitude")
    if isinstance(marker, str) and marker.strip().lower() in PASS_MARKERS:
        explicit = True
        evidence.append(f"source_marker={marker.strip().lower()}")

    if failures:
        return {
            "passed": False,
            "tier": "explicit_reject",
            "evidence": sorted(set(evidence)),
            "failures": sorted(set(failures)),
        }
    return {
        "passed": True,
        "tier": "explicit_pass" if explicit else "curated_valid",
        "evidence": sorted(set(evidence)) if evidence else ["frozen_curated_source"],
        "failures": [],
    }


def prediction_quality_decision(row: Mapping[str, Any]) -> tuple[bool, str]:
    if row.get("json_valid") is not True:
        return False, "json_invalid"
    if row.get("schema_valid") is not True:
        return False, "schema_invalid"
    if row.get("target_schema") != EXPECTED_TARGET_SCHEMA:
        return False, "target_schema_mismatch"
    if row.get("prompt_version") != EXPECTED_PROMPT_VERSION:
        return False, "prompt_version_mismatch"
    if row.get("prompt_contract_sha256") != EXPECTED_PROMPT_SHA256:
        return False, "prompt_contract_mismatch"
    if row.get("finish_reason") != "stop":
        return False, "finish_reason_not_stop"
    cleanup = row.get("hld_label_cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("version") != EXPECTED_POSTPROCESS_VERSION:
        return False, "postprocess_contract_mismatch"
    packaging = row.get("bbox_packaging")
    if not isinstance(packaging, Mapping):
        return False, "bbox_packaging_missing"
    if packaging.get("version") != EXPECTED_PACKAGING_VERSION:
        return False, "bbox_packaging_version_mismatch"
    if row.get("bbox_coordinate_order") != EXPECTED_COORDINATE_ORDER:
        return False, "bbox_coordinate_order_mismatch"
    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping):
        return False, "prediction_missing"
    for key in ("high_level_description", "normalized_edit_instruction"):
        if not isinstance(prediction.get(key), str) or not prediction[key].strip():
            return False, f"prediction_{key}_missing"
    elements = prediction.get("elements")
    if not isinstance(elements, Mapping) or not isinstance(elements.get("layout"), list):
        return False, "prediction_elements_invalid"
    return True, "passed"


def project_candidate(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    task_type: str,
    task_raw: Any,
    task_source: str,
    quality: Mapping[str, Any],
    hash_hex: str,
    bucket: int,
    candidate_numerator: int,
    candidate_denominator: int,
    candidate_reason: str = "hash_fraction",
) -> dict[str, Any]:
    keep = (
        "input_index",
        "sample_id",
        "instruction",
        "reference_image",
        "target_image",
        "target_schema",
        "prompt_version",
        "prompt_contract_sha256",
        "prediction",
        "bbox_coordinate_order",
        "bbox_packaging",
        "hld_label_cleanup",
        "dataset_source",
    )
    projected = {key: row.get(key) for key in keep}
    projected["source_metadata"] = {
        key: source_row.get(key)
        for key in SOURCE_PROJECTION_COLUMNS
        if source_row.get(key) is not None
    }
    projected["sampling"] = {
        "pipeline_version": PIPELINE_VERSION,
        "exclusion_version": EXCLUSION_VERSION,
        "quality_version": QUALITY_VERSION,
        "task_version": TASK_VERSION,
        "task_type": task_type,
        "task_type_raw": task_raw,
        "task_type_source": task_source,
        "task_bucket": bucket,
        "quality_tier": quality["tier"],
        "quality_evidence": list(quality["evidence"]),
        "selection_hash": hash_hex,
        "candidate_fraction": f"{candidate_numerator}/{candidate_denominator}",
        "candidate_reason": candidate_reason,
    }
    return projected


def hamilton_quotas(counts: Mapping[str, int], target: int) -> dict[str, int]:
    clean = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    total = sum(clean.values())
    if target < 0 or target > total:
        raise ValueError(f"target {target} must be within 0..{total}")
    if not clean:
        if target:
            raise ValueError("cannot apportion a nonzero target over no strata")
        return {}
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    allocated = 0
    for key in sorted(clean):
        numerator = clean[key] * target
        quotient, remainder = divmod(numerator, total)
        quotas[key] = quotient
        allocated += quotient
        remainders.append((remainder, key))
    for _remainder, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : target - allocated
    ]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("Hamilton apportionment did not conserve target")
    if any(quotas[key] > clean[key] for key in quotas):
        raise AssertionError("Hamilton quota exceeded a stratum population")
    return quotas


def open_deterministic_gzip(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    stream = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0)
    return raw, stream


def write_jsonl_line(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    stream.write(canonical_json_bytes(value, newline=True))


def iter_fixed_hashes(path: Path) -> Iterator[bytes]:
    with path.open("rb") as stream:
        while value := stream.read(HASH_BYTES):
            if len(value) != HASH_BYTES:
                raise ValueError(f"{path}: truncated hash index")
            yield value


def merge_hash_files(paths: Sequence[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    previous: bytes | None = None
    with output.open("wb") as destination:
        for value in heapq.merge(*(iter_fixed_hashes(path) for path in paths)):
            if value == previous:
                continue
            destination.write(value)
            previous = value
            count += 1
        destination.flush()
        os.fsync(destination.fileno())
    return count


def load_hash_set(path: Path) -> set[bytes]:
    return set(iter_fixed_hashes(path))


def validate_training_manifest(document: Mapping[str, Any], *, allow_partial: bool) -> list[dict[str, Any]]:
    if document.get("format_version") != TRAINING_MANIFEST_FORMAT or document.get("complete") is not True:
        raise ValueError("training manifest is incomplete or incompatible")
    if not allow_partial and document.get("total_rows") != EXPECTED_TRAINING_ROWS:
        raise ValueError("training manifest total_rows mismatch")
    shards = document.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("training manifest has no shards")
    if not allow_partial and len(shards) != EXPECTED_TRAINING_SHARDS:
        raise ValueError("training manifest shard count mismatch")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("part_index") != index:
            raise ValueError(f"training shard {index} identity mismatch")
        parse_s3_uri(str(shard.get("path") or ""))
        require_sha256(str(shard.get("sha256") or ""), f"training shard {index}")
        if not isinstance(shard.get("version_id"), str) or not shard["version_id"]:
            raise ValueError(f"training shard {index} has no VersionId")
    return [dict(item) for item in shards]


def validate_inference_manifest(document: Mapping[str, Any], *, allow_partial: bool) -> list[dict[str, Any]]:
    if document.get("format_version") != INFERENCE_MANIFEST_FORMAT or document.get("status") != "complete":
        raise ValueError("inference manifest is incomplete or incompatible")
    contract = document.get("contract")
    totals = document.get("totals")
    if not isinstance(contract, Mapping) or not isinstance(totals, Mapping):
        raise ValueError("inference manifest contract/totals missing")
    expected = {
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "prompt_contract_sha256": EXPECTED_PROMPT_SHA256,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"inference contract {key} mismatch")
    if not allow_partial:
        if totals.get("rows") != EXPECTED_INFERENCE_ROWS or totals.get("units") != EXPECTED_INFERENCE_UNITS:
            raise ValueError("inference manifest rows/units mismatch")
        if contract.get("expected_rows") != EXPECTED_INFERENCE_ROWS:
            raise ValueError("inference expected_rows mismatch")
    units = document.get("shards")
    if not isinstance(units, list) or not units:
        raise ValueError("inference manifest shards missing")
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or unit.get("global_unit_index") != index:
            raise ValueError(f"inference unit {index} identity mismatch")
        if unit.get("disposition") != "validated_committed":
            raise ValueError(f"inference unit {index} is not committed")
        output = unit.get("output")
        source = unit.get("source")
        if not isinstance(output, Mapping) or not isinstance(source, Mapping):
            raise ValueError(f"inference unit {index} source/output missing")
        parse_s3_uri(str(output.get("uri") or ""))
        require_sha256(str(output.get("sha256") or ""), f"inference unit {index} output")
        if int(output.get("rows") or 0) <= 0:
            raise ValueError(f"inference unit {index} has no rows")
    return [dict(item) for item in units]


def _training_extract_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    shard = dict(spec["shard"])
    staging = Path(str(spec["staging_dir"]))
    region = str(spec["region"])
    profile = str(spec.get("profile") or "") or None
    client = make_s3_client(profile=profile, region=region, max_pool_connections=4)
    index = int(shard["part_index"])
    source = parse_s3_uri(str(shard["path"]))
    local = staging / f"training-{index:05d}.jsonl"
    client.download_file(
        source.bucket,
        source.key,
        str(local),
        ExtraArgs={"VersionId": str(shard["version_id"])},
    )
    size, digest = sha256_file(local)
    if size != int(shard["bytes"]) or digest != str(shard["sha256"]):
        raise ValueError(f"training shard {index} content identity mismatch")
    id_hashes: list[bytes] = []
    composite_hashes: list[bytes] = []
    rows = 0
    with local.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"training shard {index}:{line_number} is not an object")
            id_hash, composite_hash = training_identity_digests(value)
            id_hashes.append(id_hash)
            composite_hashes.append(composite_hash)
            rows += 1
    local.unlink(missing_ok=True)
    if rows != int(shard["rows"]):
        raise ValueError(f"training shard {index} rows {rows} != {shard['rows']}")
    id_path = staging / f"training-{index:05d}.sample-id.sha256.bin"
    composite_path = staging / f"training-{index:05d}.sample-id-instruction.sha256.bin"
    id_path.write_bytes(b"".join(sorted(set(id_hashes))))
    composite_path.write_bytes(b"".join(sorted(set(composite_hashes))))
    return {
        "part_index": index,
        "rows": rows,
        "id_count": len(set(id_hashes)),
        "composite_count": len(set(composite_hashes)),
        "id_path": str(id_path),
        "composite_path": str(composite_path),
    }


def select_smoke_units(units: Sequence[dict[str, Any]], max_units: int | None) -> list[dict[str, Any]]:
    if max_units is None:
        return list(units)
    if max_units <= 0:
        raise ValueError("max_units must be positive")
    if max_units >= len(units):
        return list(units)
    indices = [(index * len(units)) // max_units for index in range(max_units)]
    if len(set(indices)) != max_units:
        raise AssertionError("smoke unit selection is not unique")
    return [units[index] for index in indices]


def minimal_unit_record(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": unit["dataset_id"],
        "dataset_unit_index": unit["dataset_unit_index"],
        "global_unit_index": unit["global_unit_index"],
        "global_row_start": unit["global_row_start"],
        "global_row_end_exclusive": unit["global_row_end_exclusive"],
        "local_row_ranges": unit["local_row_ranges"],
        "output": {
            "uri": unit["output"]["uri"],
            "sha256": unit["output"]["sha256"],
            "bytes": unit["output"]["bytes"],
            "rows": unit["output"]["rows"],
        },
        "source": {
            "bucket": unit["source"]["bucket"],
            "key": unit["source"]["key"],
            "bytes": unit["source"]["bytes"],
            "etag": unit["source"]["etag"],
            "rows": unit["source"]["rows"],
            "object_index": unit["source"]["object_index"],
        },
    }


def build_run_contract(args: argparse.Namespace, identities: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        "inputs": dict(identities),
        "expected": {
            "inference_rows": EXPECTED_INFERENCE_ROWS,
            "inference_units": EXPECTED_INFERENCE_UNITS,
            "training_rows": EXPECTED_TRAINING_ROWS,
            "training_shards": EXPECTED_TRAINING_SHARDS,
        },
        "selection": {
            "target_rows": args.target_rows,
            "seed": args.seed,
            "candidate_fraction": f"{args.candidate_numerator}/{args.candidate_denominator}",
            "rare_task_candidate_floor_per_scan_worker": (
                MIN_CANDIDATES_PER_TASK_PER_SCAN_WORKER
            ),
            "task_buckets": args.task_buckets,
            "scan_workers": args.scan_workers,
            "select_workers": args.select_workers,
            "apportionment": APPORTIONMENT_VERSION,
        },
        "contracts": {
            "training_exclusion": EXCLUSION_VERSION,
            "quality": QUALITY_VERSION,
            "task_type": TASK_VERSION,
            "prompt_version": EXPECTED_PROMPT_VERSION,
            "prompt_contract_sha256": EXPECTED_PROMPT_SHA256,
            "hld_postprocess_version": EXPECTED_POSTPROCESS_VERSION,
            "bbox_packaging_version": EXPECTED_PACKAGING_VERSION,
            "bbox_coordinate_order": EXPECTED_COORDINATE_ORDER,
        },
        "smoke": {
            "max_units": args.max_units,
            "max_training_shards": args.max_training_shards,
        },
        "output": {
            "root": args.output_root,
            "contains_raw_response": False,
            "contains_original_prediction": False,
            "format": "gzip JSONL sidecar with immutable S3 image locators",
        },
    }


def command_prepare(args: argparse.Namespace) -> dict[str, Any]:
    client = make_s3_client(
        profile=args.aws_profile, region=args.aws_region, max_pool_connections=max(64, args.prepare_processes * 2)
    )
    store = OutputStore(args.output_root, client)
    existing = store.read_json("control/prepare-summary.json", missing_ok=True)
    if existing is not None:
        return existing

    inference, inference_identity = get_pinned_json(
        client,
        uri=args.inference_manifest_uri,
        sha256=args.inference_manifest_sha256,
        version_id=args.inference_manifest_version_id,
    )
    training, training_identity = get_pinned_json(
        client,
        uri=args.training_manifest_uri,
        sha256=args.training_manifest_sha256,
        version_id=args.training_manifest_version_id,
    )
    all_units = validate_inference_manifest(inference, allow_partial=False)
    all_training_shards = validate_training_manifest(training, allow_partial=False)
    selected_units = select_smoke_units(all_units, args.max_units)
    selected_training_shards = (
        all_training_shards[: args.max_training_shards]
        if args.max_training_shards is not None
        else all_training_shards
    )

    identities = {
        "inference_manifest": inference_identity,
        "training_manifest": training_identity,
        "inference_contract": dict(inference["contract"]),
    }
    proposed_contract = build_run_contract(args, identities)
    existing_contract = store.read_json("run-contract.json", missing_ok=True)
    if existing_contract is None:
        store.put_json_once("run-contract.json", proposed_contract)
        contract = proposed_contract
    else:
        proposed_contract["created_at"] = existing_contract.get("created_at")
        if existing_contract != proposed_contract:
            raise ValueError("existing run contract does not match this invocation")
        contract = existing_contract

    with tempfile.TemporaryDirectory(prefix="image-edit-v6-5m-prepare.") as temporary:
        staging = Path(temporary)
        specs = [
            {
                "shard": shard,
                "staging_dir": str(staging),
                "region": args.aws_region,
                "profile": args.aws_profile,
            }
            for shard in selected_training_shards
        ]
        results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.prepare_processes, len(specs))
        ) as pool:
            for completed, result in enumerate(pool.map(_training_extract_worker, specs), 1):
                results.append(result)
                if completed % 8 == 0 or completed == len(specs):
                    print(
                        f"[prepare] training exclusion shards={completed}/{len(specs)} "
                        f"rows={sum(item['rows'] for item in results):,}",
                        flush=True,
                    )
        results.sort(key=lambda item: item["part_index"])
        id_index = staging / "training-sample-id.sha256.bin"
        composite_index = staging / "training-sample-id-instruction.sha256.bin"
        id_count = merge_hash_files([Path(item["id_path"]) for item in results], id_index)
        composite_count = merge_hash_files(
            [Path(item["composite_path"]) for item in results], composite_index
        )
        training_rows = sum(int(item["rows"]) for item in results)
        if args.max_training_shards is None:
            if training_rows != EXPECTED_TRAINING_ROWS:
                raise ValueError("prepared training exclusion row count mismatch")
            if id_count != EXPECTED_TRAINING_ROWS or composite_count != EXPECTED_TRAINING_ROWS:
                raise ValueError("training exclusion identities are not globally unique")
        id_file = store.put_file_once(
            "control/exclusion/training-sample-id.sha256.bin",
            id_index,
            content_type="application/octet-stream",
        )
        composite_file = store.put_file_once(
            "control/exclusion/training-sample-id-instruction.sha256.bin",
            composite_index,
            content_type="application/octet-stream",
        )

        assignment_raw: list[BinaryIO] = []
        assignment_gzip: list[gzip.GzipFile] = []
        assignment_paths: list[Path] = []
        assignment_counts = [0] * args.scan_workers
        assignment_rows = [0] * args.scan_workers
        try:
            for worker in range(args.scan_workers):
                path = staging / f"scan-worker-{worker:05d}.jsonl.gz"
                raw, compressed = open_deterministic_gzip(path)
                assignment_paths.append(path)
                assignment_raw.append(raw)
                assignment_gzip.append(compressed)
            for unit in selected_units:
                record = minimal_unit_record(unit)
                worker = int(record["global_unit_index"]) % args.scan_workers
                write_jsonl_line(assignment_gzip[worker], record)
                assignment_counts[worker] += 1
                assignment_rows[worker] += int(record["output"]["rows"])
        finally:
            for compressed in assignment_gzip:
                compressed.close()
            for raw in assignment_raw:
                raw.close()
        assignments = []
        for worker, path in enumerate(assignment_paths):
            file_record = store.put_file_once(
                f"control/scan-assignments/worker-{worker:05d}.jsonl.gz",
                path,
                content_type=JSONL_CONTENT_TYPE,
                content_encoding="gzip",
            )
            assignments.append(
                {
                    "worker_index": worker,
                    "units": assignment_counts[worker],
                    "rows": assignment_rows[worker],
                    "file": file_record,
                }
            )

    prepared_rows = sum(assignment_rows)
    expected_prepared_rows = sum(int(unit["output"]["rows"]) for unit in selected_units)
    if prepared_rows != expected_prepared_rows:
        raise AssertionError("scan assignments do not conserve inference rows")
    if args.max_units is None and prepared_rows != EXPECTED_INFERENCE_ROWS:
        raise ValueError("full scan assignments do not cover all inference rows")
    summary = {
        "format_version": f"{PIPELINE_VERSION}_prepare_summary",
        "completed_at": utc_now(),
        "run_contract_sha256": sha256_bytes(canonical_json_bytes(contract, newline=True)),
        "inference": {
            "selected_units": len(selected_units),
            "selected_rows": prepared_rows,
            "full_units": len(all_units),
            "full_rows": int(inference["totals"]["rows"]),
        },
        "training_exclusion": {
            "selected_shards": len(selected_training_shards),
            "selected_rows": training_rows,
            "full_shards": len(all_training_shards),
            "full_rows": int(training["total_rows"]),
            "sample_id_count": id_count,
            "sample_id_instruction_count": composite_count,
            "sample_id_index": id_file,
            "sample_id_instruction_index": composite_file,
        },
        "assignments": assignments,
    }
    store.put_json_once("control/prepare-summary.json", summary)
    return summary


def _iter_gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def _source_rows_for_unit(
    unit: Mapping[str, Any], s3fs: Any, client: Any
) -> dict[int, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = unit["source"]
    head = client.head_object(Bucket=str(source["bucket"]), Key=str(source["key"]))
    actual_etag = str(head.get("ETag") or "").strip('"')
    expected_etag = str(source["etag"]).strip('"')
    if actual_etag != expected_etag:
        raise ValueError(f"{source['bucket']}/{source['key']}: source ETag drifted")
    if int(head.get("ContentLength", -1)) != int(source["bytes"]):
        raise ValueError(f"{source['bucket']}/{source['key']}: source size drifted")
    parquet_path = f"{source['bucket']}/{source['key']}"
    parquet = pq.ParquetFile(parquet_path, filesystem=s3fs)
    if parquet.metadata.num_rows != int(source["rows"]):
        raise ValueError(f"{parquet_path}: source row count drifted")
    present = [name for name in SOURCE_COLUMNS if name in parquet.schema_arrow.names]
    if "sample_id" not in present:
        raise ValueError(f"{parquet_path}: source sample_id missing")
    table = parquet.read(columns=present)
    indices = [
        row
        for start, end in unit["local_row_ranges"]
        for row in range(int(start), int(end))
    ]
    if len(indices) != int(unit["output"]["rows"]):
        raise ValueError("unit local row ranges do not match output rows")
    selected = table.take(pa.array(indices, type=pa.int64())).to_pylist()
    return {index: row for index, row in zip(indices, selected, strict=True)}


def _download_and_verify_uri(client: Any, record: Mapping[str, Any], path: Path) -> None:
    source = parse_s3_uri(str(record["uri"]))
    client.download_file(source.bucket, source.key, str(path))
    size, digest = sha256_file(path)
    if size != int(record["bytes"]) or digest != str(record["sha256"]):
        raise ValueError(f"{record['uri']}: downloaded object identity mismatch")


def _validate_scan_summary_files(store: OutputStore, summary: Mapping[str, Any]) -> None:
    for file_record in summary.get("candidate_files", []):
        head = store.head(str(file_record["relative_path"]))
        metadata = head.get("Metadata") if isinstance(head, Mapping) else {}
        if int(head.get("ContentLength", -1)) != int(file_record["bytes"]):
            raise ValueError("candidate file size drifted")
        if store.is_s3 and metadata.get("sha256") != file_record["sha256"]:
            raise ValueError("candidate file SHA drifted")


def command_scan_worker(args: argparse.Namespace) -> dict[str, Any]:
    worker = args.worker_index
    if worker < 0 or worker >= args.scan_workers:
        raise ValueError("scan worker index out of range")
    client = make_s3_client(profile=args.aws_profile, region=args.aws_region, max_pool_connections=32)
    store = OutputStore(args.output_root, client)
    summary_path = f"scan/summaries/worker-{worker:05d}.json"
    existing = store.read_json(summary_path, missing_ok=True)
    if existing is not None:
        _validate_scan_summary_files(store, existing)
        return existing
    prepare = store.read_json("control/prepare-summary.json")
    assert prepare is not None
    assignment = prepare["assignments"][worker]

    import pyarrow.fs as pafs

    s3fs = pafs.S3FileSystem(region=args.aws_region)
    with tempfile.TemporaryDirectory(prefix=f"image-edit-v6-5m-scan-{worker:05d}.") as temporary:
        staging = Path(temporary)
        assignment_path = staging / "assignment.jsonl.gz"
        store.download(str(assignment["file"]["relative_path"]), assignment_path)
        id_path = staging / "training-sample-id.sha256.bin"
        composite_path = staging / "training-sample-id-instruction.sha256.bin"
        store.download(
            str(prepare["training_exclusion"]["sample_id_index"]["relative_path"]), id_path
        )
        store.download(
            str(
                prepare["training_exclusion"]["sample_id_instruction_index"]["relative_path"]
            ),
            composite_path,
        )
        training_ids = load_hash_set(id_path)
        training_composites = load_hash_set(composite_path)

        bucket_raw: dict[int, BinaryIO] = {}
        bucket_streams: dict[int, gzip.GzipFile] = {}
        bucket_paths: dict[int, Path] = {}
        input_rows = units = 0
        eligible_rows = candidate_rows = 0
        hash_fraction_candidate_rows = rare_task_floor_candidate_rows = 0
        exact_training = id_only_training = 0
        task_counts: Counter[str] = Counter()
        candidate_task_counts: Counter[str] = Counter()
        quality_tiers: Counter[str] = Counter()
        rejects: Counter[str] = Counter()
        task_sources: Counter[str] = Counter()
        fallback_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        fallback_sequence = 0
        unit_path = staging / "unit.jsonl"

        def emit_candidate(candidate: dict[str, Any], *, task: str, bucket: int) -> None:
            nonlocal candidate_rows
            if bucket not in bucket_streams:
                path = staging / f"candidate-bucket-{bucket:04d}.jsonl.gz"
                raw, compressed = open_deterministic_gzip(path)
                bucket_paths[bucket] = path
                bucket_raw[bucket] = raw
                bucket_streams[bucket] = compressed
            write_jsonl_line(bucket_streams[bucket], candidate)
            candidate_rows += 1
            candidate_task_counts[task] += 1

        try:
            for unit_number, unit in enumerate(_iter_gzip_jsonl(assignment_path), 1):
                _download_and_verify_uri(client, unit["output"], unit_path)
                source_rows = _source_rows_for_unit(unit, s3fs, client)
                observed_rows = 0
                with unit_path.open("rb") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise ValueError(f"unit {unit['global_unit_index']}:{line_number} not object")
                        observed_rows += 1
                        input_rows += 1
                        source = row.get("dataset_source")
                        if not isinstance(source, Mapping):
                            raise ValueError("inference row dataset_source missing")
                        local_row = int(source.get("source_parquet_local_row", -1))
                        source_row = source_rows.get(local_row)
                        if source_row is None:
                            raise ValueError("inference/source local-row alignment failed")
                        if str(source_row.get("sample_id")) != str(row.get("sample_id")):
                            raise ValueError("inference/source sample_id alignment failed")

                        passed, reason = prediction_quality_decision(row)
                        if not passed:
                            rejects[f"prediction:{reason}"] += 1
                            continue
                        sample_id = str(row["sample_id"])
                        instruction = str(row["instruction"])
                        id_hash = identity_digest("sample-id-v1", sample_id)
                        composite_hash = identity_digest(
                            "sample-id-instruction-v1", sample_id, instruction
                        )
                        if id_hash in training_ids:
                            if composite_hash in training_composites:
                                exact_training += 1
                            else:
                                id_only_training += 1
                            rejects["training_overlap"] += 1
                            continue
                        quality = source_quality_decision(source_row)
                        if not quality["passed"]:
                            for failure in quality["failures"]:
                                rejects[f"source_quality:{failure}"] += 1
                            continue
                        dataset_id = str(source.get("dataset_id") or unit["dataset_id"])
                        task, raw_task, task_source = normalize_task_type(
                            source_row, dataset_id=dataset_id
                        )
                        task_sources[task_source] += 1
                        quality_tiers[str(quality["tier"])] += 1
                        task_counts[task] += 1
                        eligible_rows += 1
                        hash_hex = selection_digest(row, seed=args.seed)
                        bucket = task_bucket(task, args.task_buckets)
                        if candidate_selected(
                            hash_hex,
                            numerator=args.candidate_numerator,
                            denominator=args.candidate_denominator,
                        ):
                            candidate = project_candidate(
                                row,
                                source_row,
                                task_type=task,
                                task_raw=raw_task,
                                task_source=task_source,
                                quality=quality,
                                hash_hex=hash_hex,
                                bucket=bucket,
                                candidate_numerator=args.candidate_numerator,
                                candidate_denominator=args.candidate_denominator,
                            )
                            emit_candidate(candidate, task=task, bucket=bucket)
                            hash_fraction_candidate_rows += 1
                            continue

                        heap = fallback_heaps[task]
                        hash_value = int(hash_hex, 16)
                        if (
                            len(heap) < MIN_CANDIDATES_PER_TASK_PER_SCAN_WORKER
                            or hash_value < -heap[0][0]
                        ):
                            candidate = project_candidate(
                                row,
                                source_row,
                                task_type=task,
                                task_raw=raw_task,
                                task_source=task_source,
                                quality=quality,
                                hash_hex=hash_hex,
                                bucket=bucket,
                                candidate_numerator=args.candidate_numerator,
                                candidate_denominator=args.candidate_denominator,
                                candidate_reason="rare_task_floor",
                            )
                            entry = (-hash_value, fallback_sequence, candidate)
                            fallback_sequence += 1
                            if len(heap) < MIN_CANDIDATES_PER_TASK_PER_SCAN_WORKER:
                                heapq.heappush(heap, entry)
                            else:
                                heapq.heapreplace(heap, entry)
                if observed_rows != int(unit["output"]["rows"]):
                    raise ValueError("inference unit row count mismatch")
                units += 1
                if unit_number % 100 == 0 or unit_number == int(assignment["units"]):
                    print(
                        f"[scan {worker}] units={unit_number}/{assignment['units']} "
                        f"rows={input_rows:,} eligible={eligible_rows:,} candidates={candidate_rows:,}",
                        flush=True,
                    )
                unit_path.unlink(missing_ok=True)

            for task, heap in sorted(fallback_heaps.items()):
                required = min(
                    MIN_CANDIDATES_PER_TASK_PER_SCAN_WORKER,
                    task_counts[task],
                )
                needed = max(0, required - candidate_task_counts[task])
                candidates = sorted(
                    (entry[2] for entry in heap),
                    key=lambda item: str(item["sampling"]["selection_hash"]),
                )[:needed]
                for candidate in candidates:
                    bucket = int(candidate["sampling"]["task_bucket"])
                    emit_candidate(candidate, task=task, bucket=bucket)
                    rare_task_floor_candidate_rows += 1
        finally:
            for compressed in bucket_streams.values():
                compressed.close()
            for raw in bucket_raw.values():
                raw.close()

        if units != int(assignment["units"]) or input_rows != int(assignment["rows"]):
            raise ValueError("scan worker assignment conservation failed")
        candidate_files = []
        for bucket, path in sorted(bucket_paths.items()):
            file_record = store.put_file_once(
                f"scan/candidates/bucket-{bucket:04d}/worker-{worker:05d}.jsonl.gz",
                path,
                content_type=JSONL_CONTENT_TYPE,
                content_encoding="gzip",
            )
            candidate_files.append(
                {
                    **file_record,
                    "bucket": bucket,
                    "rows": sum(
                        count
                        for task, count in candidate_task_counts.items()
                        if task_bucket(task, args.task_buckets) == bucket
                    ),
                }
            )

    summary = {
        "format_version": f"{PIPELINE_VERSION}_scan_worker_summary",
        "completed_at": utc_now(),
        "worker_index": worker,
        "worker_count": args.scan_workers,
        "input_units": units,
        "input_rows": input_rows,
        "eligible_rows": eligible_rows,
        "candidate_rows": candidate_rows,
        "candidate_source_counts": {
            "hash_fraction": hash_fraction_candidate_rows,
            "rare_task_floor": rare_task_floor_candidate_rows,
        },
        "training_excluded": {
            "sample_id_and_instruction": exact_training,
            "sample_id_only_conservative": id_only_training,
            "total": exact_training + id_only_training,
        },
        "reject_counts": dict(sorted(rejects.items())),
        "quality_tier_counts": dict(sorted(quality_tiers.items())),
        "task_type_source_counts": dict(sorted(task_sources.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "candidate_task_counts": dict(sorted(candidate_task_counts.items())),
        "candidate_files": candidate_files,
    }
    store.put_json_once(summary_path, summary)
    return summary


def _sum_counter(summaries: Sequence[Mapping[str, Any]], field: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for summary in summaries:
        result.update({str(key): int(value) for key, value in summary.get(field, {}).items()})
    return result


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    client = make_s3_client(profile=args.aws_profile, region=args.aws_region)
    store = OutputStore(args.output_root, client)
    existing = store.read_json("control/selection-plan.json", missing_ok=True)
    if existing is not None:
        return existing
    prepare = store.read_json("control/prepare-summary.json")
    assert prepare is not None
    summaries = []
    for worker in range(args.scan_workers):
        summary = store.read_json(f"scan/summaries/worker-{worker:05d}.json")
        assert summary is not None
        if summary.get("worker_index") != worker:
            raise ValueError("scan summary worker identity mismatch")
        _validate_scan_summary_files(store, summary)
        summaries.append(summary)
    input_rows = sum(int(item["input_rows"]) for item in summaries)
    if input_rows != int(prepare["inference"]["selected_rows"]):
        raise ValueError("scan summaries do not conserve prepared rows")
    task_counts = _sum_counter(summaries, "task_counts")
    candidate_counts = _sum_counter(summaries, "candidate_task_counts")
    eligible_rows = sum(task_counts.values())
    target = min(args.target_rows, eligible_rows) if args.allow_target_clamp else args.target_rows
    quotas = hamilton_quotas(task_counts, target)
    deficits = {
        task: {"quota": quota, "candidates": candidate_counts.get(task, 0)}
        for task, quota in quotas.items()
        if candidate_counts.get(task, 0) < quota
    }
    if deficits:
        examples = dict(list(sorted(deficits.items()))[:20])
        raise ValueError(
            "candidate reservoir does not cover proportional quotas; "
            f"increase --candidate-numerator: {examples}"
        )
    candidate_files_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        for file_record in summary["candidate_files"]:
            candidate_files_by_bucket[str(file_record["bucket"])].append(file_record)
    strata = {
        task: {
            "eligible_rows": task_counts[task],
            "candidate_rows": candidate_counts.get(task, 0),
            "quota": quotas[task],
            "bucket": task_bucket(task, args.task_buckets),
        }
        for task in sorted(task_counts)
    }
    plan = {
        "format_version": f"{PIPELINE_VERSION}_selection_plan",
        "created_at": utc_now(),
        "apportionment": APPORTIONMENT_VERSION,
        "target_rows": target,
        "eligible_rows": eligible_rows,
        "candidate_rows": sum(candidate_counts.values()),
        "input_rows": input_rows,
        "task_type_count": len(task_counts),
        "strata": strata,
        "candidate_files_by_bucket": {
            key: sorted(value, key=lambda item: item["relative_path"])
            for key, value in sorted(candidate_files_by_bucket.items(), key=lambda item: int(item[0]))
        },
        "aggregate": {
            "training_excluded": sum(int(s["training_excluded"]["total"]) for s in summaries),
            "reject_counts": dict(_sum_counter(summaries, "reject_counts")),
            "quality_tier_counts": dict(_sum_counter(summaries, "quality_tier_counts")),
            "task_type_source_counts": dict(
                _sum_counter(summaries, "task_type_source_counts")
            ),
        },
    }
    store.put_json_once("control/selection-plan.json", plan)
    return plan


def _download_file_record(client: Any, record: Mapping[str, Any], path: Path) -> None:
    uri = str(record["uri"])
    if uri.startswith("s3://"):
        source = parse_s3_uri(uri)
        client.download_file(source.bucket, source.key, str(path))
    else:
        shutil.copyfile(Path(uri), path)
    size, digest = sha256_file(path)
    if size != int(record["bytes"]) or digest != str(record["sha256"]):
        raise ValueError(f"candidate/output file identity mismatch: {record['uri']}")


def command_select_worker(args: argparse.Namespace) -> dict[str, Any]:
    worker = args.worker_index
    if worker < 0 or worker >= args.select_workers:
        raise ValueError("select worker index out of range")
    client = make_s3_client(profile=args.aws_profile, region=args.aws_region, max_pool_connections=32)
    store = OutputStore(args.output_root, client)
    summary_path = f"select/summaries/worker-{worker:05d}.json"
    existing = store.read_json(summary_path, missing_ok=True)
    if existing is not None:
        return existing
    plan = store.read_json("control/selection-plan.json")
    assert plan is not None
    owned_buckets = [bucket for bucket in range(args.task_buckets) if bucket % args.select_workers == worker]
    selected_by_task: Counter[str] = Counter()
    candidate_rows = 0
    selected_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"image-edit-v6-5m-select-{worker:05d}.") as temporary:
        staging = Path(temporary)
        for bucket in owned_buckets:
            by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            files = plan["candidate_files_by_bucket"].get(str(bucket), [])
            for file_index, file_record in enumerate(files):
                path = staging / f"candidate-{bucket:04d}-{file_index:05d}.jsonl.gz"
                _download_file_record(client, file_record, path)
                for row in _iter_gzip_jsonl(path):
                    sampling = row.get("sampling")
                    if not isinstance(sampling, Mapping):
                        raise ValueError("candidate sampling metadata missing")
                    task = str(sampling.get("task_type") or "")
                    if int(sampling.get("task_bucket", -1)) != bucket:
                        raise ValueError("candidate task bucket mismatch")
                    if task not in plan["strata"]:
                        raise ValueError("candidate task is absent from plan")
                    by_task[task].append(row)
                    candidate_rows += 1
                path.unlink(missing_ok=True)
            for task, rows in by_task.items():
                rows.sort(key=lambda row: str(row["sampling"]["selection_hash"]))
                quota = int(plan["strata"][task]["quota"])
                chosen = rows[:quota]
                if len(chosen) != quota:
                    raise ValueError(f"task {task} candidate coverage changed")
                for row in chosen:
                    row["sampling"]["stratum_eligible_rows"] = int(
                        plan["strata"][task]["eligible_rows"]
                    )
                    row["sampling"]["stratum_quota"] = quota
                    row["sampling"]["apportionment"] = APPORTIONMENT_VERSION
                selected_rows.extend(chosen)
                selected_by_task[task] += len(chosen)
        selected_rows.sort(
            key=lambda row: (
                str(row["sampling"]["task_type"]),
                str(row["sampling"]["selection_hash"]),
            )
        )
        output_files = []
        for shard_index, start in enumerate(range(0, len(selected_rows), args.output_shard_rows)):
            rows = selected_rows[start : start + args.output_shard_rows]
            path = staging / f"part-{shard_index:05d}.jsonl.gz"
            raw, compressed = open_deterministic_gzip(path)
            try:
                for row in rows:
                    write_jsonl_line(compressed, row)
            finally:
                compressed.close()
                raw.close()
            file_record = store.put_file_once(
                f"selected/worker-{worker:05d}/part-{shard_index:05d}.jsonl.gz",
                path,
                content_type=JSONL_CONTENT_TYPE,
                content_encoding="gzip",
            )
            output_files.append({**file_record, "rows": len(rows)})
    expected = sum(
        int(item["quota"])
        for item in plan["strata"].values()
        if int(item["bucket"]) in owned_buckets
    )
    if len(selected_rows) != expected:
        raise ValueError("select worker did not emit its planned quota")
    summary = {
        "format_version": f"{PIPELINE_VERSION}_select_worker_summary",
        "completed_at": utc_now(),
        "worker_index": worker,
        "worker_count": args.select_workers,
        "owned_buckets": owned_buckets,
        "candidate_rows_read": candidate_rows,
        "selected_rows": len(selected_rows),
        "selected_task_counts": dict(sorted(selected_by_task.items())),
        "output_files": output_files,
    }
    store.put_json_once(summary_path, summary)
    return summary


def command_finalize(args: argparse.Namespace) -> dict[str, Any]:
    client = make_s3_client(profile=args.aws_profile, region=args.aws_region, max_pool_connections=32)
    store = OutputStore(args.output_root, client)
    success = store.read_json("_SUCCESS", missing_ok=True)
    if success is not None:
        return success
    prepare = store.read_json("control/prepare-summary.json")
    plan = store.read_json("control/selection-plan.json")
    contract = store.read_json("run-contract.json")
    assert prepare is not None and plan is not None and contract is not None
    existing_manifest = store.read_json("manifest.json", missing_ok=True)
    if existing_manifest is not None:
        expected_plan_sha = sha256_bytes(canonical_json_bytes(plan, newline=True))
        expected_contract_sha = sha256_bytes(canonical_json_bytes(contract, newline=True))
        if (
            existing_manifest.get("status") != "complete"
            or int(existing_manifest.get("rows", -1)) != int(plan["target_rows"])
            or existing_manifest.get("selection_plan", {}).get("sha256")
            != expected_plan_sha
            or existing_manifest.get("run_contract", {}).get("sha256")
            != expected_contract_sha
        ):
            raise ValueError("existing manifest does not match the immutable run contract")
        manifest_record = store.put_json_once("manifest.json", existing_manifest)
        success_value = {
            "format_version": f"{PIPELINE_VERSION}_success",
            "status": "complete",
            "completed_at": existing_manifest["completed_at"],
            "rows": existing_manifest["rows"],
            "manifest": manifest_record,
            "output_root": args.output_root,
            "training_sample_id_overlap": 0,
        }
        store.put_json_once("_SUCCESS", success_value)
        return success_value
    summaries = []
    output_files = []
    for worker in range(args.select_workers):
        summary = store.read_json(f"select/summaries/worker-{worker:05d}.json")
        assert summary is not None
        if summary.get("worker_index") != worker:
            raise ValueError("select summary worker identity mismatch")
        summaries.append(summary)
        output_files.extend(summary["output_files"])
    summary_rows = sum(int(item["selected_rows"]) for item in summaries)
    if summary_rows != int(plan["target_rows"]):
        raise ValueError("select summaries do not sum to planned target")

    observed_tasks: Counter[str] = Counter()
    observed_quality: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    rows = 0
    with tempfile.TemporaryDirectory(prefix="image-edit-v6-5m-finalize.") as temporary:
        staging = Path(temporary)
        id_index = staging / "training-sample-id.sha256.bin"
        store.download(
            str(prepare["training_exclusion"]["sample_id_index"]["relative_path"]), id_index
        )
        training_ids = load_hash_set(id_index)
        for file_index, file_record in enumerate(output_files):
            path = staging / f"selected-{file_index:05d}.jsonl.gz"
            _download_file_record(client, file_record, path)
            file_rows = 0
            for row in _iter_gzip_jsonl(path):
                file_rows += 1
                rows += 1
                if "response" in row or "original_prediction" in row:
                    raise ValueError("final output leaked raw/original model output")
                sampling = row.get("sampling")
                if not isinstance(sampling, Mapping):
                    raise ValueError("final output sampling metadata missing")
                hash_hex = str(sampling.get("selection_hash") or "")
                require_sha256(hash_hex, "selection_hash")
                if hash_hex in seen_hashes:
                    raise ValueError("duplicate final selection identity")
                seen_hashes.add(hash_hex)
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("final sample_id missing")
                if identity_digest("sample-id-v1", sample_id) in training_ids:
                    raise ValueError("final output overlaps V6 training sample_id")
                task = str(sampling.get("task_type") or "")
                if task not in plan["strata"]:
                    raise ValueError("final task absent from plan")
                observed_tasks[task] += 1
                observed_quality[str(sampling.get("quality_tier") or "")] += 1
            if file_rows != int(file_record["rows"]):
                raise ValueError("final output file row count mismatch")
            path.unlink(missing_ok=True)
    expected_tasks = Counter(
        {task: int(item["quota"]) for task, item in plan["strata"].items() if int(item["quota"]) > 0}
    )
    if rows != int(plan["target_rows"]) or observed_tasks != expected_tasks:
        raise ValueError("final row count or task distribution differs from plan")
    manifest = {
        "format_version": f"{PIPELINE_VERSION}_manifest",
        "status": "complete",
        "completed_at": utc_now(),
        "rows": rows,
        "files": sorted(output_files, key=lambda item: item["relative_path"]),
        "task_counts": dict(sorted(observed_tasks.items())),
        "quality_tier_counts": dict(sorted(observed_quality.items())),
        "selection_plan": {
            "relative_path": "control/selection-plan.json",
            "sha256": sha256_bytes(canonical_json_bytes(plan, newline=True)),
            "eligible_rows": plan["eligible_rows"],
            "candidate_rows": plan["candidate_rows"],
            "target_rows": plan["target_rows"],
            "task_type_count": plan["task_type_count"],
        },
        "training_exclusion": {
            "version": EXCLUSION_VERSION,
            "training_rows": prepare["training_exclusion"]["full_rows"],
            "sample_id_overlap_in_final": 0,
            "scan_excluded_rows": plan["aggregate"]["training_excluded"],
        },
        "run_contract": {
            "relative_path": "run-contract.json",
            "sha256": sha256_bytes(canonical_json_bytes(contract, newline=True)),
        },
        "validation": {
            "all_files_read_back": True,
            "exact_target_rows": True,
            "exact_hamilton_quotas": True,
            "unique_selection_hashes": len(seen_hashes),
            "training_sample_id_overlap": 0,
            "raw_response_fields_present": 0,
        },
    }
    manifest_record = store.put_json_once("manifest.json", manifest)
    success_value = {
        "format_version": f"{PIPELINE_VERSION}_success",
        "status": "complete",
        "completed_at": manifest["completed_at"],
        "rows": rows,
        "manifest": manifest_record,
        "output_root": args.output_root,
        "training_sample_id_overlap": 0,
    }
    store.put_json_once("_SUCCESS", success_value)
    return success_value


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-manifest-uri", default=DEFAULT_INFERENCE_MANIFEST_URI)
    parser.add_argument(
        "--inference-manifest-sha256", default=DEFAULT_INFERENCE_MANIFEST_SHA256
    )
    parser.add_argument(
        "--inference-manifest-version-id", default=DEFAULT_INFERENCE_MANIFEST_VERSION_ID
    )
    parser.add_argument("--training-manifest-uri", default=DEFAULT_TRAINING_MANIFEST_URI)
    parser.add_argument(
        "--training-manifest-sha256", default=DEFAULT_TRAINING_MANIFEST_SHA256
    )
    parser.add_argument(
        "--training-manifest-version-id", default=DEFAULT_TRAINING_MANIFEST_VERSION_ID
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", default="image-edit-v6-5m-20260818-v1")
    parser.add_argument("--candidate-numerator", type=int, default=1)
    parser.add_argument("--candidate-denominator", type=int, default=10)
    parser.add_argument("--task-buckets", type=int, default=256)
    parser.add_argument("--scan-workers", type=int, default=64)
    parser.add_argument("--select-workers", type=int, default=64)
    parser.add_argument("--prepare-processes", type=int, default=32)
    parser.add_argument("--output-shard-rows", type=int, default=25_000)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-training-shards", type=int)
    parser.add_argument("--allow-target-clamp", action="store_true")
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    scan = subparsers.add_parser("scan-worker")
    scan.add_argument("--worker-index", type=int, required=True)
    subparsers.add_parser("plan")
    select = subparsers.add_parser("select-worker")
    select.add_argument("--worker-index", type=int, required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args(argv)
    for name in (
        "target_rows",
        "candidate_numerator",
        "candidate_denominator",
        "task_buckets",
        "scan_workers",
        "select_workers",
        "prepare_processes",
        "output_shard_rows",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.candidate_numerator > args.candidate_denominator:
        parser.error("candidate numerator cannot exceed denominator")
    if args.select_workers > args.task_buckets:
        parser.error("select workers cannot exceed task buckets")
    if args.max_units is not None and args.max_units <= 0:
        parser.error("--max-units must be positive")
    if args.max_training_shards is not None and args.max_training_shards <= 0:
        parser.error("--max-training-shards must be positive")
    require_sha256(args.inference_manifest_sha256, "inference manifest SHA")
    require_sha256(args.training_manifest_sha256, "training manifest SHA")
    return args


def result_for_log(command: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep terminal logs useful without serializing large per-task dictionaries."""
    summary: dict[str, Any] = {"command": command}
    for key in (
        "format_version",
        "status",
        "completed_at",
        "worker_index",
        "worker_count",
        "input_units",
        "input_rows",
        "eligible_rows",
        "candidate_rows",
        "candidate_rows_read",
        "selected_rows",
        "rows",
        "target_rows",
        "task_type_count",
        "training_sample_id_overlap",
        "output_root",
    ):
        value = result.get(key)
        if value is not None:
            summary[key] = value

    collection_keys = {
        "assignments": "assignment_count",
        "candidate_files": "candidate_file_count",
        "output_files": "output_file_count",
        "files": "output_file_count",
        "task_counts": "task_count",
        "selected_task_counts": "selected_task_count",
        "strata": "strata_count",
    }
    for key, output_key in collection_keys.items():
        value = result.get(key)
        if isinstance(value, (Mapping, Sequence)) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            summary[output_key] = len(value)

    inference = result.get("inference")
    if isinstance(inference, Mapping):
        for key in ("selected_units", "selected_rows"):
            if inference.get(key) is not None:
                summary[f"inference_{key}"] = inference[key]
    training = result.get("training_exclusion")
    if isinstance(training, Mapping):
        for key in ("full_rows", "selected_rows", "scan_excluded_rows"):
            if training.get(key) is not None:
                summary[f"training_{key}"] = training[key]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = {
        "prepare": command_prepare,
        "scan-worker": command_scan_worker,
        "plan": command_plan,
        "select-worker": command_select_worker,
        "finalize": command_finalize,
    }
    result = commands[args.command](args)
    print(
        json.dumps(
            result_for_log(args.command, result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
