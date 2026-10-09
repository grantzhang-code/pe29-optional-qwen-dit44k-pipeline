#!/usr/bin/env python3
"""Materialize the frozen image-edit-v6 Target images and JSON as aligned Parquet.

The output follows the image-edit-v5 target-caption campaign contract:

* ``main/`` contains the original Target image bytes in canonical image Parquet;
* ``sidecar/`` contains the same relative shards and row keys plus the validated
  V6 ``final_json`` in ``structured_description``;
* a unit completion marker is written only after both immutable Parquet objects
  have been uploaded and read-back validated;
* reruns validate and skip committed units without downloading their images;
* ``manifest.json`` and ``_SUCCESS`` are committed only after every frozen row
  and every output object has been audited.

The authoritative 256-shard training manifest is itself pinned to the ten
``pruned.jsonl`` objects requested for this dataset.  Workers stride those
physical shards, which avoids repeatedly downloading the ten 8--9 GiB sources.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

# Import only the locator/range-reader primitives.  This standalone materializer
# does not need the ms-swift monkeypatch that module normally applies on import.
os.environ["DLMS_S3_IMAGE_LOAD_NO_APPLY"] = "1"

from dlms.datasets.image_edit_v6_full_json import (
    ANNOTATION_CONTRACT_SHA256,
    ANNOTATION_CONTRACT_VERSION,
    MANIFEST_FORMAT,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    ROW_FORMAT,
    TARGET_SCHEMA,
    ImageEditV6FullJsonConfig,
    make_row,
)
from dlms.patches.s3_image_load import (
    _ParquetImageLocator,
    _parse_parquet_s3_uri,
    _S3RangeReader,
)

FORMAT_VERSION = "image_edit_v6_target_main_sidecar_v1"
UNIT_MARKER_FORMAT = "image_edit_v6_target_main_sidecar_unit_v1"
RUN_CONTRACT_FORMAT = "image_edit_v6_target_main_sidecar_run_contract_v1"
FINAL_MANIFEST_FORMAT = "image_edit_v6_target_main_sidecar_manifest_v1"
SUCCESS_FORMAT = "image_edit_v6_target_main_sidecar_success_v1"

DEFAULT_MANIFEST_URI = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/prune-image-edit-v6/"
    "full-gemini31-lite-10x800k-20260813-v1/"
    "training-v6-7984173a-all10-256shards-v1/"
    "image_edit_v6_full_json.manifest.json"
)
DEFAULT_MANIFEST_SHA256 = "692bf21ce252bba6f221524d628bfdc1679684935c36783d2ed8bc3f1d1548cc"
DEFAULT_MANIFEST_VERSION_ID = "5cwqU2tJqwjLv9_KUpNy8IYsBJXGGaaI"
EXPECTED_TOTAL_ROWS = 7_234_587
EXPECTED_PHYSICAL_SHARDS = 256
DEFAULT_UNIT_ROWS = 256
DATA_SOURCE = "image_edit_v6_full_gemini31_lite_pruned_20260813_v1"
TASK = "image_edit_v6_target_json"

SOURCE_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/prune-image-edit-v6/"
    "full-gemini31-lite-10x800k-20260813-v1/jobs"
)
EXPECTED_SOURCE_URIS = tuple(
    f"{SOURCE_ROOT}/job-{index:02d}/final/pruned.jsonl" for index in range(10)
)
SOURCE_ROW_FIELDS = frozenset(
    {
        "bbox_coordinate_order",
        "call2_audit",
        "contract_sha256",
        "contract_version",
        "final_json",
        "instruction",
        "sample_id",
        "source_s3_uri",
        "status",
        "target_s3_uri",
    }
)
ALLOWED_STATUSES = ("validated", "manual_review_required")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
JSON_CONTENT_TYPE = "application/json"
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"

ROW_VALIDATION_CONFIG = ImageEditV6FullJsonConfig(
    dataset_name="image_edit_v6_target_sidecar_validation",
    val_dataset_name="image_edit_v6_target_sidecar_validation_val",
    paths=(),
    allowed_statuses=ALLOWED_STATUSES,
    source_name=DATA_SOURCE,
    reject_invalid_rows=True,
)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class SourceShard:
    part_index: int
    part_count: int
    uri: str
    size: int
    rows: int
    sha256: str
    version_id: str
    global_row_start: int


@dataclass(frozen=True)
class ManifestIdentity:
    uri: str
    sha256: str
    version_id: str
    size: int


@dataclass(frozen=True)
class RowPlan:
    image_key: str
    global_input_index: int
    part_index: int
    source_local_row: int
    sample_id: str
    status: str
    instruction: str
    reference_uri: str
    target_uri: str
    structured_description: str
    structured_description_sha256: str
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class ImagePayload:
    payload: bytes
    width: int
    height: int
    is_rgba: bool
    image_format: str
    image_mode: str
    sha256: str
    content_type: str
    etag: str
    version_id: str


@dataclass(frozen=True)
class UnitSpec:
    shard: SourceShard
    unit_index: int
    local_row_start: int
    local_row_end: int

    @property
    def rows(self) -> int:
        return self.local_row_end - self.local_row_start

    @property
    def global_row_start(self) -> int:
        return self.shard.global_row_start + self.local_row_start

    @property
    def global_row_end(self) -> int:
        return self.shard.global_row_start + self.local_row_end

    @property
    def marker_relative(self) -> str:
        return (
            f"markers/part-{self.shard.part_index:05d}/"
            f"unit-{self.unit_index:06d}.complete.json"
        )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def join_output(root: str, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{root.rstrip('/')}/{suffix}" if suffix else root.rstrip("/")


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return ""
    error = response.get("Error")
    return str(error.get("Code") or "") if isinstance(error, Mapping) else ""


def make_s3_client(profile: str | None, region: str, *, max_pool_connections: int) -> Any:
    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 12, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{location} must be a lowercase SHA-256")
    return value


def validate_source_manifest(
    document: Any,
    *,
    expected_total_rows: int = EXPECTED_TOTAL_ROWS,
    expected_shard_count: int = EXPECTED_PHYSICAL_SHARDS,
    expected_source_uris: Sequence[str] = EXPECTED_SOURCE_URIS,
) -> list[SourceShard]:
    """Validate the publication and return shards with stable global offsets."""

    if not isinstance(document, Mapping):
        raise ValueError("source manifest must be an object")
    if document.get("complete") is not True or document.get("format_version") != MANIFEST_FORMAT:
        raise ValueError("source manifest is incomplete or has the wrong format")
    if document.get("total_rows") != expected_total_rows:
        raise ValueError(
            f"source manifest total_rows={document.get('total_rows')!r}, "
            f"expected={expected_total_rows}"
        )
    if document.get("physical_shard_count") != expected_shard_count:
        raise ValueError("source manifest physical_shard_count mismatch")
    status_counts = document.get("status_counts")
    if not isinstance(status_counts, Mapping) or set(status_counts) != set(ALLOWED_STATUSES):
        raise ValueError("source manifest status_counts mismatch")
    if sum(int(value) for value in status_counts.values()) != expected_total_rows:
        raise ValueError("source manifest status_counts do not sum to total_rows")

    contract = document.get("dataset_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("source manifest dataset_contract is missing")
    expected_contract = {
        "row_format_version": ROW_FORMAT,
        "runtime_loader": "dlms/datasets/image_edit_v6_full_json.py",
        "target_schema": TARGET_SCHEMA,
        "target_json_field": "final_json",
        "target_image_field": "target_s3_uri",
        "reference_image_field": "source_s3_uri",
        "instruction_field": "instruction",
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "source_stage": "offline_pruned_edit_instruction",
        "selection": "authoritative_call2_final_json",
        "bbox_coordinate_order": "xyxy",
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(
                f"source manifest dataset_contract.{field}={contract.get(field)!r}, "
                f"expected={expected!r}"
            )
    if tuple(contract.get("allowed_statuses") or ()) != ALLOWED_STATUSES:
        raise ValueError("source manifest allowed_statuses mismatch")
    source_objects = contract.get("source_objects")
    if not isinstance(source_objects, list):
        raise ValueError("source manifest source_objects must be an array")
    observed_source_uris = tuple(str(item.get("uri")) for item in source_objects)
    if observed_source_uris != tuple(expected_source_uris):
        raise ValueError("source manifest does not pin the requested ten pruned JSONL objects")
    for index, source in enumerate(source_objects):
        if not isinstance(source, Mapping) or source.get("source_index") != index:
            raise ValueError(f"source_objects[{index}] identity mismatch")
        _require_sha256(source.get("sha256"), f"source_objects[{index}].sha256")
        if not isinstance(source.get("version_id"), str) or not source["version_id"]:
            raise ValueError(f"source_objects[{index}].version_id is missing")
        if type(source.get("rows")) is not int or source["rows"] <= 0:
            raise ValueError(f"source_objects[{index}].rows is invalid")

    raw_shards = document.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != expected_shard_count:
        raise ValueError("source manifest shards length mismatch")
    paths = document.get("paths")
    if not isinstance(paths, list) or paths != [item.get("path") for item in raw_shards]:
        raise ValueError("source manifest paths do not exactly match shards")

    shards: list[SourceShard] = []
    global_start = 0
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise ValueError(f"shards[{index}] must be an object")
        if raw.get("part_index") != index or raw.get("part_count") != expected_shard_count:
            raise ValueError(f"shards[{index}] part identity mismatch")
        uri = str(raw.get("path") or "")
        parse_s3_uri(uri)
        size = raw.get("bytes")
        rows = raw.get("rows")
        version_id = raw.get("version_id")
        if type(size) is not int or size <= 0 or type(rows) is not int or rows <= 0:
            raise ValueError(f"shards[{index}] size/rows are invalid")
        if not isinstance(version_id, str) or not version_id:
            raise ValueError(f"shards[{index}].version_id is missing")
        shards.append(
            SourceShard(
                part_index=index,
                part_count=expected_shard_count,
                uri=uri,
                size=size,
                rows=rows,
                sha256=_require_sha256(raw.get("sha256"), f"shards[{index}].sha256"),
                version_id=version_id,
                global_row_start=global_start,
            )
        )
        global_start += rows
    if global_start != expected_total_rows:
        raise ValueError("source manifest shard row counts do not sum to total_rows")
    return shards


def load_source_manifest(
    client: Any,
    *,
    uri: str,
    expected_sha256: str,
    version_id: str,
) -> tuple[dict[str, Any], ManifestIdentity, list[SourceShard]]:
    _require_sha256(expected_sha256, "manifest SHA-256")
    location = parse_s3_uri(uri)
    response = client.get_object(Bucket=location.bucket, Key=location.key, VersionId=version_id)
    payload = response["Body"].read()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source manifest SHA-256 mismatch: actual={actual_sha256} expected={expected_sha256}"
        )
    actual_version = str(response.get("VersionId") or "")
    if actual_version != version_id:
        raise ValueError(
            f"source manifest VersionId mismatch: actual={actual_version!r} expected={version_id!r}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source manifest is not valid UTF-8 JSON") from exc
    shards = validate_source_manifest(document)
    return (
        dict(document),
        ManifestIdentity(uri=uri, sha256=actual_sha256, version_id=version_id, size=len(payload)),
        shards,
    )


def _schema_metadata(kind: str, manifest_sha256: str) -> dict[bytes, bytes]:
    return {
        b"contract": FORMAT_VERSION.encode("utf-8"),
        b"dataset_kind": kind.encode("utf-8"),
        b"source_manifest_sha256": manifest_sha256.encode("utf-8"),
        b"image_semantics": b"target_s3_uri-original-bytes",
        b"key_semantics": b"decimal-global-row-in-frozen-manifest-order",
        b"structured_description_semantics": b"canonical-image-edit-v6-final_json",
        b"bbox_coordinate_order": b"xyxy",
    }


def main_schema(manifest_sha256: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("image_key", pa.string(), nullable=False),
            pa.field("image", pa.binary(), nullable=False),
            pa.field("width", pa.int64(), nullable=False),
            pa.field("height", pa.int64(), nullable=False),
            pa.field("s3_image_path", pa.string(), nullable=False),
            pa.field("is_rgba", pa.bool_(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("data_source", pa.string(), nullable=False),
            pa.field("ocr_text", pa.string(), nullable=False),
            pa.field("meta_info", pa.string(), nullable=False),
        ],
        metadata=_schema_metadata("main", manifest_sha256),
    )


def sidecar_schema(manifest_sha256: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("image_key", pa.string(), nullable=False),
            pa.field("width", pa.int64(), nullable=False),
            pa.field("height", pa.int64(), nullable=False),
            pa.field("s3_image_path", pa.string(), nullable=False),
            pa.field("is_rgba", pa.bool_(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("data_source", pa.string(), nullable=False),
            pa.field("ocr_text", pa.string(), nullable=False),
            pa.field("meta_info", pa.string(), nullable=False),
            pa.field("structured_description", pa.string(), nullable=False),
        ],
        metadata=_schema_metadata("sidecar", manifest_sha256),
    )


def _schema_contract(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


class OutputStore:
    """Immutable local/S3 output helper with metadata and marker verification."""

    def __init__(self, root: str, client: Any | None):
        self.root = root.rstrip("/")
        self.is_s3 = self.root.startswith("s3://")
        self.client = client
        if self.is_s3:
            parse_s3_uri(self.root)
            if client is None:
                raise ValueError("an S3 client is required for an S3 output root")
        else:
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def absolute(self, relative: str) -> str:
        return join_output(self.root, relative)

    def _local(self, relative: str) -> Path:
        return Path(self.root) / relative

    def _head_or_none(self, relative: str) -> Mapping[str, Any] | None:
        if not self.is_s3:
            path = self._local(relative)
            if not path.is_file():
                return None
            return {"ContentLength": path.stat().st_size}
        location = parse_s3_uri(self.absolute(relative))
        try:
            return self.client.head_object(Bucket=location.bucket, Key=location.key)
        except Exception as exc:
            if _error_code(exc) in NOT_FOUND_CODES:
                return None
            raise

    def validate_file(
        self,
        relative: str,
        *,
        size: int,
        sha256: str,
        content_type: str = PARQUET_CONTENT_TYPE,
    ) -> dict[str, Any]:
        head = self._head_or_none(relative)
        if head is None:
            raise FileNotFoundError(self.absolute(relative))
        if int(head.get("ContentLength", -1)) != size:
            raise ValueError(f"{self.absolute(relative)}: object size mismatch")
        if self.is_s3:
            metadata = head.get("Metadata")
            if not isinstance(metadata, Mapping) or metadata.get("sha256") != sha256:
                raise ValueError(f"{self.absolute(relative)}: sha256 metadata mismatch")
            observed_type = str(head.get("ContentType") or "").split(";", 1)[0]
            if observed_type != content_type:
                raise ValueError(f"{self.absolute(relative)}: ContentType mismatch")
            return {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256,
                "version_id": str(head.get("VersionId") or ""),
                "etag": str(head.get("ETag") or "").strip('"'),
            }
        if sha256_file(self._local(relative)) != sha256:
            raise ValueError(f"{self.absolute(relative)}: local SHA-256 mismatch")
        return {
            "relative_path": relative,
            "bytes": size,
            "sha256": sha256,
            "version_id": "local",
            "etag": sha256,
        }

    def put_file_once(
        self,
        source: Path,
        relative: str,
        *,
        content_type: str = PARQUET_CONTENT_TYPE,
    ) -> dict[str, Any]:
        size = source.stat().st_size
        digest = sha256_file(source)
        existing = self._head_or_none(relative)
        if existing is not None:
            return self.validate_file(
                relative, size=size, sha256=digest, content_type=content_type
            )
        if self.is_s3:
            destination = parse_s3_uri(self.absolute(relative))
            self.client.upload_file(
                str(source),
                destination.bucket,
                destination.key,
                ExtraArgs={
                    "ContentType": content_type,
                    "Metadata": {"sha256": digest},
                },
            )
        else:
            destination_path = self._local(relative)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            if destination_path.exists():
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, destination_path)
        return self.validate_file(
            relative, size=size, sha256=digest, content_type=content_type
        )

    def read_json(self, relative: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        if self.is_s3:
            location = parse_s3_uri(self.absolute(relative))
            try:
                response = self.client.get_object(Bucket=location.bucket, Key=location.key)
            except Exception as exc:
                if missing_ok and _error_code(exc) in NOT_FOUND_CODES:
                    return None
                raise
            payload = response["Body"].read()
            metadata = response.get("Metadata")
            if not isinstance(metadata, Mapping) or metadata.get("sha256") != sha256_bytes(payload):
                raise ValueError(f"{self.absolute(relative)}: JSON sha256 metadata mismatch")
            observed_type = str(response.get("ContentType") or "").split(";", 1)[0]
            if observed_type != JSON_CONTENT_TYPE:
                raise ValueError(f"{self.absolute(relative)}: JSON ContentType mismatch")
        else:
            path = self._local(relative)
            if not path.is_file():
                if missing_ok:
                    return None
                raise FileNotFoundError(path)
            payload = path.read_bytes()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.absolute(relative)}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{self.absolute(relative)}: expected a JSON object")
        return value

    def put_json_once(self, relative: str, value: Mapping[str, Any]) -> str:
        payload = canonical_json_bytes(value, newline=True)
        digest = sha256_bytes(payload)
        existing = self.read_json(relative, missing_ok=True)
        if existing is not None:
            if existing != dict(value):
                raise ValueError(f"refusing to overwrite mismatched {self.absolute(relative)}")
            return digest
        if self.is_s3:
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
                if _error_code(exc) not in {"412", "PreconditionFailed"}:
                    raise
                raced = self.read_json(relative)
                if raced != dict(value):
                    raise ValueError(
                        f"concurrent writer committed mismatched {self.absolute(relative)}"
                    ) from exc
        else:
            destination_path = self._local(relative)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(
                    destination_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError:
                raced = self.read_json(relative)
                if raced != dict(value):
                    raise ValueError(
                        f"concurrent writer committed mismatched {destination_path}"
                    )
            else:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        committed = self.read_json(relative)
        if committed != dict(value):
            raise ValueError(f"JSON read-back mismatch: {self.absolute(relative)}")
        return digest


def build_run_contract(
    document: Mapping[str, Any],
    identity: ManifestIdentity,
    *,
    unit_rows: int,
) -> dict[str, Any]:
    return {
        "format_version": RUN_CONTRACT_FORMAT,
        "source_manifest": {
            "uri": identity.uri,
            "sha256": identity.sha256,
            "version_id": identity.version_id,
            "bytes": identity.size,
            "format_version": document["format_version"],
            "total_rows": document["total_rows"],
            "physical_shard_count": document["physical_shard_count"],
            "status_counts": dict(document["status_counts"]),
            "source_objects": list(document["dataset_contract"]["source_objects"]),
        },
        "unit_rows": unit_rows,
        "assignment": "source_part_index_mod_worker_count",
        "image_key": "decimal global row in frozen manifest shard order",
        "relative_shard_alignment": "main and sidecar paths, row order, and image_key are exact",
        "main_schema": _schema_contract(main_schema(identity.sha256)),
        "sidecar_schema": _schema_contract(sidecar_schema(identity.sha256)),
        "structured_description": {
            "source_field": "final_json",
            "schema": TARGET_SCHEMA,
            "serialization": "canonical UTF-8 JSON string",
            "bbox_coordinate_order": "xyxy",
        },
    }


def ensure_run_contract(
    store: OutputStore,
    document: Mapping[str, Any],
    identity: ManifestIdentity,
    *,
    unit_rows: int,
) -> dict[str, Any]:
    if unit_rows <= 0:
        raise ValueError("unit_rows must be positive")
    contract = build_run_contract(document, identity, unit_rows=unit_rows)
    store.put_json_once("run-contract.json", contract)
    return contract


def plan_source_row(
    source_row: Any,
    *,
    shard: SourceShard,
    local_row: int,
) -> RowPlan:
    location = f"{shard.uri}#row={local_row}"
    if not isinstance(source_row, Mapping):
        raise ValueError(f"{location}: source row must be an object")
    if set(source_row) != SOURCE_ROW_FIELDS:
        missing = sorted(SOURCE_ROW_FIELDS.difference(source_row))
        unknown = sorted(set(source_row).difference(SOURCE_ROW_FIELDS))
        raise ValueError(f"{location}: source fields mismatch missing={missing} unknown={unknown}")
    transformed = make_row(source_row, source_path=location, config=ROW_VALIDATION_CONFIG)
    if transformed is None:  # reject_invalid_rows=True makes this defensive only.
        raise ValueError(f"{location}: row validation unexpectedly returned None")
    canonical_target = str(transformed["messages"][-1]["content"])
    global_index = shard.global_row_start + local_row
    source_metadata = {
        "bbox_coordinate_order": source_row["bbox_coordinate_order"],
        "call2_audit": source_row["call2_audit"],
        "contract_sha256": source_row["contract_sha256"],
        "contract_version": source_row["contract_version"],
    }
    return RowPlan(
        image_key=str(global_index),
        global_input_index=global_index,
        part_index=shard.part_index,
        source_local_row=local_row,
        sample_id=str(source_row["sample_id"]),
        status=str(source_row["status"]),
        instruction=str(source_row["instruction"]),
        reference_uri=str(source_row["source_s3_uri"]),
        target_uri=str(source_row["target_s3_uri"]),
        structured_description=canonical_target,
        structured_description_sha256=sha256_bytes(canonical_target.encode("utf-8")),
        source_metadata=source_metadata,
    )


def row_identity_sha256(plans: Sequence[RowPlan]) -> str:
    identity = [
        {
            "image_key": plan.image_key,
            "source_local_row": plan.source_local_row,
            "sample_id": plan.sample_id,
            "target_s3_uri": plan.target_uri,
            "structured_description_sha256": plan.structured_description_sha256,
        }
        for plan in plans
    ]
    return sha256_bytes(canonical_json_bytes(identity))


def _decode_image_payload(
    payload: bytes,
    plan: RowPlan,
    *,
    content_type: str | None,
    etag: str,
    version_id: str,
) -> ImagePayload:
    if not payload:
        raise ValueError(f"image_key={plan.image_key}: Target image is empty")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            image_format = image.format or "unknown"
            image_mode = image.mode
            is_rgba = image_mode in {"RGBA", "LA"} or "transparency" in image.info
    except Exception as exc:
        raise ValueError(f"image_key={plan.image_key}: full Target decode failed") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"image_key={plan.image_key}: invalid decoded geometry")
    return ImagePayload(
        payload=payload,
        width=width,
        height=height,
        is_rgba=is_rgba,
        image_format=image_format,
        image_mode=image_mode,
        sha256=sha256_bytes(payload),
        content_type=content_type
        or Image.MIME.get(image_format, "application/octet-stream"),
        etag=etag,
        version_id=version_id,
    )


def _download_direct_s3_image(client: Any, plan: RowPlan) -> ImagePayload:
    location = parse_s3_uri(plan.target_uri)
    response = client.get_object(Bucket=location.bucket, Key=location.key)
    payload = response["Body"].read()
    expected_length = response.get("ContentLength")
    if type(expected_length) is int and len(payload) != expected_length:
        raise OSError(
            f"image_key={plan.image_key}: Target length mismatch {len(payload)} != {expected_length}"
        )
    return _decode_image_payload(
        payload=payload,
        content_type=str(response.get("ContentType") or "application/octet-stream"),
        etag=str(response.get("ETag") or "").strip('"'),
        version_id=str(response.get("VersionId") or ""),
        plan=plan,
    )


def _download_parquet_image_group(
    client: Any,
    items: Sequence[tuple[int, RowPlan, _ParquetImageLocator]],
) -> list[tuple[int, ImagePayload]]:
    """Load all requested cells from one immutable backing Parquet object."""

    if not items:
        return []
    first = items[0][2]
    if any(
        (locator.bucket, locator.key, locator.etag) != (first.bucket, first.key, first.etag)
        for _, _, locator in items
    ):
        raise ValueError("parquet image group contains multiple backing objects")

    head = client.head_object(Bucket=first.bucket, Key=first.key)
    actual_etag = str(head.get("ETag") or "").strip('"')
    if actual_etag != first.etag:
        raise ValueError(
            f"parquet+s3 ETag changed for s3://{first.bucket}/{first.key}: "
            f"{actual_etag!r} != {first.etag!r}"
        )
    size = int(head.get("ContentLength") or 0)
    if size < 1:
        raise ValueError(f"empty parquet+s3 object: s3://{first.bucket}/{first.key}")
    version_id = str(head.get("VersionId") or "")
    reader = _S3RangeReader(
        client,
        bucket=first.bucket,
        key=first.key,
        etag=first.etag,
        size=size,
    )
    parquet = pq.ParquetFile(reader)
    requested_columns = {locator.column for _, _, locator in items}
    missing_columns = sorted(requested_columns.difference(parquet.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"parquet+s3 object lacks columns {missing_columns}")

    row_group_ends: list[int] = []
    row_count = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_count += parquet.metadata.row_group(row_group_index).num_rows
        row_group_ends.append(row_count)

    by_row_group: dict[
        int, list[tuple[int, RowPlan, _ParquetImageLocator, int]]
    ] = defaultdict(list)
    for output_index, plan, locator in items:
        row_group_index = bisect_right(row_group_ends, locator.row)
        if row_group_index >= len(row_group_ends):
            raise ValueError(f"parquet+s3 row {locator.row} is outside row count {row_count}")
        row_group_start = 0 if row_group_index == 0 else row_group_ends[row_group_index - 1]
        by_row_group[row_group_index].append(
            (output_index, plan, locator, locator.row - row_group_start)
        )

    results: list[tuple[int, ImagePayload]] = []
    for row_group_index in sorted(by_row_group):
        group_items = by_row_group[row_group_index]
        columns = sorted({locator.column for _, _, locator, _ in group_items})
        table = parquet.read_row_group(row_group_index, columns=columns, use_threads=False)
        for output_index, plan, locator, row_offset in group_items:
            cell = table[locator.column][row_offset].as_py()
            if locator.column == "input_images":
                if (
                    not isinstance(cell, list)
                    or locator.index is None
                    or locator.index >= len(cell)
                ):
                    raise ValueError("parquet+s3 input_images cell/index is invalid")
                cell = cell[locator.index]
            if not isinstance(cell, (bytes, bytearray, memoryview)) or not cell:
                raise ValueError("parquet+s3 image cell does not contain bytes")
            results.append(
                (
                    output_index,
                    _decode_image_payload(
                        bytes(cell),
                        plan,
                        content_type=None,
                        etag=first.etag,
                        version_id=version_id,
                    ),
                )
            )
    return results


def download_image(client: Any, plan: RowPlan) -> ImagePayload:
    if plan.target_uri.startswith("parquet+s3://"):
        locator = _parse_parquet_s3_uri(plan.target_uri)
        return _download_parquet_image_group(client, [(0, plan, locator)])[0][1]
    return _download_direct_s3_image(client, plan)


def download_images(
    client: Any,
    plans: Sequence[RowPlan],
    *,
    download_workers: int,
) -> list[ImagePayload]:
    """Load a unit in order, batching immutable Parquet cells by backing object."""

    if download_workers <= 0:
        raise ValueError("download_workers must be positive")
    if not plans:
        return []

    direct: list[tuple[int, RowPlan]] = []
    parquet_groups: dict[
        tuple[str, str, str], list[tuple[int, RowPlan, _ParquetImageLocator]]
    ] = defaultdict(list)
    for index, plan in enumerate(plans):
        if plan.target_uri.startswith("parquet+s3://"):
            locator = _parse_parquet_s3_uri(plan.target_uri)
            parquet_groups[(locator.bucket, locator.key, locator.etag)].append(
                (index, plan, locator)
            )
        else:
            direct.append((index, plan))

    results: list[ImagePayload | None] = [None] * len(plans)
    task_count = len(direct) + len(parquet_groups)
    with ThreadPoolExecutor(max_workers=min(download_workers, task_count)) as executor:
        futures = [
            executor.submit(
                lambda index, plan: [(index, _download_direct_s3_image(client, plan))],
                index,
                plan,
            )
            for index, plan in direct
        ]
        futures.extend(
            executor.submit(_download_parquet_image_group, client, tuple(items))
            for _, items in sorted(parquet_groups.items())
        )
        for future in futures:
            for index, image in future.result():
                if results[index] is not None:
                    raise ValueError(f"duplicate image result at unit position {index}")
                results[index] = image
    if any(image is None for image in results):
        raise ValueError("image batch loader did not return every unit position")
    return [image for image in results if image is not None]


def _build_tables(
    plans: Sequence[RowPlan],
    images: Sequence[ImagePayload],
    *,
    unit: UnitSpec,
    manifest_identity: ManifestIdentity,
) -> tuple[pa.Table, pa.Table, list[dict[str, Any]]]:
    if len(plans) != unit.rows or len(images) != len(plans):
        raise ValueError("unit plan/image counts do not match UnitSpec")
    main_records: list[dict[str, Any]] = []
    sidecar_records: list[dict[str, Any]] = []
    target_audit: list[dict[str, Any]] = []
    for plan, image in zip(plans, images, strict=True):
        meta_info = {
            "input_index": plan.global_input_index,
            "sample_id": plan.sample_id,
            "status": plan.status,
            "instruction": plan.instruction,
            "reference_image": plan.reference_uri,
            "target_image": plan.target_uri,
            "target_sha256": image.sha256,
            "target_content_length": len(image.payload),
            "target_content_type": image.content_type,
            "target_etag": image.etag,
            "target_version_id": image.version_id,
            "image_format": image.image_format,
            "image_mode": image.image_mode,
            "structured_description_sha256": plan.structured_description_sha256,
            "annotation": plan.source_metadata,
            "provenance": {
                "source_manifest_uri": manifest_identity.uri,
                "source_manifest_sha256": manifest_identity.sha256,
                "source_manifest_version_id": manifest_identity.version_id,
                "source_shard_uri": unit.shard.uri,
                "source_shard_sha256": unit.shard.sha256,
                "source_shard_version_id": unit.shard.version_id,
                "source_part_index": unit.shard.part_index,
                "source_local_row": plan.source_local_row,
                "unit_index": unit.unit_index,
            },
        }
        metadata_text = canonical_json_bytes(meta_info).decode("utf-8")
        retained = {
            "image_key": plan.image_key,
            "width": image.width,
            "height": image.height,
            "s3_image_path": plan.target_uri,
            "is_rgba": image.is_rgba,
            "task": TASK,
            "data_source": DATA_SOURCE,
            "ocr_text": "",
            "meta_info": metadata_text,
        }
        main_records.append({"image": image.payload, **retained})
        sidecar_records.append({**retained, "structured_description": plan.structured_description})
        target_audit.append(
            {
                "image_key": plan.image_key,
                "target_sha256": image.sha256,
                "target_content_length": len(image.payload),
                "structured_description_sha256": plan.structured_description_sha256,
            }
        )
    main = pa.Table.from_pylist(main_records, schema=main_schema(manifest_identity.sha256))
    sidecar = pa.Table.from_pylist(
        sidecar_records, schema=sidecar_schema(manifest_identity.sha256)
    )
    return main, sidecar, target_audit


def _write_table(table: pa.Table, path: Path) -> None:
    pq.write_table(
        table,
        path,
        compression="zstd",
        version="2.6",
        row_group_size=min(64, max(1, table.num_rows)),
        use_dictionary=["image_key", "s3_image_path", "task", "data_source"],
        write_statistics=True,
    )


def verify_local_pair(
    main_path: Path,
    sidecar_path: Path,
    plans: Sequence[RowPlan],
    *,
    manifest_sha256: str,
) -> None:
    main = pq.read_table(main_path)
    sidecar = pq.read_table(sidecar_path)
    if main.schema != main_schema(manifest_sha256):
        raise ValueError(f"{main_path}: main schema mismatch")
    if sidecar.schema != sidecar_schema(manifest_sha256):
        raise ValueError(f"{sidecar_path}: sidecar schema mismatch")
    if main.num_rows != len(plans) or sidecar.num_rows != len(plans):
        raise ValueError("main/sidecar row count mismatch")
    expected_keys = [plan.image_key for plan in plans]
    if main["image_key"].to_pylist() != expected_keys:
        raise ValueError("main image_key order mismatch")
    if sidecar["image_key"].to_pylist() != expected_keys:
        raise ValueError("sidecar image_key order mismatch")
    retained = [name for name in main.column_names if name != "image"]
    for name in retained:
        if main[name].to_pylist() != sidecar[name].to_pylist():
            raise ValueError(f"main/sidecar retained column mismatch: {name}")
    for position, (image_bytes, metadata_text, description) in enumerate(
        zip(
            main["image"].to_pylist(),
            main["meta_info"].to_pylist(),
            sidecar["structured_description"].to_pylist(),
            strict=True,
        )
    ):
        metadata = json.loads(metadata_text)
        if sha256_bytes(image_bytes) != metadata["target_sha256"]:
            raise ValueError(f"row {position}: embedded Target SHA-256 mismatch")
        if len(image_bytes) != metadata["target_content_length"]:
            raise ValueError(f"row {position}: embedded Target length mismatch")
        if sha256_bytes(description.encode("utf-8")) != metadata[
            "structured_description_sha256"
        ]:
            raise ValueError(f"row {position}: structured_description SHA-256 mismatch")


def _unit_expected_marker(
    unit: UnitSpec,
    plans: Sequence[RowPlan],
    *,
    manifest_identity: ManifestIdentity,
    unit_rows: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "format_version": UNIT_MARKER_FORMAT,
        "source_manifest_sha256": manifest_identity.sha256,
        "source_manifest_version_id": manifest_identity.version_id,
        "source_part_index": unit.shard.part_index,
        "source_part_count": unit.shard.part_count,
        "source_shard_uri": unit.shard.uri,
        "source_shard_sha256": unit.shard.sha256,
        "source_shard_version_id": unit.shard.version_id,
        "unit_rows_contract": unit_rows,
        "unit_index": unit.unit_index,
        "source_local_row_start": unit.local_row_start,
        "source_local_row_end_exclusive": unit.local_row_end,
        "global_row_start": unit.global_row_start,
        "global_row_end_exclusive": unit.global_row_end,
        "rows": unit.rows,
        "first_image_key": plans[0].image_key,
        "last_image_key": plans[-1].image_key,
        "row_identity_sha256": row_identity_sha256(plans),
    }


def validate_completed_marker(
    store: OutputStore,
    marker: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    for field, value in expected.items():
        if marker.get(field) != value:
            raise ValueError(
                f"committed unit marker {field}={marker.get(field)!r}, expected={value!r}"
            )
    relative_path = marker.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path.endswith(".parquet"):
        raise ValueError("committed unit marker relative_path is invalid")
    for kind in ("main", "sidecar"):
        raw = marker.get(f"{kind}_object")
        if not isinstance(raw, Mapping):
            raise ValueError(f"committed unit marker lacks {kind}_object")
        if raw.get("relative_path") != f"{kind}/{relative_path}":
            raise ValueError(f"committed unit marker {kind} relative path mismatch")
        store.validate_file(
            str(raw["relative_path"]),
            size=int(raw["bytes"]),
            sha256=str(raw["sha256"]),
        )
    counts = marker.get("status_counts")
    if not isinstance(counts, Mapping) or sum(int(value) for value in counts.values()) != int(
        expected["rows"]
    ):
        raise ValueError("committed unit marker status_counts mismatch")
    return dict(marker)


ImageLoader = Callable[[RowPlan], ImagePayload]
ImageBatchLoader = Callable[[Sequence[RowPlan]], list[ImagePayload]]


def process_unit(
    *,
    store: OutputStore,
    unit: UnitSpec,
    plans: Sequence[RowPlan],
    manifest_identity: ManifestIdentity,
    unit_rows: int,
    staging_dir: Path,
    download_workers: int,
    image_loader: ImageLoader | None = None,
    image_batch_loader: ImageBatchLoader | None = None,
) -> tuple[dict[str, Any], bool]:
    expected = _unit_expected_marker(
        unit, plans, manifest_identity=manifest_identity, unit_rows=unit_rows
    )
    existing = store.read_json(unit.marker_relative, missing_ok=True)
    if existing is not None:
        return validate_completed_marker(store, existing, expected), False

    staging_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"part-{unit.shard.part_index:05d}-unit-{unit.unit_index:06d}"
    main_path = staging_dir / f"{prefix}.main.parquet"
    sidecar_path = staging_dir / f"{prefix}.sidecar.parquet"
    main_path.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)
    try:
        if (image_loader is None) == (image_batch_loader is None):
            raise ValueError("exactly one image loader must be provided")
        if image_batch_loader is not None:
            images = image_batch_loader(plans)
        else:
            assert image_loader is not None
            with ThreadPoolExecutor(max_workers=min(download_workers, len(plans))) as executor:
                images = list(executor.map(image_loader, plans))
        main, sidecar, target_audit = _build_tables(
            plans,
            images,
            unit=unit,
            manifest_identity=manifest_identity,
        )
        _write_table(main, main_path)
        _write_table(sidecar, sidecar_path)
        verify_local_pair(
            main_path,
            sidecar_path,
            plans,
            manifest_sha256=manifest_identity.sha256,
        )
        identity = str(expected["row_identity_sha256"])
        relative_path = (
            f"part-{unit.shard.part_index:05d}/"
            f"unit-{unit.unit_index:06d}-rows-{unit.local_row_start:08d}-"
            f"{unit.local_row_end:08d}-identity-{identity[:16]}.parquet"
        )
        main_object = store.put_file_once(main_path, f"main/{relative_path}")
        sidecar_object = store.put_file_once(sidecar_path, f"sidecar/{relative_path}")
        status_counts = dict(sorted(Counter(plan.status for plan in plans).items()))
        marker = {
            **expected,
            "relative_path": relative_path,
            "main_object": main_object,
            "sidecar_object": sidecar_object,
            "status_counts": status_counts,
            "target_audit_sha256": sha256_bytes(canonical_json_bytes(target_audit)),
        }
        store.put_json_once(unit.marker_relative, marker)
        committed = store.read_json(unit.marker_relative)
        assert committed is not None
        return validate_completed_marker(store, committed, expected), True
    finally:
        main_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)


def download_verified_source_shard(client: Any, shard: SourceShard, destination: Path) -> None:
    location = parse_s3_uri(shard.uri)
    response = client.get_object(
        Bucket=location.bucket,
        Key=location.key,
        VersionId=shard.version_id,
    )
    actual_version = str(response.get("VersionId") or "")
    if actual_version != shard.version_id:
        raise ValueError(f"part {shard.part_index}: source shard VersionId mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        body = response["Body"]
        while chunk := body.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    if size != shard.size or digest.hexdigest() != shard.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"part {shard.part_index}: source shard identity mismatch "
            f"bytes={size}/{shard.size} sha256={digest.hexdigest()}/{shard.sha256}"
        )
    os.replace(temporary, destination)


def iter_unit_plans(path: Path, shard: SourceShard, *, unit_rows: int) -> Iterable[tuple[UnitSpec, list[RowPlan]]]:
    plans: list[RowPlan] = []
    unit_index = 0
    observed_rows = 0
    with path.open("rb") as stream:
        for local_row, raw_line in enumerate(stream):
            if not raw_line.strip():
                raise ValueError(f"{shard.uri}#line={local_row + 1}: blank lines are forbidden")
            try:
                source_row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{shard.uri}#line={local_row + 1}: invalid JSON") from exc
            plans.append(plan_source_row(source_row, shard=shard, local_row=local_row))
            observed_rows += 1
            if len(plans) == unit_rows:
                start = local_row + 1 - len(plans)
                yield (
                    UnitSpec(
                        shard=shard,
                        unit_index=unit_index,
                        local_row_start=start,
                        local_row_end=local_row + 1,
                    ),
                    plans,
                )
                plans = []
                unit_index += 1
        if plans:
            start = observed_rows - len(plans)
            yield (
                UnitSpec(
                    shard=shard,
                    unit_index=unit_index,
                    local_row_start=start,
                    local_row_end=observed_rows,
                ),
                plans,
            )
    if observed_rows != shard.rows:
        raise ValueError(
            f"part {shard.part_index}: observed {observed_rows} rows, expected {shard.rows}"
        )


def parse_part_indices(value: str, *, shard_count: int) -> set[int] | None:
    if not value.strip():
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid part range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    invalid = sorted(index for index in result if not 0 <= index < shard_count)
    if invalid:
        raise ValueError(f"part indices outside [0, {shard_count}): {invalid}")
    return result


def run_worker(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    document, identity, shards = load_source_manifest(
        client,
        uri=args.manifest_uri,
        expected_sha256=args.manifest_sha256,
        version_id=args.manifest_version_id,
    )
    store = OutputStore(args.output_root, client)
    ensure_run_contract(store, document, identity, unit_rows=args.unit_rows)
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index/count are invalid")
    if args.download_workers <= 0:
        raise ValueError("download_workers must be positive")
    selected_parts = parse_part_indices(args.part_indices, shard_count=len(shards))
    assigned = [
        shard
        for shard in shards
        if shard.part_index % args.worker_count == args.worker_index
        and (selected_parts is None or shard.part_index in selected_parts)
    ]
    if not assigned:
        raise ValueError(f"worker {args.worker_index} has no assigned source shards")
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("max_units must be positive")

    prepared = resumed = completed_rows = units_seen = 0
    status_counts: Counter[str] = Counter()
    completed_parts: list[int] = []
    stopping = False
    staging_dir = args.staging_dir / f"worker-{args.worker_index:05d}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    image_batch_loader = lambda plans: download_images(
        client,
        plans,
        download_workers=args.download_workers,
    )

    for shard in assigned:
        source_path = staging_dir / f"part-{shard.part_index:05d}.jsonl"
        source_path.unlink(missing_ok=True)
        try:
            download_verified_source_shard(client, shard, source_path)
            for unit, plans in iter_unit_plans(source_path, shard, unit_rows=args.unit_rows):
                if args.max_units is not None and units_seen >= args.max_units:
                    stopping = True
                    break
                marker, was_prepared = process_unit(
                    store=store,
                    unit=unit,
                    plans=plans,
                    manifest_identity=identity,
                    unit_rows=args.unit_rows,
                    staging_dir=staging_dir,
                    download_workers=args.download_workers,
                    image_batch_loader=image_batch_loader,
                )
                units_seen += 1
                prepared += int(was_prepared)
                resumed += int(not was_prepared)
                completed_rows += int(marker["rows"])
                status_counts.update(
                    {key: int(value) for key, value in marker["status_counts"].items()}
                )
                print(
                    json.dumps(
                        {
                            "event": "unit_complete",
                            "worker_index": args.worker_index,
                            "part_index": shard.part_index,
                            "unit_index": unit.unit_index,
                            "rows": unit.rows,
                            "prepared": was_prepared,
                            "units_seen": units_seen,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not stopping:
                completed_parts.append(shard.part_index)
        finally:
            source_path.unlink(missing_ok=True)
        if stopping:
            break

    summary = {
        "status": "partial" if stopping else "complete",
        "format_version": "image_edit_v6_target_main_sidecar_worker_attempt_v1",
        "source_manifest_sha256": identity.sha256,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "assignment": "source_part_index_mod_worker_count",
        "assigned_parts": [shard.part_index for shard in assigned],
        "completed_parts": completed_parts,
        "units_seen": units_seen,
        "completed_rows": completed_rows,
        "prepared_units_this_attempt": prepared,
        "resumed_units_this_attempt": resumed,
        "status_counts": dict(sorted(status_counts.items())),
        "max_units": args.max_units,
    }
    if not stopping:
        worker_manifest = (
            f"worker-manifests/workers-{args.worker_count:05d}/"
            f"worker-{args.worker_index:05d}.json"
        )
        store.put_json_once(worker_manifest, summary)
    return summary


def expected_unit_specs(shards: Sequence[SourceShard], *, unit_rows: int) -> list[UnitSpec]:
    units: list[UnitSpec] = []
    for shard in shards:
        for unit_index, start in enumerate(range(0, shard.rows, unit_rows)):
            units.append(
                UnitSpec(
                    shard=shard,
                    unit_index=unit_index,
                    local_row_start=start,
                    local_row_end=min(start + unit_rows, shard.rows),
                )
            )
    return units


def _validate_final_marker(
    store: OutputStore,
    unit: UnitSpec,
    marker: Mapping[str, Any] | None,
    *,
    identity: ManifestIdentity,
    unit_rows: int,
) -> dict[str, Any]:
    if marker is None:
        raise FileNotFoundError(store.absolute(unit.marker_relative))
    expected_without_row_hash = {
        "status": "complete",
        "format_version": UNIT_MARKER_FORMAT,
        "source_manifest_sha256": identity.sha256,
        "source_manifest_version_id": identity.version_id,
        "source_part_index": unit.shard.part_index,
        "source_part_count": unit.shard.part_count,
        "source_shard_uri": unit.shard.uri,
        "source_shard_sha256": unit.shard.sha256,
        "source_shard_version_id": unit.shard.version_id,
        "unit_rows_contract": unit_rows,
        "unit_index": unit.unit_index,
        "source_local_row_start": unit.local_row_start,
        "source_local_row_end_exclusive": unit.local_row_end,
        "global_row_start": unit.global_row_start,
        "global_row_end_exclusive": unit.global_row_end,
        "rows": unit.rows,
        "first_image_key": str(unit.global_row_start),
        "last_image_key": str(unit.global_row_end - 1),
    }
    for field, expected in expected_without_row_hash.items():
        if marker.get(field) != expected:
            raise ValueError(
                f"{unit.marker_relative}: {field}={marker.get(field)!r}, expected={expected!r}"
            )
    _require_sha256(marker.get("row_identity_sha256"), "unit row_identity_sha256")
    _require_sha256(marker.get("target_audit_sha256"), "unit target_audit_sha256")
    return validate_completed_marker(store, marker, expected_without_row_hash)


def finalize_output(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    document, identity, shards = load_source_manifest(
        client,
        uri=args.manifest_uri,
        expected_sha256=args.manifest_sha256,
        version_id=args.manifest_version_id,
    )
    store = OutputStore(args.output_root, client)
    run_contract = ensure_run_contract(store, document, identity, unit_rows=args.unit_rows)
    units = expected_unit_specs(shards, unit_rows=args.unit_rows)

    def fetch_and_validate(unit: UnitSpec) -> tuple[UnitSpec, dict[str, Any]]:
        marker = store.read_json(unit.marker_relative, missing_ok=True)
        return unit, _validate_final_marker(
            store,
            unit,
            marker,
            identity=identity,
            unit_rows=args.unit_rows,
        )

    with ThreadPoolExecutor(max_workers=args.verify_workers) as executor:
        validated = list(executor.map(fetch_and_validate, units))

    total_rows = 0
    status_counts: Counter[str] = Counter()
    main_bytes = sidecar_bytes = 0
    part_summaries: list[dict[str, Any]] = []
    marker_digests: list[str] = []
    by_part: dict[int, list[dict[str, Any]]] = {}
    for unit, marker in validated:
        total_rows += int(marker["rows"])
        status_counts.update(
            {key: int(value) for key, value in marker["status_counts"].items()}
        )
        main_bytes += int(marker["main_object"]["bytes"])
        sidecar_bytes += int(marker["sidecar_object"]["bytes"])
        marker_digests.append(sha256_bytes(canonical_json_bytes(marker)))
        by_part.setdefault(unit.shard.part_index, []).append(marker)
    if total_rows != int(document["total_rows"]):
        raise ValueError(f"final row count {total_rows} != {document['total_rows']}")
    if dict(sorted(status_counts.items())) != dict(document["status_counts"]):
        raise ValueError("final status_counts differ from source manifest")

    for shard in shards:
        markers = by_part.get(shard.part_index, [])
        expected_units = math.ceil(shard.rows / args.unit_rows)
        if len(markers) != expected_units or sum(int(item["rows"]) for item in markers) != shard.rows:
            raise ValueError(f"part {shard.part_index}: final unit/row count mismatch")
        part_summaries.append(
            {
                "part_index": shard.part_index,
                "source_uri": shard.uri,
                "source_sha256": shard.sha256,
                "source_version_id": shard.version_id,
                "rows": shard.rows,
                "units": len(markers),
                "global_row_start": shard.global_row_start,
                "global_row_end_exclusive": shard.global_row_start + shard.rows,
                "unit_markers_sha256": sha256_bytes(canonical_json_bytes(markers)),
            }
        )

    final_manifest = {
        "complete": True,
        "format_version": FINAL_MANIFEST_FORMAT,
        "source_manifest": run_contract["source_manifest"],
        "run_contract_sha256": sha256_bytes(canonical_json_bytes(run_contract, newline=True)),
        "total_rows": total_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "physical_source_shards": len(shards),
        "output_units": len(units),
        "unit_rows": args.unit_rows,
        "main_prefix": join_output(args.output_root, "main") + "/",
        "sidecar_prefix": join_output(args.output_root, "sidecar") + "/",
        "key_column": "image_key",
        "image_column": "image",
        "structured_description_column": "structured_description",
        "relative_shard_alignment": "exact",
        "main_schema": run_contract["main_schema"],
        "sidecar_schema": run_contract["sidecar_schema"],
        "main_parquet_bytes": main_bytes,
        "sidecar_parquet_bytes": sidecar_bytes,
        "ordered_unit_markers_sha256": sha256_bytes(canonical_json_bytes(marker_digests)),
        "parts": part_summaries,
    }
    manifest_digest = store.put_json_once("manifest.json", final_manifest)
    success = {
        "status": "complete",
        "format_version": SUCCESS_FORMAT,
        "manifest_uri": store.absolute("manifest.json"),
        "manifest_sha256": manifest_digest,
        "source_manifest_sha256": identity.sha256,
        "total_rows": total_rows,
        "output_units": len(units),
    }
    store.put_json_once("_SUCCESS", success)
    return {**success, "main_parquet_bytes": main_bytes, "sidecar_parquet_bytes": sidecar_bytes}


def run_preflight(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    document, identity, shards = load_source_manifest(
        client,
        uri=args.manifest_uri,
        expected_sha256=args.manifest_sha256,
        version_id=args.manifest_version_id,
    )
    store = OutputStore(args.output_root, client)
    contract = ensure_run_contract(store, document, identity, unit_rows=args.unit_rows)
    return {
        "status": "ready",
        "format_version": RUN_CONTRACT_FORMAT,
        "source_manifest_sha256": identity.sha256,
        "total_rows": document["total_rows"],
        "physical_shards": len(shards),
        "output_units": len(expected_unit_specs(shards, unit_rows=args.unit_rows)),
        "run_contract_sha256": sha256_bytes(canonical_json_bytes(contract, newline=True)),
        "output_root": args.output_root,
    }


def _worker_index_default() -> int:
    raw = os.environ.get("JOB_COMPLETION_INDEX", "0")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"JOB_COMPLETION_INDEX must be an integer, got {raw!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI)
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--manifest-version-id", default=DEFAULT_MANIFEST_VERSION_ID)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--unit-rows", type=int, default=DEFAULT_UNIT_ROWS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="Validate immutable inputs and commit run-contract.json")

    worker = subparsers.add_parser("worker", help="Materialize assigned physical source shards")
    worker.add_argument("--worker-index", type=int, default=_worker_index_default())
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--download-workers", type=int, default=32)
    worker.add_argument("--staging-dir", type=Path, default=Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/v6-target-sidecar"))
    worker.add_argument(
        "--part-indices",
        default="",
        help="Optional comma/range subset such as 0,3-5; assignment modulo still applies",
    )
    worker.add_argument("--max-units", type=int, default=None, help="Canary attempt limit")

    finalize = subparsers.add_parser("finalize", help="Audit every marker/object and commit success")
    finalize.add_argument("--verify-workers", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_pool = 256
    client = make_s3_client(args.aws_profile or None, args.region, max_pool_connections=max_pool)
    if args.command == "preflight":
        result = run_preflight(args, client)
    elif args.command == "worker":
        result = run_worker(args, client)
    elif args.command == "finalize":
        if args.verify_workers <= 0:
            raise ValueError("verify_workers must be positive")
        result = finalize_output(args, client)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
