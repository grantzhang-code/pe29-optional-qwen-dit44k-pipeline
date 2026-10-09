#!/usr/bin/env python3
"""Run one global GPU rank over frozen source-Parquet objects with S3 resume.

The input inventory is produced by prepare_image_edit_full_json_shard.py.
Objects are assigned by object_index modulo global_world_size. One vLLM engine
stays alive while the rank processes all of its assigned objects. Each source
object becomes one immutable JSONL output followed by a strict success marker;
a later attempt skips an object only after validating both that marker and the
output object's S3 HEAD metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_inference_contract import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SUPPORTED_TARGET_SCHEMAS,
    TARGET_SCHEMA,
    V4_TARGET_SCHEMA,
    require_target_schema,
)
from scripts import infer_image_edit_full_json as inference  # noqa: E402
from scripts import prepare_image_edit_full_json_shard as source_prepare  # noqa: E402


SUCCESS_MARKER_FORMAT = "image_edit_full_json_parquet_shard_success_v1"
RANK_SUMMARY_FORMAT = "image_edit_full_json_parquet_rank_summary_v1"
MAX_SOURCE_OBJECT_ROWS = 500
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_CONTENT_TYPE = "application/x-ndjson"
MARKER_CONTENT_TYPE = "application/json"


class ResumeMarkerError(ValueError):
    """An existing success marker cannot be trusted for this invocation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def parse_s3_uri(uri: str, *, require_key: bool = True) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or (require_key and not key):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, key


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    joined = "/".join(value for value in (key, suffix) if value)
    return f"s3://{bucket}/{joined}"


def clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def validate_inventory(
    document: Mapping[str, Any],
    *,
    source_dataset_s3: str,
) -> list[dict[str, Any]]:
    """Validate and normalize the frozen source inventory without accessing S3."""

    if document.get("format_version") != "image_edit_source_parquet_inventory_v1":
        raise ValueError("inventory format_version is not image_edit_source_parquet_inventory_v1")
    if document.get("ordering") != "lexicographic_s3_key_then_parquet_row":
        raise ValueError("inventory ordering is not authoritative")
    if str(document.get("s3_uri", "")).rstrip("/") != source_dataset_s3.rstrip("/"):
        raise ValueError("inventory points to a different source dataset")

    source_bucket, source_prefix = parse_s3_uri(source_dataset_s3)
    raw_objects = document.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("inventory objects must be a non-empty list")
    object_count = document.get("object_count")
    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise ValueError("inventory object_count must be an integer")
    if object_count != len(raw_objects):
        raise ValueError("inventory object_count does not match objects")

    cursor = 0
    seen_keys: set[tuple[str, str]] = set()
    previous_key: str | None = None
    objects: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"inventory objects[{expected_index}] is not an object")
        item = dict(raw)
        if item.get("object_index") != expected_index:
            raise ValueError(
                f"inventory object_index is not contiguous at {expected_index}: "
                f"{item.get('object_index')!r}"
            )
        bucket = str(item.get("bucket") or "")
        key = str(item.get("key") or "")
        if bucket != source_bucket or not key:
            raise ValueError(f"inventory object {expected_index} has an invalid bucket/key")
        if not key.endswith(".parquet"):
            raise ValueError(f"inventory object {expected_index} is not a Parquet object")
        if key != source_prefix and not key.startswith(source_prefix.rstrip("/") + "/"):
            raise ValueError(f"inventory object {expected_index} is outside the source prefix")
        if previous_key is not None and key <= previous_key:
            raise ValueError("inventory objects are not in strict lexicographic key order")
        if (bucket, key) in seen_keys:
            raise ValueError(f"inventory contains duplicate source object s3://{bucket}/{key}")
        seen_keys.add((bucket, key))
        previous_key = key

        size = _positive_int(item.get("bytes"), field=f"objects[{expected_index}].bytes")
        rows = _positive_int(item.get("rows"), field=f"objects[{expected_index}].rows")
        if rows > MAX_SOURCE_OBJECT_ROWS:
            raise ValueError(
                f"inventory object {expected_index} has {rows} rows; maximum is "
                f"{MAX_SOURCE_OBJECT_ROWS}"
            )
        etag = clean_etag(item.get("etag"))
        if not etag:
            raise ValueError(f"inventory object {expected_index} has no ETag")
        start = item.get("global_row_start")
        end = item.get("global_row_end_exclusive")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ValueError(f"inventory object {expected_index} global range is not integral")
        if start != cursor or end != cursor + rows:
            raise ValueError(
                f"inventory object {expected_index} has non-contiguous global range "
                f"[{start},{end}); expected [{cursor},{cursor + rows})"
            )
        cursor += rows
        objects.append(
            {
                **item,
                "bucket": bucket,
                "key": key,
                "bytes": size,
                "rows": rows,
                "etag": etag,
                "object_index": expected_index,
                "global_row_start": start,
                "global_row_end_exclusive": end,
            }
        )

    total_rows = document.get("total_rows")
    if isinstance(total_rows, bool) or not isinstance(total_rows, int):
        raise ValueError("inventory total_rows must be an integer")
    if total_rows != cursor:
        raise ValueError(
            f"inventory total_rows mismatch: actual={cursor} declared={total_rows}"
        )
    return objects


def assigned_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    global_rank: int,
    global_world_size: int,
) -> list[dict[str, Any]]:
    if global_world_size <= 0 or global_rank < 0 or global_rank >= global_world_size:
        raise ValueError(
            f"invalid global rank/world size: rank={global_rank} world={global_world_size}"
        )
    return [
        dict(item)
        for item in objects
        if int(item["object_index"]) % global_world_size == global_rank
    ]


def source_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_index": int(item["object_index"]),
        "bucket": str(item["bucket"]),
        "key": str(item["key"]),
        "etag": clean_etag(item["etag"]),
        "bytes": int(item["bytes"]),
        "rows": int(item["rows"]),
        "global_row_start": int(item["global_row_start"]),
        "global_row_end_exclusive": int(item["global_row_end_exclusive"]),
    }


def generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return every runtime setting that can affect output or failure rows."""

    return {
        **({"target_schema": TARGET_SCHEMA} if TARGET_SCHEMA != V4_TARGET_SCHEMA else {}),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "thinking": False,
        "include_conversation": False,
        "continue_on_error": True,
        "image_order": ["reference frame_0", "target"],
        "batch_size": int(args.batch_size),
        "download_workers": int(args.download_workers),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(args.max_model_len),
        "max_new_tokens": int(args.max_tokens),
        "max_pixels": int(args.max_pixels),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "attention_backend": args.attention_backend or None,
        "enforce_eager": bool(args.enforce_eager),
        "use_mtp": bool(args.use_mtp),
        "num_speculative_tokens": int(args.num_speculative_tokens),
        "seed": int(args.seed),
    }


def shard_binding(
    item: Mapping[str, Any],
    *,
    inventory_sha256: str,
    source_dataset_s3: str,
    checkpoint_step: int,
    checkpoint_manifest_sha256: str,
    hf_artifact_manifest_sha256: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": source_identity(item),
        "source_dataset_s3": source_dataset_s3.rstrip("/"),
        "inventory": {"sha256": inventory_sha256},
        "checkpoint": {
            "step": checkpoint_step,
            "manifest_sha256": checkpoint_manifest_sha256,
        },
        "hf_artifact": {
            "manifest_sha256": hf_artifact_manifest_sha256,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
        },
        "generation": dict(generation),
    }


def output_uris(output_s3: str, object_index: int) -> tuple[str, str]:
    stem = f"part-{object_index:05d}"
    return (
        join_s3(output_s3, "shards", f"{stem}.jsonl"),
        join_s3(output_s3, "shards", f"{stem}.success.json"),
    )


def summarize_output_file(
    path: Path,
    *,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete local shard and calculate its immutable metadata."""

    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{path}: output must be non-empty newline-terminated JSONL")
    rows = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: result row is not an object")
        rows.append(row)

    source = source_identity(item)
    if len(rows) != source["rows"]:
        raise ValueError(f"{path}: result rows={len(rows)} expected={source['rows']}")
    expected_indices = list(
        range(source["global_row_start"], source["global_row_end_exclusive"])
    )
    actual_indices = [row.get("input_index") for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(f"{path}: input_index coverage/order does not match source global range")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError(f"{path}: every result must contain a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{path}: duplicate sample_id within output shard")

    parquet_uri = f"s3://{source['bucket']}/{source['key']}"
    for local_row, row in enumerate(rows):
        if "messages" in row or "images" in row:
            raise ValueError(f"{path}: output unexpectedly contains conversation payloads")
        dataset_source = row.get("dataset_source")
        expected_dataset_source = {
            "parquet_s3_uri": parquet_uri,
            "parquet_object_index": source["object_index"],
            "parquet_local_row": local_row,
            "dataset_global_row": source["global_row_start"] + local_row,
        }
        if not isinstance(dataset_source, Mapping):
            raise ValueError(f"{path}: result {local_row} has no dataset_source")
        for key, value in expected_dataset_source.items():
            if dataset_source.get(key) != value:
                raise ValueError(
                    f"{path}: result {local_row} dataset_source.{key}="
                    f"{dataset_source.get(key)!r}, expected {value!r}"
                )

    counts = {
        "rows": len(rows),
        "json_valid": sum(bool(row.get("json_valid")) for row in rows),
        "schema_valid": sum(bool(row.get("schema_valid")) for row in rows),
        "failed": sum(not bool(row.get("schema_valid")) for row in rows),
        "inference_errors": sum(bool(row.get("inference_error")) for row in rows),
    }
    return {
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "rows": len(rows),
        "counts": counts,
    }


def build_success_marker(
    *,
    binding: Mapping[str, Any],
    output_uri: str,
    output_summary: Mapping[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    return {
        "format_version": SUCCESS_MARKER_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "binding": dict(binding),
        "output": {
            "uri": output_uri,
            "sha256": output_summary["sha256"],
            "bytes": output_summary["bytes"],
            "rows": output_summary["rows"],
            "counts": dict(output_summary["counts"]),
        },
    }


def output_head_metadata(marker: Mapping[str, Any]) -> dict[str, str]:
    binding = marker["binding"]
    output = marker["output"]
    counts = output["counts"]
    return {
        "sha256": str(output["sha256"]),
        "rows": str(output["rows"]),
        "json-valid": str(counts["json_valid"]),
        "schema-valid": str(counts["schema_valid"]),
        "failed": str(counts["failed"]),
        "inference-errors": str(counts["inference_errors"]),
        "source-etag": str(binding["source"]["etag"]),
        "inventory-sha256": str(binding["inventory"]["sha256"]),
        "checkpoint-step": str(binding["checkpoint"]["step"]),
        "checkpoint-sha256": str(binding["checkpoint"]["manifest_sha256"]),
        "hf-artifact-sha256": str(binding["hf_artifact"]["manifest_sha256"]),
        "prompt-sha256": str(binding["prompt"]["contract_sha256"]),
    }


def _validate_counts(counts: Any, *, expected_rows: int) -> dict[str, int]:
    expected_keys = {
        "rows",
        "json_valid",
        "schema_valid",
        "failed",
        "inference_errors",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_keys:
        raise ResumeMarkerError("marker output counts have an unexpected schema")
    normalized: dict[str, int] = {}
    for key in sorted(expected_keys):
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResumeMarkerError(f"marker output counts.{key} is invalid")
        normalized[key] = value
    if normalized["rows"] != expected_rows:
        raise ResumeMarkerError("marker output row count does not match source rows")
    if not (
        normalized["schema_valid"]
        <= normalized["json_valid"]
        <= normalized["rows"]
    ):
        raise ResumeMarkerError("marker JSON/schema validity counts are inconsistent")
    if normalized["failed"] != normalized["rows"] - normalized["schema_valid"]:
        raise ResumeMarkerError("marker failed count is inconsistent")
    if normalized["inference_errors"] > normalized["failed"]:
        raise ResumeMarkerError("marker inference error count is inconsistent")
    return normalized


def validate_source_head(source: Mapping[str, Any], head: Mapping[str, Any]) -> None:
    """Ensure a resumed shard still refers to the frozen source object bytes."""

    if head.get("ContentLength") != source.get("bytes"):
        raise ResumeMarkerError("source HEAD ContentLength differs from frozen inventory")
    if clean_etag(head.get("ETag")) != source.get("etag"):
        raise ResumeMarkerError("source HEAD ETag differs from frozen inventory")


def validate_resume_marker(
    marker: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
    expected_output_uri: str,
    output_head: Mapping[str, Any],
) -> None:
    """Strictly validate a marker and the corresponding output HEAD response."""

    if marker.get("format_version") != SUCCESS_MARKER_FORMAT or marker.get("status") != "complete":
        raise ResumeMarkerError("existing marker is not a complete supported success marker")
    if marker.get("binding") != expected_binding:
        raise ResumeMarkerError("existing marker binding differs from this invocation")
    if not isinstance(marker.get("completed_at"), str) or not marker["completed_at"]:
        raise ResumeMarkerError("existing marker has no completion timestamp")
    output = marker.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "uri",
        "sha256",
        "bytes",
        "rows",
        "counts",
    }:
        raise ResumeMarkerError("existing marker output has an unexpected schema")
    if output.get("uri") != expected_output_uri:
        raise ResumeMarkerError("existing marker points to a different output URI")
    if SHA256_RE.fullmatch(str(output.get("sha256", ""))) is None:
        raise ResumeMarkerError("existing marker output SHA-256 is invalid")
    output_bytes = output.get("bytes")
    output_rows = output.get("rows")
    if isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes <= 0:
        raise ResumeMarkerError("existing marker output byte count is invalid")
    if isinstance(output_rows, bool) or not isinstance(output_rows, int):
        raise ResumeMarkerError("existing marker output row count is invalid")
    _validate_counts(
        output.get("counts"),
        expected_rows=int(expected_binding["source"]["rows"]),
    )
    if output_rows != expected_binding["source"]["rows"]:
        raise ResumeMarkerError("existing marker output rows differ from source rows")

    if output_head.get("ContentLength") != output_bytes:
        raise ResumeMarkerError("output HEAD ContentLength differs from marker")
    if output_head.get("ContentType") != OUTPUT_CONTENT_TYPE:
        raise ResumeMarkerError("output HEAD ContentType differs from the committed contract")
    raw_metadata = output_head.get("Metadata")
    if not isinstance(raw_metadata, Mapping):
        raise ResumeMarkerError("output HEAD has no user metadata")
    actual_metadata = {str(key).lower(): str(value) for key, value in raw_metadata.items()}
    for key, expected in output_head_metadata(marker).items():
        if actual_metadata.get(key) != expected:
            raise ResumeMarkerError(
                f"output HEAD metadata {key!r}={actual_metadata.get(key)!r}, "
                f"expected {expected!r}"
            )


def compact_shard_record(
    marker: Mapping[str, Any],
    *,
    marker_uri: str,
    disposition: str,
) -> dict[str, Any]:
    return {
        "object_index": marker["binding"]["source"]["object_index"],
        "source": dict(marker["binding"]["source"]),
        "output": dict(marker["output"]),
        "success_marker_uri": marker_uri,
        "disposition": disposition,
    }


def aggregate_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "failed": 0,
        "inference_errors": 0,
        "bytes": 0,
    }
    for record in records:
        output = record["output"]
        counts = output["counts"]
        for key in ("rows", "json_valid", "schema_valid", "failed", "inference_errors"):
            totals[key] += int(counts[key])
        totals["bytes"] += int(output["bytes"])
    return totals


def _not_found(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    detail = response.get("Error")
    if not isinstance(detail, Mapping):
        return False
    return str(detail.get("Code")) in {"404", "NoSuchKey", "NotFound"}


def s3_client(args: argparse.Namespace) -> Any:
    import boto3
    from botocore.config import Config

    session = (
        boto3.Session(profile_name=args.aws_profile, region_name=args.aws_region)
        if args.aws_profile
        else boto3.Session(region_name=args.aws_region)
    )
    return session.client(
        "s3",
        region_name=args.aws_region,
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=max(32, args.download_workers * 2),
        ),
    )


def read_inventory_bytes(uri: str, *, client: Any) -> bytes:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).expanduser().resolve(strict=True).read_bytes()


def load_existing_marker(
    client: Any,
    *,
    marker_uri: str,
    expected_binding: Mapping[str, Any],
    expected_output_uri: str,
) -> dict[str, Any] | None:
    marker_bucket, marker_key = parse_s3_uri(marker_uri)
    try:
        response = client.get_object(Bucket=marker_bucket, Key=marker_key)
    except BaseException as exc:
        if _not_found(exc):
            return None
        raise
    payload = response["Body"].read()
    metadata = response.get("Metadata")
    marker_sha256 = sha256_bytes(payload)
    if not isinstance(metadata, Mapping) or metadata.get("sha256") != marker_sha256:
        raise ResumeMarkerError(f"{marker_uri}: marker body SHA-256 metadata is missing or wrong")
    if response.get("ContentType") != MARKER_CONTENT_TYPE:
        raise ResumeMarkerError(f"{marker_uri}: marker ContentType is invalid")
    try:
        marker = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeMarkerError(f"{marker_uri}: marker is not valid JSON") from exc
    if not isinstance(marker, Mapping):
        raise ResumeMarkerError(f"{marker_uri}: marker is not an object")
    source = expected_binding["source"]
    try:
        source_head = client.head_object(Bucket=source["bucket"], Key=source["key"])
    except BaseException as exc:
        if _not_found(exc):
            raise ResumeMarkerError(
                f"{marker_uri}: marker exists but source object is missing"
            ) from exc
        raise
    validate_source_head(source, source_head)
    output_bucket, output_key = parse_s3_uri(expected_output_uri)
    try:
        output_head = client.head_object(Bucket=output_bucket, Key=output_key)
    except BaseException as exc:
        if _not_found(exc):
            raise ResumeMarkerError(
                f"{marker_uri}: marker exists but output object is missing"
            ) from exc
        raise
    validate_resume_marker(
        marker,
        expected_binding=expected_binding,
        expected_output_uri=expected_output_uri,
        output_head=output_head,
    )
    return compact_shard_record(
        marker,
        marker_uri=marker_uri,
        disposition="resumed",
    )


def upload_output_and_marker(
    client: Any,
    *,
    output_path: Path,
    output_uri: str,
    marker_path: Path,
    marker_uri: str,
    marker: Mapping[str, Any],
) -> None:
    output_bucket, output_key = parse_s3_uri(output_uri)
    client.upload_file(
        str(output_path),
        output_bucket,
        output_key,
        ExtraArgs={
            "ContentType": OUTPUT_CONTENT_TYPE,
            "Metadata": output_head_metadata(marker),
        },
    )
    output_head = client.head_object(Bucket=output_bucket, Key=output_key)
    validate_resume_marker(
        marker,
        expected_binding=marker["binding"],
        expected_output_uri=output_uri,
        output_head=output_head,
    )

    marker_payload = json_bytes(marker)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_bytes(marker_payload)
    marker_bucket, marker_key = parse_s3_uri(marker_uri)
    client.put_object(
        Bucket=marker_bucket,
        Key=marker_key,
        Body=marker_payload,
        ContentType=MARKER_CONTENT_TYPE,
        Metadata={"sha256": sha256_bytes(marker_payload)},
    )


def records_from_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    item: Mapping[str, Any],
) -> list[inference.InferenceRecord]:
    source = source_identity(item)
    if len(rows) != source["rows"]:
        raise ValueError(
            f"source object {source['object_index']} returned {len(rows)} rows; "
            f"expected {source['rows']}"
        )
    records: list[inference.InferenceRecord] = []
    for local_row, row in enumerate(rows):
        input_index = source["global_row_start"] + local_row
        record = inference.normalize_input_row(row, input_index)
        dataset_source = record.dataset_source
        if not isinstance(dataset_source, Mapping):
            raise ValueError(
                f"source object {source['object_index']} row {local_row} lost provenance"
            )
        if (
            dataset_source.get("parquet_object_index") != source["object_index"]
            or dataset_source.get("parquet_local_row") != local_row
            or dataset_source.get("dataset_global_row") != input_index
        ):
            raise ValueError(
                f"source object {source['object_index']} row {local_row} has wrong provenance"
            )
        records.append(record)
    return records


def process_object(
    args: argparse.Namespace,
    runtime: inference.InferenceRuntime,
    client: Any,
    *,
    item: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    source = source_identity(item)
    object_index = source["object_index"]
    rank_dir = args.work_dir / f"rank-{args.rank:03d}"
    output_path = rank_dir / "shards" / f"part-{object_index:05d}.jsonl"
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    marker_path = rank_dir / "markers" / f"part-{object_index:05d}.success.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = source_prepare._read_source_object(
        item,
        s3_client=client,
        source_dataset_uri=args.source_dataset_s3,
        global_start=source["global_row_start"],
        global_end=source["global_row_end_exclusive"],
    )
    records = records_from_source_rows(rows, item=item)
    with partial_path.open("w", encoding="utf-8") as output_handle:
        execution = inference.infer_records(args, runtime, records, output_handle)
    os.replace(partial_path, output_path)
    output_summary = summarize_output_file(output_path, item=item)
    for key, value in execution.counts.items():
        if output_summary["counts"][key] != value:
            raise RuntimeError(
                f"object {object_index} output count {key}="
                f"{output_summary['counts'][key]} differs from inference count {value}"
            )

    output_uri, marker_uri = output_uris(args.output_s3, object_index)
    marker = build_success_marker(
        binding=binding,
        output_uri=output_uri,
        output_summary=output_summary,
        completed_at=utc_now(),
    )
    upload_output_and_marker(
        client,
        output_path=output_path,
        output_uri=output_uri,
        marker_path=marker_path,
        marker_uri=marker_uri,
        marker=marker,
    )
    return compact_shard_record(
        marker,
        marker_uri=marker_uri,
        disposition="generated",
    )


def rank_summary_uri(output_s3: str, global_rank: int) -> str:
    return join_s3(output_s3, "ranks", f"rank-{global_rank:03d}", "summary.json")


def upload_rank_summary(
    client: Any,
    *,
    path: Path,
    output_s3: str,
    global_rank: int,
) -> str:
    uri = rank_summary_uri(output_s3, global_rank)
    bucket, key = parse_s3_uri(uri)
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": MARKER_CONTENT_TYPE,
            "Metadata": {"sha256": file_sha256(path)},
        },
    )
    return uri


def run(args: argparse.Namespace) -> dict[str, Any]:
    inference.configure_aws_environment(args)
    inference.validate_prompt_contract()
    client = s3_client(args)
    inventory_payload = read_inventory_bytes(args.inventory, client=client)
    actual_inventory_sha256 = sha256_bytes(inventory_payload)
    if actual_inventory_sha256 != args.inventory_sha256:
        raise ValueError(
            "inventory SHA-256 mismatch: "
            f"actual={actual_inventory_sha256} expected={args.inventory_sha256}"
        )
    try:
        inventory = json.loads(inventory_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("inventory is not valid JSON") from exc
    if not isinstance(inventory, Mapping):
        raise ValueError("inventory document is not an object")
    objects = validate_inventory(inventory, source_dataset_s3=args.source_dataset_s3)
    selected = assigned_objects(
        objects,
        global_rank=args.rank,
        global_world_size=args.world_size,
    )
    if not selected:
        raise ValueError(f"global rank {args.rank} has no assigned source objects")

    started_at = utc_now()
    generation = generation_contract(args)
    bindings = {
        int(item["object_index"]): shard_binding(
            item,
            inventory_sha256=actual_inventory_sha256,
            source_dataset_s3=args.source_dataset_s3,
            checkpoint_step=args.checkpoint_step,
            checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            hf_artifact_manifest_sha256=args.hf_artifact_manifest_sha256,
            generation=generation,
        )
        for item in selected
    }

    shard_records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for item in selected:
        object_index = int(item["object_index"])
        output_uri, marker_uri = output_uris(args.output_s3, object_index)
        resumed = load_existing_marker(
            client,
            marker_uri=marker_uri,
            expected_binding=bindings[object_index],
            expected_output_uri=output_uri,
        )
        if resumed is None:
            pending.append(item)
        else:
            shard_records.append(resumed)
            print(
                json.dumps(
                    {
                        "event": "object_resume",
                        "global_rank": args.rank,
                        "object_index": object_index,
                        "output": output_uri,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    runtime: inference.InferenceRuntime | None = None
    if pending:
        runtime = inference.initialize_inference_runtime(args)
    for item in pending:
        assert runtime is not None
        object_index = int(item["object_index"])
        print(
            json.dumps(
                {
                    "event": "object_start",
                    "global_rank": args.rank,
                    "object_index": object_index,
                    "rows": item["rows"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        record = process_object(
            args,
            runtime,
            client,
            item=item,
            binding=bindings[object_index],
        )
        shard_records.append(record)
        print(
            json.dumps(
                {
                    "event": "object_complete",
                    "global_rank": args.rank,
                    "object_index": object_index,
                    "counts": record["output"]["counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    shard_records.sort(key=lambda record: int(record["object_index"]))
    expected_indices = [int(item["object_index"]) for item in selected]
    actual_indices = [int(record["object_index"]) for record in shard_records]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"rank shard coverage mismatch: actual={actual_indices} expected={expected_indices}"
        )
    totals = aggregate_counts(shard_records)
    expected_rows = sum(int(item["rows"]) for item in selected)
    if totals["rows"] != expected_rows:
        raise RuntimeError(f"rank result rows={totals['rows']} expected={expected_rows}")

    rank_dir = args.work_dir / f"rank-{args.rank:03d}"
    summary_path = rank_dir / "summary.json"
    summary = {
        "format_version": RANK_SUMMARY_FORMAT,
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "global_rank": args.rank,
        "global_world_size": args.world_size,
        "source_dataset_s3": args.source_dataset_s3.rstrip("/"),
        "inventory": {
            "uri": args.inventory,
            "sha256": actual_inventory_sha256,
            "total_objects": len(objects),
            "total_rows": inventory["total_rows"],
        },
        "checkpoint": {
            "step": args.checkpoint_step,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        "hf_artifact": {
            "manifest_sha256": args.hf_artifact_manifest_sha256,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
        },
        "generation": generation,
        "runtime_initialized": runtime is not None,
        "assigned_objects": len(selected),
        "generated_objects": len(pending),
        "resumed_objects": len(selected) - len(pending),
        "totals": totals,
        "shards": shard_records,
        "summary_s3_uri": rank_summary_uri(args.output_s3, args.rank),
    }
    atomic_json(summary_path, summary)
    uploaded_summary_uri = upload_rank_summary(
        client,
        path=summary_path,
        output_s3=args.output_s3,
        global_rank=args.rank,
    )
    if uploaded_summary_uri != summary["summary_s3_uri"]:
        raise RuntimeError(
            "uploaded rank summary URI differs from the persisted summary_s3_uri"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-schema",
        choices=SUPPORTED_TARGET_SCHEMAS,
        default=TARGET_SCHEMA,
    )
    parser.add_argument("--model", required=True, help="Converted local Hugging Face model")
    parser.add_argument("--tokenizer", help="Tokenizer directory; defaults to --model")
    parser.add_argument("--inventory", required=True, help="Local path or s3:// inventory JSON")
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--source-dataset-s3", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--global-rank", "--rank", dest="rank", type=int, required=True)
    parser.add_argument(
        "--global-world-size",
        "--world-size",
        dest="world_size",
        type=int,
        required=True,
    )
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--hf-artifact-manifest-sha256", required=True)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=inference.DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=inference.DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-pixels", type=int, default=inference.DEFAULT_MAX_PIXELS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_target_schema(args.target_schema)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (
        "checkpoint_step",
        "world_size",
        "batch_size",
        "download_workers",
        "tensor_parallel_size",
        "max_model_len",
        "max_tokens",
        "max_pixels",
        "num_speculative_tokens",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--global-rank must be in [0, --global-world-size)")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    for name in (
        "inventory_sha256",
        "checkpoint_manifest_sha256",
        "hf_artifact_manifest_sha256",
    ):
        if SHA256_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase 64-hex")
    try:
        parse_s3_uri(args.source_dataset_s3)
        parse_s3_uri(args.output_s3)
        if args.inventory.startswith("s3://"):
            parse_s3_uri(args.inventory)
    except ValueError as exc:
        parser.error(str(exc))
    args.source_dataset_s3 = args.source_dataset_s3.rstrip("/")
    args.output_s3 = args.output_s3.rstrip("/")
    args.include_conversation = False
    args.continue_on_error = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
