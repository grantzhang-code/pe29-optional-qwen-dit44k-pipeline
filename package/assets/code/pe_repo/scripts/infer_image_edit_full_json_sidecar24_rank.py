#!/usr/bin/env python3
"""Run one GPU rank over a frozen sidecar24 single-reference unit catalog.

Planner units first enter one deterministic data part, then part-local unit
ordinal assigns each unit to exactly one global GPU rank.  Each unit
produces one immutable JSONL shard followed by a strictly bound success marker.
URI-backed and embedded-image Parquet are both read through
``image_edit_single_source_adapter``; one vLLM runtime is initialized only when
this rank has at least one unit that cannot be resumed.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

from dlms.datasets.image_edit_inference_contract import (
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SUPPORTED_TARGET_SCHEMAS,
    TARGET_SCHEMA,
    require_target_schema,
)
from dlms.datasets.image_edit_v6_bbox_packaging import (
    PACKAGED_BBOX_COORDINATE_ORDER,
    PACKAGING_VERSION,
    SOURCE_BBOX_COORDINATE_ORDER,
    package_v6_inference_row_yxyx,
)
from scripts import image_edit_sidecar24_catalog as source_catalog_module
from scripts import image_edit_sidecar24_partition as partition_module
from scripts import image_edit_single_source_adapter as source_adapter
from scripts import infer_image_edit_full_json as inference
from scripts import infer_image_edit_full_json_parquet_rank as base_worker
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module


SUCCESS_MARKER_FORMAT = "image_edit_full_json_sidecar24_unit_success_v1"
RANK_SUMMARY_FORMAT = "image_edit_full_json_sidecar24_rank_summary_v1"
REBALANCED_ASSIGNMENT_FORMAT = "image_edit_sidecar24_rebalanced_assignment_v1"
REBALANCED_WORKER_SUMMARY_FORMAT = "image_edit_sidecar24_rebalanced_worker_summary_v1"
OUTPUT_CONTENT_TYPE = "application/x-ndjson"
MARKER_CONTENT_TYPE = "application/json"
DEFAULT_REQUEST_WINDOW_SIZE = 32
CROSS_UNIT_STREAM_VERSION = "image_edit_full_json_sidecar24_cross_unit_stream_v1"
DEFAULT_UNIT_UPLOAD_WORKERS = 2
DEFAULT_MAX_PENDING_UNIT_WRITES = 8
DEFAULT_UNIT_READ_WORKERS = 2
DEFAULT_UNIT_PREFETCH_DEPTH = 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPACT_SHARD_FIELDS = frozenset(
    {
        "global_unit_index",
        "dataset_id",
        "dataset_unit_index",
        "source",
        "local_row_ranges",
        "dataset_row_start",
        "dataset_row_end_exclusive",
        "global_row_start",
        "global_row_end_exclusive",
        "output",
        "success_marker_uri",
        "disposition",
    }
)
RANK_SUMMARY_FIELDS = frozenset(
    {
        "format_version",
        "status",
        "started_at",
        "completed_at",
        "global_rank",
        "global_world_size",
        "catalogs",
        "checkpoint",
        "hf_artifact",
        "prompt",
        "generation",
        "runtime_initialized",
        "assigned_units",
        "generated_units",
        "resumed_units",
        "totals",
        "shards",
        "summary_s3_uri",
    }
)


class ResumeMarkerError(ValueError):
    """An existing unit success marker cannot be trusted."""


# Keep the established S3, JSON, hashing, timestamp, and generation semantics.
utc_now = base_worker.utc_now
json_bytes = base_worker.json_bytes
sha256_bytes = base_worker.sha256_bytes
file_sha256 = base_worker.file_sha256
atomic_json = base_worker.atomic_json
parse_s3_uri = base_worker.parse_s3_uri
join_s3 = base_worker.join_s3
clean_etag = base_worker.clean_etag
_not_found = base_worker._not_found


def generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Bind sidecar24 window scheduling in addition to the shared settings."""

    contract = {
        **base_worker.generation_contract(args),
        "request_window_size": int(getattr(args, "request_window_size", None) or args.batch_size),
        "prefetch_images": bool(getattr(args, "prefetch_images", False)),
        "cross_unit_streaming": bool(getattr(args, "cross_unit_streaming", False)),
        "cross_unit_stream_version": (
            CROSS_UNIT_STREAM_VERSION if getattr(args, "cross_unit_streaming", False) else None
        ),
        "max_pending_unit_writes": int(
            getattr(args, "max_pending_unit_writes", DEFAULT_MAX_PENDING_UNIT_WRITES)
        ),
        "unit_upload_workers": int(getattr(args, "unit_upload_workers", DEFAULT_UNIT_UPLOAD_WORKERS)),
        # The data selection is part of resumability.  Without this binding a
        # 100-case canary could incorrectly resume a full-part unit marker.
        "data_partition": partition_module.basic_contract(
            partition_count=int(getattr(args, "partition_count", 1)),
            partition_index=int(getattr(args, "partition_index", 0)),
            global_world_size=int(getattr(args, "world_size", 1)),
            case_limit=getattr(args, "case_limit", None),
        ),
    }
    if bool(getattr(args, "package_v6_bboxes_yxyx", False)):
        contract["hld_postprocess_version"] = str(
            args.expected_hld_postprocess_version
        )
        contract["bbox_packaging"] = {
            "version": PACKAGING_VERSION,
            "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
            "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
            "stage": "after_schema_validation_and_hld_postprocess_before_unit_commit",
        }
    for field in (
        "gdn_prefill_backend",
        "max_num_batched_tokens",
        "mm_processor_cache_gb",
        "mm_encoder_attn_backend",
        "speculative_attention_backend",
        "cudagraph_mode",
        "max_cudagraph_capture_size",
        "cudagraph_mm_encoder",
        "cudagraph_specialize_lora",
        "enable_tower_connector_lora",
        "enable_prefix_caching",
    ):
        value = getattr(args, field, None)
        if value is not None:
            contract[field] = None if value == "auto" else value
    runtime_stack = inference.expected_runtime_stack(args)
    if runtime_stack:
        contract["runtime_stack"] = {
            "verification": inference.RUNTIME_STACK_VERIFICATION,
            "versions": runtime_stack,
        }
    cudagraph_mode = getattr(args, "cudagraph_mode", None)
    if cudagraph_mode:
        contract["cuda_graph_contract"] = {
            "requested_mode": cudagraph_mode,
            "required_effective_mode": cudagraph_mode,
            "verification": inference.CUDA_GRAPH_WORKER_VERIFICATION,
        }
    return contract


def _strict_json_object(payload: bytes, *, location: str) -> dict[str, Any]:
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{location}: duplicate JSON object key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{location}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected a JSON object")
    return value


def s3_client(args: argparse.Namespace) -> Any:
    return base_worker.s3_client(args)


def read_document_bytes(uri: str, *, client: Any) -> bytes:
    return base_worker.read_inventory_bytes(uri, client=client)


def load_frozen_catalogs(
    args: argparse.Namespace,
    *,
    client: Any,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Load, validate, and canonically fingerprint both catalog layers."""

    source_payload = read_document_bytes(args.source_catalog, client=client)
    source_catalog = _strict_json_object(source_payload, location="source catalog")
    source_catalog_module.validate_catalog(source_catalog)
    source_sha256 = source_catalog_module.canonical_sha256(source_catalog)
    if source_sha256 != args.source_catalog_sha256:
        raise ValueError(
            "source catalog canonical SHA-256 mismatch: "
            f"actual={source_sha256} expected={args.source_catalog_sha256}"
        )

    unit_payload = read_document_bytes(args.unit_catalog, client=client)
    unit_catalog = _strict_json_object(unit_payload, location="unit catalog")
    unit_catalog_module.validate_unit_catalog(
        unit_catalog,
        source_catalog=source_catalog,
    )
    unit_sha256 = unit_catalog_module.canonical_unit_catalog_sha256(
        unit_catalog,
        source_catalog=source_catalog,
    )
    if unit_sha256 != args.unit_catalog_sha256:
        raise ValueError(
            "unit catalog canonical SHA-256 mismatch: "
            f"actual={unit_sha256} expected={args.unit_catalog_sha256}"
        )
    return source_catalog, unit_catalog, source_sha256, unit_sha256


def catalog_bindings(
    *,
    source_catalog_uri: str,
    source_catalog: Mapping[str, Any],
    source_catalog_sha256: str,
    unit_catalog_uri: str,
    unit_catalog: Mapping[str, Any],
    unit_catalog_sha256: str,
) -> dict[str, Any]:
    """Build the exact catalog section shared by rank and node summaries."""

    parse_s3_uri(source_catalog_uri)
    parse_s3_uri(unit_catalog_uri)
    return {
        "source": {
            "uri": source_catalog_uri,
            "sha256": source_catalog_sha256,
            "format_version": source_catalog["schema_version"],
            "total_rows": source_catalog["total_rows"],
            "dataset_count": len(source_catalog["datasets"]),
        },
        "unit": {
            "uri": unit_catalog_uri,
            "sha256": unit_catalog_sha256,
            "format_version": unit_catalog["format_version"],
            "total_units": unit_catalog["total_units"],
            "total_rows": unit_catalog["total_selected_rows"],
        },
    }


def selected_local_rows(unit: Mapping[str, Any]) -> list[int]:
    normalized = source_adapter.validate_unit(unit)
    return [local_row for start, end in normalized.local_row_ranges for local_row in range(start, end)]


def assigned_units(
    units: Sequence[Mapping[str, Any]],
    *,
    global_rank: int,
    global_world_size: int,
) -> list[dict[str, Any]]:
    if global_world_size <= 0 or global_rank < 0 or global_rank >= global_world_size:
        raise ValueError(f"invalid global rank/world size: rank={global_rank} world={global_world_size}")
    return [dict(unit) for unit in units if int(unit["global_unit_index"]) % global_world_size == global_rank]


def partition_plan(
    units: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> partition_module.PartitionPlan:
    """Resolve the invocation's part, global canary limit, and rank lanes."""

    return partition_module.build_partition_plan(
        units,
        partition_count=int(getattr(args, "partition_count", 1)),
        partition_index=int(getattr(args, "partition_index", 0)),
        global_world_size=int(args.world_size),
        case_limit=getattr(args, "case_limit", None),
    )


def source_identity(unit: Mapping[str, Any]) -> dict[str, Any]:
    source = source_adapter.validate_unit(unit).source
    return {
        "bucket": source.bucket,
        "key": source.key,
        "etag": source.etag,
        "bytes": source.bytes,
        "rows": source.rows,
        "object_index": source.object_index,
    }


def unit_identity(unit: Mapping[str, Any]) -> dict[str, Any]:
    normalized = source_adapter.validate_unit(unit)
    return {
        "dataset_id": normalized.dataset_id,
        "dataset_ordinal": normalized.dataset_ordinal,
        "dataset_unit_index": normalized.dataset_unit_index,
        "global_unit_index": normalized.global_unit_index,
        "dataset_row_start": normalized.dataset_row_start,
        "dataset_row_end_exclusive": normalized.dataset_row_end_exclusive,
        "global_row_start": normalized.global_row_start,
        "global_row_end_exclusive": normalized.global_row_end_exclusive,
        "local_row_ranges": [list(row_range) for row_range in normalized.local_row_ranges],
        "source": source_identity(unit),
        "adapter": normalized.adapter,
        "main_s3_uri": normalized.main_s3_uri,
        "selected_rows": normalized.selected_rows,
    }


def unit_binding(
    unit: Mapping[str, Any],
    *,
    source_catalog_sha256: str,
    unit_catalog_sha256: str,
    checkpoint_step: int,
    checkpoint_manifest_sha256: str,
    hf_artifact_manifest_sha256: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "unit": unit_identity(unit),
        "catalogs": {
            "source": {"sha256": source_catalog_sha256},
            "unit": {"sha256": unit_catalog_sha256},
        },
        "checkpoint": {
            "step": checkpoint_step,
            "manifest_sha256": checkpoint_manifest_sha256,
        },
        "hf_artifact": {"manifest_sha256": hf_artifact_manifest_sha256},
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
        },
        "generation": dict(generation),
    }


def output_uris(
    output_s3: str,
    dataset_id: str,
    dataset_unit_index: int,
) -> tuple[str, str]:
    if not isinstance(dataset_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", dataset_id) is None:
        raise ValueError(f"invalid dataset_id for output key: {dataset_id!r}")
    if type(dataset_unit_index) is not int or dataset_unit_index < 0:
        raise ValueError("dataset_unit_index must be a non-negative integer")
    stem = f"unit-{dataset_unit_index:06d}"
    root = join_s3(output_s3, "datasets", dataset_id, "shards")
    return f"{root}/{stem}.jsonl", f"{root}/{stem}.success.json"


def _expected_dataset_source(
    unit: Mapping[str, Any],
    *,
    selected_ordinal: int,
    local_row: int,
) -> dict[str, Any]:
    normalized = source_adapter.validate_unit(unit)
    return {
        "dataset_id": normalized.dataset_id,
        "dataset_ordinal": normalized.dataset_ordinal,
        "dataset_unit_index": normalized.dataset_unit_index,
        "global_unit_index": normalized.global_unit_index,
        "dataset_row_index": normalized.dataset_row_start + selected_ordinal,
        "global_input_index": normalized.global_row_start + selected_ordinal,
        "parquet_s3_uri": normalized.source.s3_uri,
        "parquet_object_index": normalized.source.object_index,
        "source_parquet_local_row": local_row,
    }


def validate_records_for_unit(
    records: Sequence[inference.InferenceRecord],
    *,
    unit: Mapping[str, Any],
) -> None:
    normalized = source_adapter.validate_unit(unit)
    local_rows = selected_local_rows(unit)
    if len(records) != normalized.selected_rows:
        raise ValueError(
            f"unit {normalized.global_unit_index} returned {len(records)} records; "
            f"expected {normalized.selected_rows}"
        )
    for selected_ordinal, (record, local_row) in enumerate(zip(records, local_rows, strict=True)):
        expected_source = _expected_dataset_source(
            unit,
            selected_ordinal=selected_ordinal,
            local_row=local_row,
        )
        if record.input_index != expected_source["global_input_index"]:
            raise ValueError(
                f"unit {normalized.global_unit_index} record {selected_ordinal} input_index drifted"
            )
        if record.dataset_source != expected_source:
            raise ValueError(
                f"unit {normalized.global_unit_index} record {selected_ordinal} provenance drifted"
            )


def summarize_output_file(
    path: Path,
    *,
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate complete unit output coverage, provenance, and stable media identities."""

    normalized = source_adapter.validate_unit(unit)
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{path}: output must be non-empty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: result row is not an object")
        rows.append(row)
    if len(rows) != normalized.selected_rows:
        raise ValueError(f"{path}: result rows={len(rows)} expected={normalized.selected_rows}")

    expected_indices = list(range(normalized.global_row_start, normalized.global_row_end_exclusive))
    if [row.get("input_index") for row in rows] != expected_indices:
        raise ValueError(f"{path}: input_index/global_input_index coverage is not contiguous")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError(f"{path}: every result must contain a non-empty sample_id")

    # ``sample_id`` is a source label, not the sidecar24 row identity.  The
    # frozen planner intentionally performs an exact multiset join and permits
    # repeated IDs when source/sidecar multiplicities agree.  Preserve those
    # labels verbatim; the contiguous ``input_index`` coverage above and the
    # exact per-row frozen provenance below are the unique identity guards.

    local_rows = selected_local_rows(unit)
    for selected_ordinal, (row, local_row) in enumerate(zip(rows, local_rows, strict=True)):
        if "messages" in row or "images" in row:
            raise ValueError(f"{path}: output unexpectedly contains conversation payloads")
        expected_source = _expected_dataset_source(
            unit,
            selected_ordinal=selected_ordinal,
            local_row=local_row,
        )
        if row.get("dataset_source") != expected_source:
            raise ValueError(f"{path}: result {selected_ordinal} dataset_source differs from unit")
        reference_image = row.get("reference_image")
        target_image = row.get("target_image")
        if not isinstance(reference_image, str) or not reference_image:
            raise ValueError(f"{path}: result {selected_ordinal} lacks reference_image identity")
        if not isinstance(target_image, str) or not target_image:
            raise ValueError(f"{path}: result {selected_ordinal} lacks target_image identity")
        if normalized.adapter in {
            source_adapter.EMBEDDED_SINGLE_V1,
            source_adapter.EMBEDDED_SYNTH_V1,
        }:
            if normalized.adapter == source_adapter.EMBEDDED_SYNTH_V1:
                reference_column = "source_image"
                reference_list_index = None
                target_column = "target_image"
            else:
                reference_column = "input_images"
                reference_list_index = 0
                target_column = "output_image"
            expected_reference = source_adapter.parquet_media_identity(
                normalized.source,
                local_row=local_row,
                column=reference_column,
                list_index=reference_list_index,
            )
            expected_target = source_adapter.parquet_media_identity(
                normalized.source,
                local_row=local_row,
                column=target_column,
            )
            if (reference_image, target_image) != (expected_reference, expected_target):
                raise ValueError(
                    f"{path}: result {selected_ordinal} leaked or changed embedded media identity"
                )
        elif not (reference_image.startswith("s3://") and target_image.startswith("s3://")):
            raise ValueError(f"{path}: URI adapter result {selected_ordinal} has non-S3 media")

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


def package_output_file_yxyx(
    path: Path,
    *,
    expected_hld_postprocess_version: str,
) -> dict[str, Any]:
    """Atomically convert the final unit projection after V6 HLD cleanup."""

    packaged_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: result row is not an object")
            packaged, audit = package_v6_inference_row_yxyx(
                value,
                expected_hld_postprocess_version=(
                    expected_hld_postprocess_version
                ),
            )
            packaged_rows.append(packaged)
            audits.append(audit)
    temporary = path.with_suffix(path.suffix + ".packaging")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in packaged_rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "version": PACKAGING_VERSION,
        "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
        "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
        "rows": len(packaged_rows),
        "packaged_rows": sum(
            audit.get("status") == "packaged" for audit in audits
        ),
        "skipped_invalid_rows": sum(
            audit.get("status") == "skipped_invalid_prediction" for audit in audits
        ),
        "bboxes_converted": sum(
            int(audit.get("total_bboxes_converted", 0)) for audit in audits
        ),
        "hld_postprocess_version": expected_hld_postprocess_version,
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
    unit = binding["unit"]
    output = marker["output"]
    counts = output["counts"]
    return {
        "sha256": str(output["sha256"]),
        "rows": str(output["rows"]),
        "json-valid": str(counts["json_valid"]),
        "schema-valid": str(counts["schema_valid"]),
        "failed": str(counts["failed"]),
        "inference-errors": str(counts["inference_errors"]),
        "source-etag": str(unit["source"]["etag"]),
        "source-catalog-sha256": str(binding["catalogs"]["source"]["sha256"]),
        "unit-catalog-sha256": str(binding["catalogs"]["unit"]["sha256"]),
        "global-unit-index": str(unit["global_unit_index"]),
        "dataset-unit-index": str(unit["dataset_unit_index"]),
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
        if type(value) is not int or value < 0:
            raise ResumeMarkerError(f"marker output counts.{key} is invalid")
        normalized[key] = value
    if normalized["rows"] != expected_rows:
        raise ResumeMarkerError("marker output row count does not match selected unit rows")
    if not normalized["schema_valid"] <= normalized["json_valid"] <= normalized["rows"]:
        raise ResumeMarkerError("marker JSON/schema validity counts are inconsistent")
    if normalized["failed"] != normalized["rows"] - normalized["schema_valid"]:
        raise ResumeMarkerError("marker failed count is inconsistent")
    if normalized["inference_errors"] > normalized["failed"]:
        raise ResumeMarkerError("marker inference error count is inconsistent")
    if normalized["inference_errors"] != 0:
        raise ResumeMarkerError("success marker inference error count must be zero")
    return normalized


def validate_source_head(source: Mapping[str, Any], head: Mapping[str, Any]) -> None:
    if head.get("ContentLength") != source.get("bytes"):
        raise ResumeMarkerError("source HEAD ContentLength differs from frozen unit catalog")
    if clean_etag(head.get("ETag")) != source.get("etag"):
        raise ResumeMarkerError("source HEAD ETag differs from frozen unit catalog")


def validate_resume_marker(
    marker: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
    expected_output_uri: str,
    output_head: Mapping[str, Any],
) -> None:
    if set(marker) != {"format_version", "status", "completed_at", "binding", "output"}:
        raise ResumeMarkerError("existing marker has an unexpected top-level schema")
    if marker.get("format_version") != SUCCESS_MARKER_FORMAT or marker.get("status") != "complete":
        raise ResumeMarkerError("existing marker is not a complete supported unit marker")
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
    if type(output_bytes) is not int or output_bytes <= 0:
        raise ResumeMarkerError("existing marker output byte count is invalid")
    expected_rows = int(expected_binding["unit"]["selected_rows"])
    if output_rows != expected_rows:
        raise ResumeMarkerError("existing marker output rows differ from selected unit rows")
    _validate_counts(output.get("counts"), expected_rows=expected_rows)
    if output_head.get("ContentLength") != output_bytes:
        raise ResumeMarkerError("output HEAD ContentLength differs from marker")
    if output_head.get("ContentType") != OUTPUT_CONTENT_TYPE:
        raise ResumeMarkerError("output HEAD ContentType differs from committed contract")
    raw_metadata = output_head.get("Metadata")
    if not isinstance(raw_metadata, Mapping):
        raise ResumeMarkerError("output HEAD has no user metadata")
    actual_metadata = {str(key).lower(): str(value) for key, value in raw_metadata.items()}
    for key, expected in output_head_metadata(marker).items():
        if actual_metadata.get(key) != expected:
            raise ResumeMarkerError(
                f"output HEAD metadata {key!r}={actual_metadata.get(key)!r}, expected {expected!r}"
            )


def compact_shard_record(
    marker: Mapping[str, Any],
    *,
    marker_uri: str,
    disposition: str,
) -> dict[str, Any]:
    if disposition not in {"generated", "resumed"}:
        raise ValueError(f"invalid shard disposition: {disposition!r}")
    unit = marker["binding"]["unit"]
    record = {
        "global_unit_index": unit["global_unit_index"],
        "dataset_id": unit["dataset_id"],
        "dataset_unit_index": unit["dataset_unit_index"],
        "source": dict(unit["source"]),
        "local_row_ranges": [list(row_range) for row_range in unit["local_row_ranges"]],
        "dataset_row_start": unit["dataset_row_start"],
        "dataset_row_end_exclusive": unit["dataset_row_end_exclusive"],
        "global_row_start": unit["global_row_start"],
        "global_row_end_exclusive": unit["global_row_end_exclusive"],
        "output": dict(marker["output"]),
        "success_marker_uri": marker_uri,
        "disposition": disposition,
    }
    if set(record) != COMPACT_SHARD_FIELDS:
        raise AssertionError("internal compact shard schema drift")
    return record


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


def unit_inference_telemetry_event(
    args: argparse.Namespace,
    unit: Mapping[str, Any],
    execution: inference.InferenceExecution,
) -> dict[str, Any]:
    """Render one generated unit's inference-only telemetry as a log event."""

    normalized = source_adapter.validate_unit(unit)
    telemetry = execution.telemetry
    available = bool(telemetry)
    generated_requests = int(telemetry.get("generated_requests", execution.counts.get("rows", 0)))
    prompt_tokens = telemetry.get("prompt_tokens")
    output_tokens = telemetry.get("output_tokens")
    elapsed_seconds = telemetry.get("elapsed_seconds")
    rows_per_second = None
    output_tokens_per_second = None
    if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds > 0:
        rows_per_second = generated_requests / float(elapsed_seconds)
        if isinstance(output_tokens, int):
            output_tokens_per_second = output_tokens / float(elapsed_seconds)

    raw_speculative = telemetry.get("speculative")
    speculative = raw_speculative if isinstance(raw_speculative, Mapping) else {}
    return {
        "event": "unit_inference_telemetry",
        "global_rank": args.rank,
        "global_world_size": args.world_size,
        "dataset_id": normalized.dataset_id,
        "dataset_unit_index": normalized.dataset_unit_index,
        "global_unit_index": normalized.global_unit_index,
        "rows": execution.counts.get("rows", 0),
        "generated_requests": generated_requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed_seconds,
        "request_window_size": int(args.request_window_size),
        "active_batch_size": int(args.batch_size),
        "prefetch_images": bool(args.prefetch_images),
        "image_wait_seconds": telemetry.get("image_wait_seconds"),
        "generation_seconds": telemetry.get("generation_seconds"),
        "postprocess_seconds": telemetry.get("postprocess_seconds"),
        "write_seconds": telemetry.get("write_seconds"),
        "rows_per_second": rows_per_second,
        "output_tokens_per_second": output_tokens_per_second,
        "mtp": {
            "enabled": speculative.get("enabled", bool(args.use_mtp)),
            "drafts": speculative.get("drafts"),
            "draft_tokens": speculative.get("draft_tokens"),
            "accepted_tokens": speculative.get("accepted_tokens"),
            "acceptance_rate": speculative.get("acceptance_rate"),
        },
        "telemetry_available": available,
    }


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
    marker_sha256 = sha256_bytes(payload)
    metadata = response.get("Metadata")
    if not isinstance(metadata, Mapping) or metadata.get("sha256") != marker_sha256:
        raise ResumeMarkerError(f"{marker_uri}: marker body SHA-256 metadata is missing or wrong")
    if response.get("ContentType") != MARKER_CONTENT_TYPE:
        raise ResumeMarkerError(f"{marker_uri}: marker ContentType is invalid")
    marker = _strict_json_object(payload, location=marker_uri)

    source = expected_binding["unit"]["source"]
    try:
        source_head = client.head_object(Bucket=source["bucket"], Key=source["key"])
    except BaseException as exc:
        if _not_found(exc):
            raise ResumeMarkerError(f"{marker_uri}: source object is missing") from exc
        raise
    validate_source_head(source, source_head)

    output_bucket, output_key = parse_s3_uri(expected_output_uri)
    try:
        output_head = client.head_object(Bucket=output_bucket, Key=output_key)
    except BaseException as exc:
        if _not_found(exc):
            raise ResumeMarkerError(f"{marker_uri}: output object is missing") from exc
        raise
    validate_resume_marker(
        marker,
        expected_binding=expected_binding,
        expected_output_uri=expected_output_uri,
        output_head=output_head,
    )
    return compact_shard_record(marker, marker_uri=marker_uri, disposition="resumed")


def upload_output_object(
    client: Any,
    *,
    output_path: Path,
    output_uri: str,
    marker: Mapping[str, Any],
) -> None:
    from boto3.s3.transfer import TransferConfig

    output_bucket, output_key = parse_s3_uri(output_uri)
    client.upload_file(
        str(output_path),
        output_bucket,
        output_key,
        ExtraArgs={
            "ContentType": OUTPUT_CONTENT_TYPE,
            "Metadata": output_head_metadata(marker),
        },
        # Cross-unit mode already bounds two outer writer threads per rank.
        # Avoid nesting boto3's default transfer pool below every unit upload.
        Config=TransferConfig(max_concurrency=1, use_threads=False),
    )
    output_head = client.head_object(Bucket=output_bucket, Key=output_key)
    validate_resume_marker(
        marker,
        expected_binding=marker["binding"],
        expected_output_uri=output_uri,
        output_head=output_head,
    )


def upload_success_marker(
    client: Any,
    *,
    marker_path: Path,
    marker_uri: str,
    marker: Mapping[str, Any],
) -> None:
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


def upload_output_and_marker(
    client: Any,
    *,
    output_path: Path,
    output_uri: str,
    marker_path: Path,
    marker_uri: str,
    marker: Mapping[str, Any],
) -> None:
    upload_output_object(
        client,
        output_path=output_path,
        output_uri=output_uri,
        marker=marker,
    )
    upload_success_marker(
        client,
        marker_path=marker_path,
        marker_uri=marker_uri,
        marker=marker,
    )


def unit_temp_dir(args: argparse.Namespace, unit: Mapping[str, Any]) -> Path:
    normalized = source_adapter.validate_unit(unit)
    return args.work_dir / f"rank-{args.rank:03d}" / "unit-work" / f"unit-{normalized.global_unit_index:08d}"


def cleanup_unit_temp_dir(args: argparse.Namespace, unit: Mapping[str, Any]) -> None:
    """Remove only the exact per-unit scratch directory after durable success."""

    target = unit_temp_dir(args, unit).resolve()
    parent = (args.work_dir / f"rank-{args.rank:03d}" / "unit-work").resolve()
    if target.parent != parent or re.fullmatch(r"unit-[0-9]{8}", target.name) is None:
        raise RuntimeError(f"refusing unsafe unit scratch cleanup: {target}")
    if target.exists():
        shutil.rmtree(target)


def _local_unit_paths(
    args: argparse.Namespace,
    unit: Mapping[str, Any],
) -> tuple[Path, Path]:
    normalized = source_adapter.validate_unit(unit)
    root = args.work_dir / f"rank-{args.rank:03d}" / "datasets" / normalized.dataset_id / "shards"
    stem = f"unit-{normalized.dataset_unit_index:06d}"
    return root / f"{stem}.jsonl", root / f"{stem}.success.json"


@dataclass
class _CrossUnitState:
    """Per-unit write state while one rank-long request stream stays full."""

    unit: Mapping[str, Any]
    binding: Mapping[str, Any]
    normalized: source_adapter.SourceUnit
    output_path: Path
    partial_path: Path
    marker_path: Path
    scratch: Path
    output_handle: Any | None = None
    rows_written: int = 0


def _new_cross_unit_state(
    args: argparse.Namespace,
    unit: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> _CrossUnitState:
    normalized = source_adapter.validate_unit(unit)
    output_path, marker_path = _local_unit_paths(args, unit)
    return _CrossUnitState(
        unit=unit,
        binding=binding,
        normalized=normalized,
        output_path=output_path,
        partial_path=output_path.with_suffix(output_path.suffix + ".partial"),
        marker_path=marker_path,
        scratch=unit_temp_dir(args, unit),
    )


def _finalize_streamed_unit(
    args: argparse.Namespace,
    client: Any,
    state: _CrossUnitState,
) -> dict[str, Any]:
    """Commit one unit after its generating window passed MTP validation."""

    os.replace(state.partial_path, state.output_path)
    packaging_summary = None
    if bool(getattr(args, "package_v6_bboxes_yxyx", False)):
        packaging_summary = package_output_file_yxyx(
            state.output_path,
            expected_hld_postprocess_version=(
                args.expected_hld_postprocess_version
            ),
        )
    output_summary = summarize_output_file(state.output_path, unit=state.unit)
    if packaging_summary is not None:
        output_summary["bbox_packaging"] = packaging_summary
    inference_errors = output_summary["counts"]["inference_errors"]
    if inference_errors:
        raise RuntimeError(
            f"unit {state.normalized.global_unit_index} contains {inference_errors} "
            "inference errors; retaining local diagnostics without publishing a success marker"
        )
    output_uri, marker_uri = output_uris(
        args.output_s3,
        state.normalized.dataset_id,
        state.normalized.dataset_unit_index,
    )
    marker = build_success_marker(
        binding=state.binding,
        output_uri=output_uri,
        output_summary=output_summary,
        completed_at=utc_now(),
    )
    upload_output_object(
        client,
        output_path=state.output_path,
        output_uri=output_uri,
        marker=marker,
    )
    upload_success_marker(
        client,
        marker_path=state.marker_path,
        marker_uri=marker_uri,
        marker=marker,
    )
    cleanup_unit_temp_dir(args, state.unit)
    return compact_shard_record(
        marker,
        marker_uri=marker_uri,
        disposition="generated",
    )


class _CrossUnitJsonlRouter:
    """Scatter ordered rank-stream JSONL rows back to immutable unit shards.

    The route queue is bounded by vLLM's current and prefetched request windows.
    Completed units are validated and uploaded on a bounded background queue so
    S3 marker latency does not leave the H200 idle between generation windows.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        client: Any,
        states: Sequence[_CrossUnitState],
    ) -> None:
        self.args = args
        self.client = client
        self.states = list(states)
        self.routes: deque[tuple[_CrossUnitState, int]] = deque()
        self.buffer = ""
        self.executor = ThreadPoolExecutor(
            max_workers=int(args.unit_upload_workers),
            thread_name_prefix=f"sidecar24-rank-{args.rank}-unit-writer",
        )
        self.pending_writes: deque[tuple[_CrossUnitState, Future[dict[str, Any]]]] = deque()
        self.shard_records: list[dict[str, Any]] = []

    def register_record(self, state: _CrossUnitState, record: inference.InferenceRecord) -> None:
        self.routes.append((state, record.input_index))

    def _resolve_oldest_write(self) -> None:
        state, future = self.pending_writes.popleft()
        record = future.result()
        self.shard_records.append(record)
        print(
            json.dumps(
                {
                    "event": "unit_complete",
                    "global_rank": self.args.rank,
                    "global_unit_index": state.normalized.global_unit_index,
                    "counts": record["output"]["counts"],
                    "cross_unit_streaming": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _submit_completed_unit(self, state: _CrossUnitState) -> None:
        while len(self.pending_writes) >= int(self.args.max_pending_unit_writes):
            self._resolve_oldest_write()
        future = self.executor.submit(
            _finalize_streamed_unit,
            self.args,
            self.client,
            state,
        )
        self.pending_writes.append((state, future))

    def write(self, payload: str) -> int:
        if not isinstance(payload, str):
            raise TypeError("cross-unit output router accepts text JSONL only")
        # Split the whole window once. Repeated split(..., 1) copies the long
        # JSONL suffix for every row and becomes quadratic at H200 window sizes.
        lines = (self.buffer + payload).split("\n")
        self.buffer = lines.pop()
        for line in lines:
            if not line:
                raise RuntimeError("inference emitted an empty JSONL row")
            if not self.routes:
                raise RuntimeError("inference emitted more JSONL rows than routed input records")
            state, _expected_input_index = self.routes.popleft()
            if state.output_handle is None:
                raise RuntimeError(f"unit {state.normalized.global_unit_index} output handle is not open")
            state.output_handle.write(line + "\n")
            state.rows_written += 1
            if state.rows_written > state.normalized.selected_rows:
                raise RuntimeError(f"unit {state.normalized.global_unit_index} received too many result rows")
            if state.rows_written == state.normalized.selected_rows:
                state.output_handle.close()
                state.output_handle = None
                self._submit_completed_unit(state)
        return len(payload)

    def flush(self) -> None:
        for state in self.states:
            if state.output_handle is not None:
                state.output_handle.flush()

    def finish(self) -> list[dict[str, Any]]:
        if self.buffer:
            raise RuntimeError("inference output ended with a non-newline-terminated JSONL row")
        if self.routes:
            raise RuntimeError(f"inference omitted {len(self.routes)} routed result rows")
        incomplete = [
            state.normalized.global_unit_index
            for state in self.states
            if state.rows_written != state.normalized.selected_rows
        ]
        if incomplete:
            raise RuntimeError(f"cross-unit stream ended with incomplete units: {incomplete[:8]}")
        while self.pending_writes:
            self._resolve_oldest_write()
        self.executor.shutdown(wait=True)
        self.shard_records.sort(key=lambda record: int(record["global_unit_index"]))
        return self.shard_records

    def abort(self) -> None:
        for state in self.states:
            if state.output_handle is not None:
                state.output_handle.close()
                state.output_handle = None
        self.executor.shutdown(wait=True, cancel_futures=False)


def _load_cross_unit_records(
    state: _CrossUnitState,
    *,
    client: Any,
) -> list[inference.InferenceRecord]:
    records = source_adapter.load_unit_records(
        state.unit,
        s3_client=client,
        work_dir=state.scratch,
    )
    validate_records_for_unit(records, unit=state.unit)
    return records


def _iter_prefetched_unit_records(
    states: Sequence[_CrossUnitState],
    *,
    client: Any,
    workers: int,
    depth: int,
) -> Iterable[tuple[_CrossUnitState, list[inference.InferenceRecord]]]:
    """Load bounded future Parquet units while vLLM consumes the current one."""

    if workers <= 0 or depth <= 0:
        raise ValueError("unit read workers and prefetch depth must be positive")
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="sidecar24-unit-reader",
    )
    state_iter = iter(states)
    pending: deque[tuple[_CrossUnitState, Future[list[inference.InferenceRecord]]]] = deque()

    def fill() -> None:
        while len(pending) < depth:
            try:
                state = next(state_iter)
            except StopIteration:
                return
            pending.append(
                (
                    state,
                    executor.submit(_load_cross_unit_records, state, client=client),
                )
            )

    try:
        fill()
        while pending:
            state, future = pending.popleft()
            records = future.result()
            fill()
            yield state, records
    finally:
        for _state, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def process_units_cross_stream(
    args: argparse.Namespace,
    runtime: inference.InferenceRuntime,
    client: Any,
    *,
    units: Sequence[Mapping[str, Any]],
    bindings: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one continuous vLLM request stream across unit boundaries."""

    states = [_new_cross_unit_state(args, unit, bindings[int(unit["global_unit_index"])]) for unit in units]
    router = _CrossUnitJsonlRouter(args, client, states)

    def iter_records() -> Iterable[inference.InferenceRecord]:
        for state, records in _iter_prefetched_unit_records(
            states,
            client=client,
            workers=int(getattr(args, "unit_read_workers", DEFAULT_UNIT_READ_WORKERS)),
            depth=int(getattr(args, "unit_prefetch_depth", DEFAULT_UNIT_PREFETCH_DEPTH)),
        ):
            print(
                json.dumps(
                    {
                        "event": "unit_start",
                        "global_rank": args.rank,
                        "global_unit_index": state.normalized.global_unit_index,
                        "dataset_id": state.normalized.dataset_id,
                        "rows": state.normalized.selected_rows,
                        "cross_unit_streaming": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            state.output_path.parent.mkdir(parents=True, exist_ok=True)
            state.output_handle = state.partial_path.open("w", encoding="utf-8")
            for record in records:
                router.register_record(state, record)
                yield record

    try:
        execution = inference.infer_records(args, runtime, iter_records(), router)
        shard_records = router.finish()
    except BaseException:
        router.abort()
        raise

    totals = aggregate_counts(shard_records)
    for key, value in execution.counts.items():
        if totals[key] != value:
            raise RuntimeError(
                f"cross-unit output count {key}={totals[key]} differs from inference count {value}"
            )
    telemetry = execution.telemetry
    print(
        json.dumps(
            {
                "event": "rank_cross_unit_inference_telemetry",
                "global_rank": args.rank,
                "global_world_size": args.world_size,
                "units": len(states),
                "rows": totals["rows"],
                "request_window_size": args.request_window_size,
                "active_batch_size": args.batch_size,
                "max_pending_unit_writes": args.max_pending_unit_writes,
                "unit_upload_workers": args.unit_upload_workers,
                "unit_read_workers": int(
                    getattr(args, "unit_read_workers", DEFAULT_UNIT_READ_WORKERS)
                ),
                "unit_prefetch_depth": int(
                    getattr(args, "unit_prefetch_depth", DEFAULT_UNIT_PREFETCH_DEPTH)
                ),
                "generated_requests": telemetry.get("generated_requests"),
                "prompt_tokens": telemetry.get("prompt_tokens"),
                "output_tokens": telemetry.get("output_tokens"),
                "elapsed_seconds": telemetry.get("elapsed_seconds"),
                "generation_seconds": telemetry.get("generation_seconds"),
                "image_wait_seconds": telemetry.get("image_wait_seconds"),
                "speculative": telemetry.get("speculative"),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return shard_records


def process_unit(
    args: argparse.Namespace,
    runtime: inference.InferenceRuntime,
    client: Any,
    *,
    unit: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = source_adapter.validate_unit(unit)
    output_path, marker_path = _local_unit_paths(args, unit)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = unit_temp_dir(args, unit)

    records = source_adapter.load_unit_records(
        unit,
        s3_client=client,
        work_dir=scratch,
    )
    validate_records_for_unit(records, unit=unit)
    with partial_path.open("w", encoding="utf-8") as output_handle:
        execution = inference.infer_records(args, runtime, records, output_handle)
    os.replace(partial_path, output_path)
    packaging_summary = None
    if bool(getattr(args, "package_v6_bboxes_yxyx", False)):
        packaging_summary = package_output_file_yxyx(
            output_path,
            expected_hld_postprocess_version=(
                args.expected_hld_postprocess_version
            ),
        )
    output_summary = summarize_output_file(output_path, unit=unit)
    if packaging_summary is not None:
        output_summary["bbox_packaging"] = packaging_summary
    if output_summary["counts"]["inference_errors"] != 0:
        raise RuntimeError(
            f"unit {normalized.global_unit_index} contains "
            f"{output_summary['counts']['inference_errors']} inference errors; "
            "retaining local diagnostics without publishing a success marker"
        )
    for key, value in execution.counts.items():
        if output_summary["counts"][key] != value:
            raise RuntimeError(
                f"unit {normalized.global_unit_index} output count {key}="
                f"{output_summary['counts'][key]} differs from inference count {value}"
            )

    output_uri, marker_uri = output_uris(
        args.output_s3,
        normalized.dataset_id,
        normalized.dataset_unit_index,
    )
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
    cleanup_unit_temp_dir(args, unit)
    print(
        json.dumps(
            unit_inference_telemetry_event(args, unit, execution),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return compact_shard_record(marker, marker_uri=marker_uri, disposition="generated")


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


def _load_rebalanced_assignment(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    assignment_path = getattr(args, "assigned_unit_indices", None)
    if assignment_path is None:
        return None
    path = Path(assignment_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "physical_global_rank",
        "selection_sha256",
        "global_unit_indices",
    }:
        raise ValueError(f"{path}: rebalanced assignment has an unexpected schema")
    if value["format_version"] != REBALANCED_ASSIGNMENT_FORMAT:
        raise ValueError(f"{path}: rebalanced assignment format is invalid")
    if value["physical_global_rank"] != args.rank:
        raise ValueError(f"{path}: rebalanced assignment physical rank mismatch")
    indices = value["global_unit_indices"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError(f"{path}: rebalanced assignment indices are invalid")
    return value


def _selected_units_for_run(
    args: argparse.Namespace,
    units: Sequence[Mapping[str, Any]],
    plan: partition_module.PartitionPlan | None,
) -> list[dict[str, Any]]:
    assignment = _load_rebalanced_assignment(args)
    if assignment is None:
        if plan is not None:
            return [dict(unit) for unit in plan.units_by_rank[args.rank]]
        partition_count = int(getattr(args, "partition_count", 1))
        partition_index = int(getattr(args, "partition_index", 0))
        selected: list[dict[str, Any]] = []
        partition_ordinal = 0
        for unit in units:
            index = int(unit["global_unit_index"])
            if index % partition_count != partition_index:
                continue
            if partition_ordinal % args.world_size == args.rank:
                selected.append(dict(unit))
            partition_ordinal += 1
        return selected

    indices = assignment["global_unit_indices"]
    if plan is not None:
        if assignment["selection_sha256"] != plan.contract["selection_sha256"]:
            raise ValueError("rebalanced assignment selection mismatch")
        candidate_lanes = plan.units_by_rank
    else:
        partition_count = int(getattr(args, "partition_count", 1))
        partition_index = int(getattr(args, "partition_index", 0))
        candidate_lanes = (
            tuple(
                unit
                for unit in units
                if int(unit["global_unit_index"]) % partition_count == partition_index
            ),
        )
    wanted = set(indices)
    units_by_index: dict[int, dict[str, Any]] = {}
    for lane in candidate_lanes:
        for unit in lane:
            index = int(unit["global_unit_index"])
            if index in wanted:
                units_by_index[index] = dict(unit)
        if len(units_by_index) == len(wanted):
            break
    unknown = [index for index in indices if index not in units_by_index]
    if unknown:
        raise ValueError(
            "rebalanced assignment contains units outside the selection: "
            f"{unknown[:8]}"
        )
    return [units_by_index[index] for index in indices]


def run(args: argparse.Namespace) -> dict[str, Any]:
    inference.configure_aws_environment(args)
    inference.validate_prompt_contract()
    client = s3_client(args)
    source_catalog, unit_catalog, source_sha256, unit_sha256 = load_frozen_catalogs(
        args,
        client=client,
    )
    units = unit_catalog["units"]
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"invalid global rank/world size: rank={args.rank} world={args.world_size}")
    # Full runs can resolve one lane in a single catalog scan. Avoid building
    # and hashing all world-size lanes independently in every GPU subprocess.
    plan = (
        partition_plan(units, args)
        if getattr(args, "case_limit", None) is not None
        else None
    )
    selected = _selected_units_for_run(args, units, plan)
    # Part and canary outputs use disjoint namespaces.  All existing output URI
    # helpers can then retain their strict dataset/unit layout unchanged.
    selection_contract = (
        plan.contract
        if plan is not None
        else partition_module.basic_contract(
            partition_count=int(getattr(args, "partition_count", 1)),
            partition_index=int(getattr(args, "partition_index", 0)),
            global_world_size=args.world_size,
            case_limit=None,
        )
    )
    args.output_s3 = partition_module.selection_output_root(args.output_s3, selection_contract)
    started_at = utc_now()
    generation = generation_contract(args)
    bindings = {
        int(unit["global_unit_index"]): unit_binding(
            unit,
            source_catalog_sha256=source_sha256,
            unit_catalog_sha256=unit_sha256,
            checkpoint_step=args.checkpoint_step,
            checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            hf_artifact_manifest_sha256=args.hf_artifact_manifest_sha256,
            generation=generation,
        )
        for unit in selected
    }

    shard_records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for unit in selected:
        normalized = source_adapter.validate_unit(unit)
        output_uri, marker_uri = output_uris(
            args.output_s3,
            normalized.dataset_id,
            normalized.dataset_unit_index,
        )
        resumed = load_existing_marker(
            client,
            marker_uri=marker_uri,
            expected_binding=bindings[normalized.global_unit_index],
            expected_output_uri=output_uri,
        )
        if resumed is None:
            pending.append(unit)
        else:
            cleanup_unit_temp_dir(args, unit)
            shard_records.append(resumed)
            print(
                json.dumps(
                    {
                        "event": "unit_resume",
                        "global_rank": args.rank,
                        "global_unit_index": normalized.global_unit_index,
                        "output": output_uri,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    runtime: inference.InferenceRuntime | None = None
    if pending:
        runtime = inference.initialize_inference_runtime(args)
    if pending and bool(getattr(args, "cross_unit_streaming", False)):
        assert runtime is not None
        shard_records.extend(
            process_units_cross_stream(
                args,
                runtime,
                client,
                units=pending,
                bindings=bindings,
            )
        )
    else:
        for unit in pending:
            assert runtime is not None
            normalized = source_adapter.validate_unit(unit)
            print(
                json.dumps(
                    {
                        "event": "unit_start",
                        "global_rank": args.rank,
                        "global_unit_index": normalized.global_unit_index,
                        "dataset_id": normalized.dataset_id,
                        "rows": normalized.selected_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            record = process_unit(
                args,
                runtime,
                client,
                unit=unit,
                binding=bindings[normalized.global_unit_index],
            )
            shard_records.append(record)
            print(
                json.dumps(
                    {
                        "event": "unit_complete",
                        "global_rank": args.rank,
                        "global_unit_index": normalized.global_unit_index,
                        "counts": record["output"]["counts"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    shard_records.sort(key=lambda record: int(record["global_unit_index"]))
    expected_indices = [int(unit["global_unit_index"]) for unit in selected]
    actual_indices = [int(record["global_unit_index"]) for record in shard_records]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"rank unit coverage mismatch: actual={actual_indices} expected={expected_indices}"
        )
    totals = aggregate_counts(shard_records)
    expected_rows = sum(source_adapter.validate_unit(unit).selected_rows for unit in selected)
    if totals["rows"] != expected_rows:
        raise RuntimeError(f"rank result rows={totals['rows']} expected={expected_rows}")

    rank_dir = args.work_dir / f"rank-{args.rank:03d}"
    if bool(getattr(args, "defer_rank_summary", False)):
        assignment = _load_rebalanced_assignment(args)
        if assignment is None:
            raise AssertionError("a deferred rank summary has no rebalanced assignment")
        operational_summary = {
            "format_version": REBALANCED_WORKER_SUMMARY_FORMAT,
            "status": "complete",
            "started_at": started_at,
            "completed_at": utc_now(),
            "physical_global_rank": args.rank,
            "global_world_size": args.world_size,
            "selection_sha256": assignment["selection_sha256"],
            "assigned_units": len(selected),
            "generated_units": len(pending),
            "resumed_units": len(selected) - len(pending),
            "global_unit_indices": actual_indices,
            "totals": totals,
        }
        atomic_json(rank_dir / "operational-summary.json", operational_summary)
        return operational_summary
    summary_path = rank_dir / "summary.json"
    summary = {
        "format_version": RANK_SUMMARY_FORMAT,
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "global_rank": args.rank,
        "global_world_size": args.world_size,
        "catalogs": catalog_bindings(
            source_catalog_uri=args.source_catalog_uri,
            source_catalog=source_catalog,
            source_catalog_sha256=source_sha256,
            unit_catalog_uri=args.unit_catalog_uri,
            unit_catalog=unit_catalog,
            unit_catalog_sha256=unit_sha256,
        ),
        "checkpoint": {
            "step": args.checkpoint_step,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        "hf_artifact": {"manifest_sha256": args.hf_artifact_manifest_sha256},
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
        },
        "generation": generation,
        "runtime_initialized": runtime is not None,
        "assigned_units": len(selected),
        "generated_units": len(pending),
        "resumed_units": len(selected) - len(pending),
        "totals": totals,
        "shards": shard_records,
        "summary_s3_uri": rank_summary_uri(args.output_s3, args.rank),
    }
    if set(summary) != RANK_SUMMARY_FIELDS:
        raise AssertionError("internal rank summary schema drift")
    atomic_json(summary_path, summary)
    uploaded_uri = upload_rank_summary(
        client,
        path=summary_path,
        output_s3=args.output_s3,
        global_rank=args.rank,
    )
    if uploaded_uri != summary["summary_s3_uri"]:
        raise RuntimeError("uploaded rank summary URI differs from persisted summary")
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
    parser.add_argument("--source-catalog", required=True, help="Local path or s3:// source catalog")
    parser.add_argument(
        "--source-catalog-uri",
        required=True,
        help="Stable s3:// identity of the frozen source catalog",
    )
    parser.add_argument("--source-catalog-sha256", required=True)
    parser.add_argument("--unit-catalog", required=True, help="Local path or s3:// unit catalog")
    parser.add_argument(
        "--unit-catalog-uri",
        required=True,
        help="Stable s3:// identity of the frozen unit catalog",
    )
    parser.add_argument("--unit-catalog-sha256", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--assigned-unit-indices",
        type=Path,
        help="Operational rebalancing manifest; excluded from the generation contract",
    )
    parser.add_argument(
        "--defer-rank-summary",
        action="store_true",
        help="Commit units without publishing a canonical fixed-rank summary",
    )
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
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Number of mutually exclusive data parts (unit-index modulo assignment)",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Zero-based part to process",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Global canary row limit across all ranks, never per-rank",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--request-window-size",
        type=int,
        default=DEFAULT_REQUEST_WINDOW_SIZE,
        help="Requests queued per blocking vLLM call; active batch remains --batch-size",
    )
    parser.add_argument(
        "--prefetch-images",
        action="store_true",
        help="Load/decode the next request window while the current window generates",
    )
    parser.add_argument(
        "--cross-unit-streaming",
        action="store_true",
        help="Keep request windows full across unit boundaries and scatter rows atomically",
    )
    parser.add_argument(
        "--max-pending-unit-writes",
        type=int,
        default=DEFAULT_MAX_PENDING_UNIT_WRITES,
        help="Bounded async unit validation/upload queue",
    )
    parser.add_argument(
        "--unit-upload-workers",
        type=int,
        default=DEFAULT_UNIT_UPLOAD_WORKERS,
        help="Concurrent unit validators/uploaders behind the bounded queue",
    )
    parser.add_argument(
        "--unit-read-workers",
        type=int,
        default=DEFAULT_UNIT_READ_WORKERS,
        help="Concurrent Parquet unit readers feeding a cross-unit stream",
    )
    parser.add_argument(
        "--unit-prefetch-depth",
        type=int,
        default=DEFAULT_UNIT_PREFETCH_DEPTH,
        help="Maximum prefetched Parquet units retained per GPU rank",
    )
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
    parser.add_argument("--speculative-attention-backend")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--expected-hld-postprocess-version")
    parser.add_argument("--package-v6-bboxes-yxyx", action="store_true")
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
        "request_window_size",
        "max_pending_unit_writes",
        "unit_upload_workers",
        "unit_read_workers",
        "unit_prefetch_depth",
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
    if args.request_window_size < args.batch_size:
        parser.error("--request-window-size must be at least --batch-size")
    if args.unit_upload_workers > args.max_pending_unit_writes:
        parser.error("--unit-upload-workers must not exceed --max-pending-unit-writes")
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
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--global-rank must be in [0, --global-world-size)")
    if (args.assigned_unit_indices is None) != (not args.defer_rank_summary):
        parser.error(
            "--assigned-unit-indices and --defer-rank-summary must be supplied together"
        )
    if args.assigned_unit_indices is not None and not args.assigned_unit_indices.is_file():
        parser.error("--assigned-unit-indices must name an existing file")
    if args.partition_count <= 0:
        parser.error("--partition-count must be positive")
    if args.partition_index < 0 or args.partition_index >= args.partition_count:
        parser.error("--partition-index must be in [0, --partition-count)")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("--case-limit must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    if args.package_v6_bboxes_yxyx:
        if TARGET_SCHEMA != "image_edit_v6_direct_root_xyxy":
            parser.error("--package-v6-bboxes-yxyx requires the V6 XYXY schema")
        if not args.expected_hld_postprocess_version:
            parser.error(
                "--package-v6-bboxes-yxyx requires "
                "--expected-hld-postprocess-version"
            )
    for name in (
        "source_catalog_sha256",
        "unit_catalog_sha256",
        "checkpoint_manifest_sha256",
        "hf_artifact_manifest_sha256",
    ):
        if SHA256_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase 64-hex")
    try:
        parse_s3_uri(args.output_s3)
        parse_s3_uri(args.source_catalog_uri)
        parse_s3_uri(args.unit_catalog_uri)
        for location in (args.source_catalog, args.unit_catalog):
            if location.startswith("s3://"):
                parse_s3_uri(location)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_s3 = args.output_s3.rstrip("/")
    args.include_conversation = False
    args.continue_on_error = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def fail_fast_entrypoint(argv: Sequence[str] | None = None) -> int:
    """Exit immediately on an unhandled rank error.

    vLLM owns multiprocessing children.  Normal interpreter teardown waits for
    those children, so a rank traceback can otherwise retain its GPU forever
    while the node supervisor still sees a live process.  The node supervisor
    observes this immediate non-zero exit, terminates sibling ranks, and lets
    the Job controller retry from immutable success markers.
    """

    try:
        return main(argv)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
        raise AssertionError("os._exit returned")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(fail_fast_entrypoint())
