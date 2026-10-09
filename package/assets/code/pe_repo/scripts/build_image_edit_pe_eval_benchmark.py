#!/usr/bin/env python3
"""Build the frozen image-edit PE benchmark from the exact train-5M complement.

The source manifest defines a global row order.  The immutable training
selection bitset identifies the 5,000,000 training rows in that order.  This
program streams every source shard, rejects every bitset member, and selects
the lowest deterministic SHA-256 scores independently for every authoritative
top-level ``task_type``.  Bottom-k selection is mergeable, so distributed
workers produce exactly the same benchmark as a single-process full scan.

Only Call-2 ``validated`` rows are eligible.  Finalization publishes exact
source JSONL rows, aligned audit metadata, and an evaluation-ready manifest
with Target canvas sizes and the same sparse non-``keep`` label projection as
the current PE training loader.  ``_SUCCESS`` is written last.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import unquote, urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


FORMAT_VERSION = "image-edit-pe-task-type-eval-benchmark-v1"
ALGORITHM_VERSION = "train-bitset-complement-validated-sha256-bottom-k-v1"
SAMPLING_ALGORITHM = "python-mt19937-global-srs-bitset-v1"
PROMPT_VERSION = "image_edit_pe_sparse_edited_regions_target_size_v4"
PROMPT_CONTRACT_SHA256 = (
    "be81ac83b4779d068fda638abe693f54ba7948d341b68bc633e670e2d981c610"
)
LABEL_PROJECTION_VERSION = "non_keep_layout_only_deleted_parent_to_root_v2"
ANNOTATION_CONTRACT_VERSION = "image_edit_v6_edit_evidence_remove_repair_xy_v7"
ANNOTATION_CONTRACT_SHA256 = (
    "121b58cb71123e2c2ce78850455d208b626d313792768f5bb8d260beb936156f"
)
TARGET_SCHEMA = "image_edit_v6_direct_root_xyxy"
ELIGIBLE_STATUS = "validated"

EXPECTED_TASK_TYPES = (
    "background_or_environment",
    "blur_focus_or_sharpness",
    "cleanup_damage_or_aging",
    "color_temperature_or_photo_filter",
    "composition_or_layout",
    "crop_or_zoom_in",
    "facial_expression",
    "lighting_or_exposure",
    "material_or_texture",
    "move_or_reposition",
    "object_addition",
    "object_removal",
    "object_replacement",
    "other",
    "outpaint_or_zoom_out",
    "perspective_or_viewpoint",
    "pose_gesture_or_articulation",
    "recolor_or_palette",
    "reflection_shadow_or_glow",
    "resize_or_scale",
    "rotation_flip_or_mirror",
    "season_or_time_of_day",
    "shape_geometry_or_deformation",
    "style_or_medium_transfer",
    "text_content_or_translation",
    "transparency_or_occlusion",
    "typography_or_text_style",
    "visual_effect_or_overlay",
    "weather_or_atmosphere",
)

_SCAN_BITSET: bytes | None = None
_SCAN_CONFIG: dict[str, Any] | None = None


def canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    kwargs["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def canonical_jsonl(rows: Iterable[Any]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def validate_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.port is not None
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def normalize_s3_prefix(uri: str) -> str:
    bucket, key = parse_s3_uri(uri.rstrip("/"))
    return f"s3://{bucket}/{key}/"


def join_s3(prefix: str, relative: str) -> str:
    return normalize_s3_prefix(prefix) + relative.lstrip("/")


def s3_client(*, max_pool_connections: int = 128):
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region != "us-east-1":
        raise ValueError("AWS_REGION/AWS_DEFAULT_REGION must be us-east-1")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=5,
            read_timeout=180,
            tcp_keepalive=True,
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


def get_object_bytes(
    client: Any,
    uri_or_descriptor: str | Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> bytes:
    if isinstance(uri_or_descriptor, str):
        uri = uri_or_descriptor
        descriptor: Mapping[str, Any] = {}
    else:
        descriptor = uri_or_descriptor
        uri = descriptor.get("s3_uri") or descriptor.get("path")
        if not isinstance(uri, str):
            raise ValueError("S3 object descriptor has no URI")
    bucket, key = parse_s3_uri(uri)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if descriptor.get("version_id"):
        kwargs["VersionId"] = str(descriptor["version_id"])
    payload = client.get_object(**kwargs)["Body"].read()
    expected_bytes = descriptor.get("bytes")
    if expected_bytes is not None and len(payload) != int(expected_bytes):
        raise ValueError(f"object byte count changed for {uri}")
    digest = hashlib.sha256(payload).hexdigest()
    pinned = expected_sha256 or descriptor.get("sha256")
    if pinned is not None and digest != validate_sha256(str(pinned), "object SHA-256"):
        raise ValueError(f"object SHA-256 changed for {uri}: {digest} != {pinned}")
    return payload


def maybe_get_json(client: Any, uri: str) -> dict[str, Any] | None:
    bucket, key = parse_s3_uri(uri)
    try:
        payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {uri}")
    return value


def object_descriptor(client: Any, uri: str, *, sha256: str) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    response = client.head_object(Bucket=bucket, Key=key)
    result: dict[str, Any] = {
        "bytes": int(response["ContentLength"]),
        "etag": str(response.get("ETag", "")).strip('"'),
        "s3_uri": uri,
        "sha256": validate_sha256(sha256, "object SHA-256"),
    }
    if response.get("VersionId"):
        result["version_id"] = str(response["VersionId"])
    return result


def put_bytes_if_absent(
    client: Any,
    uri: str,
    payload: bytes,
    *,
    content_type: str,
) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata={"algorithm": ALGORITHM_VERSION, "sha256": digest},
            IfNoneMatch="*",
        )
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code not in {"ConditionalRequestConflict", "PreconditionFailed"} and status not in {
            409,
            412,
        }:
            raise
        existing = get_object_bytes(client, uri)
        if existing != payload:
            raise ValueError(
                f"immutable output already exists with different bytes: {uri}"
            ) from error
    descriptor = object_descriptor(client, uri, sha256=digest)
    if int(descriptor["bytes"]) != len(payload):
        raise ValueError(f"output byte count changed for {uri}")
    return descriptor


def put_json_if_absent(client: Any, uri: str, value: Any) -> dict[str, Any]:
    return put_bytes_if_absent(
        client,
        uri,
        canonical_json(value, pretty=True),
        content_type="application/json",
    )


def bit_is_set(bitset: bytes, index: int) -> bool:
    if index < 0 or index >= len(bitset) * 8:
        raise IndexError(index)
    return bool(bitset[index >> 3] & (1 << (index & 7)))


def count_set_bits(bitset: bytes, start: int, count: int) -> int:
    if start < 0 or count < 0 or start + count > len(bitset) * 8:
        raise ValueError("bitset count range is invalid")
    return sum(bit_is_set(bitset, index) for index in range(start, start + count))


def _ordered_source_outputs(source_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = source_manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("source manifest outputs are missing")
    if not all(isinstance(item, Mapping) for item in outputs):
        raise ValueError("source manifest output is not an object")
    ordered = [dict(item) for item in sorted(outputs, key=lambda item: int(item["part_index"]))]
    if [int(item["part_index"]) for item in ordered] != list(range(len(ordered))):
        raise ValueError("source manifest part indexes are not contiguous")
    if any(int(item["part_count"]) != len(ordered) for item in ordered):
        raise ValueError("source manifest part_count is inconsistent")
    for item in ordered:
        if int(item.get("rows", 0)) <= 0:
            raise ValueError("source shard has a non-positive row count")
        if int(item.get("bytes", 0)) <= 0:
            raise ValueError("source shard has a non-positive byte count")
        validate_sha256(str(item.get("sha256", "")), "source shard SHA-256")
        if not isinstance(item.get("path"), str):
            raise ValueError("source shard has no path")
    return ordered


def load_frozen_contract(client: Any, args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_uri = join_s3(args.source, "manifest.json")
    source_payload = get_object_bytes(
        client,
        source_manifest_uri,
        expected_sha256=args.source_manifest_sha256,
    )
    source_manifest = json.loads(source_payload)
    if not isinstance(source_manifest, dict) or source_manifest.get("complete") is not True:
        raise ValueError("source manifest is not complete")
    outputs = _ordered_source_outputs(source_manifest)
    population_rows = sum(int(item["rows"]) for item in outputs)
    if population_rows != args.expected_population_rows:
        raise ValueError(
            f"source population rows changed: {population_rows} != "
            f"{args.expected_population_rows}"
        )

    plan_payload = get_object_bytes(
        client,
        args.training_sampling_plan,
        expected_sha256=args.training_sampling_plan_sha256,
    )
    plan = json.loads(plan_payload)
    if not isinstance(plan, dict):
        raise ValueError("training sampling plan is not an object")
    if (
        plan.get("complete") is not True
        or plan.get("algorithm") != SAMPLING_ALGORITHM
        or plan.get("source_manifest_sha256") != args.source_manifest_sha256
        or int(plan.get("population_rows", -1)) != population_rows
        or int(plan.get("sample_count", -1)) != args.expected_training_rows
        or int(plan.get("seed", -1)) != args.training_sample_seed
    ):
        raise ValueError("training sampling plan contract mismatch")
    parts = plan.get("source_parts")
    if not isinstance(parts, list) or len(parts) != len(outputs):
        raise ValueError("training sampling plan part coverage mismatch")
    offset = 0
    ordered_parts: list[dict[str, Any]] = []
    for output, raw_part in zip(outputs, parts, strict=True):
        if not isinstance(raw_part, Mapping):
            raise ValueError("training sampling plan part is not an object")
        part = dict(raw_part)
        if (
            int(part.get("part_index", -1)) != int(output["part_index"])
            or int(part.get("global_row_start", -1)) != offset
            or int(part.get("population_rows", -1)) != int(output["rows"])
            or not 0 <= int(part.get("sampled_rows", -1)) <= int(output["rows"])
        ):
            raise ValueError("training sampling plan part metadata mismatch")
        ordered_parts.append(part)
        offset += int(output["rows"])

    selection_descriptor = plan.get("selection_bitset")
    if not isinstance(selection_descriptor, Mapping):
        raise ValueError("training sampling plan has no selection bitset")
    if selection_descriptor.get("sha256") != args.training_sampling_bitset_sha256:
        raise ValueError("training sampling bitset descriptor SHA-256 mismatch")
    bitset = get_object_bytes(client, selection_descriptor)
    if hashlib.sha256(bitset).hexdigest() != args.training_sampling_bitset_sha256:
        raise ValueError("training sampling bitset SHA-256 mismatch")
    if len(bitset) != (population_rows + 7) // 8:
        raise ValueError("training sampling bitset byte count mismatch")
    if population_rows & 7 and bitset[-1] & ~((1 << (population_rows & 7)) - 1):
        raise ValueError("training sampling bitset has selected padding bits")
    if sum(value.bit_count() for value in bitset) != args.expected_training_rows:
        raise ValueError("training sampling bitset selected-row count mismatch")
    for part in ordered_parts:
        actual = count_set_bits(
            bitset,
            int(part["global_row_start"]),
            int(part["population_rows"]),
        )
        if actual != int(part["sampled_rows"]):
            raise ValueError("training sampling bitset per-part count mismatch")
    return {
        "source_manifest": {**source_manifest, "outputs": outputs},
        "source_manifest_uri": source_manifest_uri,
        "sampling_plan": {**plan, "source_parts": ordered_parts},
        "selection_bitset": bitset,
        "population_rows": population_rows,
    }


def selection_hash(
    *,
    seed: int,
    task_type: str,
    sample_id: str,
    global_row_index: int,
) -> str:
    payload = canonical_json(
        {
            "algorithm": ALGORITHM_VERSION,
            "global_row_index": global_row_index,
            "sample_id": sample_id,
            "seed": seed,
            "task_type": task_type,
        }
    )
    return hashlib.sha256(payload).hexdigest()


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[int, int]:
    return int(str(candidate["selection_hash"]), 16), int(candidate["global_row_index"])


def push_bottom_k(
    heap: list[tuple[int, int, dict[str, Any]]],
    candidate: Mapping[str, Any],
    k: int,
) -> None:
    score, global_index = _candidate_key(candidate)
    entry = (-score, -global_index, dict(candidate))
    if len(heap) < k:
        heapq.heappush(heap, entry)
        return
    worst = (-heap[0][0], -heap[0][1])
    if (score, global_index) < worst:
        heapq.heapreplace(heap, entry)


def ordered_heap(heap: Sequence[tuple[int, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    return sorted((entry[2] for entry in heap), key=_candidate_key)


def _bbox_order_is_xyxy(value: Any) -> bool:
    if value == "xyxy":
        return True
    if not isinstance(value, Mapping):
        return False
    return all(
        value.get(field) == "xyxy"
        for field in ("target_layout", "reference_bbox", "remove_reference_bbox")
    )


def validate_eligible_row(row: Mapping[str, Any], *, context: str) -> tuple[str, str]:
    if row.get("status") != ELIGIBLE_STATUS:
        raise ValueError(f"{context}: eligible row status changed")
    sample_id = row.get("sample_id")
    instruction = row.get("instruction")
    source_uri = row.get("source_s3_uri")
    target_uri = row.get("target_s3_uri")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{context}: missing sample_id")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{context}: missing instruction")
    if not isinstance(source_uri, str) or not source_uri.startswith(("s3://", "parquet+s3://")):
        raise ValueError(f"{context}: invalid source_s3_uri")
    if not isinstance(target_uri, str) or not target_uri.startswith(("s3://", "parquet+s3://")):
        raise ValueError(f"{context}: invalid target_s3_uri")
    if source_uri == target_uri:
        raise ValueError(f"{context}: Source and Target locators are identical")
    audit = row.get("call2_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("valid") is not True
        or audit.get("errors") != []
    ):
        raise ValueError(f"{context}: Gemini Call-2 audit is not clean")
    if row.get("contract_version") != ANNOTATION_CONTRACT_VERSION:
        raise ValueError(f"{context}: annotation contract version mismatch")
    if row.get("contract_sha256") != ANNOTATION_CONTRACT_SHA256:
        raise ValueError(f"{context}: annotation contract SHA-256 mismatch")
    if not _bbox_order_is_xyxy(row.get("bbox_coordinate_order")):
        raise ValueError(f"{context}: bbox coordinate order is not XYXY")
    if row.get("schema_valid", True) is not True:
        raise ValueError(f"{context}: schema_valid is false")
    if row.get("target_schema", TARGET_SCHEMA) != TARGET_SCHEMA:
        raise ValueError(f"{context}: Target schema mismatch")
    if "final_json" not in row:
        raise ValueError(f"{context}: final_json is missing")
    return sample_id, instruction


def canonical_sparse_edit_target(value: Any) -> str:
    """Apply the PE loader's non-keep layout projection and serialization."""

    if isinstance(value, str):
        value = json.loads(
            value,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {token}")
            ),
        )
    # JSON round-trip provides a deep mutable copy and rejects non-JSON values.
    target = json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {token}")
        ),
    )
    if not isinstance(target, dict):
        raise ValueError("final_json must be an object")
    elements = target.get("elements")
    if not isinstance(elements, dict) or not isinstance(elements.get("layout"), list):
        raise ValueError("final_json.elements.layout must be a list")
    layout = elements["layout"]
    edited_labels: set[str] = set()
    for index, region in enumerate(layout):
        if not isinstance(region, dict):
            raise ValueError(f"layout[{index}] must be an object")
        label = region.get("label")
        edit = region.get("edit")
        if not isinstance(label, str) or not label:
            raise ValueError(f"layout[{index}].label is invalid")
        if not isinstance(edit, dict) or not isinstance(edit.get("instruction"), str):
            raise ValueError(f"layout[{index}].edit.instruction is invalid")
        if edit["instruction"] != "keep":
            edited_labels.add(label)
    sparse_layout: list[dict[str, Any]] = []
    for region in layout:
        if region["label"] not in edited_labels:
            continue
        parent = region.get("parent")
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"region {region['label']} has invalid parent")
        if parent != "ROOT" and parent not in edited_labels:
            region["parent"] = "ROOT"
        sparse_layout.append(region)
    elements["layout"] = sparse_layout
    return json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _init_scan_process(bitset: bytes, config: Mapping[str, Any]) -> None:
    global _SCAN_BITSET, _SCAN_CONFIG
    _SCAN_BITSET = bitset
    _SCAN_CONFIG = dict(config)


def iter_s3_object_lines_by_range(
    client: Any,
    output: Mapping[str, Any],
    *,
    digest: Any,
    range_size_bytes: int,
) -> Iterable[bytes]:
    """Yield exact lines while bounding any one S3 connection's slow tail."""

    if range_size_bytes <= 0:
        raise ValueError("range_size_bytes must be positive")
    uri = str(output["path"])
    bucket, key = parse_s3_uri(uri)
    object_bytes = int(output["bytes"])
    carry = b""
    for start in range(0, object_bytes, range_size_bytes):
        stop = min(start + range_size_bytes, object_bytes) - 1
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Range": f"bytes={start}-{stop}",
        }
        if output.get("version_id"):
            request["VersionId"] = str(output["version_id"])
        response = client.get_object(**request)
        body = response["Body"]
        chunk = body.read()
        body.close()
        expected = stop - start + 1
        if len(chunk) != expected:
            raise ValueError(
                f"short ranged read for {uri}: bytes={start}-{stop}, "
                f"got {len(chunk)}"
            )
        digest.update(chunk)
        pieces = (carry + chunk).split(b"\n")
        carry = pieces.pop()
        for piece in pieces:
            yield piece + b"\n"
    if carry:
        yield carry


def _scan_source_part(
    output: Mapping[str, Any],
    plan_part: Mapping[str, Any],
    local_dir: str,
) -> dict[str, Any]:
    if _SCAN_BITSET is None or _SCAN_CONFIG is None:
        raise RuntimeError("scan subprocess was not initialized")
    bitset = _SCAN_BITSET
    config = _SCAN_CONFIG
    part_index = int(output["part_index"])
    global_start = int(plan_part["global_row_start"])
    expected_rows = int(output["rows"])
    samples_per_task = int(config["samples_per_task"])
    expected_tasks = set(config["expected_task_types"])
    client = s3_client(max_pool_connections=4)
    uri = str(output["path"])
    digest = hashlib.sha256()
    byte_count = int(output["bytes"])
    line_count = 0
    counters: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    complement_task_counts: Counter[str] = Counter()
    eligible_task_counts: Counter[str] = Counter()
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        task: [] for task in expected_tasks
    }
    source_lines = iter_s3_object_lines_by_range(
        client,
        output,
        digest=digest,
        range_size_bytes=int(config["range_size_mb"]) * 1024 * 1024,
    )
    for line_count, raw_line in enumerate(source_lines, start=1):
        payload = raw_line.rstrip(b"\r\n")
        if not payload:
            raise ValueError(f"empty row at {uri}:{line_count}")
        global_index = global_start + line_count - 1
        counters["source_rows_scanned"] += 1
        if bit_is_set(bitset, global_index):
            counters["training_rows_excluded"] += 1
            continue
        counters["complement_rows"] += 1
        try:
            row = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON at {uri}:{line_count}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSON object expected at {uri}:{line_count}")
        task_type = row.get("task_type")
        if not isinstance(task_type, str) or task_type not in expected_tasks:
            raise ValueError(
                f"unexpected task_type={task_type!r} at {uri}:{line_count}"
            )
        complement_task_counts[task_type] += 1
        status = str(row.get("status") or "<missing>")
        status_counts[status] += 1
        if status != ELIGIBLE_STATUS:
            counters["status_ineligible_rows"] += 1
            continue
        context = f"{uri}:{line_count}"
        sample_id, _instruction = validate_eligible_row(row, context=context)
        eligible_task_counts[task_type] += 1
        counters["eligible_rows"] += 1
        score = selection_hash(
            seed=int(config["benchmark_seed"]),
            task_type=task_type,
            sample_id=sample_id,
            global_row_index=global_index,
        )
        candidate = {
            "global_row_index": global_index,
            "line_number": line_count,
            "part_index": part_index,
            "row_json": payload.decode("utf-8"),
            "row_sha256": hashlib.sha256(payload).hexdigest(),
            "sample_id": sample_id,
            "selection_hash": score,
            "status": status,
            "task_type": task_type,
        }
        push_bottom_k(heaps[task_type], candidate, samples_per_task)
    if line_count != expected_rows:
        raise ValueError(
            f"source row count changed for {uri}: {line_count} != {expected_rows}"
        )
    expected_bytes = int(output["bytes"])
    if byte_count != expected_bytes:
        raise ValueError(
            f"source byte count changed for {uri}: {byte_count} != {expected_bytes}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != output["sha256"]:
        raise ValueError(
            f"source SHA-256 changed for {uri}: {actual_sha256} != {output['sha256']}"
        )
    if counters["training_rows_excluded"] != int(plan_part["sampled_rows"]):
        raise ValueError(f"training bitset count changed for source part {part_index}")
    if counters["complement_rows"] != expected_rows - int(plan_part["sampled_rows"]):
        raise ValueError(f"complement row count changed for source part {part_index}")

    candidate_rows = [
        candidate
        for task in sorted(heaps)
        for candidate in ordered_heap(heaps[task])
    ]
    candidate_path = Path(local_dir) / f"part-{part_index:05d}.candidates.jsonl"
    candidate_path.write_bytes(canonical_jsonl(candidate_rows))
    return {
        "candidate_path": str(candidate_path),
        "candidate_rows": len(candidate_rows),
        "complement_task_counts": dict(sorted(complement_task_counts.items())),
        "counters": dict(sorted(counters.items())),
        "eligible_task_counts": dict(sorted(eligible_task_counts.items())),
        "part_index": part_index,
        "source": {
            "bytes": byte_count,
            "global_row_start": global_start,
            "part_index": part_index,
            "path": uri,
            "rows": line_count,
            "sha256": actual_sha256,
            "training_rows": int(plan_part["sampled_rows"]),
        },
        "status_counts": dict(sorted(status_counts.items())),
    }


def _rank_relative(rank: int, world_size: int) -> str:
    return f"workers/rank-{rank:05d}-of-{world_size:05d}.manifest.json"


def _rank_candidates_relative(rank: int, world_size: int) -> str:
    return f"workers/rank-{rank:05d}-of-{world_size:05d}.candidates.jsonl"


def _static_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "benchmark_seed": args.benchmark_seed,
        "eligible_status": ELIGIBLE_STATUS,
        "expected_population_rows": args.expected_population_rows,
        "expected_training_rows": args.expected_training_rows,
        "format_version": FORMAT_VERSION,
        "samples_per_task": args.samples_per_task,
        "source_manifest_sha256": args.source_manifest_sha256,
        "training_sample_seed": args.training_sample_seed,
        "training_sampling_bitset_sha256": args.training_sampling_bitset_sha256,
        "training_sampling_plan_sha256": args.training_sampling_plan_sha256,
        "world_size": args.world_size,
    }


def _rank_manifest_is_compatible(
    value: Mapping[str, Any], *, args: argparse.Namespace, rank: int
) -> bool:
    return (
        value.get("complete") is True
        and int(value.get("rank", -1)) == rank
        and value.get("contract") == _static_contract(args)
        and isinstance(value.get("candidates"), Mapping)
        and isinstance(value.get("source_parts"), list)
    )


def run_worker(args: argparse.Namespace) -> None:
    client = s3_client()
    output_prefix = normalize_s3_prefix(args.output)
    rank_uri = join_s3(output_prefix, _rank_relative(args.rank, args.world_size))
    existing = maybe_get_json(client, rank_uri)
    if existing is not None:
        if _rank_manifest_is_compatible(existing, args=args, rank=args.rank):
            print(json.dumps({"status": "rank_already_complete", "rank": args.rank}))
            return
        raise ValueError(f"rank {args.rank} has a conflicting completion manifest")

    frozen = load_frozen_contract(client, args)
    outputs = frozen["source_manifest"]["outputs"]
    parts = frozen["sampling_plan"]["source_parts"]
    if args.world_size > len(outputs):
        raise ValueError("world_size cannot exceed source shard count")
    assigned = [
        (output, parts[int(output["part_index"])])
        for output in outputs
        if int(output["part_index"]) % args.world_size == args.rank
    ]
    if not assigned:
        raise ValueError(f"rank {args.rank} has no assigned source shards")
    processes = min(args.processes or (os.cpu_count() or 8), len(assigned))
    scan_config = {
        "benchmark_seed": args.benchmark_seed,
        "expected_task_types": EXPECTED_TASK_TYPES,
        "range_size_mb": args.range_size_mb,
        "samples_per_task": args.samples_per_task,
    }
    work_parent = Path(args.work_dir)
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"image-edit-pe-eval-{args.rank:05d}-", dir=work_parent
    ) as temporary:
        print(
            json.dumps(
                {
                    "event": "rank_scan_start",
                    "parts": len(assigned),
                    "processes": processes,
                    "rank": args.rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        summaries: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=processes,
            initializer=_init_scan_process,
            initargs=(frozen["selection_bitset"], scan_config),
        ) as pool:
            futures = {
                pool.submit(_scan_source_part, output, part, temporary): int(
                    output["part_index"]
                )
                for output, part in assigned
            }
            for future in concurrent.futures.as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(
                    json.dumps(
                        {
                            "event": "source_part_complete",
                            "part_index": summary["part_index"],
                            "rank": args.rank,
                            "rows": summary["counters"]["source_rows_scanned"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        summaries.sort(key=lambda item: int(item["part_index"]))
        aggregate_counters: Counter[str] = Counter()
        aggregate_status: Counter[str] = Counter()
        aggregate_complement_tasks: Counter[str] = Counter()
        aggregate_eligible_tasks: Counter[str] = Counter()
        heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
            task: [] for task in EXPECTED_TASK_TYPES
        }
        for summary in summaries:
            aggregate_counters.update(
                {key: int(value) for key, value in summary["counters"].items()}
            )
            aggregate_status.update(
                {key: int(value) for key, value in summary["status_counts"].items()}
            )
            aggregate_complement_tasks.update(
                {
                    key: int(value)
                    for key, value in summary["complement_task_counts"].items()
                }
            )
            aggregate_eligible_tasks.update(
                {key: int(value) for key, value in summary["eligible_task_counts"].items()}
            )
            candidate_path = Path(summary["candidate_path"])
            with candidate_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    candidate = json.loads(line)
                    push_bottom_k(
                        heaps[candidate["task_type"]], candidate, args.samples_per_task
                    )
        candidates = [
            candidate
            for task in EXPECTED_TASK_TYPES
            for candidate in ordered_heap(heaps[task])
        ]
        candidate_payload = canonical_jsonl(candidates)
        candidate_descriptor = put_bytes_if_absent(
            client,
            join_s3(
                output_prefix, _rank_candidates_relative(args.rank, args.world_size)
            ),
            candidate_payload,
            content_type="application/x-ndjson",
        )
        rank_manifest = {
            "candidates": candidate_descriptor,
            "complete": True,
            "complement_task_counts": dict(sorted(aggregate_complement_tasks.items())),
            "contract": _static_contract(args),
            "counters": dict(sorted(aggregate_counters.items())),
            "eligible_task_counts": dict(sorted(aggregate_eligible_tasks.items())),
            "rank": args.rank,
            "source_parts": [summary["source"] for summary in summaries],
            "status_counts": dict(sorted(aggregate_status.items())),
        }
        descriptor = put_json_if_absent(client, rank_uri, rank_manifest)
    print(
        json.dumps(
            {
                "candidate_rows": len(candidates),
                "rank": args.rank,
                "rank_manifest": descriptor,
                "status": "rank_complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_rank_manifest(
    client: Any, *, args: argparse.Namespace, rank: int
) -> dict[str, Any] | None:
    uri = join_s3(args.output, _rank_relative(rank, args.world_size))
    value = maybe_get_json(client, uri)
    if value is None:
        return None
    if not _rank_manifest_is_compatible(value, args=args, rank=rank):
        raise ValueError(f"rank {rank} completion manifest is incompatible")
    return value


def _resolve_target_sizes(
    row_payloads: Sequence[bytes], *, threads: int, max_inflight: int
) -> list[Any]:
    from scripts.build_image_edit_pe_target_size_sidecar import resolve_in_order

    return list(
        resolve_in_order(row_payloads, threads=threads, max_inflight=max_inflight)
    )


def _validate_success(value: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return (
        value.get("complete") is True
        and value.get("contract") == _static_contract(args)
        and int(value.get("rows", -1))
        == len(EXPECTED_TASK_TYPES) * args.samples_per_task
    )


def run_finalize(args: argparse.Namespace) -> None:
    client = s3_client(max_pool_connections=max(args.target_size_threads * 2, 128))
    output_prefix = normalize_s3_prefix(args.output)
    success_uri = join_s3(output_prefix, "_SUCCESS")
    existing_success = maybe_get_json(client, success_uri)
    if existing_success is not None:
        if _validate_success(existing_success, args):
            print(json.dumps({"status": "already_complete"}), flush=True)
            return
        raise ValueError("output prefix contains a conflicting _SUCCESS marker")
    frozen = load_frozen_contract(client, args)

    rank_manifests: list[dict[str, Any]] = []
    covered_parts: list[int] = []
    aggregate_counters: Counter[str] = Counter()
    aggregate_status: Counter[str] = Counter()
    aggregate_complement_tasks: Counter[str] = Counter()
    aggregate_eligible_tasks: Counter[str] = Counter()
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        task: [] for task in EXPECTED_TASK_TYPES
    }
    for rank in range(args.world_size):
        manifest = _load_rank_manifest(client, args=args, rank=rank)
        if manifest is None:
            raise RuntimeError(f"rank {rank} is incomplete")
        rank_manifests.append(manifest)
        aggregate_counters.update(
            {key: int(value) for key, value in manifest["counters"].items()}
        )
        aggregate_status.update(
            {key: int(value) for key, value in manifest["status_counts"].items()}
        )
        aggregate_complement_tasks.update(
            {
                key: int(value)
                for key, value in manifest["complement_task_counts"].items()
            }
        )
        aggregate_eligible_tasks.update(
            {key: int(value) for key, value in manifest["eligible_task_counts"].items()}
        )
        covered_parts.extend(int(item["part_index"]) for item in manifest["source_parts"])
        candidate_payload = get_object_bytes(client, manifest["candidates"])
        for line_number, raw_line in enumerate(candidate_payload.splitlines(), start=1):
            candidate = json.loads(raw_line)
            if not isinstance(candidate, dict):
                raise ValueError(f"rank {rank} candidate {line_number} is not an object")
            task_type = candidate.get("task_type")
            if task_type not in heaps:
                raise ValueError(f"rank {rank} candidate has unexpected task type")
            push_bottom_k(heaps[task_type], candidate, args.samples_per_task)

    outputs = frozen["source_manifest"]["outputs"]
    if sorted(covered_parts) != list(range(len(outputs))):
        raise ValueError("source part coverage is incomplete or duplicated")
    if aggregate_counters["source_rows_scanned"] != args.expected_population_rows:
        raise ValueError("full source row scan count mismatch")
    if aggregate_counters["training_rows_excluded"] != args.expected_training_rows:
        raise ValueError("training exclusion row count mismatch")
    expected_complement = args.expected_population_rows - args.expected_training_rows
    if aggregate_counters["complement_rows"] != expected_complement:
        raise ValueError("training complement row count mismatch")
    if sum(aggregate_status.values()) != expected_complement:
        raise ValueError("complement status accounting mismatch")
    if sum(aggregate_complement_tasks.values()) != expected_complement:
        raise ValueError("complement task accounting mismatch")
    if set(aggregate_complement_tasks) != set(EXPECTED_TASK_TYPES):
        raise ValueError("complement does not contain exactly the expected task types")
    if set(aggregate_eligible_tasks) != set(EXPECTED_TASK_TYPES):
        raise ValueError("validated complement does not cover every expected task type")
    insufficient = {
        task: aggregate_eligible_tasks[task]
        for task in EXPECTED_TASK_TYPES
        if aggregate_eligible_tasks[task] < args.samples_per_task
    }
    if insufficient:
        raise ValueError(f"task types have fewer than requested validated rows: {insufficient}")

    selected = [
        candidate
        for task in EXPECTED_TASK_TYPES
        for candidate in ordered_heap(heaps[task])
    ]
    expected_selected = len(EXPECTED_TASK_TYPES) * args.samples_per_task
    if len(selected) != expected_selected:
        raise ValueError(f"selected {len(selected)} rows, expected {expected_selected}")
    selected_ids: set[str] = set()
    selected_indexes: set[int] = set()
    selected_task_counts: Counter[str] = Counter()
    row_payloads: list[bytes] = []
    parsed_rows: list[dict[str, Any]] = []
    projected_labels: list[str] = []
    for candidate in selected:
        sample_id = str(candidate["sample_id"])
        global_index = int(candidate["global_row_index"])
        task_type = str(candidate["task_type"])
        payload = str(candidate["row_json"]).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != candidate["row_sha256"]:
            raise ValueError(f"candidate source-row SHA-256 mismatch: {sample_id}")
        if bit_is_set(frozen["selection_bitset"], global_index):
            raise ValueError(f"selected benchmark row is in training bitset: {sample_id}")
        if sample_id in selected_ids or global_index in selected_indexes:
            raise ValueError(f"duplicate selected benchmark row: {sample_id}")
        row = json.loads(payload)
        if not isinstance(row, dict):
            raise ValueError(f"selected source row is not an object: {sample_id}")
        validated_id, _instruction = validate_eligible_row(
            row, context=f"selected sample_id={sample_id}"
        )
        if (
            validated_id != sample_id
            or row.get("task_type") != task_type
            or selection_hash(
                seed=args.benchmark_seed,
                task_type=task_type,
                sample_id=sample_id,
                global_row_index=global_index,
            )
            != candidate["selection_hash"]
        ):
            raise ValueError(f"candidate identity mismatch: {sample_id}")
        projected = canonical_sparse_edit_target(row["final_json"])
        selected_ids.add(sample_id)
        selected_indexes.add(global_index)
        selected_task_counts[task_type] += 1
        row_payloads.append(payload)
        parsed_rows.append(row)
        projected_labels.append(projected)
    if any(
        selected_task_counts[task] != args.samples_per_task
        for task in EXPECTED_TASK_TYPES
    ):
        raise ValueError("selected task counts are not exactly balanced")

    print(
        json.dumps(
            {
                "event": "target_size_resolution_start",
                "rows": len(row_payloads),
                "threads": args.target_size_threads,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    target_sizes = _resolve_target_sizes(
        row_payloads,
        threads=args.target_size_threads,
        max_inflight=args.target_size_max_inflight,
    )
    if len(target_sizes) != len(selected):
        raise ValueError("Target-size result count mismatch")

    source_rows_payload = b"".join(payload + b"\n" for payload in row_payloads)
    index_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    task_raw_payloads: dict[str, list[bytes]] = {
        task: [] for task in EXPECTED_TASK_TYPES
    }
    task_indexes: Counter[str] = Counter()
    target_size_methods: Counter[str] = Counter()
    for candidate, row, payload, label, target_size in zip(
        selected,
        parsed_rows,
        row_payloads,
        projected_labels,
        target_sizes,
        strict=True,
    ):
        task = str(candidate["task_type"])
        within_task = task_indexes[task]
        task_indexes[task] += 1
        if target_size.row_sha256.hex() != candidate["row_sha256"]:
            raise ValueError(f"Target size row binding mismatch: {candidate['sample_id']}")
        target_size_methods[str(target_size.method)] += 1
        index_row = {
            "global_row_index": int(candidate["global_row_index"]),
            "line_number": int(candidate["line_number"]),
            "part_index": int(candidate["part_index"]),
            "sample_id": str(candidate["sample_id"]),
            "selection_hash": str(candidate["selection_hash"]),
            "selection_index_within_task": within_task,
            "source_row_sha256": str(candidate["row_sha256"]),
            "status": ELIGIBLE_STATUS,
            "task_type": task,
            "training_bitset_member": False,
        }
        index_rows.append(index_row)
        task_raw_payloads[task].append(payload + b"\n")
        evaluation_rows.append(
            {
                "assistant_target_json": label,
                "instruction": row["instruction"],
                "label_projection_version": LABEL_PROJECTION_VERSION,
                "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
                "prompt_version": PROMPT_VERSION,
                "sample_id": row["sample_id"],
                "selection": index_row,
                "source_s3_uri": row["source_s3_uri"],
                "target_canvas_size": {
                    "height": int(target_size.height),
                    "width": int(target_size.width),
                },
                "target_s3_uri": row["target_s3_uri"],
                "target_size_provenance": {
                    "bytes_read": int(target_size.bytes_read),
                    "encoded_format": str(target_size.encoded_format),
                    "method": str(target_size.method),
                    "object_bytes": int(target_size.object_bytes),
                    "target_identity": str(target_size.target_identity),
                },
                "task_type": task,
            }
        )

    output_descriptors: dict[str, Any] = {}
    output_descriptors["source_rows"] = put_bytes_if_absent(
        client,
        join_s3(output_prefix, "source_rows.jsonl"),
        source_rows_payload,
        content_type="application/x-ndjson",
    )
    output_descriptors["benchmark_index"] = put_bytes_if_absent(
        client,
        join_s3(output_prefix, "benchmark_index.jsonl"),
        canonical_jsonl(index_rows),
        content_type="application/x-ndjson",
    )
    output_descriptors["evaluation_manifest"] = put_bytes_if_absent(
        client,
        join_s3(output_prefix, "evaluation_manifest.jsonl"),
        canonical_jsonl(evaluation_rows),
        content_type="application/x-ndjson",
    )
    task_outputs: list[dict[str, Any]] = []
    for task in EXPECTED_TASK_TYPES:
        descriptor = put_bytes_if_absent(
            client,
            join_s3(output_prefix, f"tasks/{task}.jsonl"),
            b"".join(task_raw_payloads[task]),
            content_type="application/x-ndjson",
        )
        task_outputs.append({"rows": args.samples_per_task, "task_type": task, **descriptor})
    output_descriptors["tasks"] = task_outputs

    checks = {
        "all_source_shards_hash_verified": True,
        "all_target_sizes_bound_to_source_row_sha256": True,
        "all_tasks_covered": set(selected_task_counts) == set(EXPECTED_TASK_TYPES),
        "annotation_contract_pinned": True,
        "exactly_samples_per_task": all(
            selected_task_counts[task] == args.samples_per_task
            for task in EXPECTED_TASK_TYPES
        ),
        "full_global_row_order_scanned_once": sorted(covered_parts)
        == list(range(len(outputs))),
        "labels_use_training_projection": True,
        "selected_global_rows_unique": len(selected_indexes) == len(selected),
        "selected_sample_ids_unique": len(selected_ids) == len(selected),
        "training_intersection_rows": 0,
        "training_selection_bitset_hash_verified": True,
    }
    boolean_checks = {
        key: value for key, value in checks.items() if key != "training_intersection_rows"
    }
    if (
        not all(value is True for value in boolean_checks.values())
        or checks["training_intersection_rows"] != 0
    ):
        raise AssertionError(f"benchmark verification failed: {checks}")
    verification = {
        "checks": checks,
        "complete": True,
        "complement_rows": expected_complement,
        "complement_status_counts": dict(sorted(aggregate_status.items())),
        "complement_task_counts": dict(sorted(aggregate_complement_tasks.items())),
        "eligible_status": ELIGIBLE_STATUS,
        "eligible_task_counts": dict(sorted(aggregate_eligible_tasks.items())),
        "selected_rows": len(selected),
        "selected_task_counts": dict(sorted(selected_task_counts.items())),
        "source_rows": args.expected_population_rows,
        "target_size_methods": dict(sorted(target_size_methods.items())),
        "training_rows_excluded": args.expected_training_rows,
    }
    verification_descriptor = put_json_if_absent(
        client, join_s3(output_prefix, "verification.json"), verification
    )
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "complete": True,
        "contract": _static_contract(args),
        "evaluation_contract": {
            "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
            "label_projection_version": LABEL_PROJECTION_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "prompt_version": PROMPT_VERSION,
            "target_schema": TARGET_SCHEMA,
        },
        "format_version": FORMAT_VERSION,
        "outputs": output_descriptors,
        "selection": {
            "authoritative_stratum_field": "task_type",
            "eligible_status": ELIGIBLE_STATUS,
            "expected_task_types": list(EXPECTED_TASK_TYPES),
            "method": "per-task deterministic SHA-256 bottom-k without replacement",
            "rows": len(selected),
            "samples_per_task": args.samples_per_task,
            "seed": args.benchmark_seed,
        },
        "source": {
            "manifest_sha256": args.source_manifest_sha256,
            "manifest_uri": frozen["source_manifest_uri"],
            "prefix": normalize_s3_prefix(args.source),
            "rows": args.expected_population_rows,
            "shards": len(outputs),
        },
        "training_exclusion": {
            "algorithm": SAMPLING_ALGORITHM,
            "rows": args.expected_training_rows,
            "seed": args.training_sample_seed,
            "selection_bitset_sha256": args.training_sampling_bitset_sha256,
            "selection_plan_sha256": args.training_sampling_plan_sha256,
            "selection_plan_uri": args.training_sampling_plan,
        },
        "verification": verification_descriptor,
        "worker_manifests": len(rank_manifests),
    }
    manifest_descriptor = put_json_if_absent(
        client, join_s3(output_prefix, "manifest.json"), manifest
    )
    success = {
        "complete": True,
        "contract": _static_contract(args),
        "manifest": manifest_descriptor,
        "rows": len(selected),
        "task_types": len(EXPECTED_TASK_TYPES),
        "training_intersection_rows": 0,
        "verification": verification_descriptor,
    }
    success_descriptor = put_json_if_absent(client, success_uri, success)
    print(
        json.dumps(
            {
                "output": output_prefix,
                "rows": len(selected),
                "status": "complete",
                "success": success_descriptor,
                "task_types": len(EXPECTED_TASK_TYPES),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_wait_finalize(args: argparse.Namespace) -> None:
    client = s3_client()
    deadline = time.monotonic() + args.wait_timeout_seconds
    while True:
        missing = [
            rank
            for rank in range(args.world_size)
            if _load_rank_manifest(client, args=args, rank=rank) is None
        ]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ranks: {missing}")
        print(
            json.dumps(
                {
                    "complete_ranks": args.world_size - len(missing),
                    "event": "waiting_for_ranks",
                    "missing_ranks": missing,
                    "world_size": args.world_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(min(args.poll_seconds, max(deadline - time.monotonic(), 1)))
    run_finalize(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True)
    common.add_argument("--source-manifest-sha256", required=True)
    common.add_argument("--training-sampling-plan", required=True)
    common.add_argument("--training-sampling-plan-sha256", required=True)
    common.add_argument("--training-sampling-bitset-sha256", required=True)
    common.add_argument("--training-sample-seed", type=int, required=True)
    common.add_argument("--expected-population-rows", type=int, required=True)
    common.add_argument("--expected-training-rows", type=int, required=True)
    common.add_argument("--output", required=True)
    common.add_argument("--world-size", type=int, required=True)
    common.add_argument("--samples-per-task", type=int, default=40)
    common.add_argument("--benchmark-seed", type=int, default=20260828)
    common.add_argument("--range-size-mb", type=int, default=32)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", parents=[common])
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--processes", type=int, default=0)
    worker.add_argument("--work-dir", default="/mnt/ephemeral/pcb-v2-9c4d-17000/data")
    worker.set_defaults(func=run_worker)

    finalize = subparsers.add_parser("finalize", parents=[common])
    finalize.add_argument("--target-size-threads", type=int, default=64)
    finalize.add_argument("--target-size-max-inflight", type=int, default=256)
    finalize.set_defaults(func=run_finalize)

    wait_finalize = subparsers.add_parser("wait-finalize", parents=[common])
    wait_finalize.add_argument("--target-size-threads", type=int, default=64)
    wait_finalize.add_argument("--target-size-max-inflight", type=int, default=256)
    wait_finalize.add_argument("--wait-timeout-seconds", type=int, default=172800)
    wait_finalize.add_argument("--poll-seconds", type=int, default=30)
    wait_finalize.set_defaults(func=run_wait_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for value, field in (
        (args.source_manifest_sha256, "source manifest SHA-256"),
        (args.training_sampling_plan_sha256, "training sampling plan SHA-256"),
        (args.training_sampling_bitset_sha256, "training sampling bitset SHA-256"),
    ):
        validate_sha256(value, field)
    normalize_s3_prefix(args.source)
    normalize_s3_prefix(args.output)
    parse_s3_uri(args.training_sampling_plan)
    for field in (
        "benchmark_seed",
        "expected_population_rows",
        "expected_training_rows",
        "range_size_mb",
        "samples_per_task",
        "training_sample_seed",
        "world_size",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.expected_training_rows >= args.expected_population_rows:
        raise ValueError("training rows must be smaller than population rows")
    if hasattr(args, "rank") and not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if hasattr(args, "processes") and args.processes < 0:
        raise ValueError("processes must be non-negative")
    if hasattr(args, "target_size_threads"):
        if args.target_size_threads <= 0:
            raise ValueError("target-size threads must be positive")
        if args.target_size_max_inflight < args.target_size_threads:
            raise ValueError("target-size max-inflight must be >= threads")
    if hasattr(args, "wait_timeout_seconds") and args.wait_timeout_seconds <= 0:
        raise ValueError("wait timeout must be positive")
    if hasattr(args, "poll_seconds") and args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
