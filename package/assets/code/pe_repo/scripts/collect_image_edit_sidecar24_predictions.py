#!/usr/bin/env python3
"""Validate and merge a completed sidecar24 selection into viewer JSONL.

This tool is deliberately local-only: first mirror the completed selection
root (``manifest.json`` plus ``datasets/**/shards/*.jsonl``), then point
``--selection-root`` at that directory.  It verifies the root contract, every
shard's byte count and SHA-256, canonical row identity, and provenance before
atomically writing a ``predictions.snapshot.jsonl`` for
``demos/inference_result_viewer``.

Source ``sample_id`` values are preserved labels and may repeat: sidecar24's
planner uses an exact multiset join.  ``input_index`` plus frozen source-row
provenance, rather than ``sample_id``, defines a unique prediction record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


RESULT_FORMAT = "image_edit_full_json_sidecar24_results_v1"
PARTITION_FORMAT = "image_edit_sidecar24_partition_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CollectionError(ValueError):
    """The completed manifest or one of its JSONL shards is inconsistent."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_without_duplicates(payload: bytes, *, location: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CollectionError(f"{location}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=object_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"{location}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"{location}: expected a JSON object")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CollectionError(f"root manifest does not exist: {path}")
    return _json_without_duplicates(path.read_bytes(), location=str(path))


def _required_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CollectionError(f"{location} must be an integer >= {minimum}")
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_records: int,
    expected_partition_count: int | None,
    expected_partition_index: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("format_version") != RESULT_FORMAT or manifest.get("status") != "complete":
        raise CollectionError("root manifest is not a complete sidecar24 result")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise CollectionError("root manifest has no contract object")
    partition = contract.get("data_partition")
    if not isinstance(partition, Mapping) or partition.get("format_version") != PARTITION_FORMAT:
        raise CollectionError("root manifest has no supported data_partition contract")

    selected_rows = _required_int(
        partition.get("selected_rows"), location="contract.data_partition.selected_rows", minimum=1
    )
    if selected_rows != expected_records:
        raise CollectionError(
            f"partition selected_rows={selected_rows}, expected exactly {expected_records}"
        )
    if partition.get("case_limit") != expected_records:
        raise CollectionError(
            f"partition case_limit={partition.get('case_limit')!r}, expected {expected_records}"
        )
    if contract.get("expected_rows") != expected_records:
        raise CollectionError("root contract expected_rows differs from requested record count")
    if expected_partition_count is not None and partition.get("partition_count") != expected_partition_count:
        raise CollectionError(
            f"partition_count={partition.get('partition_count')!r}, expected {expected_partition_count}"
        )
    if expected_partition_index is not None and partition.get("partition_index") != expected_partition_index:
        raise CollectionError(
            f"partition_index={partition.get('partition_index')!r}, expected {expected_partition_index}"
        )
    rank_rows = partition.get("rank_row_counts")
    if not isinstance(rank_rows, list) or any(type(value) is not int or value < 0 for value in rank_rows):
        raise CollectionError("partition rank_row_counts is invalid")
    if sum(rank_rows) != expected_records:
        raise CollectionError("partition rank_row_counts do not sum to selected_rows")
    selection_sha256 = partition.get("selection_sha256")
    if not isinstance(selection_sha256, str) or SHA256_RE.fullmatch(selection_sha256) is None:
        raise CollectionError("partition selection_sha256 is invalid")
    all_part_rows = partition.get("all_partition_row_counts")
    catalog_rows = partition.get("catalog_rows")
    if (
        not isinstance(all_part_rows, list)
        or len(all_part_rows) != partition.get("partition_count")
        or any(type(value) is not int or value < 0 for value in all_part_rows)
        or sum(all_part_rows) != catalog_rows
    ):
        raise CollectionError("all_partition_row_counts do not cover catalog_rows")

    totals = manifest.get("totals")
    if not isinstance(totals, Mapping) or totals.get("rows") != expected_records:
        raise CollectionError("root totals.rows differs from requested record count")
    if totals.get("inference_errors") != 0:
        raise CollectionError("completed result declares inference errors")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CollectionError("root manifest has no shards")
    if totals.get("units") != len(shards) or contract.get("expected_units") != len(shards):
        raise CollectionError("root shard count differs from contract/totals units")
    return [dict(shard) for shard in shards], dict(totals)


def shard_relative_path(shard: Mapping[str, Any]) -> Path:
    dataset_id = shard.get("dataset_id")
    unit_index = shard.get("dataset_unit_index")
    if not isinstance(dataset_id, str) or DATASET_ID_RE.fullmatch(dataset_id) is None:
        raise CollectionError(f"invalid shard dataset_id: {dataset_id!r}")
    if type(unit_index) is not int or unit_index < 0:
        raise CollectionError(f"invalid shard dataset_unit_index: {unit_index!r}")
    return Path("datasets") / dataset_id / "shards" / f"unit-{unit_index:06d}.jsonl"


def _load_shard(
    root: Path,
    shard: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = shard.get("output")
    if not isinstance(output, Mapping):
        raise CollectionError("manifest shard has no output object")
    relative = shard_relative_path(shard)
    uri = output.get("uri")
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise CollectionError(f"{relative}: output.uri is not an S3 URI")
    uri_path = urlparse(uri).path.lstrip("/")
    if not uri_path.endswith(relative.as_posix()):
        raise CollectionError(f"{relative}: output.uri does not match shard identity")
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root != path and resolved_root not in path.parents:
        raise CollectionError(f"{relative}: resolved shard path escapes selection root")
    if not path.is_file():
        raise CollectionError(f"local shard is missing: {path}")
    payload = path.read_bytes()
    expected_bytes = _required_int(output.get("bytes"), location=f"{relative}.bytes", minimum=1)
    if len(payload) != expected_bytes:
        raise CollectionError(f"{relative}: byte count differs from root manifest")
    expected_sha = output.get("sha256")
    if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
        raise CollectionError(f"{relative}: root manifest SHA-256 is invalid")
    if sha256_bytes(payload) != expected_sha:
        raise CollectionError(f"{relative}: SHA-256 differs from root manifest")
    if not payload or not payload.endswith(b"\n"):
        raise CollectionError(f"{relative}: JSONL must be non-empty and newline terminated")

    rows = [
        _json_without_duplicates(line, location=f"{path}:{line_number}")
        for line_number, line in enumerate(payload.splitlines(), start=1)
    ]
    expected_rows = _required_int(output.get("rows"), location=f"{relative}.rows", minimum=1)
    if len(rows) != expected_rows:
        raise CollectionError(f"{relative}: decoded rows differ from root manifest")
    start = _required_int(shard.get("global_row_start"), location=f"{relative}.global_row_start")
    end = _required_int(
        shard.get("global_row_end_exclusive"),
        location=f"{relative}.global_row_end_exclusive",
        minimum=1,
    )
    if end - start != expected_rows:
        raise CollectionError(f"{relative}: shard interval length differs from output rows")
    if [row.get("input_index") for row in rows] != list(range(start, end)):
        raise CollectionError(f"{relative}: input_index coverage is not its declared interval")

    global_unit_index = shard.get("global_unit_index")
    dataset_unit_index = shard.get("dataset_unit_index")
    dataset_id = shard.get("dataset_id")
    dataset_row_start = _required_int(
        shard.get("dataset_row_start"), location=f"{relative}.dataset_row_start"
    )
    source_identity = shard.get("source")
    if not isinstance(source_identity, Mapping):
        raise CollectionError(f"{relative}: source identity is missing")
    source_bucket = source_identity.get("bucket")
    source_key = source_identity.get("key")
    source_object_index = source_identity.get("object_index")
    if not isinstance(source_bucket, str) or not source_bucket:
        raise CollectionError(f"{relative}: source bucket is invalid")
    if not isinstance(source_key, str) or not source_key:
        raise CollectionError(f"{relative}: source key is invalid")
    if type(source_object_index) is not int or source_object_index < 0:
        raise CollectionError(f"{relative}: source object_index is invalid")
    source_parquet_uri = f"s3://{source_bucket}/{source_key}"

    raw_ranges = shard.get("local_row_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise CollectionError(f"{relative}: local_row_ranges is invalid")
    local_rows: list[int] = []
    previous_end = -1
    for range_index, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise CollectionError(f"{relative}: local_row_ranges[{range_index}] is invalid")
        range_start = _required_int(
            raw_range[0], location=f"{relative}.local_row_ranges[{range_index}][0]"
        )
        range_end = _required_int(
            raw_range[1], location=f"{relative}.local_row_ranges[{range_index}][1]", minimum=1
        )
        if range_start >= range_end or range_start < previous_end:
            raise CollectionError(f"{relative}: local_row_ranges is not ordered and disjoint")
        local_rows.extend(range(range_start, range_end))
        previous_end = range_end
    if len(local_rows) != expected_rows:
        raise CollectionError(f"{relative}: local_row_ranges length differs from output rows")

    for row_offset, (row, local_row) in enumerate(zip(rows, local_rows, strict=True)):
        row_number = row_offset + 1
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise CollectionError(f"{relative}:{row_number}: sample_id is missing")
        if not isinstance(row.get("reference_image"), str) or not row["reference_image"]:
            raise CollectionError(f"{relative}:{row_number}: reference_image is missing")
        if not isinstance(row.get("target_image"), str) or not row["target_image"]:
            raise CollectionError(f"{relative}:{row_number}: target_image is missing")
        source = row.get("dataset_source")
        expected_source = {
            "dataset_id": dataset_id,
            "dataset_unit_index": dataset_unit_index,
            "global_unit_index": global_unit_index,
            "dataset_row_index": dataset_row_start + row_offset,
            "global_input_index": row.get("input_index"),
            "parquet_s3_uri": source_parquet_uri,
            "parquet_object_index": source_object_index,
            "source_parquet_local_row": local_row,
        }
        if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in expected_source.items()):
            raise CollectionError(f"{relative}:{row_number}: dataset_source differs from shard")

    counts = output.get("counts")
    if not isinstance(counts, Mapping):
        raise CollectionError(f"{relative}: output counts are missing")
    actual_counts = {
        "rows": len(rows),
        "json_valid": sum(row.get("json_valid") is True for row in rows),
        "schema_valid": sum(row.get("schema_valid") is True for row in rows),
        "failed": sum(row.get("schema_valid") is not True for row in rows),
        "inference_errors": sum(bool(row.get("inference_error")) for row in rows),
    }
    if dict(counts) != actual_counts:
        raise CollectionError(f"{relative}: decoded validity counts differ from root manifest")
    return rows, actual_counts


def collect(
    selection_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_records: int = 100,
    expected_partition_count: int | None = 16,
    expected_partition_index: int | None = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shards, root_totals = validate_manifest(
        manifest,
        expected_records=expected_records,
        expected_partition_count=expected_partition_count,
        expected_partition_index=expected_partition_index,
    )
    indices = [shard.get("global_unit_index") for shard in shards]
    if any(type(value) is not int or value < 0 for value in indices):
        raise CollectionError("manifest contains invalid global_unit_index values")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise CollectionError("manifest shards are not uniquely ordered by global_unit_index")

    merged: list[dict[str, Any]] = []
    aggregate = {
        "rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "failed": 0,
        "inference_errors": 0,
        "bytes": 0,
    }
    for shard in shards:
        rows, counts = _load_shard(selection_root, shard)
        merged.extend(rows)
        for key, value in counts.items():
            aggregate[key] += value
        aggregate["bytes"] += int(shard["output"]["bytes"])
    aggregate_with_units = {"units": len(shards), **aggregate}
    if aggregate_with_units != root_totals:
        raise CollectionError("decoded aggregate totals differ from root manifest")

    merged.sort(key=lambda row: int(row["input_index"]))
    input_indices = [row["input_index"] for row in merged]
    sample_ids = [row["sample_id"] for row in merged]
    provenance = [
        (
            row["dataset_source"]["dataset_id"],
            row["dataset_source"]["parquet_s3_uri"],
            row["dataset_source"]["source_parquet_local_row"],
        )
        for row in merged
    ]
    if len(merged) != expected_records:
        raise CollectionError(f"merged rows={len(merged)}, expected exactly {expected_records}")
    if len(set(input_indices)) != len(input_indices):
        raise CollectionError("merged predictions contain duplicate input_index")
    if len(set(provenance)) != len(provenance):
        raise CollectionError("merged predictions contain duplicate source-row provenance")

    distinct_sample_ids = len(set(sample_ids))

    audit = {
        "format_version": "image_edit_sidecar24_prediction_snapshot_v1",
        "records": len(merged),
        "input_index_min": min(input_indices),
        "input_index_max": max(input_indices),
        "sample_ids_unique": distinct_sample_ids == len(sample_ids),
        "distinct_sample_ids": distinct_sample_ids,
        "duplicate_sample_id_occurrences": len(sample_ids) - distinct_sample_ids,
        "input_indices_unique": True,
        "source_rows_unique": True,
        "record_identity": "input_index_and_frozen_source_row_v1",
        "root_manifest_selection_sha256": manifest["contract"]["data_partition"][
            "selection_sha256"
        ],
        "counts": {key: aggregate[key] for key in aggregate if key != "bytes"},
    }
    return merged, audit


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-root",
        type=Path,
        required=True,
        help="Local mirror containing manifest.json and datasets/**/shards/*.jsonl",
    )
    parser.add_argument("--manifest", type=Path, help="Defaults to SELECTION_ROOT/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--expect-records", type=int, default=100)
    parser.add_argument("--expected-partition-count", type=int, default=16)
    parser.add_argument("--expected-partition-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expect_records <= 0:
        parser.error("--expect-records must be positive")
    if args.expected_partition_count <= 0:
        parser.error("--expected-partition-count must be positive")
    if not 0 <= args.expected_partition_index < args.expected_partition_count:
        parser.error("--expected-partition-index must be in [0, --expected-partition-count)")
    args.selection_root = args.selection_root.resolve(strict=True)
    args.manifest = (args.manifest or args.selection_root / "manifest.json").resolve(strict=True)
    args.output = args.output.resolve()
    args.audit_output = (
        args.audit_output.resolve()
        if args.audit_output is not None
        else args.output.with_suffix(args.output.suffix + ".audit.json")
    )
    if args.output == args.audit_output:
        parser.error("--output and --audit-output must differ")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.overwrite:
        existing = [path for path in (args.output, args.audit_output) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    manifest = load_manifest(args.manifest)
    rows, audit = collect(
        args.selection_root,
        manifest,
        expected_records=args.expect_records,
        expected_partition_count=args.expected_partition_count,
        expected_partition_index=args.expected_partition_index,
    )
    payload = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    audit = {
        **audit,
        "output": str(args.output),
        "output_sha256": sha256_bytes(payload),
        "output_bytes": len(payload),
    }
    _atomic_write(args.output, payload, overwrite=args.overwrite)
    _atomic_write(
        args.audit_output,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
