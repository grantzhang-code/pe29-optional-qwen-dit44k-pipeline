#!/usr/bin/env python3
"""Sample committed predictions from an in-progress sidecar24 S3 selection.

The unit success marker is the commit boundary.  JSONL objects without a
matching marker are ignored; a marker without its declared JSONL is treated as
corruption.  Sampling is uniform over the rows declared by a frozen, sorted
marker listing and is deterministic for a fixed marker set and seed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Protocol, Sequence


MARKER_FORMAT = "image_edit_full_json_sidecar24_unit_success_v1"
AUDIT_FORMAT = "image_edit_sidecar24_partial_prediction_snapshot_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_ID_RE = r"[a-z0-9][a-z0-9_-]*"


class SamplingError(ValueError):
    """The partial result set does not satisfy the committed-unit contract."""


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int


class ObjectStore(Protocol):
    def list_objects(self, bucket: str, prefix: str) -> list[ObjectInfo]: ...

    def get_bytes(self, bucket: str, key: str) -> bytes: ...


class Boto3ObjectStore:
    def __init__(self, *, region: str) -> None:
        import boto3
        from botocore.config import Config

        self._client = boto3.client(
            "s3",
            region_name=region,
            config=Config(max_pool_connections=32),
        )

    def list_objects(self, bucket: str, prefix: str) -> list[ObjectInfo]:
        objects: list[ObjectInfo] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for value in page.get("Contents", []):
                key = value.get("Key")
                size = value.get("Size")
                if isinstance(key, str) and type(size) is int and size >= 0:
                    objects.append(ObjectInfo(key=key, size=size))
        return objects

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self._client.get_object(Bucket=bucket, Key=key)["Body"].read()


@dataclass(frozen=True)
class CommittedUnit:
    marker_key: str
    jsonl_key: str
    rows: int
    size: int
    sha256: str
    dataset_id: str
    dataset_unit_index: int
    global_row_start: int
    marker_payload: bytes


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_s3_root(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SamplingError(f"invalid S3 root: {uri!r}")
    bucket, separator, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/")
    if not bucket or not separator or not prefix:
        raise SamplingError(f"S3 root must include a bucket and non-empty prefix: {uri!r}")
    return bucket, prefix


def _strict_json_object(payload: bytes, *, location: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SamplingError(f"{location}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=object_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SamplingError(f"{location}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise SamplingError(f"{location}: expected a JSON object")
    return value


def _required_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SamplingError(f"{location} must be an integer >= {minimum}")
    return value


def _committed_units(
    store: ObjectStore,
    *,
    bucket: str,
    prefix: str,
) -> tuple[list[CommittedUnit], int, str]:
    dataset_prefix = f"{prefix}/datasets/"
    listed = store.list_objects(bucket, dataset_prefix)
    by_key = {item.key: item for item in listed}
    if len(by_key) != len(listed):
        raise SamplingError("S3 listing returned duplicate object keys")

    marker_re = re.compile(
        rf"^{re.escape(dataset_prefix)}(?P<dataset>{DATASET_ID_RE})/shards/"
        r"unit-(?P<unit>[0-9]{6})\.success\.json$"
    )
    jsonl_re = re.compile(
        rf"^{re.escape(dataset_prefix)}{DATASET_ID_RE}/shards/unit-[0-9]{{6}}\.jsonl$"
    )
    marker_keys = sorted(key for key in by_key if key.endswith(".success.json"))
    if not marker_keys:
        raise SamplingError(f"no committed unit markers under s3://{bucket}/{dataset_prefix}")

    units: list[CommittedUnit] = []
    snapshot_hash = hashlib.sha256()
    paired_jsonl_keys: set[str] = set()
    for marker_key in marker_keys:
        match = marker_re.fullmatch(marker_key)
        if match is None:
            raise SamplingError(f"unexpected success-marker path: s3://{bucket}/{marker_key}")
        dataset_id = match.group("dataset")
        dataset_unit_index = int(match.group("unit"))
        jsonl_key = marker_key.removesuffix(".success.json") + ".jsonl"
        jsonl_info = by_key.get(jsonl_key)
        if jsonl_info is None:
            raise SamplingError(
                f"committed marker has no paired JSONL: s3://{bucket}/{marker_key}"
            )

        marker_payload = store.get_bytes(bucket, marker_key)
        marker = _strict_json_object(
            marker_payload, location=f"s3://{bucket}/{marker_key}"
        )
        if marker.get("format_version") != MARKER_FORMAT or marker.get("status") != "complete":
            raise SamplingError(f"s3://{bucket}/{marker_key}: marker is not complete")
        output = marker.get("output")
        if not isinstance(output, Mapping):
            raise SamplingError(f"s3://{bucket}/{marker_key}: marker has no output object")
        rows = _required_int(
            output.get("rows"), location=f"s3://{bucket}/{marker_key}: output.rows", minimum=1
        )
        size = _required_int(
            output.get("bytes"), location=f"s3://{bucket}/{marker_key}: output.bytes", minimum=1
        )
        digest = output.get("sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise SamplingError(f"s3://{bucket}/{marker_key}: output.sha256 is invalid")
        expected_uri = f"s3://{bucket}/{jsonl_key}"
        if output.get("uri") != expected_uri:
            raise SamplingError(
                f"s3://{bucket}/{marker_key}: output.uri does not identify its paired JSONL"
            )
        if jsonl_info.size != size:
            raise SamplingError(
                f"s3://{bucket}/{jsonl_key}: listed byte count differs from marker"
            )
        counts = output.get("counts")
        if isinstance(counts, Mapping) and counts.get("rows") != rows:
            raise SamplingError(
                f"s3://{bucket}/{marker_key}: output counts.rows differs from output.rows"
            )

        binding = marker.get("binding")
        bound_unit = binding.get("unit") if isinstance(binding, Mapping) else None
        if not isinstance(bound_unit, Mapping):
            raise SamplingError(f"s3://{bucket}/{marker_key}: marker has no unit binding")
        if bound_unit.get("dataset_id") != dataset_id:
            raise SamplingError(f"s3://{bucket}/{marker_key}: bound dataset_id differs from path")
        if bound_unit.get("dataset_unit_index") != dataset_unit_index:
            raise SamplingError(
                f"s3://{bucket}/{marker_key}: bound dataset_unit_index differs from path"
            )
        bound_rows = _required_int(
            bound_unit.get("selected_rows"),
            location=f"s3://{bucket}/{marker_key}: binding.unit.selected_rows",
            minimum=1,
        )
        global_row_start = _required_int(
            bound_unit.get("global_row_start"),
            location=f"s3://{bucket}/{marker_key}: binding.unit.global_row_start",
        )
        global_row_end = _required_int(
            bound_unit.get("global_row_end_exclusive"),
            location=f"s3://{bucket}/{marker_key}: binding.unit.global_row_end_exclusive",
            minimum=1,
        )
        if bound_rows != rows or global_row_end - global_row_start != rows:
            raise SamplingError(
                f"s3://{bucket}/{marker_key}: unit binding interval differs from output.rows"
            )

        snapshot_hash.update(marker_key.encode("utf-8"))
        snapshot_hash.update(b"\0")
        snapshot_hash.update(marker_payload)
        snapshot_hash.update(b"\0")
        paired_jsonl_keys.add(jsonl_key)
        units.append(
            CommittedUnit(
                marker_key=marker_key,
                jsonl_key=jsonl_key,
                rows=rows,
                size=size,
                sha256=digest,
                dataset_id=dataset_id,
                dataset_unit_index=dataset_unit_index,
                global_row_start=global_row_start,
                marker_payload=marker_payload,
            )
        )

    listed_jsonl = {key for key in by_key if jsonl_re.fullmatch(key)}
    uncommitted_jsonl = len(listed_jsonl - paired_jsonl_keys)
    intervals = sorted(
        (unit.global_row_start, unit.global_row_start + unit.rows, unit.marker_key)
        for unit in units
    )
    previous_end = -1
    previous_key = ""
    for start, end, marker_key in intervals:
        if start < previous_end:
            raise SamplingError(
                "committed unit global-row intervals overlap: "
                f"s3://{bucket}/{previous_key} and s3://{bucket}/{marker_key}"
            )
        previous_end = end
        previous_key = marker_key
    return units, uncommitted_jsonl, snapshot_hash.hexdigest()


def _load_selected_rows(
    store: ObjectStore,
    *,
    bucket: str,
    unit: CommittedUnit,
    local_offsets: tuple[int, ...],
) -> list[dict[str, Any]]:
    payload = store.get_bytes(bucket, unit.jsonl_key)
    location = f"s3://{bucket}/{unit.jsonl_key}"
    if len(payload) != unit.size:
        raise SamplingError(f"{location}: downloaded byte count differs from marker")
    if sha256_bytes(payload) != unit.sha256:
        raise SamplingError(f"{location}: downloaded SHA-256 differs from marker")
    if not payload.endswith(b"\n"):
        raise SamplingError(f"{location}: JSONL is not newline terminated")
    lines = payload.splitlines()
    if len(lines) != unit.rows or any(not line.strip() for line in lines):
        raise SamplingError(f"{location}: decoded row count differs from marker")

    selected: list[dict[str, Any]] = []
    for local_offset in local_offsets:
        row = _strict_json_object(
            lines[local_offset], location=f"{location}:{local_offset + 1}"
        )
        input_index = row.get("input_index")
        expected_input_index = unit.global_row_start + local_offset
        if type(input_index) is not int or input_index != expected_input_index:
            raise SamplingError(
                f"{location}:{local_offset + 1}: input_index differs from marker binding"
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SamplingError(f"{location}:{local_offset + 1}: sample_id is missing")
        selected.append(row)
    return selected


def sample_partial_predictions(
    store: ObjectStore,
    *,
    s3_root: str,
    sample_size: int,
    seed: int,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_size <= 0:
        raise SamplingError("sample_size must be positive")
    if workers <= 0:
        raise SamplingError("workers must be positive")
    bucket, prefix = parse_s3_root(s3_root)
    units, uncommitted_jsonl, marker_snapshot_sha256 = _committed_units(
        store, bucket=bucket, prefix=prefix
    )
    population_rows = sum(unit.rows for unit in units)
    if population_rows < sample_size:
        raise SamplingError(
            f"committed population has {population_rows} rows, fewer than requested {sample_size}"
        )

    sampled_ordinals = sorted(random.Random(seed).sample(range(population_rows), sample_size))
    offsets_by_unit: dict[CommittedUnit, list[int]] = {}
    position_index = 0
    population_offset = 0
    for unit in units:
        unit_end = population_offset + unit.rows
        while (
            position_index < len(sampled_ordinals)
            and sampled_ordinals[position_index] < unit_end
        ):
            offsets_by_unit.setdefault(unit, []).append(
                sampled_ordinals[position_index] - population_offset
            )
            position_index += 1
        population_offset = unit_end
    if position_index != len(sampled_ordinals):
        raise AssertionError("sampled ordinal mapping did not cover every selected row")

    rows_by_marker: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(offsets_by_unit))) as executor:
        futures = {
            executor.submit(
                _load_selected_rows,
                store,
                bucket=bucket,
                unit=unit,
                local_offsets=tuple(offsets),
            ): unit
            for unit, offsets in offsets_by_unit.items()
        }
        for future in as_completed(futures):
            unit = futures[future]
            rows_by_marker[unit.marker_key] = future.result()

    sampled_rows = [
        row
        for unit in units
        if unit.marker_key in rows_by_marker
        for row in rows_by_marker[unit.marker_key]
    ]
    if len(sampled_rows) != sample_size:
        raise AssertionError("decoded sample size differs from requested sample size")
    sampled_rows.sort(key=lambda row: row["input_index"])
    input_indices = [row["input_index"] for row in sampled_rows]
    if len(set(input_indices)) != len(input_indices):
        raise SamplingError("sampled predictions contain duplicate input_index")
    sample_ids = [row["sample_id"] for row in sampled_rows]
    distinct_sample_ids = len(set(sample_ids))

    audit = {
        "format_version": AUDIT_FORMAT,
        "source_s3_root": f"s3://{bucket}/{prefix}/",
        "seed": seed,
        "sample_size": sample_size,
        "population": {
            "committed_units": len(units),
            "committed_rows": population_rows,
            "uncommitted_jsonl_ignored": uncommitted_jsonl,
            "marker_snapshot_sha256": marker_snapshot_sha256,
        },
        "verification": {
            "all_markers_complete": True,
            "all_output_uris_exact": True,
            "all_listed_byte_counts_match": True,
            "sampled_unit_sha256_and_rows_match": True,
        },
        "sampled_units": len(offsets_by_unit),
        "input_indices_unique": True,
        "input_index_min": min(input_indices),
        "input_index_max": max(input_indices),
        "sample_ids_unique": distinct_sample_ids == sample_size,
        "distinct_sample_ids": distinct_sample_ids,
        "duplicate_sample_id_occurrences": sample_size - distinct_sample_ids,
        "record_identity": "input_index_and_frozen_source_row_v1",
    }
    return sampled_rows, audit


def _snapshot_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
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
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    output = args.output.resolve()
    audit_output = (
        args.audit_output.resolve()
        if args.audit_output is not None
        else output.with_suffix(output.suffix + ".audit.json")
    )
    if output == audit_output:
        raise SystemExit("--output and --audit-output must differ")
    if not args.overwrite:
        existing = [path for path in (output, audit_output) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    rows, audit = sample_partial_predictions(
        Boto3ObjectStore(region=args.region),
        s3_root=args.s3_root,
        sample_size=args.sample_size,
        seed=args.seed,
        workers=args.workers,
    )
    snapshot = _snapshot_payload(rows)
    audit = {
        **audit,
        "output": {
            "records": len(rows),
            "bytes": len(snapshot),
            "sha256": sha256_bytes(snapshot),
        },
    }
    audit_payload = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    _atomic_write(output, snapshot, overwrite=args.overwrite)
    _atomic_write(audit_output, audit_payload, overwrite=args.overwrite)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
