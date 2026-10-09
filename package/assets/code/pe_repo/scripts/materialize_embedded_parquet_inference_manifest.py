#!/usr/bin/env python3
"""Materialize one embedded-Parquet inference cohort with one source read.

The input JSONL keeps immutable ``parquet+s3`` media identities.  The runtime
JSONL replaces only the load paths with deterministic local files and stores
the original locators in ``*_image_identity``.  Predictions therefore retain
the frozen source identity while image bytes are decoded from Parquet once,
instead of once per image cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZATION_FORMAT = "embedded_parquet_inference_materialization_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MaterializationError(ValueError):
    """The frozen smoke cohort cannot be safely materialized."""


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _load_manifest(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise MaterializationError(f"{path}:{line_number}: row must be an object")
            if value.get("input_index") != len(rows):
                raise MaterializationError(
                    f"{path}:{line_number}: input_index must be contiguous from zero"
                )
            rows.append(value)
    if len(rows) != expected_count:
        raise MaterializationError(
            f"manifest rows={len(rows)}, expected exactly {expected_count}"
        )
    return rows


def _source_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_dataset_s3: str,
    s3_client: Any,
) -> tuple[dict[str, Any], list[int]]:
    from scripts.image_edit_single_source_adapter import SourceIdentity, parquet_media_identity
    from scripts.infer_image_edit_full_json import parse_s3_media_uri

    if not source_dataset_s3.endswith("/"):
        source_dataset_s3 += "/"
    source_uris: set[str] = set()
    etags: set[str] = set()
    dataset_ids: set[str] = set()
    local_rows: list[int] = []
    for index, row in enumerate(rows):
        source = row.get("dataset_source")
        if not isinstance(source, Mapping):
            raise MaterializationError(f"row {index}: dataset_source must be an object")
        uri = source.get("parquet_s3_uri")
        etag = source.get("parquet_etag")
        local_row = source.get("parquet_local_row")
        dataset_id = source.get("dataset_id")
        if not isinstance(uri, str) or not uri.startswith(source_dataset_s3):
            raise MaterializationError(
                f"row {index}: source object is outside {source_dataset_s3}"
            )
        if not isinstance(etag, str) or not etag:
            raise MaterializationError(f"row {index}: missing source ETag")
        if type(local_row) is not int or local_row < 0:
            raise MaterializationError(f"row {index}: invalid Parquet-local row")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise MaterializationError(f"row {index}: missing dataset_id")
        source_uris.add(uri)
        etags.add(etag)
        dataset_ids.add(dataset_id)
        local_rows.append(local_row)
    if len(source_uris) != 1 or len(etags) != 1 or len(dataset_ids) != 1:
        raise MaterializationError(
            "single-object materialization requires one source URI, ETag, and dataset_id"
        )
    if local_rows != list(range(len(rows))):
        raise MaterializationError(
            "single-object smoke rows must be the complete contiguous [0, case_count) range"
        )

    source_uri = next(iter(source_uris))
    expected_etag = next(iter(etags))
    bucket, key = parse_s3_media_uri(source_uri)
    head = s3_client.head_object(Bucket=bucket, Key=key)
    actual_etag = _clean_etag(head.get("ETag"))
    size = head.get("ContentLength")
    if actual_etag != expected_etag:
        raise MaterializationError(
            f"source ETag changed: actual={actual_etag!r} expected={expected_etag!r}"
        )
    if type(size) is not int or size <= 0:
        raise MaterializationError(f"source has invalid ContentLength: {size!r}")

    identity = SourceIdentity(
        bucket=bucket,
        key=key,
        etag=actual_etag,
        bytes=size,
        rows=len(rows),
        object_index=0,
    )
    for index, row in enumerate(rows):
        expected_reference = parquet_media_identity(
            identity,
            local_row=index,
            column="input_images",
            list_index=0,
        )
        expected_target = parquet_media_identity(
            identity,
            local_row=index,
            column="output_image",
        )
        if row.get("reference_image") != expected_reference:
            raise MaterializationError(f"row {index}: reference Parquet locator drifted")
        if row.get("target_image") != expected_target:
            raise MaterializationError(f"row {index}: target Parquet locator drifted")
    return (
        {
            "dataset_id": next(iter(dataset_ids)),
            "source_uri": source_uri,
            "bucket": bucket,
            "key": key,
            "etag": actual_etag,
            "bytes": size,
            "rows": len(rows),
        },
        local_rows,
    )


def materialize_manifest(
    source_manifest: Path,
    runtime_manifest: Path,
    *,
    work_dir: Path,
    source_dataset_s3: str,
    expected_count: int,
    s3_client: Any,
) -> dict[str, Any]:
    """Write a local-path runtime manifest and return non-secret audit data."""

    from scripts import image_edit_single_source_adapter as adapter
    from scripts.infer_image_edit_full_json import normalize_input_row

    source_manifest = source_manifest.resolve(strict=True)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(source_manifest, expected_count=expected_count)
    source, local_rows = _source_contract(
        rows,
        source_dataset_s3=source_dataset_s3,
        s3_client=s3_client,
    )
    unit = {
        "dataset_id": source["dataset_id"],
        "dataset_ordinal": 0,
        "dataset_unit_index": 0,
        "global_unit_index": 0,
        "dataset_row_start": 0,
        "dataset_row_end_exclusive": expected_count,
        "global_row_start": 0,
        "global_row_end_exclusive": expected_count,
        "local_row_ranges": [[0, expected_count]],
        "source": {
            "bucket": source["bucket"],
            "key": source["key"],
            "etag": source["etag"],
            "bytes": source["bytes"],
            "rows": source["rows"],
            "object_index": 0,
        },
        "adapter": adapter.EMBEDDED_SINGLE_V1,
        "main_s3_uri": source_dataset_s3.rstrip("/") + "/",
    }
    materialized = adapter.load_unit_records(
        unit,
        s3_client=s3_client,
        work_dir=work_dir,
    )
    if len(materialized) != len(rows):
        raise MaterializationError(
            f"adapter returned {len(materialized)} records for {len(rows)} source rows"
        )

    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_manifest.with_name(f".{runtime_manifest.name}.tmp")
    digest = hashlib.sha256()
    materialized_files: set[Path] = set()
    with temporary.open("wb") as handle:
        for index, (source_row, runtime_record) in enumerate(
            zip(rows, materialized, strict=True)
        ):
            frozen = normalize_input_row(source_row, index)
            if runtime_record.sample_id != frozen.sample_id:
                raise MaterializationError(f"row {index}: sample_id changed while decoding Parquet")
            if runtime_record.instruction != frozen.instruction:
                raise MaterializationError(f"row {index}: instruction changed while decoding Parquet")
            if runtime_record.reference_image_identity != frozen.reference_image:
                raise MaterializationError(f"row {index}: reference identity changed")
            if runtime_record.target_image_identity != frozen.target_image:
                raise MaterializationError(f"row {index}: target identity changed")
            reference_path = Path(runtime_record.reference_image).resolve(strict=True)
            target_path = Path(runtime_record.target_image).resolve(strict=True)
            if work_dir not in reference_path.parents or work_dir not in target_path.parents:
                raise MaterializationError(f"row {index}: materialized media escaped work_dir")
            materialized_files.update((reference_path, target_path))
            value = {
                "input_index": index,
                "sample_id": frozen.sample_id,
                "instruction": frozen.instruction,
                "reference_image": str(reference_path),
                "target_image": str(target_path),
                "reference_image_identity": frozen.reference_image,
                "target_image_identity": frozen.target_image,
                "dataset_source": dict(source_row["dataset_source"]),
            }
            payload = _canonical_line(value)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, runtime_manifest)
    if len(materialized_files) != expected_count * 2:
        raise MaterializationError(
            f"materialized unique files={len(materialized_files)}, expected={expected_count * 2}"
        )
    return {
        "format_version": MATERIALIZATION_FORMAT,
        "strategy": "single_parquet_object_single_decode_v1",
        "rows": expected_count,
        "source_objects": 1,
        "source": {
            "uri": source["source_uri"],
            "etag": source["etag"],
            "bytes": source["bytes"],
            "rows": source["rows"],
            "local_rows_sha256": hashlib.sha256(
                json.dumps(local_rows, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "runtime_manifest_sha256": digest.hexdigest(),
        "materialized_files": len(materialized_files),
        "materialized_bytes": sum(path.stat().st_size for path in materialized_files),
        "output_identity": "original_parquet_s3_locators",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--source-dataset-s3", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if not args.source_dataset_s3.startswith("s3://"):
        parser.error("--source-dataset-s3 must be an S3 URI")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import boto3

    summary = materialize_manifest(
        args.source_manifest,
        args.runtime_manifest,
        work_dir=args.work_dir,
        source_dataset_s3=args.source_dataset_s3,
        expected_count=args.expected_count,
        s3_client=boto3.client("s3", region_name=args.aws_region),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
