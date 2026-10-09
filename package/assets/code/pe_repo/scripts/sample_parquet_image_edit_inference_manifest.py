#!/usr/bin/env python3
"""Uniformly sample label-free image-edit inference rows from S3 Parquet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.image_edit_single_source_adapter import SourceIdentity, parquet_media_identity
from scripts.prepare_image_edit_full_json_shard import S3RangeReader


FORMAT_VERSION = "parquet_image_edit_inference_sample_v2"
BASE_READ_COLUMNS = (
    "sample_id",
    "instructions",
    "num_inputs",
)
URI_MEDIA_COLUMNS = ("input_s3_paths", "output_s3_path")
EMBEDDED_MEDIA_COLUMNS = ("input_images", "output_image")


class SamplingError(ValueError):
    """The source inventory or a sampled row violates the frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _parse_s3_prefix(value: str) -> tuple[str, str]:
    if not value.startswith("s3://"):
        raise SamplingError("--source-prefix must be an s3:// URI")
    bucket, separator, key = value[5:].partition("/")
    if not separator or not bucket or not key:
        raise SamplingError("--source-prefix must include a bucket and key")
    return bucket, key.rstrip("/") + "/"


def _list_parquet_objects(client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for value in page.get("Contents", []):
            key = value.get("Key")
            size = value.get("Size")
            etag = _clean_etag(value.get("ETag"))
            if not isinstance(key, str) or not key.endswith(".parquet"):
                continue
            if type(size) is not int or size <= 0 or not etag:
                raise SamplingError(f"invalid object identity for {key!r}")
            objects.append({"bucket": bucket, "key": key, "bytes": size, "etag": etag})
    objects.sort(key=lambda item: item["key"])
    if not objects:
        raise SamplingError(f"no Parquet objects found under s3://{bucket}/{prefix}")
    return objects


def _parquet_rows(client: Any, item: Mapping[str, Any]) -> int:
    import pyarrow.parquet as pq

    reader = S3RangeReader(
        client,
        bucket=item["bucket"],
        key=item["key"],
        size=item["bytes"],
    )
    parquet = pq.ParquetFile(reader)
    rows = parquet.metadata.num_rows
    if type(rows) is not int or rows <= 0:
        raise SamplingError(f"s3://{item['bucket']}/{item['key']}: invalid row count {rows!r}")
    return rows


def _read_selected_rows(
    client: Any,
    item: Mapping[str, Any],
    local_rows: Sequence[int],
    *,
    dataset_id: str,
    global_rows: Mapping[int, int],
    media_mode: str,
    skip_invalid_rows: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    import pyarrow.parquet as pq

    head = client.head_object(Bucket=item["bucket"], Key=item["key"])
    if head.get("ContentLength") != item["bytes"] or _clean_etag(head.get("ETag")) != item["etag"]:
        raise SamplingError(f"source object changed: s3://{item['bucket']}/{item['key']}")
    reader = S3RangeReader(
        client,
        bucket=item["bucket"],
        key=item["key"],
        size=item["bytes"],
    )
    parquet = pq.ParquetFile(reader)
    media_columns = URI_MEDIA_COLUMNS if media_mode == "uri" else EMBEDDED_MEDIA_COLUMNS
    required_columns = BASE_READ_COLUMNS + media_columns
    missing = sorted(set(required_columns).difference(parquet.schema_arrow.names))
    if missing:
        raise SamplingError(f"{item['key']}: missing columns {missing}")
    # Embedded image columns can be very large. Their immutable Parquet cell identities
    # are enough for inference, so only decode the small metadata columns in that mode.
    read_columns = required_columns if media_mode == "uri" else BASE_READ_COLUMNS
    table = parquet.read(columns=list(read_columns), use_threads=False)
    if table.num_rows != item["rows"]:
        raise SamplingError(
            f"{item['key']}: decoded rows={table.num_rows}, expected={item['rows']}"
        )
    parquet_uri = f"s3://{item['bucket']}/{item['key']}"
    result: dict[int, dict[str, Any]] = {}
    invalid: dict[int, str] = {}
    source_identity = SourceIdentity(
        bucket=item["bucket"],
        key=item["key"],
        etag=item["etag"],
        bytes=item["bytes"],
        rows=item["rows"],
        object_index=item["object_index"],
    )
    for local_row in local_rows:
        raw = table.slice(local_row, 1).to_pylist()[0]
        context = f"{parquet_uri} row {local_row}"
        try:
            sample_id = raw.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise SamplingError("invalid_sample_id")
            if raw.get("num_inputs") != 1:
                raise SamplingError("num_inputs_not_one")
            instructions = raw.get("instructions")
            if isinstance(instructions, str):
                instructions = [instructions]
            if not isinstance(instructions, list):
                raise SamplingError("invalid_instructions_type")
            instruction = next(
                (
                    value.strip()
                    for value in instructions
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
            if instruction is None:
                raise SamplingError("no_non_empty_instruction")
            if media_mode == "uri":
                inputs = raw.get("input_s3_paths")
                if (
                    not isinstance(inputs, list)
                    or len(inputs) != 1
                    or not isinstance(inputs[0], str)
                    or not inputs[0].startswith("s3://")
                ):
                    raise SamplingError("invalid_input_s3_paths")
                target = raw.get("output_s3_path")
                if not isinstance(target, str) or not target.startswith("s3://"):
                    raise SamplingError("invalid_output_s3_path")
                reference_image = inputs[0]
                target_image = target
            else:
                reference_image = parquet_media_identity(
                    source_identity,
                    local_row=local_row,
                    column="input_images",
                    list_index=0,
                )
                target_image = parquet_media_identity(
                    source_identity,
                    local_row=local_row,
                    column="output_image",
                )
        except SamplingError as exc:
            if not skip_invalid_rows:
                raise SamplingError(f"{context}: {exc}") from exc
            invalid[global_rows[local_row]] = str(exc)
            continue
        result[global_rows[local_row]] = {
            "sample_id": sample_id.strip(),
            "instruction": instruction,
            "reference_image": reference_image,
            "target_image": target_image,
            "dataset_source": {
                "dataset_id": dataset_id,
                "parquet_s3_uri": parquet_uri,
                "parquet_etag": item["etag"],
                "parquet_local_row": local_row,
                "dataset_global_row": global_rows[local_row],
            },
        }
    return result, invalid


def _group_global_rows(
    objects: Sequence[Mapping[str, Any]], global_rows: Sequence[int]
) -> tuple[dict[int, list[int]], dict[int, dict[int, int]]]:
    selected_by_object: dict[int, list[int]] = defaultdict(list)
    global_by_object_local: dict[int, dict[int, int]] = defaultdict(dict)
    object_index = 0
    for global_row in sorted(global_rows):
        while global_row >= objects[object_index]["global_row_end_exclusive"]:
            object_index += 1
        local_row = global_row - objects[object_index]["global_row_start"]
        selected_by_object[object_index].append(local_row)
        global_by_object_local[object_index][local_row] = global_row
    return selected_by_object, global_by_object_local


def _read_global_rows(
    client: Any,
    objects: Sequence[Mapping[str, Any]],
    global_rows: Sequence[int],
    *,
    dataset_id: str,
    media_mode: str,
    skip_invalid_rows: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    selected_by_object, global_by_object_local = _group_global_rows(objects, global_rows)
    selected_rows: dict[int, dict[str, Any]] = {}
    invalid_rows: dict[int, str] = {}
    for index in sorted(selected_by_object):
        selected, invalid = _read_selected_rows(
            client,
            objects[index],
            selected_by_object[index],
            dataset_id=dataset_id,
            global_rows=global_by_object_local[index],
            media_mode=media_mode,
            skip_invalid_rows=skip_invalid_rows,
        )
        selected_rows.update(selected)
        invalid_rows.update(invalid)
    return selected_rows, invalid_rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            line = canonical_json_bytes(row) + b"\n"
            handle.write(line)
            digest.update(line)
    temporary.replace(path)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-order",
        choices=("uniform", "catalog"),
        default="uniform",
        help=(
            "Choose rows uniformly without replacement or take the first valid rows in "
            "the frozen lexicographic-Parquet/row order."
        ),
    )
    parser.add_argument("--expected-source-rows", type=int)
    parser.add_argument("--aws-profile", required=True)
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument(
        "--media-mode",
        choices=("uri", "embedded"),
        default="uri",
        help="Read direct S3 media URIs or emit immutable embedded-Parquet cell locators.",
    )
    parser.add_argument(
        "--skip-invalid-rows",
        action="store_true",
        help="Continue a deterministic random permutation until enough valid rows are found.",
    )
    args = parser.parse_args(argv)
    if args.sample_count <= 0 or args.inventory_workers <= 0:
        parser.error("--sample-count and --inventory-workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=args.aws_profile)
    bucket, prefix = _parse_s3_prefix(args.source_prefix)
    client = session.client(
        "s3",
        region_name="us-east-1",
        config=Config(max_pool_connections=max(16, args.inventory_workers * 2)),
    )
    objects = _list_parquet_objects(client, bucket=bucket, prefix=prefix)
    with ThreadPoolExecutor(max_workers=args.inventory_workers) as executor:
        row_counts = list(executor.map(lambda item: _parquet_rows(client, item), objects))
    offset = 0
    for object_index, (item, rows) in enumerate(zip(objects, row_counts, strict=True)):
        item["object_index"] = object_index
        item["rows"] = rows
        item["global_row_start"] = offset
        offset += rows
        item["global_row_end_exclusive"] = offset
    total_rows = offset
    if args.expected_source_rows is not None and total_rows != args.expected_source_rows:
        raise SamplingError(
            f"source rows={total_rows}, expected={args.expected_source_rows}"
        )
    if args.sample_count > total_rows:
        raise SamplingError("sample count exceeds source rows")

    rng = random.Random(args.seed)
    invalid_reasons: Counter[str] = Counter()
    inspected_global_rows: list[int] = []
    sampled_global_rows: list[int] = []
    rows: list[dict[str, Any]] = []
    sample_ids_seen: set[str] = set()
    if args.selection_order == "catalog":
        # A catalog-order smoke keeps the cohort deterministic and, unlike a
        # dataset-wide random sample, lets embedded Parquet media be decoded
        # once per physical object.  This is important for GPU utilization:
        # one image-cell range read can otherwise decompress an entire large
        # row group for every sampled case.
        candidate_start = 0
        candidate_batch_size = max(500, args.sample_count)
        while len(rows) < args.sample_count and candidate_start < total_rows:
            candidate_end = min(candidate_start + candidate_batch_size, total_rows)
            candidates = list(range(candidate_start, candidate_end))
            selected_rows, invalid_rows = _read_global_rows(
                client,
                objects,
                candidates,
                dataset_id=args.dataset_id,
                media_mode=args.media_mode,
                skip_invalid_rows=args.skip_invalid_rows,
            )
            for global_row in candidates:
                inspected_global_rows.append(global_row)
                if global_row in invalid_rows:
                    invalid_reasons[invalid_rows[global_row]] += 1
                    continue
                row = selected_rows[global_row]
                if row["sample_id"] in sample_ids_seen:
                    if not args.skip_invalid_rows:
                        raise SamplingError("catalog-order sample_id values are not unique")
                    invalid_reasons["duplicate_sample_id"] += 1
                    continue
                sample_ids_seen.add(row["sample_id"])
                sampled_global_rows.append(global_row)
                rows.append(row)
                if len(rows) == args.sample_count:
                    break
            candidate_start = candidate_end
        if len(rows) != args.sample_count:
            raise SamplingError(
                f"only found {len(rows)} valid unique rows after inspecting "
                f"{len(inspected_global_rows)} source rows in catalog order"
            )
        sample_ids = [row["sample_id"] for row in rows]
    elif not args.skip_invalid_rows:
        sampled_global_rows = rng.sample(range(total_rows), args.sample_count)
        selected_rows, _ = _read_global_rows(
            client,
            objects,
            sampled_global_rows,
            dataset_id=args.dataset_id,
            media_mode=args.media_mode,
            skip_invalid_rows=False,
        )
        rows = [selected_rows[index] for index in sampled_global_rows]
        inspected_global_rows = list(sampled_global_rows)
        sample_ids = [row["sample_id"] for row in rows]
        if len(set(sample_ids)) != len(sample_ids):
            raise SamplingError("sampled sample_id values are not unique")
    else:
        # Rejection sampling over a deterministic uniform random permutation keeps
        # the accepted cohort uniform over valid rows without loading the dataset.
        seen_candidates: set[int] = set()
        candidate_batch_size = max(256, args.sample_count * 2)
        while len(rows) < args.sample_count and len(seen_candidates) < total_rows:
            candidates: list[int] = []
            while (
                len(candidates) < candidate_batch_size
                and len(seen_candidates) < total_rows
            ):
                candidate = rng.randrange(total_rows)
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                candidates.append(candidate)
            selected_rows, invalid_rows = _read_global_rows(
                client,
                objects,
                candidates,
                dataset_id=args.dataset_id,
                media_mode=args.media_mode,
                skip_invalid_rows=True,
            )
            for global_row in candidates:
                inspected_global_rows.append(global_row)
                if global_row in invalid_rows:
                    invalid_reasons[invalid_rows[global_row]] += 1
                    continue
                row = selected_rows[global_row]
                if row["sample_id"] in sample_ids_seen:
                    invalid_reasons["duplicate_sample_id"] += 1
                    continue
                sample_ids_seen.add(row["sample_id"])
                sampled_global_rows.append(global_row)
                rows.append(row)
                if len(rows) == args.sample_count:
                    break
        if len(rows) != args.sample_count:
            raise SamplingError(
                f"only found {len(rows)} valid unique rows after inspecting "
                f"{len(inspected_global_rows)} source rows"
            )
        sample_ids = [row["sample_id"] for row in rows]
    for input_index, row in enumerate(rows):
        row["input_index"] = input_index

    output_sha256 = _write_jsonl(args.output_jsonl.resolve(), rows)
    inventory = [
        {
            "key": item["key"],
            "etag": item["etag"],
            "bytes": item["bytes"],
            "rows": item["rows"],
        }
        for item in objects
    ]
    summary = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "source": {
            "dataset_id": args.dataset_id,
            "prefix": args.source_prefix.rstrip("/") + "/",
            "objects": len(objects),
            "rows": total_rows,
            "bytes": sum(item["bytes"] for item in objects),
            "inventory_sha256": hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
        },
        "sampling": {
            "method": (
                "first_valid_rows_in_catalog_order_v1"
                if args.selection_order == "catalog"
                else (
                    "uniform_valid_rows_via_random_permutation_rejection_v1"
                    if args.skip_invalid_rows
                    else "uniform_without_replacement_python_random_v1"
                )
            ),
            "seed": args.seed if args.selection_order == "uniform" else None,
            "sample_count": len(rows),
            "media_mode": args.media_mode,
            "candidate_rows_inspected": len(inspected_global_rows),
            "invalid_rows_skipped": sum(invalid_reasons.values()),
            "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
            "sampled_global_rows_sha256": hashlib.sha256(
                canonical_json_bytes(sampled_global_rows)
            ).hexdigest(),
            "inspected_global_rows_sha256": hashlib.sha256(
                canonical_json_bytes(inspected_global_rows)
            ).hexdigest(),
            "ordered_sample_ids_sha256": hashlib.sha256(
                canonical_json_bytes(sample_ids)
            ).hexdigest(),
        },
        "output": {
            "path": str(args.output_jsonl.resolve()),
            "rows": len(rows),
            "sha256": output_sha256,
        },
    }
    _write_json(args.summary_json.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
