#!/usr/bin/env python3
"""Build a deterministic, label-free V6 inference sample from CRAFT strict Parquet.

The sampler lists the immutable published shard inventory, then reads only the
small URI/instruction columns from a deterministic subset of Parquet objects.
Image bytes and annotation-like metadata never enter the output manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_image_edit_full_json_shard import S3RangeReader  # noqa: E402


FORMAT_VERSION = "craft_strict_parquet_v1_inference_sample_v1"
SOURCE_SCHEMA_VERSION = "craft-image-edit-pack-dataset-1.0"
DIMENSIONS = (
    "audience_adaptation",
    "chart_editing",
    "design_polish",
    "relation_change",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
READ_COLUMNS = (
    "sample_id",
    "input_s3_paths",
    "output_s3_path",
    "instructions",
    "num_inputs",
    "input_roles",
    "operation_type",
)


class SamplingError(ValueError):
    """The frozen source or a sampled row violates the inference contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _parse_s3_prefix(value: str) -> tuple[str, str]:
    if not value.startswith("s3://"):
        raise SamplingError("--source-prefix must be an s3:// URI")
    bucket, separator, key = value[5:].partition("/")
    if not separator or not bucket or not key:
        raise SamplingError("--source-prefix must include bucket and key")
    return bucket, key.rstrip("/") + "/"


def dimension_quotas(
    dimension_rows: Mapping[str, Any], sample_count: int
) -> dict[str, int]:
    """Allocate exact proportional quotas using the largest-remainder method."""

    if sample_count <= 0:
        raise SamplingError("sample_count must be positive")
    rows: dict[str, int] = {}
    for dimension in DIMENSIONS:
        value = dimension_rows.get(dimension)
        if type(value) is not int or value <= 0:
            raise SamplingError(f"invalid row count for {dimension!r}: {value!r}")
        rows[dimension] = value
    total = sum(rows.values())
    if sample_count > total:
        raise SamplingError("sample_count exceeds source rows")
    exact = {dimension: sample_count * rows[dimension] / total for dimension in DIMENSIONS}
    quotas = {dimension: int(exact[dimension]) for dimension in DIMENSIONS}
    remaining = sample_count - sum(quotas.values())
    order = sorted(
        DIMENSIONS,
        key=lambda dimension: (-(exact[dimension] - quotas[dimension]), dimension),
    )
    for dimension in order[:remaining]:
        quotas[dimension] += 1
    if sum(quotas.values()) != sample_count:
        raise AssertionError("dimension quota allocation drifted")
    return quotas


def _hash_rank(seed: int, namespace: str, *values: Any) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("utf-8"))
    for value in values:
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
    return digest.digest()


def _validate_row(
    raw: Mapping[str, Any],
    *,
    dimension: str,
    parquet_uri: str,
    parquet_etag: str,
    local_row: int,
) -> dict[str, Any]:
    location = f"{parquet_uri} row {local_row}"
    sample_id = raw.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SamplingError(f"{location}: invalid sample_id")
    if raw.get("num_inputs") != 1:
        raise SamplingError(f"{location}: num_inputs must equal 1")
    roles = raw.get("input_roles")
    if roles != ["source"]:
        raise SamplingError(f"{location}: input_roles must equal ['source']")
    operations = raw.get("operation_type")
    if operations != [dimension]:
        raise SamplingError(
            f"{location}: operation_type={operations!r}, expected {[dimension]!r}"
        )
    input_paths = raw.get("input_s3_paths")
    if (
        not isinstance(input_paths, list)
        or len(input_paths) != 1
        or not isinstance(input_paths[0], str)
        or not input_paths[0].startswith("s3://")
    ):
        raise SamplingError(f"{location}: input_s3_paths must contain one S3 URI")
    target = raw.get("output_s3_path")
    if not isinstance(target, str) or not target.startswith("s3://"):
        raise SamplingError(f"{location}: output_s3_path must be an S3 URI")
    instructions = raw.get("instructions")
    if not isinstance(instructions, list):
        raise SamplingError(f"{location}: instructions must be a list")
    instruction = next(
        (
            value.strip()
            for value in instructions
            if isinstance(value, str) and value.strip()
        ),
        None,
    )
    if instruction is None:
        raise SamplingError(f"{location}: instructions contains no non-empty value")
    return {
        "sample_id": sample_id.strip(),
        "instruction": instruction,
        "reference_image": input_paths[0],
        "target_image": target,
        "dataset_source": {
            "dataset_id": "craft_parquet_strict_v1",
            "dimension": dimension,
            "parquet_s3_uri": parquet_uri,
            "parquet_etag": parquet_etag,
            "parquet_local_row": local_row,
        },
    }


def select_rows(
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        rows = list(candidates.get(dimension, ()))
        quota = quotas[dimension]
        if len(rows) < quota:
            raise SamplingError(
                f"dimension {dimension!r} has {len(rows)} candidates, needs {quota}"
            )
        ranked = sorted(
            rows,
            key=lambda row: _hash_rank(
                seed,
                "row",
                dimension,
                row["sample_id"],
                row["dataset_source"]["parquet_s3_uri"],
                row["dataset_source"]["parquet_local_row"],
            ),
        )
        selected.extend(dict(row) for row in ranked[:quota])
    selected.sort(
        key=lambda row: _hash_rank(
            seed,
            "global-order",
            row["dataset_source"]["dimension"],
            row["sample_id"],
        )
    )
    sample_ids = [row["sample_id"] for row in selected]
    if len(set(sample_ids)) != len(sample_ids):
        duplicates = sorted(
            sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
        )
        raise SamplingError(f"selected sample_id values are not unique: {duplicates[:20]}")
    for input_index, row in enumerate(selected):
        row["input_index"] = input_index
    return selected


def _list_parquet_objects(s3_client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "data/"):
        for item in page.get("Contents", []):
            key = item.get("Key")
            size = item.get("Size")
            etag = _clean_etag(item.get("ETag"))
            if not isinstance(key, str) or not key.endswith(".parquet"):
                continue
            if type(size) is not int or size <= 0 or not etag:
                raise SamplingError(f"invalid S3 identity for {key!r}")
            relative = key[len(prefix + "data/") :]
            dimension = relative.partition("/")[0]
            if dimension not in DIMENSIONS:
                raise SamplingError(f"unexpected dimension in key {key!r}")
            objects.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "bytes": size,
                    "etag": etag,
                    "dimension": dimension,
                }
            )
    objects.sort(key=lambda item: item["key"])
    return objects


def _read_object_rows(s3_client: Any, item: Mapping[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    head = s3_client.head_object(Bucket=item["bucket"], Key=item["key"])
    if head.get("ContentLength") != item["bytes"] or _clean_etag(head.get("ETag")) != item["etag"]:
        raise SamplingError(f"S3 object identity changed: s3://{item['bucket']}/{item['key']}")
    reader = S3RangeReader(
        s3_client,
        bucket=item["bucket"],
        key=item["key"],
        size=item["bytes"],
    )
    parquet = pq.ParquetFile(reader)
    missing = sorted(set(READ_COLUMNS).difference(parquet.schema_arrow.names))
    if missing:
        raise SamplingError(f"{item['key']}: missing required columns {missing}")
    table = parquet.read(columns=list(READ_COLUMNS), use_threads=False)
    parquet_uri = f"s3://{item['bucket']}/{item['key']}"
    return [
        _validate_row(
            raw,
            dimension=item["dimension"],
            parquet_uri=parquet_uri,
            parquet_etag=item["etag"],
            local_row=local_row,
        )
        for local_row, raw in enumerate(table.to_pylist())
    ]


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
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aws-profile", required=True)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    args = parser.parse_args(argv)
    if args.sample_count <= 0 or args.candidate_multiplier < 1:
        parser.error("--sample-count and --candidate-multiplier must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import boto3

    bucket, prefix = _parse_s3_prefix(args.source_prefix)
    s3_client = boto3.Session(profile_name=args.aws_profile).client(
        "s3", region_name="us-east-1"
    )
    marker_key = prefix + "_SUCCESS.json"
    marker_response = s3_client.get_object(Bucket=bucket, Key=marker_key)
    marker_bytes = marker_response["Body"].read()
    marker = json.loads(marker_bytes)
    if (
        marker.get("schema_version") != SOURCE_SCHEMA_VERSION
        or marker.get("status") != "success"
        or marker.get("dataset_s3_prefix") != args.source_prefix.rstrip("/") + "/"
        or marker.get("gates") is None
        or not all(marker["gates"].values())
    ):
        raise SamplingError("source _SUCCESS.json is incomplete or incompatible")
    quotas = dimension_quotas(marker.get("dimension_rows", {}), args.sample_count)
    objects = _list_parquet_objects(s3_client, bucket=bucket, prefix=prefix)
    if len(objects) != marker.get("parquet_shards"):
        raise SamplingError(
            f"listed {len(objects)} Parquet shards, expected {marker.get('parquet_shards')}"
        )
    if sum(item["bytes"] for item in objects) != marker.get("parquet_bytes"):
        raise SamplingError("listed Parquet bytes do not match _SUCCESS.json")

    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        by_dimension[item["dimension"]].append(item)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_objects: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        ranked_objects = sorted(
            by_dimension[dimension],
            key=lambda item: _hash_rank(args.seed, "object", item["key"]),
        )
        target_candidates = max(
            quotas[dimension] * args.candidate_multiplier,
            quotas[dimension],
        )
        for item in ranked_objects:
            rows = _read_object_rows(s3_client, item)
            candidates[dimension].extend(rows)
            selected_objects.append(
                {
                    "s3_uri": f"s3://{item['bucket']}/{item['key']}",
                    "etag": item["etag"],
                    "bytes": item["bytes"],
                    "rows": len(rows),
                    "dimension": dimension,
                }
            )
            if len(candidates[dimension]) >= target_candidates:
                break
        if len(candidates[dimension]) < quotas[dimension]:
            raise SamplingError(f"could not collect enough candidates for {dimension}")

    rows = select_rows(candidates, quotas=quotas, seed=args.seed)
    output_path = args.output_jsonl.resolve()
    output_sha = _write_jsonl(output_path, rows)
    ordered_ids_sha = sha256_bytes(canonical_json_bytes([row["sample_id"] for row in rows]))
    summary = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "sampling": {
            "method": "proportional_dimension_then_sha256_ranked_object_and_row",
            "seed": args.seed,
            "sample_count": len(rows),
            "candidate_multiplier": args.candidate_multiplier,
            "dimension_counts": dict(Counter(row["dataset_source"]["dimension"] for row in rows)),
        },
        "source": {
            "prefix": args.source_prefix.rstrip("/") + "/",
            "run_id": marker["run_id"],
            "rows": marker["rows"],
            "parquet_shards": marker["parquet_shards"],
            "parquet_bytes": marker["parquet_bytes"],
            "success_sha256": sha256_bytes(marker_bytes),
            "success_version_id": marker_response.get("VersionId"),
            "success_etag": _clean_etag(marker_response.get("ETag")),
        },
        "selected_objects": selected_objects,
        "ordered_ids_sha256": ordered_ids_sha,
        "output": {
            "path": str(output_path),
            "rows": len(rows),
            "sha256": output_sha,
        },
    }
    _write_json(args.summary_json.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
