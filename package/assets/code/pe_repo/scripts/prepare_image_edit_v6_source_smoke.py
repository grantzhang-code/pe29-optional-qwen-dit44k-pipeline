#!/usr/bin/env python3
"""Prepare a deterministic source-only image-edit-v6 inference smoke set.

The input rows contain image-edit pairs, but this program deliberately reads
and publishes only ``source_s3_uri``.  It balances the smoke set between the
PE64 and Full72 portions of the immutable merged training manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from urllib.parse import urlparse

from PIL import Image


EXPECTED_MANIFEST_SHA256 = "0bb33959861f4464759a73153f2134274694b7499f2b08dfa23456fa8cbece3e"
EXPECTED_ROWS = 11_771_901
EXPECTED_SHARDS = 136
DEFAULT_RANGE_BYTES = 8 * 1024 * 1024
SOURCE_COUNTS = {"pe64": 50, "full72": 50}
SOURCE_SHARD_INDICES = {"pe64": 0, "full72": 64}
FORMAT_SUFFIXES = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_manifest(client: Any, args: argparse.Namespace) -> tuple[Mapping[str, Any], bytes]:
    bucket, key = parse_s3_uri(args.training_manifest_uri)
    response = client.get_object(Bucket=bucket, Key=key, VersionId=args.training_manifest_version_id)
    payload = response["Body"].read()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != args.training_manifest_sha256:
        raise ValueError(
            f"training manifest SHA-256 mismatch: actual={actual_sha256} "
            f"expected={args.training_manifest_sha256}"
        )
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError("training manifest must be a JSON object")
    if value.get("format_version") != "image_edit_v6_training_manifest_v1":
        raise ValueError(f"unexpected manifest format: {value.get('format_version')!r}")
    if value.get("complete") is not True:
        raise ValueError("training manifest is not complete")
    shards = value.get("shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_SHARDS:
        raise ValueError(f"training manifest must contain {EXPECTED_SHARDS} shards")
    total_rows = int(value.get("total_rows") or sum(int(item["rows"]) for item in shards))
    if total_rows != EXPECTED_ROWS:
        raise ValueError(f"training row count is {total_rows}, expected {EXPECTED_ROWS}")
    return value, payload


def read_candidate_rows(
    client: Any,
    *,
    shard: Mapping[str, Any],
    dataset_source: str,
    seed: int,
    range_bytes: int,
) -> list[dict[str, Any]]:
    bucket, key = parse_s3_uri(str(shard["path"]))
    response = client.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=str(shard["version_id"]),
        Range=f"bytes=0-{range_bytes - 1}",
    )
    payload = response["Body"].read()
    # The final range line can be truncated; every preceding non-empty line is
    # a complete row because the range begins at byte zero.
    lines = payload.split(b"\n")
    if not payload.endswith(b"\n"):
        lines.pop()
    candidates: list[dict[str, Any]] = []
    for source_line, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{shard['path']}:{source_line}: row is not an object")
        sample_id = row.get("sample_id")
        source_uri = row.get("source_s3_uri")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if not isinstance(source_uri, str) or not source_uri:
            continue
        if not source_uri.startswith(("s3://", "parquet+s3://")):
            continue
        rank = sha256_bytes(f"{seed}\0{dataset_source}\0{sample_id}".encode("utf-8"))
        candidates.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "source_s3_uri": source_uri,
                "source_line_number": source_line,
            }
        )
    candidates.sort(key=lambda item: (item["rank"], item["sample_id"]))
    return candidates


def validate_image(payload: bytes, *, source_uri: str) -> tuple[str, int, int]:
    if not payload:
        raise ValueError(f"empty source image: {source_uri}")
    with Image.open(io.BytesIO(payload)) as opened:
        image_format = str(opened.format or "").upper()
        width, height = opened.size
        opened.load()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid source image dimensions: {source_uri}")
    suffix = FORMAT_SUFFIXES.get(image_format)
    if suffix is None:
        raise ValueError(f"unsupported source image format {image_format!r}: {source_uri}")
    return suffix, width, height


def put_bytes(client: Any, uri: str, payload: bytes, *, content_type: str) -> None:
    bucket, key = parse_s3_uri(uri)
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    import boto3
    from botocore.config import Config

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from dlms.datasets.image_edit_media import (  # noqa: PLC0415
        read_media_bytes,
        resolve_parquet_direct_media_uri,
    )

    client = boto3.client(
        "s3",
        region_name=args.aws_region,
        config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "adaptive"}),
    )
    manifest, manifest_payload = load_manifest(client, args)
    shards = manifest["shards"]
    output_prefix = args.output_prefix.rstrip("/")
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()

    for dataset_source, required_count in SOURCE_COUNTS.items():
        shard = shards[SOURCE_SHARD_INDICES[dataset_source]]
        if shard.get("source_dataset_id") != dataset_source:
            raise ValueError(
                f"shard {SOURCE_SHARD_INDICES[dataset_source]} has source_dataset_id "
                f"{shard.get('source_dataset_id')!r}, expected {dataset_source!r}"
            )
        candidates = read_candidate_rows(
            client,
            shard=shard,
            dataset_source=dataset_source,
            seed=args.seed,
            range_bytes=args.range_bytes,
        )
        accepted = 0
        for candidate in candidates:
            if accepted >= required_count:
                break
            if candidate["sample_id"] in seen_sample_ids:
                continue
            source_uri = candidate["source_s3_uri"]
            resolved_uri: str | None = None
            try:
                if source_uri.startswith("parquet+s3://"):
                    resolved_uri = resolve_parquet_direct_media_uri(source_uri)
                payload = read_media_bytes(resolved_uri or source_uri)
                suffix, width, height = validate_image(payload, source_uri=source_uri)
            except Exception as exc:  # noqa: BLE001 - skip bad smoke candidates with audit
                failures.append(
                    {
                        "dataset_source": dataset_source,
                        "sample_id": candidate["sample_id"],
                        "source_s3_uri": source_uri,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            output_index = len(selected)
            output_uri = f"{output_prefix}/sources/{output_index:06d}{suffix}"
            content_type = "image/jpeg" if suffix == ".jpg" else f"image/{suffix.lstrip('.')}"
            put_bytes(client, output_uri, payload, content_type=content_type)
            selected.append(
                {
                    "s3_image_path": output_uri,
                    "sha256": sha256_bytes(payload),
                    "content_length": len(payload),
                    "source": {
                        "sample_id": candidate["sample_id"],
                        "dataset_source": dataset_source,
                        "source_s3_uri": source_uri,
                        "resolved_source_s3_uri": resolved_uri,
                        "source_width": width,
                        "source_height": height,
                        "training_shard_uri": shard["path"],
                        "training_shard_version_id": shard["version_id"],
                        "training_shard_sha256": shard["sha256"],
                        "training_shard_line": candidate["source_line_number"],
                    },
                }
            )
            seen_sample_ids.add(candidate["sample_id"])
            accepted += 1
        if accepted != required_count:
            raise RuntimeError(
                f"materialized only {accepted}/{required_count} {dataset_source} source images; "
                f"candidate_count={len(candidates)} failure_count={len(failures)}"
            )

    if len(selected) != sum(SOURCE_COUNTS.values()):
        raise AssertionError("internal smoke row count mismatch")
    manifest_bytes = b"".join(canonical_json_bytes(row) for row in selected)
    smoke_manifest_uri = f"{output_prefix}/source_manifest.jsonl"
    put_bytes(client, smoke_manifest_uri, manifest_bytes, content_type="application/x-ndjson")
    report = {
        "complete": True,
        "format_version": "image_edit_v6_source_only_smoke_v1",
        "output_prefix": f"{output_prefix}/",
        "source_manifest_uri": smoke_manifest_uri,
        "source_manifest_sha256": sha256_bytes(manifest_bytes),
        "source_rows": len(selected),
        "source_counts": SOURCE_COUNTS,
        "selection_seed": args.seed,
        "selection_range_bytes_per_shard": args.range_bytes,
        "training_manifest": {
            "uri": args.training_manifest_uri,
            "version_id": args.training_manifest_version_id,
            "sha256": sha256_bytes(manifest_payload),
            "rows": EXPECTED_ROWS,
            "shards": EXPECTED_SHARDS,
        },
        "source_bytes_published": sum(int(row["content_length"]) for row in selected),
        "failed_candidates": failures,
        "target_media_read_count": 0,
        "target_fields_emitted": [],
    }
    report_bytes = canonical_json_bytes(report)
    report_uri = f"{output_prefix}/_PREP_SUCCESS.json"
    put_bytes(client, report_uri, report_bytes, content_type="application/json")
    report["report_uri"] = report_uri
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest-uri", required=True)
    parser.add_argument("--training-manifest-version-id", required=True)
    parser.add_argument("--training-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--range-bytes", type=int, default=DEFAULT_RANGE_BYTES)
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.range_bytes < 1024 * 1024:
        raise SystemExit("--range-bytes must be at least 1 MiB")
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
