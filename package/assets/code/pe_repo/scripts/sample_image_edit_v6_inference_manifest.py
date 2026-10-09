#!/usr/bin/env python3
"""Select one immutable V6 training shard and reservoir-sample inference rows.

The published assistant label is intentionally omitted from the inference
manifest.  Only the two ordered image URIs, raw instruction, sample identity,
and immutable source provenance are retained, preventing label leakage while
making the random cohort exactly reproducible.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 object URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def validate_media_uri(uri: str) -> None:
    if uri.startswith("s3://"):
        parse_s3_uri(uri)
        return
    parsed = urlparse(uri)
    if (
        parsed.scheme == "parquet+s3"
        and parsed.netloc
        and parsed.path.lstrip("/").endswith(".parquet")
        and parsed.query
        and parsed.fragment
    ):
        return
    raise ValueError(f"unsupported training media URI: {uri}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_training_manifest(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("training manifest must be an object")
    if value.get("format_version") != "image_edit_v6_training_manifest_v1":
        raise ValueError("unexpected training manifest format_version")
    if value.get("complete") is not True:
        raise ValueError("training manifest is not complete")
    shards = value.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("training manifest has no shards")
    expected_count = len(shards)
    seen_paths: set[str] = set()
    for index, raw in enumerate(shards):
        if not isinstance(raw, Mapping):
            raise ValueError(f"shards[{index}] must be an object")
        path = str(raw.get("path") or "")
        version_id = str(raw.get("version_id") or "")
        sha256 = str(raw.get("sha256") or "")
        if raw.get("part_index") != index or raw.get("part_count") != expected_count:
            raise ValueError(f"shards[{index}] has inconsistent part identity")
        parse_s3_uri(path)
        if path in seen_paths or not version_id:
            raise ValueError(f"shards[{index}] path/version identity is invalid")
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError(f"shards[{index}] sha256 is invalid")
        if int(raw.get("rows") or 0) <= 0 or int(raw.get("bytes") or 0) <= 0:
            raise ValueError(f"shards[{index}] rows/bytes must be positive")
        seen_paths.add(path)
    return shards


def project_inference_row(
    row: Any,
    *,
    shard: Mapping[str, Any],
    source_line: int,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"source line {source_line} must be an object")
    required_strings = ("sample_id", "instruction", "source_s3_uri", "target_s3_uri")
    values: dict[str, str] = {}
    for field in required_strings:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source line {source_line}: {field} must be a non-empty string")
        values[field] = value.strip()
    validate_media_uri(values["source_s3_uri"])
    validate_media_uri(values["target_s3_uri"])
    final_json = row.get("final_json")
    if not isinstance(final_json, Mapping):
        raise ValueError(f"source line {source_line}: final_json must be an object")
    return {
        "sample_id": values["sample_id"],
        "instruction": values["instruction"],
        "reference_image": values["source_s3_uri"],
        "target_image": values["target_s3_uri"],
        "dataset_source": {
            "training_shard_uri": str(shard["path"]),
            "training_shard_version_id": str(shard["version_id"]),
            "training_shard_sha256": str(shard["sha256"]),
            "training_shard_line": source_line,
            "training_status": str(row.get("status") or ""),
            "target_label_sha256": canonical_json_sha256(final_json),
        },
    }


def sample_local_shard(
    path: Path,
    *,
    shard: Mapping[str, Any],
    sample_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    reservoir: list[tuple[int, dict[str, Any]]] = []
    source_rows = 0
    with path.open("rb") as handle:
        for physical_line, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            source_rows += 1
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{physical_line}: invalid JSON: {exc}") from exc
            projected = project_inference_row(value, shard=shard, source_line=physical_line)
            if source_rows <= sample_count:
                reservoir.append((physical_line, projected))
            else:
                replacement = rng.randrange(source_rows)
                if replacement < sample_count:
                    reservoir[replacement] = (physical_line, projected)
    if source_rows < sample_count:
        raise ValueError(f"shard has only {source_rows} rows; requested {sample_count}")
    expected_rows = int(shard["rows"])
    if source_rows != expected_rows:
        raise ValueError(f"shard row count mismatch: actual={source_rows} expected={expected_rows}")
    reservoir.sort(key=lambda item: item[0])
    rows = [row for _, row in reservoir]
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != sample_count:
        raise ValueError("sampled inference rows contain duplicate sample_id values")
    status_counts = Counter(row["dataset_source"]["training_status"] for row in rows)
    selected_lines = [line for line, _ in reservoir]
    return rows, {
        "source_rows": source_rows,
        "sampled_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "selected_line_min": min(selected_lines),
        "selected_line_max": max(selected_lines),
        "selected_lines_sha256": sha256_bytes(
            ("\n".join(str(line) for line in selected_lines) + "\n").encode("ascii")
        ),
        "selected_sample_ids_sha256": sha256_bytes(
            ("\n".join(sample_ids) + "\n").encode("utf-8")
        ),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_identity(path)


def s3_client(profile: str | None, region: str) -> Any:
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client(
        "s3",
        region_name=region,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = s3_client(args.aws_profile, args.aws_region)
    manifest_bucket, manifest_key = parse_s3_uri(args.training_manifest_uri)
    manifest_response = client.get_object(
        Bucket=manifest_bucket,
        Key=manifest_key,
        VersionId=args.training_manifest_version_id,
    )
    manifest_bytes = manifest_response["Body"].read()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if manifest_sha256 != args.training_manifest_sha256:
        raise ValueError(
            "training manifest SHA-256 mismatch: "
            f"actual={manifest_sha256} expected={args.training_manifest_sha256}"
        )
    if str(manifest_response.get("VersionId") or "") != args.training_manifest_version_id:
        raise ValueError("training manifest VersionId changed")
    manifest = json.loads(manifest_bytes)
    shards = validate_training_manifest(manifest)

    rng = random.Random(args.seed)
    shard_index = rng.randrange(len(shards))
    shard = shards[shard_index]
    shard_bucket, shard_key = parse_s3_uri(str(shard["path"]))
    head = client.head_object(
        Bucket=shard_bucket,
        Key=shard_key,
        VersionId=str(shard["version_id"]),
    )
    if str(head.get("VersionId") or "") != str(shard["version_id"]):
        raise ValueError("selected shard VersionId changed")
    if int(head.get("ContentLength") or -1) != int(shard["bytes"]):
        raise ValueError("selected shard remote byte count changed")

    with tempfile.TemporaryDirectory(prefix="image-edit-v6-random-shard.") as temporary:
        local_shard = Path(temporary) / "source.jsonl"
        client.download_file(
            shard_bucket,
            shard_key,
            str(local_shard),
            ExtraArgs={"VersionId": str(shard["version_id"])},
        )
        actual_bytes, actual_sha256 = file_identity(local_shard)
        if actual_bytes != int(shard["bytes"]) or actual_sha256 != str(shard["sha256"]):
            raise ValueError(
                "selected shard identity mismatch: "
                f"bytes={actual_bytes}/{shard['bytes']} sha256={actual_sha256}/{shard['sha256']}"
            )
        rows, sample_summary = sample_local_shard(
            local_shard,
            shard=shard,
            sample_count=args.sample_count,
            rng=rng,
        )

    output_bytes, output_sha256 = write_jsonl(args.output_jsonl, rows)
    summary = {
        "format_version": "image_edit_v6_random_training_shard_inference_sample_v1",
        "complete": True,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "training_manifest": {
            "uri": args.training_manifest_uri,
            "version_id": args.training_manifest_version_id,
            "sha256": manifest_sha256,
            "total_rows": int(manifest["total_rows"]),
            "shard_count": len(shards),
        },
        "selected_shard": dict(shard),
        "sampling": sample_summary,
        "output": {
            "path": str(args.output_jsonl),
            "rows": len(rows),
            "bytes": output_bytes,
            "sha256": output_sha256,
            "contains_target_labels": False,
        },
    }
    atomic_json(args.summary_json, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest-uri", required=True)
    parser.add_argument("--training-manifest-version-id", required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")
    if len(args.training_manifest_sha256) != 64:
        parser.error("--training-manifest-sha256 must contain 64 hex characters")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
