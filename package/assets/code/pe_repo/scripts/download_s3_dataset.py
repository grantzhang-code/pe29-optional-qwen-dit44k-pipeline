#!/usr/bin/env python3
"""Atomically download one training JSONL from S3 and verify its SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"S3 URI must start with s3://, got {uri!r}")
    bucket, separator, key = uri[5:].partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"S3 URI must include a bucket and key, got {uri!r}")
    return bucket, key


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    s3_uri: str,
    destination: Path,
    expected_sha256: str,
    *,
    region: str,
) -> dict[str, object]:
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return {
                "status": "reused",
                "s3_uri": s3_uri,
                "local_path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": actual,
            }
        raise RuntimeError(
            f"existing destination SHA-256 mismatch: path={destination} "
            f"expected={expected_sha256} actual={actual}"
        )

    bucket, key = parse_s3_uri(s3_uri)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        boto3.client("s3", region_name=region).download_file(bucket, key, str(temporary_path))
        actual = sha256_file(temporary_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"downloaded dataset SHA-256 mismatch: uri={s3_uri} "
                f"expected={expected_sha256} actual={actual}"
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "status": "downloaded",
        "s3_uri": s3_uri,
        "local_path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": expected_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    result = download_verified(args.s3_uri, args.output, args.sha256, region=args.region)
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(f"[download_s3_dataset] {rendered}", flush=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
