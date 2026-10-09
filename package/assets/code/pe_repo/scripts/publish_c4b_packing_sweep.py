#!/usr/bin/env python3
"""Publish small packing-sweep results to S3, committing ``_SUCCESS`` last."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    SHA256_METADATA_KEY,
    S3Location,
    canonical_json_bytes,
    sha256_bytes,
    upload_verified,
    utc_now,
    verify_object_stat,
)


def result_files(root: Path) -> list[Path]:
    required = [root / "summary.json", root / "summary.md"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required sweep results: {', '.join(map(str, missing))}")
    files = list(required)
    for directory in sorted(root.glob("packing-*")):
        status = directory / "status.json"
        if status.is_file():
            files.append(status)
        direct_log = directory / "logging.jsonl"
        if direct_log.is_file():
            files.append(direct_log)
        files.extend(sorted(path for path in directory.glob("*/logging.jsonl") if path.is_file()))
    return files


def publish(root: Path, output_s3: str, *, region: str) -> str:
    destination = S3Location.parse(output_s3.rstrip("/"))
    store = Boto3ObjectStore(region=region, max_concurrency=2)
    records = []
    for source in result_files(root):
        relative = source.relative_to(root).as_posix()
        uri = destination.join(relative).uri
        size, digest = upload_verified(store, source, uri)
        records.append({"path": relative, "uri": uri, "size": size, "sha256": digest})

    # Readers only trust a sweep after this object exists.  Every referenced object has
    # already been uploaded and HEAD-verified, so publishing the marker is the commit.
    success = {
        "schema_version": 1,
        "created_at": utc_now(),
        "root": destination.uri,
        "files": records,
    }
    payload = canonical_json_bytes(success)
    digest = sha256_bytes(payload)
    success_uri = destination.join("_SUCCESS").uri
    store.put_bytes(success_uri, payload, metadata={SHA256_METADATA_KEY: digest})
    verify_object_stat(store, success_uri, size=len(payload), sha256=digest)
    return success_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(publish(args.root, args.output_s3, region=args.region))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
