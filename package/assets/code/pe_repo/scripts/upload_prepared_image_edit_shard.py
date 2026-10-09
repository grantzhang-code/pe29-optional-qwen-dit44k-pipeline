#!/usr/bin/env python3
"""Upload one completed prepared shard to S3 and register it in a local manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_image_edit_full_json_shard import (  # noqa: E402
    S3Location,
    parse_s3_uri,
    update_prepared_manifest as update_v4_prepared_manifest,
)
from scripts.prepare_image_edit_v5_full_json_shard import (  # noqa: E402
    FORMAT_VERSION as V5_SHARD_FORMAT_VERSION,
    update_prepared_manifest as update_v5_prepared_manifest,
)


def _s3_destination(prefix: S3Location, path: Path) -> S3Location:
    return S3Location(prefix.bucket, f"{prefix.key.rstrip('/')}/{path.name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_one(
    client: Any,
    *,
    source: Path,
    destination: S3Location,
    sha256: str,
    rows: int | None,
    transfer_config: Any,
) -> tuple[str, str, int]:
    size = source.stat().st_size
    try:
        existing = client.head_object(Bucket=destination.bucket, Key=destination.key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        metadata = existing.get("Metadata") or {}
        if int(existing.get("ContentLength", -1)) == size and metadata.get("sha256") == sha256:
            return destination.uri, "already-present", size
        raise ValueError(
            f"refusing to overwrite mismatched S3 object: {destination.uri}; "
            f"local_size={size}, remote_size={existing.get('ContentLength')}, "
            f"local_sha256={sha256}, remote_sha256={metadata.get('sha256')}"
        )

    metadata = {"sha256": sha256}
    if rows is not None:
        metadata["rows"] = str(rows)
    client.upload_file(
        str(source),
        destination.bucket,
        destination.key,
        ExtraArgs={"ContentType": "application/x-ndjson", "Metadata": metadata},
        Config=transfer_config,
    )
    uploaded = client.head_object(Bucket=destination.bucket, Key=destination.key)
    if int(uploaded.get("ContentLength", -1)) != size:
        raise ValueError(f"uploaded S3 size mismatch: {destination.uri}")
    if (uploaded.get("Metadata") or {}).get("sha256") != sha256:
        raise ValueError(f"uploaded S3 sha256 metadata mismatch: {destination.uri}")
    return destination.uri, "uploaded", size


def upload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("boto3 is required for S3 publication") from exc

    summary_path = args.summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete") is not True or summary.get("output_format") != "prepared":
        raise ValueError(f"incomplete or non-prepared summary: {summary_path}")
    part_paths = [Path(item).resolve() for item in summary["physical_outputs"]]
    part_rows = [int(item) for item in summary["physical_output_rows"]]
    part_sha256 = [str(item) for item in summary["physical_output_sha256"]]
    if not (len(part_paths) == len(part_rows) == len(part_sha256)):
        raise ValueError("summary physical part metadata lengths differ")
    for path in part_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    prefix = parse_s3_uri(args.s3_prefix.rstrip("/"))
    session = boto3.Session(profile_name=args.aws_profile or None, region_name=args.region)
    client = session.client(
        "s3",
        region_name=args.region,
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=max(32, args.workers * args.multipart_concurrency * 2),
        ),
    )
    transfer_config = TransferConfig(
        max_concurrency=args.multipart_concurrency,
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        use_threads=True,
    )

    work = [
        (path, rows, digest)
        for path, rows, digest in zip(part_paths, part_rows, part_sha256, strict=True)
    ]
    results: list[tuple[str, str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _upload_one,
                client,
                source=path,
                destination=_s3_destination(prefix, path),
                sha256=digest,
                rows=rows,
                transfer_config=transfer_config,
            )
            for path, rows, digest in work
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[prepared-upload] {result[1]} bytes={result[2]:,} uri={result[0]}",
                flush=True,
            )

    summary_digest = _file_sha256(summary_path)
    summary_result = _upload_one(
        client,
        source=summary_path,
        destination=_s3_destination(prefix, summary_path),
        sha256=summary_digest,
        rows=None,
        transfer_config=transfer_config,
    )
    print(
        f"[prepared-upload] {summary_result[1]} bytes={summary_result[2]:,} "
        f"uri={summary_result[0]}",
        flush=True,
    )

    manifest_updater = (
        update_v5_prepared_manifest
        if summary.get("format_version") == V5_SHARD_FORMAT_VERSION
        else update_v4_prepared_manifest
    )
    manifest = manifest_updater(
        args.manifest.resolve(),
        summary=summary,
        part_paths=part_paths,
        part_rows=part_rows,
        part_sha256=part_sha256,
        entry_prefix=args.s3_prefix.rstrip("/"),
    )
    report = {
        "annotation_shard": int(summary["selected_shard"]),
        "parts": len(part_paths),
        "rows": sum(part_rows),
        "bytes": sum(item[2] for item in results),
        "uploaded": sum(item[1] == "uploaded" for item in results),
        "already_present": sum(item[1] == "already-present" for item in results),
        "manifest_complete": bool(manifest["complete"]),
        "manifest_annotation_shards": manifest["annotation_shards"],
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--aws-profile")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--multipart-concurrency", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.multipart_concurrency <= 0:
        raise ValueError("worker counts must be positive")
    upload(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
