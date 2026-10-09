#!/usr/bin/env python3
"""Shard and immutably publish externally reviewed V6 training JSONLs.

This publisher intentionally performs no semantic quality review.  It preserves
each non-empty JSONL row byte-for-byte (apart from adding a missing final
newline), distributes rows from one or more pinned sources round-robin across
physical shards, records object identities, and emits the minimal immutable
manifest consumed by the V6 loader.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    ANNOTATION_CONTRACT_SHA256,
    ANNOTATION_CONTRACT_VERSION,
    MANIFEST_FORMAT,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    ROW_FORMAT,
    TARGET_SCHEMA,
)
from scripts.publish_image_edit_v6_full64 import (  # noqa: E402
    S3Location,
    _client,
    _upload_file,
    child_location,
    parse_s3_uri,
)
from scripts.publish_image_edit_v6_training_manifest import (  # noqa: E402
    publish_training_manifest,
)


ALLOWED_STATUSES = ("validated", "manual_review_required")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(
    client: Any,
    source: S3Location,
    *,
    version_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    head_args: dict[str, Any] = {"Bucket": source.bucket, "Key": source.key}
    if version_id:
        head_args["VersionId"] = version_id
    head = client.head_object(**head_args)
    actual_version = str(head.get("VersionId") or "")
    actual_sha256 = str((head.get("Metadata") or {}).get("sha256") or "")
    if not actual_version or actual_version == "null":
        raise ValueError(f"source object has no VersionId: {source.uri}")
    if version_id and actual_version != version_id:
        raise ValueError(
            f"source VersionId changed: actual={actual_version!r} expected={version_id!r}"
        )
    if not actual_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(
            f"source SHA-256 metadata mismatch: actual={actual_sha256!r} "
            f"expected={expected_sha256!r}"
        )
    return {
        "uri": source.uri,
        "version_id": actual_version,
        "sha256": actual_sha256,
        "bytes": int(head["ContentLength"]),
    }


def split_jsonl(source: Path, shard_dir: Path, shard_count: int) -> tuple[list[dict], Counter]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shard_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        shard_dir / f"part-{index:05d}-of-{shard_count:05d}.jsonl"
        for index in range(shard_count)
    ]
    streams = [path.open("wb") for path in paths]
    row_counts = [0] * shard_count
    status_counts: Counter[str] = Counter()
    try:
        with source.open("rb") as input_stream:
            row_index = 0
            for line_number, line in enumerate(input_stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
                if not isinstance(value, Mapping):
                    raise ValueError(f"{source}:{line_number}: row must be an object")
                status = str(value.get("status") or "")
                if status not in ALLOWED_STATUSES:
                    raise ValueError(
                        f"{source}:{line_number}: unsupported status {status!r}"
                    )
                shard_index = row_index % shard_count
                streams[shard_index].write(line if line.endswith(b"\n") else line + b"\n")
                row_counts[shard_index] += 1
                status_counts[status] += 1
                row_index += 1
    finally:
        for stream in streams:
            stream.close()
    if not status_counts or any(count < 1 for count in row_counts):
        raise ValueError(
            f"source has too few rows for {shard_count} non-empty shards: "
            f"rows={sum(row_counts)}"
        )
    records = [
        {
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "rows": row_counts[index],
            "sha256": _sha256_file(path),
            "part_index": index,
            "part_count": shard_count,
        }
        for index, path in enumerate(paths)
    ]
    return records, status_counts


def split_s3_jsonls(
    client: Any,
    sources: Sequence[Mapping[str, Any]],
    shard_dir: Path,
    shard_count: int,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    """Stream pinned sources into globally balanced shards without rewriting rows."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not sources:
        raise ValueError("at least one source is required")
    shard_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        shard_dir / f"part-{index:05d}-of-{shard_count:05d}.jsonl"
        for index in range(shard_count)
    ]
    streams = [path.open("wb") for path in paths]
    row_counts = [0] * shard_count
    status_counts: Counter[str] = Counter()
    completed_sources: list[dict[str, Any]] = []
    global_row_index = 0
    try:
        for source_index, source_record in enumerate(sources):
            source = parse_s3_uri(str(source_record["uri"]))
            response = client.get_object(
                Bucket=source.bucket,
                Key=source.key,
                VersionId=str(source_record["version_id"]),
            )
            body = response["Body"]
            source_digest = hashlib.sha256()
            source_bytes = 0
            source_rows = 0
            source_status_counts: Counter[str] = Counter()
            print(
                f"[split] source {source_index + 1}/{len(sources)} begin: "
                f"{source.uri} bytes={source_record['bytes']}",
                flush=True,
            )
            try:
                for line_number, line in enumerate(
                    body.iter_lines(chunk_size=8 * 1024 * 1024, keepends=True), 1
                ):
                    source_digest.update(line)
                    source_bytes += len(line)
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{source.uri}:{line_number}: invalid JSON"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise ValueError(
                            f"{source.uri}:{line_number}: row must be an object"
                        )
                    status = str(value.get("status") or "")
                    if status not in ALLOWED_STATUSES:
                        raise ValueError(
                            f"{source.uri}:{line_number}: unsupported status {status!r}"
                        )
                    shard_index = global_row_index % shard_count
                    streams[shard_index].write(
                        line if line.endswith(b"\n") else line + b"\n"
                    )
                    row_counts[shard_index] += 1
                    status_counts[status] += 1
                    source_status_counts[status] += 1
                    source_rows += 1
                    global_row_index += 1
                    if global_row_index % 100_000 == 0:
                        print(
                            f"[split] rows={global_row_index:,} "
                            f"source={source_index:02d} source_rows={source_rows:,}",
                            flush=True,
                        )
            finally:
                body.close()
            actual_digest = source_digest.hexdigest()
            if source_bytes != int(source_record["bytes"]):
                raise ValueError(
                    f"{source.uri}: streamed byte count mismatch: "
                    f"actual={source_bytes} expected={source_record['bytes']}"
                )
            if actual_digest != str(source_record["sha256"]):
                raise ValueError(
                    f"{source.uri}: streamed SHA-256 mismatch: "
                    f"actual={actual_digest} expected={source_record['sha256']}"
                )
            completed_sources.append(
                {
                    **dict(source_record),
                    "rows": source_rows,
                    "status_counts": dict(sorted(source_status_counts.items())),
                    "source_index": source_index,
                    "source_count": len(sources),
                }
            )
            print(
                f"[split] source {source_index + 1}/{len(sources)} complete: "
                f"rows={source_rows:,} sha256={actual_digest}",
                flush=True,
            )
    finally:
        for stream in streams:
            stream.close()
    if not status_counts or any(count < 1 for count in row_counts):
        raise ValueError(
            f"sources have too few rows for {shard_count} non-empty shards: "
            f"rows={sum(row_counts)}"
        )
    records = [
        {
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "rows": row_counts[index],
            "sha256": _sha256_file(path),
            "part_index": index,
            "part_count": shard_count,
        }
        for index, path in enumerate(paths)
    ]
    return records, status_counts, completed_sources


def _publish_shard(
    client: Any,
    output_prefix: S3Location,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(str(record["local_path"]))
    destination = child_location(output_prefix, f"shards/{source.name}")
    head = _upload_file(
        client,
        source,
        destination,
        sha256=str(record["sha256"]),
        content_type="application/x-ndjson",
        rows=int(record["rows"]),
    )
    return {
        "path": destination.uri,
        "version_id": str(head["VersionId"]),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
        "rows": int(record["rows"]),
        "part_index": int(record["part_index"]),
        "part_count": int(record["part_count"]),
    }


def _manifest_document(
    sources: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    status_counts: Mapping[str, int],
) -> dict[str, Any]:
    ordered = sorted(shards, key=lambda item: int(item["part_index"]))
    total_rows = sum(int(item["rows"]) for item in ordered)
    if total_rows != sum(int(value) for value in status_counts.values()):
        raise ValueError("status counts do not conserve rows")
    contract: dict[str, Any] = {
        "output_format": "prepared",
        "row_format_version": ROW_FORMAT,
        "runtime_loader": "dlms/datasets/image_edit_v6_full_json.py",
        "target_schema": TARGET_SCHEMA,
        "target_json_field": "final_json",
        "instruction_field": "instruction",
        "reference_image_field": "source_s3_uri",
        "target_image_field": "target_s3_uri",
        "audit_valid_field": "call2_audit.valid",
        "image_order": ["reference frame_0", "target"],
        "selection": "authoritative_call2_final_json",
        "allowed_statuses": list(ALLOWED_STATUSES),
        "bbox_coordinate_order": "xyxy",
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "source_stage": "offline_pruned_edit_instruction",
        "review_policy": "externally_reviewed_no_reaudit_requested",
        "edit_instruction_only_change": True,
    }
    if len(sources) == 1:
        contract["source_object"] = dict(sources[0])
    else:
        contract["source_objects"] = [dict(source) for source in sources]
    return {
        "format_version": MANIFEST_FORMAT,
        "complete": True,
        "total_rows": total_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "physical_shard_count": len(ordered),
        "paths": [str(item["path"]) for item in ordered],
        "shards": [dict(item) for item in ordered],
        "dataset_contract": contract,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", action="append", required=True)
    parser.add_argument("--source-version-id", action="append", required=True)
    parser.add_argument("--source-sha256", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--region", default="us-east-1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not (
        len(args.source_uri)
        == len(args.source_version_id)
        == len(args.source_sha256)
    ):
        raise ValueError(
            "--source-uri, --source-version-id, and --source-sha256 counts must match"
        )
    output_prefix = parse_s3_uri(args.output_prefix)
    client = _client(args.aws_profile, args.region)
    source_records = [
        _source_identity(
            client,
            parse_s3_uri(uri),
            version_id=version_id,
            expected_sha256=sha256,
        )
        for uri, version_id, sha256 in zip(
            args.source_uri,
            args.source_version_id,
            args.source_sha256,
            strict=True,
        )
    ]
    local_records, status_counts, completed_sources = split_s3_jsonls(
        client, source_records, work_dir / "shards", args.shards
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        published = list(
            executor.map(
                lambda record: _publish_shard(client, output_prefix, record),
                local_records,
            )
        )
    manifest = _manifest_document(completed_sources, published, status_counts)
    manifest_path = work_dir / "image_edit_v6_full_json.manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    pins = publish_training_manifest(
        manifest_path,
        args.manifest_uri,
        aws_profile=args.aws_profile,
        region=args.region,
    )
    print(
        json.dumps(
            {
                "sources": completed_sources,
                "output_prefix": output_prefix.uri,
                "manifest_uri": args.manifest_uri,
                **pins,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
