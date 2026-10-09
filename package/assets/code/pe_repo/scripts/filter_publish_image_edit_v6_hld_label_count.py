#!/usr/bin/env python3
"""Filter excessive HLD label enumerations and publish immutable V6 shards.

The input is an already-published image-edit-v6 training manifest.  Every
source shard is read at its pinned S3 VersionId and verified against its byte
count and SHA-256.  Rows whose ``final_json.high_level_description`` contains
more than the configured number of numbered angle-bracket label references
are removed; every accepted JSONL row is otherwise copied byte-for-byte.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import MANIFEST_FORMAT  # noqa: E402
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


FILTER_VERSION = "image_edit_v6_hld_label_reference_count_v1"
HLD_NUMBERED_LABEL_PATTERN = r"<[^<>\r\n]+ [1-9][0-9]*>"
HLD_NUMBERED_LABEL_RE = re.compile(HLD_NUMBERED_LABEL_PATTERN)
ALLOWED_STATUSES = ("validated", "manual_review_required")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_hld_label_references(row: Mapping[str, Any], *, context: str) -> int:
    target = row.get("final_json")
    if not isinstance(target, Mapping):
        raise ValueError(f"{context}: final_json must be an object")
    hld = target.get("high_level_description")
    if not isinstance(hld, str):
        raise ValueError(
            f"{context}: final_json.high_level_description must be a string"
        )
    return sum(1 for _match in HLD_NUMBERED_LABEL_RE.finditer(hld))


def filter_jsonl_stream(
    lines: Any,
    output_stream: Any,
    rejection_stream: Any,
    *,
    source_context: str,
    source_part_index: int,
    maximum_label_references: int,
) -> dict[str, Any]:
    """Filter one JSONL byte stream while preserving accepted row bytes."""

    if maximum_label_references < 0:
        raise ValueError("maximum_label_references must be non-negative")
    source_digest = hashlib.sha256()
    source_bytes = 0
    output_digest = hashlib.sha256()
    output_bytes = 0
    input_rows = 0
    output_rows = 0
    input_status_counts: Counter[str] = Counter()
    output_status_counts: Counter[str] = Counter()
    rejected_status_counts: Counter[str] = Counter()
    rejected_label_reference_counts: Counter[int] = Counter()

    for line_number, line in enumerate(lines, 1):
        if not isinstance(line, bytes):
            raise TypeError(f"{source_context}:{line_number}: expected bytes")
        source_digest.update(line)
        source_bytes += len(line)
        if not line.strip():
            continue
        context = f"{source_context}:{line_number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{context}: row must be an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{context}: sample_id must be a non-empty string")
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{context}: unsupported status {status!r}")
        reference_count = count_hld_label_references(row, context=context)

        input_rows += 1
        input_status_counts[str(status)] += 1
        if reference_count > maximum_label_references:
            rejection = {
                "filter_version": FILTER_VERSION,
                "hld_label_reference_count": reference_count,
                "maximum_allowed_label_references": maximum_label_references,
                "sample_id": sample_id,
                "source_line_number": line_number,
                "source_part_index": source_part_index,
                "status": status,
            }
            rejection_stream.write(
                json.dumps(
                    rejection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            rejected_status_counts[str(status)] += 1
            rejected_label_reference_counts[reference_count] += 1
            continue

        output_line = line if line.endswith(b"\n") else line + b"\n"
        output_stream.write(output_line)
        output_digest.update(output_line)
        output_bytes += len(output_line)
        output_rows += 1
        output_status_counts[str(status)] += 1

    if input_rows != output_rows + sum(rejected_status_counts.values()):
        raise ValueError(f"{source_context}: row conservation failed")
    if output_rows < 1:
        raise ValueError(f"{source_context}: filtering produced an empty shard")
    return {
        "source_sha256": source_digest.hexdigest(),
        "source_bytes": source_bytes,
        "input_rows": input_rows,
        "input_status_counts": dict(sorted(input_status_counts.items())),
        "output_sha256": output_digest.hexdigest(),
        "output_bytes": output_bytes,
        "output_rows": output_rows,
        "output_status_counts": dict(sorted(output_status_counts.items())),
        "rejected_rows": sum(rejected_status_counts.values()),
        "rejected_status_counts": dict(sorted(rejected_status_counts.items())),
        "rejected_label_reference_counts": {
            str(count): occurrences
            for count, occurrences in sorted(rejected_label_reference_counts.items())
        },
    }


def _validate_source_manifest(
    manifest: Mapping[str, Any], *, expected_sha256: str, payload: bytes
) -> list[dict[str, Any]]:
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "source manifest SHA-256 mismatch: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    if manifest.get("format_version") != MANIFEST_FORMAT:
        raise ValueError("source manifest format is incompatible")
    if manifest.get("complete") is not True:
        raise ValueError("source manifest is incomplete")
    shards = manifest.get("shards")
    paths = manifest.get("paths")
    if not isinstance(shards, list) or not isinstance(paths, list) or not shards:
        raise ValueError("source manifest must contain shards and paths")
    if len(shards) != int(manifest.get("physical_shard_count") or 0):
        raise ValueError("source shard count does not match physical_shard_count")
    if [str(item.get("path") or "") for item in shards] != [str(path) for path in paths]:
        raise ValueError("source paths do not match ordered shard records")
    ordered = sorted((dict(item) for item in shards), key=lambda item: int(item["part_index"]))
    if [int(item["part_index"]) for item in ordered] != list(range(len(ordered))):
        raise ValueError("source shard part indexes are not contiguous")
    total_rows = 0
    for item in ordered:
        if (
            int(item.get("part_count") or 0) != len(ordered)
            or int(item.get("rows") or 0) < 1
            or int(item.get("bytes") or 0) < 1
            or len(str(item.get("sha256") or "")) != 64
            or not str(item.get("path") or "").startswith("s3://")
            or str(item.get("version_id") or "") in {"", "null"}
        ):
            raise ValueError(f"invalid source shard record: {item}")
        total_rows += int(item["rows"])
    if total_rows != int(manifest.get("total_rows") or 0):
        raise ValueError("source shard rows do not match total_rows")
    status_counts = manifest.get("status_counts")
    if not isinstance(status_counts, Mapping) or sum(
        int(value) for value in status_counts.values()
    ) != total_rows:
        raise ValueError("source status_counts do not conserve rows")
    return ordered


def _verify_source_head(client: Any, source: Mapping[str, Any]) -> None:
    location = parse_s3_uri(str(source["path"]))
    head = client.head_object(
        Bucket=location.bucket,
        Key=location.key,
        VersionId=str(source["version_id"]),
    )
    if (
        int(head.get("ContentLength", -1)) != int(source["bytes"])
        or str(head.get("VersionId") or "") != str(source["version_id"])
        or str((head.get("Metadata") or {}).get("sha256") or "")
        != str(source["sha256"])
    ):
        raise ValueError(f"source shard identity mismatch: {location.uri}")


def _filter_publish_shard(
    source: Mapping[str, Any],
    *,
    output_prefix: S3Location,
    work_dir: Path,
    maximum_label_references: int,
    aws_profile: str,
    region: str,
) -> dict[str, Any]:
    client = _client(aws_profile, region)
    _verify_source_head(client, source)
    source_location = parse_s3_uri(str(source["path"]))
    part_index = int(source["part_index"])
    filename = f"part-{part_index:05d}-of-{int(source['part_count']):05d}.jsonl"
    destination = child_location(output_prefix, f"shards/{filename}")
    if destination.uri == source_location.uri:
        raise ValueError("output shard path must differ from the source shard path")

    output_path = work_dir / "active" / filename
    rejection_path = work_dir / "rejections" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    response = client.get_object(
        Bucket=source_location.bucket,
        Key=source_location.key,
        VersionId=str(source["version_id"]),
    )
    body = response["Body"]
    try:
        with output_path.open("wb") as output_stream, rejection_path.open(
            "wb"
        ) as rejection_stream:
            stats = filter_jsonl_stream(
                body.iter_lines(chunk_size=8 * 1024 * 1024, keepends=True),
                output_stream,
                rejection_stream,
                source_context=source_location.uri,
                source_part_index=part_index,
                maximum_label_references=maximum_label_references,
            )
    finally:
        body.close()

    if (
        stats["source_bytes"] != int(source["bytes"])
        or stats["source_sha256"] != str(source["sha256"])
        or stats["input_rows"] != int(source["rows"])
    ):
        raise ValueError(f"streamed source identity mismatch: {source_location.uri}")
    if output_path.stat().st_size != stats["output_bytes"]:
        raise ValueError(f"output byte accounting mismatch: {output_path}")
    head = _upload_file(
        client,
        output_path,
        destination,
        sha256=str(stats["output_sha256"]),
        content_type="application/x-ndjson",
        rows=int(stats["output_rows"]),
    )
    output_path.unlink()
    return {
        "source": dict(source),
        "shard": {
            "path": destination.uri,
            "version_id": str(head["VersionId"]),
            "sha256": str(stats["output_sha256"]),
            "bytes": int(stats["output_bytes"]),
            "rows": int(stats["output_rows"]),
            "part_index": part_index,
            "part_count": int(source["part_count"]),
        },
        "rejection_path": str(rejection_path),
        "stats": stats,
    }


def _aggregate_rejections(
    results: Sequence[Mapping[str, Any]], destination: Path
) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with destination.open("wb") as output:
        for result in sorted(results, key=lambda item: int(item["shard"]["part_index"])):
            path = Path(str(result["rejection_path"]))
            with path.open("rb") as stream:
                for line in stream:
                    output.write(line)
                    digest.update(line)
                    if line.strip():
                        rows += 1
    return rows, digest.hexdigest()


def _build_manifest(
    source_manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    source_manifest_identity: Mapping[str, Any],
    rejection_audit_identity: Mapping[str, Any],
    maximum_label_references: int,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item["shard"]["part_index"]))
    input_status_counts: Counter[str] = Counter()
    output_status_counts: Counter[str] = Counter()
    rejected_status_counts: Counter[str] = Counter()
    rejected_label_reference_counts: Counter[int] = Counter()
    for result in ordered:
        stats = result["stats"]
        input_status_counts.update(stats["input_status_counts"])
        output_status_counts.update(stats["output_status_counts"])
        rejected_status_counts.update(stats["rejected_status_counts"])
        rejected_label_reference_counts.update(
            {
                int(count): int(occurrences)
                for count, occurrences in stats[
                    "rejected_label_reference_counts"
                ].items()
            }
        )
    input_rows = sum(input_status_counts.values())
    output_rows = sum(output_status_counts.values())
    rejected_rows = sum(rejected_status_counts.values())
    if input_rows != int(source_manifest["total_rows"]):
        raise ValueError("global input rows differ from source manifest")
    if input_rows != output_rows + rejected_rows:
        raise ValueError("global row conservation failed")
    if dict(sorted(input_status_counts.items())) != dict(
        sorted(source_manifest["status_counts"].items())
    ):
        raise ValueError("global input status counts differ from source manifest")
    if rejected_rows != int(rejection_audit_identity["rows"]):
        raise ValueError("rejection audit row count mismatch")

    manifest = copy.deepcopy(dict(source_manifest))
    shards = [dict(result["shard"]) for result in ordered]
    manifest.update(
        {
            "complete": True,
            "total_rows": output_rows,
            "status_counts": dict(sorted(output_status_counts.items())),
            "physical_shard_count": len(shards),
            "paths": [str(shard["path"]) for shard in shards],
            "shards": shards,
        }
    )
    contract = manifest.get("dataset_contract")
    if not isinstance(contract, dict):
        raise ValueError("source manifest dataset_contract must be an object")
    contract["source_stage"] = "offline_pruned_edit_instruction_hld_label_filter"
    contract["source_manifest"] = dict(source_manifest_identity)
    contract["hld_label_reference_filter"] = {
        "complete": True,
        "comparison": "reject_if_count_greater_than_maximum",
        "count_semantics": "regex_occurrences_not_unique_labels",
        "field": "final_json.high_level_description",
        "filter_version": FILTER_VERSION,
        "input_rows": input_rows,
        "maximum_allowed_label_references": maximum_label_references,
        "numbered_label_regex": HLD_NUMBERED_LABEL_PATTERN,
        "output_rows": output_rows,
        "rejected_label_reference_counts": {
            str(count): occurrences
            for count, occurrences in sorted(rejected_label_reference_counts.items())
        },
        "rejected_rows": rejected_rows,
        "rejected_status_counts": dict(sorted(rejected_status_counts.items())),
        "rejection_audit": dict(rejection_audit_identity),
    }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-uri", required=True)
    parser.add_argument("--source-manifest-version-id", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--maximum-label-references", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--region", default="us-east-1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.maximum_label_references < 0:
        raise ValueError("--maximum-label-references must be non-negative")
    source_manifest_path = args.source_manifest.expanduser().resolve(strict=True)
    source_payload = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_payload)
    if not isinstance(source_manifest, Mapping):
        raise ValueError("source manifest must be a JSON object")
    sources = _validate_source_manifest(
        source_manifest,
        expected_sha256=args.source_manifest_sha256,
        payload=source_payload,
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = parse_s3_uri(args.output_prefix)
    client = _client(args.aws_profile, args.region)
    source_manifest_location = parse_s3_uri(args.source_manifest_uri)
    source_manifest_head = client.head_object(
        Bucket=source_manifest_location.bucket,
        Key=source_manifest_location.key,
        VersionId=args.source_manifest_version_id,
    )
    if (
        int(source_manifest_head.get("ContentLength", -1)) != len(source_payload)
        or str(source_manifest_head.get("VersionId") or "")
        != args.source_manifest_version_id
        or str((source_manifest_head.get("Metadata") or {}).get("sha256") or "")
        != args.source_manifest_sha256
    ):
        raise ValueError("source manifest S3 identity does not match its pinned bytes")

    print(
        f"[filter] begin shards={len(sources)} rows={source_manifest['total_rows']:,} "
        f"maximum_label_references={args.maximum_label_references}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _filter_publish_shard,
                source,
                output_prefix=output_prefix,
                work_dir=work_dir,
                maximum_label_references=args.maximum_label_references,
                aws_profile=args.aws_profile,
                region=args.region,
            ): int(source["part_index"])
            for source in sources
        }
        results: list[dict[str, Any]] = []
        for completed_index, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            result = future.result()
            results.append(result)
            stats = result["stats"]
            print(
                f"[filter] shard={futures[future]:05d} "
                f"output={stats['output_rows']:,} rejected={stats['rejected_rows']:,} "
                f"complete={completed_index}/{len(sources)}",
                flush=True,
            )

    rejection_path = work_dir / "hld_label_reference_filter_rejections.jsonl"
    rejection_rows, rejection_sha256 = _aggregate_rejections(results, rejection_path)
    rejection_destination = child_location(
        output_prefix, "audits/hld_label_reference_filter_rejections.jsonl"
    )
    rejection_head = _upload_file(
        client,
        rejection_path,
        rejection_destination,
        sha256=rejection_sha256,
        content_type="application/x-ndjson",
        rows=rejection_rows,
    )
    rejection_identity = {
        "bytes": rejection_path.stat().st_size,
        "path": rejection_destination.uri,
        "rows": rejection_rows,
        "sha256": rejection_sha256,
        "version_id": str(rejection_head["VersionId"]),
    }
    source_manifest_identity = {
        "bytes": len(source_payload),
        "sha256": args.source_manifest_sha256,
        "uri": args.source_manifest_uri,
        "version_id": args.source_manifest_version_id,
    }
    manifest = _build_manifest(
        source_manifest,
        results,
        source_manifest_identity=source_manifest_identity,
        rejection_audit_identity=rejection_identity,
        maximum_label_references=args.maximum_label_references,
    )
    manifest_path = work_dir / "image_edit_v6_full_json.manifest.json"
    manifest_path.write_bytes(
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    pins = publish_training_manifest(
        manifest_path,
        args.manifest_uri,
        aws_profile=args.aws_profile,
        region=args.region,
        client=client,
    )
    summary = {
        "filter": manifest["dataset_contract"]["hld_label_reference_filter"],
        "manifest_uri": args.manifest_uri,
        "output_prefix": output_prefix.uri,
        **pins,
    }
    summary_path = work_dir / "publication_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
