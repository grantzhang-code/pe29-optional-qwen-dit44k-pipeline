#!/usr/bin/env python3
"""Prepare and publish all image-edit-v6 Call-2 partitions without overwrites.

Each source ``final_results.jsonl`` is downloaded at an immutable S3 VersionId,
validated and projected by ``prepare_image_edit_v6_full_json_partition.py``,
split into four deterministic training shards, and uploaded below a new prefix.
The publisher also creates versioned source snapshots with SHA-256 metadata so a
later manifest finalizer can prove the full publication is immutable.

With ``--prepared-root`` and ``--reuse-source-publication-identities``, it
instead publishes an already-filtered prepared tree.  That mode re-hashes and
re-counts every local shard and re-verifies each reused source at its pinned S3
VersionId before writing a new publication identity document.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_image_edit_v6_full64_manifest import (  # noqa: E402
    PUBLICATION_IDENTITIES_FORMAT,
)
from scripts.prepare_image_edit_v6_full_json_partition import (  # noqa: E402
    PreparePartitionOptions,
    json_document_bytes,
    prepare_partition,
    sha256_file,
)


DEFAULT_SOURCE_PREFIX = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/"
    "full64-v7-20260812-r6fix1/results"
)
DEFAULT_OUTPUT_PREFIX = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
    "full64-v7-call2-final-prepared-256shards-gavin-disjoint-v1"
)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class WorkerOptions:
    partition: int
    partition_count: int
    output_parts: int
    local_root: Path
    source_prefix: S3Location
    output_prefix: S3Location
    aws_profile: str
    region: str


@dataclass(frozen=True)
class PreparedOnlyWorkerOptions:
    partition: int
    partition_count: int
    output_parts: int
    prepared_root: Path
    output_prefix: S3Location
    source_identity: Mapping[str, Any]
    aws_profile: str
    region: str


def parse_s3_uri(value: str) -> S3Location:
    if not value.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI: {value}")
    bucket_key = value[5:]
    bucket, separator, key = bucket_key.partition("/")
    if not bucket or not separator or not key.strip("/"):
        raise ValueError(f"incomplete S3 URI: {value}")
    return S3Location(bucket, key.strip("/"))


def child_location(prefix: S3Location, relative: str) -> S3Location:
    return S3Location(prefix.bucket, f"{prefix.key.rstrip('/')}/{relative.lstrip('/')}")


def _client(profile: str, region: str):
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=profile, region_name=region).client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 12, "mode": "adaptive"},
            max_pool_connections=32,
        ),
    )


def _head_or_none(client: Any, location: S3Location) -> Mapping[str, Any] | None:
    from botocore.exceptions import ClientError

    try:
        return client.head_object(Bucket=location.bucket, Key=location.key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise


def _verified_head(
    client: Any,
    location: S3Location,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Mapping[str, Any]:
    head = _head_or_none(client, location)
    if head is None:
        raise ValueError(f"uploaded object is missing: {location.uri}")
    actual_bytes = int(head.get("ContentLength", -1))
    actual_sha256 = str((head.get("Metadata") or {}).get("sha256") or "")
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(
            f"published identity mismatch: {location.uri}; "
            f"bytes={actual_bytes}/{expected_bytes} sha256={actual_sha256}/{expected_sha256}"
        )
    version_id = str(head.get("VersionId") or "")
    if not version_id or version_id == "null":
        raise ValueError(f"S3 VersionId is missing: {location.uri}")
    return head


def _existing_exact(
    client: Any,
    location: S3Location,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Mapping[str, Any] | None:
    head = _head_or_none(client, location)
    if head is None:
        return None
    actual_bytes = int(head.get("ContentLength", -1))
    actual_sha256 = str((head.get("Metadata") or {}).get("sha256") or "")
    if actual_bytes == expected_bytes and actual_sha256 == expected_sha256:
        return head
    raise ValueError(
        f"refusing to overwrite a mismatched S3 object: {location.uri}; "
        f"bytes={actual_bytes}/{expected_bytes} sha256={actual_sha256}/{expected_sha256}"
    )


def _download_source(
    client: Any,
    source: S3Location,
    destination: Path,
    *,
    version_id: str,
    expected_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        client.download_file(
            source.bucket,
            source.key,
            str(temporary),
            ExtraArgs={"VersionId": version_id},
        )
        if temporary.stat().st_size != expected_bytes:
            raise ValueError(f"downloaded source size mismatch: {source.uri}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_reusable_summary(path: Path, partition: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("complete") is not True
        or int(value.get("partition_index", -1)) != partition
    ):
        raise ValueError(f"incomplete local partition summary: {path}")
    for shard in value.get("shards") or []:
        local_path = Path(str(shard.get("local_path") or ""))
        if (
            not local_path.is_file()
            or local_path.stat().st_size != int(shard.get("bytes") or -1)
            or sha256_file(local_path) != str(shard.get("sha256") or "")
        ):
            raise ValueError(f"local prepared shard differs from its summary: {local_path}")
    return value


def _upload_file(
    client: Any,
    source: Path,
    destination: S3Location,
    *,
    sha256: str,
    content_type: str,
    rows: int | None = None,
) -> Mapping[str, Any]:
    from boto3.s3.transfer import TransferConfig

    size = source.stat().st_size
    existing = _existing_exact(
        client,
        destination,
        expected_bytes=size,
        expected_sha256=sha256,
    )
    if existing is None:
        metadata = {"sha256": sha256}
        if rows is not None:
            metadata["rows"] = str(rows)
        client.upload_file(
            str(source),
            destination.bucket,
            destination.key,
            ExtraArgs={"ContentType": content_type, "Metadata": metadata},
            Config=TransferConfig(
                multipart_threshold=16 * 1024 * 1024,
                multipart_chunksize=32 * 1024 * 1024,
                max_concurrency=4,
                use_threads=True,
            ),
        )
    return _verified_head(
        client,
        destination,
        expected_bytes=size,
        expected_sha256=sha256,
    )


def _copy_source_snapshot(
    client: Any,
    source: S3Location,
    destination: S3Location,
    *,
    source_version_id: str,
    expected_bytes: int,
    expected_sha256: str,
    rows: int,
) -> Mapping[str, Any]:
    existing = _existing_exact(
        client,
        destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    if existing is None:
        client.copy_object(
            Bucket=destination.bucket,
            Key=destination.key,
            CopySource={
                "Bucket": source.bucket,
                "Key": source.key,
                "VersionId": source_version_id,
            },
            MetadataDirective="REPLACE",
            Metadata={"sha256": expected_sha256, "rows": str(rows)},
            ContentType="application/x-ndjson",
        )
    return _verified_head(
        client,
        destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )


def _identity(head: Mapping[str, Any], sha256: str) -> dict[str, Any]:
    return {
        "version_id": str(head["VersionId"]),
        "metadata_sha256": sha256,
        "head_verified": True,
    }


def _write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_document_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_exact_json(path: Path, value: Any) -> None:
    """Create an identity document once, permitting only byte-identical retries."""
    payload = json_document_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite a mismatched local identity: {path}")
        return
    _write_atomic_json(path, value)


def _source_identity_index(
    path: Path, partition_count: int
) -> dict[int, Mapping[str, Any]]:
    document = json.loads(path.expanduser().resolve(strict=True).read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("format_version") != PUBLICATION_IDENTITIES_FORMAT
        or document.get("complete") is not True
        or not isinstance(document.get("source_partitions"), list)
    ):
        raise ValueError(f"incomplete source publication identities: {path}")
    result: dict[int, Mapping[str, Any]] = {}
    for raw in document["source_partitions"]:
        if not isinstance(raw, Mapping):
            raise ValueError("source publication identity must be an object")
        partition = int(raw.get("partition_index", -1))
        if partition in result:
            raise ValueError(f"duplicate source publication identity: {partition}")
        result[partition] = raw
    expected = set(range(partition_count))
    if set(result) != expected:
        raise ValueError(
            "source publication identities do not cover every partition: "
            f"missing={sorted(expected - set(result))} extra={sorted(set(result) - expected)}"
        )
    return result


def _verify_reused_source_identity(
    client: Any,
    identity: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    *,
    partition: int,
) -> dict[str, Any]:
    if int(identity.get("partition_index", -1)) != partition:
        raise ValueError(f"partition-{partition:03d}: source identity partition differs")
    uri = str(identity.get("uri") or "")
    location = parse_s3_uri(uri)
    version_id = str(identity.get("version_id") or "")
    if not version_id or version_id in {"null", "PENDING_S3_UPLOAD"}:
        raise ValueError(f"{uri}: reusable source VersionId is missing")
    expected_bytes = int(source_summary.get("bytes") or 0)
    expected_rows = int(source_summary.get("rows") or 0)
    expected_sha256 = str(source_summary.get("sha256") or "")
    declared = {
        "bytes": int(identity.get("bytes") or 0),
        "rows": int(identity.get("rows") or 0),
        "sha256": str(identity.get("sha256") or ""),
        "metadata_sha256": str(identity.get("metadata_sha256") or ""),
    }
    expected = {
        "bytes": expected_bytes,
        "rows": expected_rows,
        "sha256": expected_sha256,
        "metadata_sha256": expected_sha256,
    }
    if expected_bytes < 1 or expected_rows < 1 or declared != expected:
        raise ValueError(
            f"partition-{partition:03d}: reusable source identity differs from summary; "
            f"declared={declared} expected={expected}"
        )
    head = client.head_object(
        Bucket=location.bucket,
        Key=location.key,
        VersionId=version_id,
    )
    actual = {
        "version_id": str(head.get("VersionId") or ""),
        "bytes": int(head.get("ContentLength", -1)),
        "metadata_sha256": str((head.get("Metadata") or {}).get("sha256") or ""),
    }
    head_expected = {
        "version_id": version_id,
        "bytes": expected_bytes,
        "metadata_sha256": expected_sha256,
    }
    if actual != head_expected:
        raise ValueError(
            f"{uri}: pinned source HEAD differs; actual={actual} expected={head_expected}"
        )
    return {
        "partition_index": partition,
        "uri": uri,
        "version_id": version_id,
        "bytes": expected_bytes,
        "rows": expected_rows,
        "sha256": expected_sha256,
        "metadata_sha256": expected_sha256,
        "head_verified": True,
    }


def _inspect_prepared_summary(
    prepared_root: Path,
    *,
    partition: int,
    partition_count: int,
    output_parts: int,
) -> dict[str, Any]:
    tag = f"partition-{partition:03d}"
    prepared_dir = prepared_root / tag / "prepared"
    summary_path = prepared_dir / f"{tag}.summary.json"
    summary = json.loads(summary_path.resolve(strict=True).read_bytes())
    if (
        not isinstance(summary, dict)
        or summary.get("complete") is not True
        or int(summary.get("partition_index", -1)) != partition
    ):
        raise ValueError(f"incomplete local partition summary: {summary_path}")
    if int(summary.get("partition_count") or 0) != partition_count:
        raise ValueError(f"{tag}: partition_count differs")
    shards = summary.get("shards")
    if not isinstance(shards, list) or len(shards) != output_parts:
        raise ValueError(f"{tag}: expected exactly {output_parts} prepared shards")

    status_counts: Counter[str] = Counter()
    total_rows = 0
    seen_parts: set[int] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError(f"{tag}: shard summary must be an object")
        part = int(shard.get("part_index", -1))
        expected_path = (
            prepared_dir / f"part-{part:03d}-of-{output_parts:03d}.jsonl"
        ).resolve(strict=True)
        actual_path = Path(str(shard.get("local_path") or "")).resolve(strict=True)
        if (
            part not in range(output_parts)
            or part in seen_parts
            or int(shard.get("source_partition", -1)) != partition
            or int(shard.get("part_count") or 0) != output_parts
            or actual_path != expected_path
        ):
            raise ValueError(f"{tag}: invalid prepared shard identity for part {part}")
        seen_parts.add(part)
        digest = hashlib.sha256()
        rows = 0
        byte_count = 0
        with actual_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.strip() or not raw_line.endswith(b"\n"):
                    raise ValueError(f"{actual_path}:{line_number}: invalid JSONL framing")
                value = json.loads(raw_line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{actual_path}:{line_number}: expected an object")
                status = value.get("status")
                if not isinstance(status, str) or not status:
                    raise ValueError(f"{actual_path}:{line_number}: status is missing")
                rows += 1
                status_counts[status] += 1
        actual_identity = {
            "rows": rows,
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
        }
        declared_identity = {
            "rows": int(shard.get("rows") or 0),
            "bytes": int(shard.get("bytes") or 0),
            "sha256": str(shard.get("sha256") or ""),
        }
        if rows < 1 or actual_identity != declared_identity:
            raise ValueError(
                f"{tag} part {part}: local shard differs from summary; "
                f"actual={actual_identity} declared={declared_identity}"
            )
        total_rows += rows
    if seen_parts != set(range(output_parts)):
        raise ValueError(f"{tag}: prepared part indexes are incomplete")
    declared_statuses = {
        str(key): int(value)
        for key, value in dict(summary.get("status_counts") or {}).items()
    }
    if (
        total_rows != int(summary.get("output_rows") or 0)
        or dict(sorted(status_counts.items())) != declared_statuses
    ):
        raise ValueError(f"{tag}: recomputed rows/status counts differ from summary")
    return summary


def publish_partition(options: WorkerOptions) -> dict[str, Any]:
    partition = options.partition
    tag = f"partition-{partition:03d}"
    client = _client(options.aws_profile, options.region)
    source = child_location(
        options.source_prefix, f"{tag}/call2/final_results.jsonl"
    )
    source_head = _head_or_none(client, source)
    if source_head is None:
        raise FileNotFoundError(source.uri)
    source_bytes = int(source_head.get("ContentLength", 0))
    source_version_id = str(source_head.get("VersionId") or "")
    if source_bytes < 1 or not source_version_id or source_version_id == "null":
        raise ValueError(f"source object is not immutable/versioned: {source.uri}")

    partition_root = options.local_root / tag
    local_source = partition_root / "source" / "final_results.jsonl"
    output_dir = partition_root / "prepared"
    summary_path = output_dir / f"{tag}.summary.json"
    state_path = partition_root / f"{tag}.publication.json"
    print(f"[v6-publish] {tag} download bytes={source_bytes:,}", flush=True)
    _download_source(
        client,
        source,
        local_source,
        version_id=source_version_id,
        expected_bytes=source_bytes,
    )

    summary = _load_reusable_summary(summary_path, partition)
    if summary is None:
        print(f"[v6-publish] {tag} validate-and-split", flush=True)
        summary = prepare_partition(
            PreparePartitionOptions(
                input_jsonl=local_source,
                source_uri=source.uri,
                source_version_id=source_version_id,
                output_dir=output_dir,
                summary_path=summary_path,
                partition_index=partition,
                partition_count=options.partition_count,
                output_parts=options.output_parts,
                invalid_policy="skip",
            )
        )
    elif str(summary["source"]["version_id"]) != source_version_id:
        raise ValueError(f"{tag}: source VersionId changed since local preparation")

    local_source_sha256 = str(summary["source"]["sha256"])
    if sha256_file(local_source) != local_source_sha256:
        raise ValueError(f"{tag}: local source SHA-256 changed after preparation")

    print(f"[v6-publish] {tag} upload {options.output_parts} shards", flush=True)
    shard_identities: list[dict[str, Any]] = []
    for shard in sorted(summary["shards"], key=lambda item: int(item["part_index"])):
        part_index = int(shard["part_index"])
        local_path = Path(shard["local_path"])
        destination = child_location(
            options.output_prefix,
            f"shards/{tag}/{local_path.name}",
        )
        digest = str(shard["sha256"])
        head = _upload_file(
            client,
            local_path,
            destination,
            sha256=digest,
            content_type="application/x-ndjson",
            rows=int(shard["rows"]),
        )
        shard_identities.append(
            {
                "source_partition": partition,
                "part_index": part_index,
                "path": destination.uri,
                "bytes": int(shard["bytes"]),
                "rows": int(shard["rows"]),
                "sha256": digest,
                **_identity(head, digest),
            }
        )

    snapshot = child_location(
        options.output_prefix, f"sources/{tag}/final_results.jsonl"
    )
    print(f"[v6-publish] {tag} immutable source snapshot", flush=True)
    snapshot_head = _copy_source_snapshot(
        client,
        source,
        snapshot,
        source_version_id=source_version_id,
        expected_bytes=int(summary["source"]["bytes"]),
        expected_sha256=local_source_sha256,
        rows=int(summary["source"]["rows"]),
    )

    summary_digest = sha256_file(summary_path)
    summary_destination = child_location(
        options.output_prefix, f"summaries/{tag}.summary.json"
    )
    _upload_file(
        client,
        summary_path,
        summary_destination,
        sha256=summary_digest,
        content_type="application/json",
    )

    state = {
        "partition_index": partition,
        "source": {
            "partition_index": partition,
            "uri": snapshot.uri,
            "bytes": int(summary["source"]["bytes"]),
            "rows": int(summary["source"]["rows"]),
            "sha256": local_source_sha256,
            **_identity(snapshot_head, local_source_sha256),
        },
        "shards": shard_identities,
        "summary_path": str(summary_path),
        "summary_uri": summary_destination.uri,
        "summary_sha256": summary_digest,
        "input_rows": int(summary["input_rows"]),
        "excluded_rows": int(summary["excluded_rows"]),
        "output_rows": int(summary["output_rows"]),
        "status_counts": dict(summary["status_counts"]),
    }
    _write_atomic_json(state_path, state)
    print(
        f"[v6-publish] {tag} complete rows={state['output_rows']:,} "
        f"excluded={state['excluded_rows']:,}",
        flush=True,
    )
    return state


def publish_prepared_partition(options: PreparedOnlyWorkerOptions) -> dict[str, Any]:
    """Publish already-filtered prepared shards without re-reading Call-2 inputs."""
    partition = options.partition
    tag = f"partition-{partition:03d}"
    client = _client(options.aws_profile, options.region)
    summary = _inspect_prepared_summary(
        options.prepared_root,
        partition=partition,
        partition_count=options.partition_count,
        output_parts=options.output_parts,
    )
    source_summary = summary.get("source")
    if not isinstance(source_summary, Mapping):
        raise ValueError(f"{tag}: source summary is missing")
    source_identity = _verify_reused_source_identity(
        client,
        options.source_identity,
        source_summary,
        partition=partition,
    )

    shard_identities: list[dict[str, Any]] = []
    for shard in sorted(summary["shards"], key=lambda item: int(item["part_index"])):
        part_index = int(shard["part_index"])
        local_path = Path(str(shard["local_path"])).resolve(strict=True)
        destination = child_location(
            options.output_prefix,
            f"shards/{tag}/{local_path.name}",
        )
        digest = str(shard["sha256"])
        rows = int(shard["rows"])
        head = _upload_file(
            client,
            local_path,
            destination,
            sha256=digest,
            content_type="application/x-ndjson",
            rows=rows,
        )
        uploaded_rows = str((head.get("Metadata") or {}).get("rows") or "")
        if uploaded_rows != str(rows):
            raise ValueError(
                f"refusing mismatched published row metadata: {destination.uri}; "
                f"rows={uploaded_rows!r}/{rows}"
            )
        shard_identities.append(
            {
                "source_partition": partition,
                "part_index": part_index,
                "path": destination.uri,
                "bytes": int(shard["bytes"]),
                "rows": rows,
                "sha256": digest,
                **_identity(head, digest),
            }
        )

    summary_path = (
        options.prepared_root / tag / "prepared" / f"{tag}.summary.json"
    ).resolve(strict=True)
    summary_digest = sha256_file(summary_path)
    summary_destination = child_location(
        options.output_prefix, f"summaries/{tag}.summary.json"
    )
    _upload_file(
        client,
        summary_path,
        summary_destination,
        sha256=summary_digest,
        content_type="application/json",
    )
    state = {
        "partition_index": partition,
        "source": source_identity,
        "shards": shard_identities,
        "summary_path": str(summary_path),
        "summary_uri": summary_destination.uri,
        "summary_sha256": summary_digest,
        "input_rows": int(summary["input_rows"]),
        "excluded_rows": int(summary["excluded_rows"]),
        "output_rows": int(summary["output_rows"]),
        "status_counts": dict(summary["status_counts"]),
    }
    state_path = options.prepared_root / tag / f"{tag}.publication.json"
    _write_exact_json(state_path, state)
    print(
        f"[v6-publish-prepared] {tag} complete rows={state['output_rows']:,}",
        flush=True,
    )
    return state


def _aggregate_states(
    states: list[dict[str, Any]], partition_count: int
) -> dict[str, Any]:
    states.sort(key=lambda item: int(item["partition_index"]))
    actual = [int(item["partition_index"]) for item in states]
    expected = list(range(partition_count))
    if actual != expected:
        raise ValueError(
            f"publication states are incomplete: expected={expected} actual={actual}"
        )
    return {
        "format_version": PUBLICATION_IDENTITIES_FORMAT,
        "complete": True,
        "source_partitions": [item["source"] for item in states],
        "shards": [shard for item in states for shard in item["shards"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/tmp/image-edit-v6-full64-v7-call2-prepared-20260813"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="publish already-filtered partition summaries/shards without preparing Call-2 inputs",
    )
    parser.add_argument(
        "--reuse-source-publication-identities",
        type=Path,
        help="publication-identities.json whose pinned source records are reused and re-HEADed",
    )
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--partition-count", type=int, default=64)
    parser.add_argument("--output-parts", type=int, default=4)
    parser.add_argument("--start-partition", type=int, default=0)
    parser.add_argument("--end-partition", type=int, default=63)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.partition_count < 1 or args.output_parts < 1 or args.workers < 1:
        raise ValueError("partition-count, output-parts, and workers must be positive")
    if not 0 <= args.start_partition <= args.end_partition < args.partition_count:
        raise ValueError("requested partition range is invalid")
    output_prefix = parse_s3_uri(args.output_prefix)
    prepared_only = args.prepared_root is not None
    if prepared_only != (args.reuse_source_publication_identities is not None):
        raise ValueError(
            "--prepared-root and --reuse-source-publication-identities must be used together"
        )
    if prepared_only:
        publication_root = args.prepared_root.expanduser().resolve(strict=True)
        source_identities = _source_identity_index(
            args.reuse_source_publication_identities,
            args.partition_count,
        )
        work: Sequence[WorkerOptions | PreparedOnlyWorkerOptions] = [
            PreparedOnlyWorkerOptions(
                partition=partition,
                partition_count=args.partition_count,
                output_parts=args.output_parts,
                prepared_root=publication_root,
                output_prefix=output_prefix,
                source_identity=source_identities[partition],
                aws_profile=args.aws_profile,
                region=args.region,
            )
            for partition in range(args.start_partition, args.end_partition + 1)
        ]
        worker = publish_prepared_partition
    else:
        source_prefix = parse_s3_uri(args.source_prefix)
        if source_prefix == output_prefix or output_prefix.key.startswith(
            f"{source_prefix.key.rstrip('/')}/"
        ):
            raise ValueError("output prefix must be separate from the source prefix")
        args.local_root.mkdir(parents=True, exist_ok=True)
        publication_root = args.local_root.resolve()
        work = [
            WorkerOptions(
                partition=partition,
                partition_count=args.partition_count,
                output_parts=args.output_parts,
                local_root=publication_root,
                source_prefix=source_prefix,
                output_prefix=output_prefix,
                aws_profile=args.aws_profile,
                region=args.region,
            )
            for partition in range(args.start_partition, args.end_partition + 1)
        ]
        worker = publish_partition
    completed: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, item): item.partition for item in work}
        for future in concurrent.futures.as_completed(futures):
            partition = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report every failed partition.
                failures.append((partition, repr(exc)))
                print(f"[v6-publish] partition-{partition:03d} FAILED: {exc!r}", flush=True)
    if failures:
        raise RuntimeError(f"partition publication failures: {failures}")

    if args.start_partition == 0 and args.end_partition == args.partition_count - 1:
        identities = _aggregate_states(completed, args.partition_count)
        identity_path = publication_root / "publication-identities.json"
        if prepared_only:
            _write_exact_json(identity_path, identities)
        else:
            _write_atomic_json(identity_path, identities)
        digest = hashlib.sha256(identity_path.read_bytes()).hexdigest()
        client = _client(args.aws_profile, args.region)
        destination = child_location(output_prefix, "publication-identities.json")
        head = _upload_file(
            client,
            identity_path,
            destination,
            sha256=digest,
            content_type="application/json",
        )
        print(
            json.dumps(
                {
                    "complete": True,
                    "partitions": len(completed),
                    "physical_shards": len(identities["shards"]),
                    "total_rows": sum(int(item["output_rows"]) for item in completed),
                    "publication_identities": destination.uri,
                    "publication_identities_version_id": str(head["VersionId"]),
                    "publication_identities_sha256": digest,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
