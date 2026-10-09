#!/usr/bin/env python3
"""Atomically publish and restore node-local native MCore checkpoints through S3.

The protocol never treats uploaded shard objects as a committed checkpoint. Every node
publishes into an immutable attempt namespace and writes its node manifest last. A
coordinator validates all node manifests and objects, writes a global manifest, then
creates ``_SUCCESS`` with an S3 conditional write. Resume only considers checkpoints
with a valid ``_SUCCESS`` and reconstructs the complete checkpoint on each node-local disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    canonical_json_bytes,
    download_verified,
    local_path,
    parse_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    validate_file_entries,
    verify_object_stat,
    write_json_atomic,
)


PROTOCOL_VERSION = 1
SUCCESS_NAME = "_SUCCESS"
LATEST_CANDIDATE_DIR = "_LATEST"
GLOBAL_MANIFEST_NAME = "manifest.json"
NODE_MANIFEST_NAME = "node_manifest.json"
RANK_MANIFEST_NAME = "rank_manifest.json"
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DISTCP_SHARD_RE = re.compile(r"iter_(\d{7})/__(\d+)_\d+\.distcp$")


def _checkpoint_step_from_name(name: str) -> int:
    match = CHECKPOINT_RE.fullmatch(name)
    if match is None:
        raise CheckpointError(f"Expected checkpoint-<iteration>, got {name!r}")
    return int(match.group(1))


def _checkpoint_step_from_uri(uri: str) -> int:
    location = S3Location.parse(uri)
    return _checkpoint_step_from_name(location.key.rstrip("/").rsplit("/", 1)[-1])


def _validate_attempt_id(attempt_id: str) -> None:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise CheckpointError(
            "attempt_id must be 1-128 characters containing only letters, digits, '.', '_', or '-'"
        )


def _checkpoint_root(uri: str) -> S3Location:
    location = S3Location.parse(uri)
    return S3Location(location.bucket, location.key.rstrip("/"))


def _checkpoint_parent(uri: str) -> S3Location:
    root = _checkpoint_root(uri)
    parent, separator, _ = root.key.rpartition("/")
    if not separator or not parent:
        raise CheckpointError(f"Checkpoint URI has no parent namespace: {uri}")
    return S3Location(root.bucket, parent)


def _node_root(checkpoint_uri: str, attempt_id: str, node_rank: int) -> S3Location:
    return _checkpoint_root(checkpoint_uri).join("_attempts", attempt_id, "nodes", f"node-{node_rank:05d}")


def _node_manifest_uri(checkpoint_uri: str, attempt_id: str, node_rank: int) -> str:
    return _node_root(checkpoint_uri, attempt_id, node_rank).join(NODE_MANIFEST_NAME).uri


def _rank_root(checkpoint_uri: str, attempt_id: str, rank: int) -> S3Location:
    return _checkpoint_root(checkpoint_uri).join("_attempts", attempt_id, "ranks", f"rank-{rank:05d}")


def _rank_manifest_uri(checkpoint_uri: str, attempt_id: str, rank: int) -> str:
    return _rank_root(checkpoint_uri, attempt_id, rank).join(RANK_MANIFEST_NAME).uri


def _global_manifest_uri(checkpoint_uri: str, attempt_id: str) -> str:
    return _checkpoint_root(checkpoint_uri).join("_attempts", attempt_id, GLOBAL_MANIFEST_NAME).uri


def _success_uri(checkpoint_uri: str) -> str:
    return _checkpoint_root(checkpoint_uri).join(SUCCESS_NAME).uri


def _write_latest_candidate(store: ObjectStore, checkpoint_uri: str, *, step: int) -> None:
    success_uri = _success_uri(checkpoint_uri)
    latest_uri = (
        _checkpoint_parent(checkpoint_uri).join(LATEST_CANDIDATE_DIR, f"checkpoint-{step:010d}.json").uri
    )
    _put_json_if_absent(
        store,
        latest_uri,
        {
            "version": PROTOCOL_VERSION,
            "kind": "mcore-latest-candidate",
            "status": "complete",
            "step": step,
            "checkpoint_uri": _checkpoint_root(checkpoint_uri).uri + "/",
            "success_uri": success_uri,
        },
    )


def validate_local_mcore_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int | None = None,
    partial_node: bool = False,
) -> int:
    checkpoint = checkpoint.resolve()
    name_match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if name_match is not None:
        step = int(name_match.group(1))
    elif expected_step is not None:
        # Restores are validated in a random temporary directory before the
        # directory is atomically renamed to checkpoint-<iteration>.
        step = expected_step
    else:
        raise CheckpointError(f"Expected checkpoint-<iteration>, got {checkpoint.name!r}")
    if expected_step is not None and step != expected_step:
        raise CheckpointError(f"Checkpoint iteration mismatch: directory={step} expected={expected_step}")
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        try:
            tracker_step = int(tracker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"Invalid MCore iteration tracker: {tracker}") from exc
        if tracker_step != step:
            raise CheckpointError(f"MCore tracker mismatch: directory={step} tracker={tracker_step}")
    elif not partial_node:
        raise CheckpointError(f"Missing MCore iteration tracker: {tracker}")
    iteration_dir = checkpoint / f"iter_{step:07d}"
    if not iteration_dir.is_dir():
        raise CheckpointError(f"Missing MCore iteration directory: {iteration_dir}")
    shards = [path for path in iteration_dir.glob("*.distcp") if path.is_file() and path.stat().st_size > 0]
    if not shards:
        raise CheckpointError(f"No non-empty .distcp shards in {iteration_dir}")
    if not partial_node:
        required = (iteration_dir / "common.pt", iteration_dir / ".metadata")
        missing = [path.name for path in required if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise CheckpointError(f"Incomplete MCore checkpoint {checkpoint}: missing/empty {missing}")
    return step


def _local_checkpoint_files(checkpoint: Path, step: int) -> list[Path]:
    iteration_dir = checkpoint / f"iter_{step:07d}"
    candidates = [path for path in checkpoint.iterdir() if path.is_file()]
    candidates.extend(path for path in iteration_dir.rglob("*") if path.is_file())
    files: list[Path] = []
    for path in sorted(candidates):
        if path.is_symlink():
            raise CheckpointError(f"Refusing checkpoint symlink: {path}")
        if path.name.startswith(".s3_") or path.name == SUCCESS_NAME:
            continue
        if path.stat().st_size <= 0:
            raise CheckpointError(f"Refusing empty checkpoint file: {path}")
        files.append(path)
    return files


def _put_json_if_absent(store: ObjectStore, uri: str, value: dict[str, Any]) -> str:
    data = canonical_json_bytes(value)
    digest = sha256_bytes(data)
    created = store.put_bytes(uri, data, metadata={"sha256": digest}, if_absent=True)
    if not created:
        existing = store.read_bytes(uri)
        if existing != data:
            raise CheckpointError(f"Immutable commit object already exists with different content: {uri}")
    stat = store.stat(uri)
    if stat.size != len(data) or stat.metadata.get("sha256") != digest:
        raise CheckpointError(f"Failed to verify committed JSON object: {uri}")
    return digest


def _read_committed_json(store: ObjectStore, uri: str) -> tuple[dict[str, Any], str]:
    data = store.read_bytes(uri)
    digest = sha256_bytes(data)
    stat = store.stat(uri)
    if stat.size != len(data) or stat.metadata.get("sha256") != digest:
        raise CheckpointError(
            f"JSON object envelope mismatch for {uri}: size={stat.size}/{len(data)} "
            f"sha256={stat.metadata.get('sha256')!r}/{digest}"
        )
    return parse_json_bytes(data, source=uri), digest


def _local_records(
    checkpoint: Path,
    step: int,
    object_root: S3Location,
    *,
    files: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, int]]]:
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, tuple[int, int, int]] = {}
    for path in files if files is not None else _local_checkpoint_files(checkpoint, step):
        relative = path.relative_to(checkpoint).as_posix()
        safe_relative_path(relative)
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        before_fingerprint = (before.st_size, before.st_mtime_ns, before.st_ino)
        after_fingerprint = (after.st_size, after.st_mtime_ns, after.st_ino)
        if before_fingerprint != after_fingerprint:
            raise CheckpointError(f"Local checkpoint changed while hashing: {path}")
        records.append(
            {
                "path": relative,
                "size": after.st_size,
                "sha256": digest,
                "object_uri": object_root.join("files", relative).uri,
            }
        )
        fingerprints[relative] = after_fingerprint
    return records, fingerprints


def _upload_prehashed(
    store: ObjectStore,
    checkpoint: Path,
    record: dict[str, Any],
    fingerprint: tuple[int, int, int],
) -> None:
    """Upload a record without reading a multi-GB file a second time for SHA-256."""

    source = local_path(checkpoint, record["path"])
    before = source.stat()
    before_fingerprint = (before.st_size, before.st_mtime_ns, before.st_ino)
    if before_fingerprint != fingerprint or before.st_size != record["size"]:
        raise CheckpointError(f"Local checkpoint changed before upload: {source}")
    store.upload_file(source, record["object_uri"], metadata={"sha256": record["sha256"]})
    after = source.stat()
    after_fingerprint = (after.st_size, after.st_mtime_ns, after.st_ino)
    if after_fingerprint != fingerprint:
        raise CheckpointError(f"Local checkpoint changed during upload: {source}")
    verify_object_stat(
        store,
        record["object_uri"],
        size=record["size"],
        sha256=record["sha256"],
    )


def _validate_node_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
    step: int,
    attempt_id: str,
    node_rank: int,
    node_count: int,
) -> dict[str, dict[str, Any]]:
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-shards",
        "status": "uploaded",
        "step": step,
        "attempt_id": attempt_id,
        "node_rank": node_rank,
        "node_count": node_count,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"Node manifest {key} mismatch in {source}: expected={value!r} actual={manifest.get(key)!r}"
            )
    entries = validate_file_entries(manifest, source=source)
    for path, record in entries.items():
        uri = record.get("object_uri")
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise CheckpointError(f"Invalid object_uri for {path!r} in {source}")
    return entries


def publish_node(
    store: ObjectStore,
    checkpoint: Path,
    checkpoint_uri: str,
    *,
    attempt_id: str,
    node_rank: int,
    node_count: int,
    world_size: int | None = None,
) -> dict[str, Any]:
    _validate_attempt_id(attempt_id)
    if node_count <= 0 or node_rank < 0 or node_rank >= node_count:
        raise CheckpointError(f"Invalid node rank/count: rank={node_rank} count={node_count}")
    checkpoint = checkpoint.resolve()
    remote_step = _checkpoint_step_from_uri(checkpoint_uri)
    step = validate_local_mcore_checkpoint(checkpoint, expected_step=remote_step, partial_node=True)
    node_root = _node_root(checkpoint_uri, attempt_id, node_rank)
    records, fingerprints = _local_records(checkpoint, step, node_root)
    manifest = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-shards",
        "status": "uploaded",
        "step": step,
        "attempt_id": attempt_id,
        "node_rank": node_rank,
        "node_count": node_count,
        "world_size": world_size,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    manifest_uri = _node_manifest_uri(checkpoint_uri, attempt_id, node_rank)
    if store.exists(manifest_uri):
        existing, _ = _read_committed_json(store, manifest_uri)
        if existing != manifest:
            raise CheckpointError(
                f"Node attempt is already sealed with different local files; use a new attempt_id: {manifest_uri}"
            )
        entries = _validate_node_manifest(
            existing,
            source=manifest_uri,
            step=step,
            attempt_id=attempt_id,
            node_rank=node_rank,
            node_count=node_count,
        )
        for record in entries.values():
            verify_object_stat(
                store,
                record["object_uri"],
                size=record["size"],
                sha256=record["sha256"],
            )
        return existing

    for record in records:
        _upload_prehashed(store, checkpoint, record, fingerprints[record["path"]])
    _put_json_if_absent(store, manifest_uri, manifest)
    return manifest


def _distcp_owner(path: str, *, step: int, source: str) -> int | None:
    if not path.endswith(".distcp"):
        return None
    match = DISTCP_SHARD_RE.fullmatch(path)
    if match is None or int(match.group(1)) != step:
        raise CheckpointError(f"Unrecognized distributed-checkpoint shard path in {source}: {path!r}")
    return int(match.group(2))


def _validate_rank_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
    step: int,
    attempt_id: str,
    rank: int,
    world_size: int,
) -> dict[str, dict[str, Any]]:
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-rank-shards",
        "status": "uploaded",
        "step": step,
        "attempt_id": attempt_id,
        "rank": rank,
        "world_size": world_size,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"Rank manifest {key} mismatch in {source}: "
                f"expected={value!r} actual={manifest.get(key)!r}"
            )
    entries = validate_file_entries(manifest, source=source)
    shard_count = 0
    for path, record in entries.items():
        uri = record.get("object_uri")
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise CheckpointError(f"Invalid object_uri for {path!r} in {source}")
        owner = _distcp_owner(path, step=step, source=source)
        if owner is None:
            if rank != 0:
                raise CheckpointError(f"Only global rank 0 may publish common file {path!r} in {source}")
        else:
            shard_count += 1
            if owner != rank:
                raise CheckpointError(
                    f"Rank {rank} attempted to publish rank {owner} shard {path!r} in {source}"
                )
    if shard_count == 0:
        raise CheckpointError(f"Rank manifest contains no owned .distcp shard: {source}")
    for key in ("node_rank", "node_count", "local_rank"):
        value = manifest.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise CheckpointError(f"Invalid {key} in {source}: {value!r}")
    node_count = manifest.get("node_count")
    node_rank = manifest.get("node_rank")
    if isinstance(node_count, int) and node_count <= 0:
        raise CheckpointError(f"Invalid node_count in {source}: {node_count!r}")
    if isinstance(node_count, int) and isinstance(node_rank, int) and node_rank >= node_count:
        raise CheckpointError(f"Invalid node rank/count in {source}: rank={node_rank} count={node_count}")
    return entries


def publish_rank(
    store: ObjectStore,
    checkpoint: Path,
    checkpoint_uri: str,
    *,
    attempt_id: str,
    rank: int,
    world_size: int,
    node_rank: int | None = None,
    node_count: int | None = None,
    local_rank: int | None = None,
) -> dict[str, Any]:
    """Publish only one global rank's shard; global rank 0 also owns common files."""

    _validate_attempt_id(attempt_id)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise CheckpointError(f"Invalid global rank/world size: rank={rank} world_size={world_size}")
    if node_count is not None and (node_count <= 0 or node_rank is None):
        raise CheckpointError("node_count requires a valid node_rank")
    if node_rank is not None and (node_rank < 0 or node_count is None or node_rank >= node_count):
        raise CheckpointError(f"Invalid node rank/count: rank={node_rank} count={node_count}")
    if local_rank is not None and local_rank < 0:
        raise CheckpointError(f"Invalid local_rank: {local_rank}")

    checkpoint = checkpoint.resolve()
    remote_step = _checkpoint_step_from_uri(checkpoint_uri)
    step = validate_local_mcore_checkpoint(checkpoint, expected_step=remote_step, partial_node=True)
    owned_files: list[Path] = []
    owned_shards = 0
    for path in _local_checkpoint_files(checkpoint, step):
        relative = path.relative_to(checkpoint).as_posix()
        owner = _distcp_owner(relative, step=step, source=str(checkpoint))
        if owner is not None:
            if owner < 0 or owner >= world_size:
                raise CheckpointError(
                    f"Shard owner {owner} is outside world_size={world_size}: {relative!r}"
                )
            if owner == rank:
                owned_files.append(path)
                owned_shards += 1
        elif rank == 0:
            owned_files.append(path)
    if owned_shards == 0:
        raise CheckpointError(f"No .distcp shard owned by global rank {rank} in {checkpoint}")

    rank_root = _rank_root(checkpoint_uri, attempt_id, rank)
    records, fingerprints = _local_records(
        checkpoint,
        step,
        rank_root,
        files=owned_files,
    )
    manifest = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-rank-shards",
        "status": "uploaded",
        "step": step,
        "attempt_id": attempt_id,
        "rank": rank,
        "world_size": world_size,
        "node_rank": node_rank,
        "node_count": node_count,
        "local_rank": local_rank,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    manifest_uri = _rank_manifest_uri(checkpoint_uri, attempt_id, rank)
    if store.exists(manifest_uri):
        existing, _ = _read_committed_json(store, manifest_uri)
        if existing != manifest:
            raise CheckpointError(
                f"Rank attempt is already sealed with different local files; use a new attempt_id: "
                f"{manifest_uri}"
            )
        entries = _validate_rank_manifest(
            existing,
            source=manifest_uri,
            step=step,
            attempt_id=attempt_id,
            rank=rank,
            world_size=world_size,
        )
        for record in entries.values():
            verify_object_stat(
                store,
                record["object_uri"],
                size=record["size"],
                sha256=record["sha256"],
            )
        return existing

    for record in records:
        _upload_prehashed(store, checkpoint, record, fingerprints[record["path"]])
    _put_json_if_absent(store, manifest_uri, manifest)
    return manifest


def _validate_logical_layout(files: dict[str, dict[str, Any]], *, step: int, source: str) -> None:
    iteration_prefix = f"iter_{step:07d}/"
    required = {
        "latest_checkpointed_iteration.txt",
        f"{iteration_prefix}common.pt",
        f"{iteration_prefix}.metadata",
    }
    missing = sorted(required - files.keys())
    shards = [path for path in files if path.startswith(iteration_prefix) and path.endswith(".distcp")]
    if missing or not shards:
        raise CheckpointError(
            f"Incomplete logical MCore checkpoint in {source}: missing={missing}, distcp_shards={len(shards)}"
        )


def _validate_global_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
    checkpoint_uri: str,
) -> dict[str, dict[str, Any]]:
    step = _checkpoint_step_from_uri(checkpoint_uri)
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint",
        "status": "complete",
        "step": step,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"Global manifest {key} mismatch in {source}: expected={value!r} actual={manifest.get(key)!r}"
            )
    entries = validate_file_entries(manifest, source=source)
    for path, record in entries.items():
        sources = record.get("source_uris")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(uri, str) and uri.startswith("s3://") for uri in sources)
        ):
            raise CheckpointError(f"Invalid source_uris for {path!r} in {source}")
    _validate_logical_layout(entries, step=step, source=source)
    return entries


def _read_success(store: ObjectStore, checkpoint_uri: str) -> tuple[dict[str, Any], dict[str, Any]]:
    success_uri = _success_uri(checkpoint_uri)
    marker, _ = _read_committed_json(store, success_uri)
    step = _checkpoint_step_from_uri(checkpoint_uri)
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint-commit",
        "status": "complete",
        "step": step,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Commit marker {key} mismatch in {success_uri}: expected={value!r} actual={marker.get(key)!r}"
            )
    manifest_uri = marker.get("manifest_uri")
    manifest_digest = marker.get("manifest_sha256")
    if not isinstance(manifest_uri, str) or not isinstance(manifest_digest, str):
        raise CheckpointError(f"Invalid commit marker pointers in {success_uri}")
    attempt_id = marker.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise CheckpointError(f"Invalid commit marker attempt_id in {success_uri}: {attempt_id!r}")
    _validate_attempt_id(attempt_id)
    root = _checkpoint_root(checkpoint_uri)
    manifest_location = S3Location.parse(manifest_uri)
    expected_prefix = root.join("_attempts").key.rstrip("/") + "/"
    if manifest_location.bucket != root.bucket or not manifest_location.key.startswith(expected_prefix):
        raise CheckpointError(f"Commit marker points outside its checkpoint namespace: {manifest_uri}")
    expected_manifest_uri = _global_manifest_uri(checkpoint_uri, attempt_id)
    if manifest_uri != expected_manifest_uri:
        raise CheckpointError(
            f"Commit marker manifest URI does not match attempt {attempt_id!r}: "
            f"expected={expected_manifest_uri} actual={manifest_uri}"
        )
    manifest, actual_digest = _read_committed_json(store, manifest_uri)
    if actual_digest != manifest_digest:
        raise CheckpointError(
            f"Global manifest digest mismatch: marker={manifest_digest} actual={actual_digest}"
        )
    _validate_global_manifest(manifest, source=manifest_uri, checkpoint_uri=checkpoint_uri)
    if manifest.get("attempt_id") != attempt_id:
        raise CheckpointError(
            f"Commit marker/global manifest attempt mismatch: "
            f"marker={attempt_id!r} manifest={manifest.get('attempt_id')!r}"
        )
    return marker, manifest


def inspect_committed_checkpoint(
    store: ObjectStore,
    checkpoint_uri: str,
    *,
    verify_objects: bool = True,
) -> dict[str, Any]:
    marker, manifest = _read_success(store, checkpoint_uri)
    entries = _validate_global_manifest(
        manifest,
        source=str(marker["manifest_uri"]),
        checkpoint_uri=checkpoint_uri,
    )
    if verify_objects:
        for record in entries.values():
            for uri in record["source_uris"]:
                verify_object_stat(store, uri, size=record["size"], sha256=record["sha256"])
    return {
        "checkpoint_uri": _checkpoint_root(checkpoint_uri).uri + "/",
        "step": manifest["step"],
        "attempt_id": manifest["attempt_id"],
        "node_count": manifest["node_count"],
        "world_size": manifest.get("world_size"),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_uri": marker["manifest_uri"],
        "verified_objects": verify_objects,
    }


def commit_attempt(
    store: ObjectStore,
    checkpoint_uri: str,
    *,
    attempt_id: str,
    node_count: int,
    world_size: int | None = None,
    wait_seconds: float = 0,
    poll_seconds: float = 5,
) -> dict[str, Any]:
    _validate_attempt_id(attempt_id)
    if node_count <= 0:
        raise CheckpointError(f"node_count must be positive, got {node_count}")
    if wait_seconds < 0 or poll_seconds <= 0:
        raise CheckpointError("wait_seconds must be non-negative and poll_seconds must be positive")
    step = _checkpoint_step_from_uri(checkpoint_uri)
    success_uri = _success_uri(checkpoint_uri)
    if store.exists(success_uri):
        marker, _ = _read_success(store, checkpoint_uri)
        _write_latest_candidate(store, checkpoint_uri, step=step)
        return marker

    deadline = time.monotonic() + wait_seconds
    manifest_uris = [
        _node_manifest_uri(checkpoint_uri, attempt_id, node_rank) for node_rank in range(node_count)
    ]
    while True:
        missing_manifests = [uri for uri in manifest_uris if not store.exists(uri)]
        if not missing_manifests:
            break
        if time.monotonic() >= deadline:
            raise CheckpointError(f"Missing node manifests for attempt {attempt_id}: {missing_manifests}")
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))

    logical: dict[str, dict[str, Any]] = {}
    observed_world_sizes: set[int] = set()
    for node_rank, manifest_uri in enumerate(manifest_uris):
        node_manifest, _ = _read_committed_json(store, manifest_uri)
        entries = _validate_node_manifest(
            node_manifest,
            source=manifest_uri,
            step=step,
            attempt_id=attempt_id,
            node_rank=node_rank,
            node_count=node_count,
        )
        node_world_size = node_manifest.get("world_size")
        if node_world_size is not None:
            if not isinstance(node_world_size, int) or node_world_size <= 0:
                raise CheckpointError(f"Invalid world_size in {manifest_uri}: {node_world_size!r}")
            observed_world_sizes.add(node_world_size)
        for path, record in entries.items():
            verify_object_stat(
                store,
                record["object_uri"],
                size=record["size"],
                sha256=record["sha256"],
            )
            existing = logical.get(path)
            if existing is None:
                logical[path] = {
                    "path": path,
                    "size": record["size"],
                    "sha256": record["sha256"],
                    "source_uris": [record["object_uri"]],
                }
            elif (existing["size"], existing["sha256"]) == (record["size"], record["sha256"]):
                existing["source_uris"].append(record["object_uri"])
            else:
                raise CheckpointError(
                    f"Conflicting node-local files for logical path {path!r}: "
                    f"{existing['sha256']} != {record['sha256']}"
                )
    if len(observed_world_sizes) > 1:
        raise CheckpointError(f"Nodes disagree on world_size: {sorted(observed_world_sizes)}")
    observed_world_size = next(iter(observed_world_sizes), None)
    if world_size is not None and observed_world_size not in {None, world_size}:
        raise CheckpointError(
            f"Published node world_size {observed_world_size} does not match commit world_size {world_size}"
        )
    committed_world_size = world_size if world_size is not None else observed_world_size
    _validate_logical_layout(logical, step=step, source=f"attempt {attempt_id}")
    records = [logical[path] for path in sorted(logical)]
    global_manifest = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint",
        "status": "complete",
        "step": step,
        "attempt_id": attempt_id,
        "node_count": node_count,
        "world_size": committed_world_size,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    manifest_uri = _global_manifest_uri(checkpoint_uri, attempt_id)
    manifest_digest = _put_json_if_absent(store, manifest_uri, global_manifest)
    marker = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint-commit",
        "status": "complete",
        "step": step,
        "attempt_id": attempt_id,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "file_count": global_manifest["file_count"],
        "total_bytes": global_manifest["total_bytes"],
    }
    try:
        _put_json_if_absent(store, success_uri, marker)
    except CheckpointError:
        # Another valid attempt may have committed this immutable logical step
        # after our initial existence check. In that case its commit wins and is
        # idempotently accepted; malformed/incomplete commits still fail here.
        if not store.exists(success_uri):
            raise
        marker, _ = _read_success(store, checkpoint_uri)
    # Latest markers are immutable candidates rather than one mutable pointer:
    # concurrent commits cannot make an older step overwrite a newer one.
    _write_latest_candidate(store, checkpoint_uri, step=step)
    return marker


def commit_rank_attempt(
    store: ObjectStore,
    checkpoint_uri: str,
    *,
    attempt_id: str,
    world_size: int,
    node_count: int | None = None,
    wait_seconds: float = 0,
    poll_seconds: float = 5,
) -> dict[str, Any]:
    """Commit a checkpoint after all global-rank manifests are present and verified."""

    _validate_attempt_id(attempt_id)
    if world_size <= 0:
        raise CheckpointError(f"world_size must be positive, got {world_size}")
    if node_count is not None and node_count <= 0:
        raise CheckpointError(f"node_count must be positive, got {node_count}")
    if wait_seconds < 0 or poll_seconds <= 0:
        raise CheckpointError("wait_seconds must be non-negative and poll_seconds must be positive")
    step = _checkpoint_step_from_uri(checkpoint_uri)
    success_uri = _success_uri(checkpoint_uri)
    if store.exists(success_uri):
        marker, _ = _read_success(store, checkpoint_uri)
        _write_latest_candidate(store, checkpoint_uri, step=step)
        return marker

    deadline = time.monotonic() + wait_seconds
    manifest_uris = [
        _rank_manifest_uri(checkpoint_uri, attempt_id, rank) for rank in range(world_size)
    ]
    while True:
        missing_manifests = [uri for uri in manifest_uris if not store.exists(uri)]
        if not missing_manifests:
            break
        if time.monotonic() >= deadline:
            raise CheckpointError(
                f"Missing rank manifests for attempt {attempt_id}: {missing_manifests}"
            )
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))

    logical: dict[str, dict[str, Any]] = {}
    observed_node_counts: set[int] = set()
    for rank, manifest_uri in enumerate(manifest_uris):
        rank_manifest, _ = _read_committed_json(store, manifest_uri)
        entries = _validate_rank_manifest(
            rank_manifest,
            source=manifest_uri,
            step=step,
            attempt_id=attempt_id,
            rank=rank,
            world_size=world_size,
        )
        manifest_node_count = rank_manifest.get("node_count")
        if isinstance(manifest_node_count, int):
            observed_node_counts.add(manifest_node_count)
        manifest_node_rank = rank_manifest.get("node_rank")
        manifest_local_rank = rank_manifest.get("local_rank")
        if (
            isinstance(manifest_node_count, int)
            and isinstance(manifest_node_rank, int)
            and isinstance(manifest_local_rank, int)
        ):
            if world_size % manifest_node_count != 0:
                raise CheckpointError(
                    f"world_size={world_size} is not divisible by node_count={manifest_node_count}"
                )
            ranks_per_node = world_size // manifest_node_count
            expected_rank = manifest_node_rank * ranks_per_node + manifest_local_rank
            if manifest_local_rank >= ranks_per_node or expected_rank != rank:
                raise CheckpointError(
                    f"Rank topology mismatch in {manifest_uri}: global={rank} "
                    f"node={manifest_node_rank} local={manifest_local_rank}"
                )
        for path, record in entries.items():
            verify_object_stat(
                store,
                record["object_uri"],
                size=record["size"],
                sha256=record["sha256"],
            )
            existing = logical.get(path)
            if existing is not None:
                raise CheckpointError(f"Duplicate rank-published logical path {path!r}")
            logical[path] = {
                "path": path,
                "size": record["size"],
                "sha256": record["sha256"],
                "source_uris": [record["object_uri"]],
            }

    if len(observed_node_counts) > 1:
        raise CheckpointError(f"Ranks disagree on node_count: {sorted(observed_node_counts)}")
    observed_node_count = next(iter(observed_node_counts), None)
    if node_count is not None and observed_node_count not in {None, node_count}:
        raise CheckpointError(
            f"Published rank node_count {observed_node_count} does not match commit node_count "
            f"{node_count}"
        )
    committed_node_count = node_count or observed_node_count or 1
    _validate_logical_layout(logical, step=step, source=f"rank attempt {attempt_id}")
    records = [logical[path] for path in sorted(logical)]
    global_manifest = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint",
        "status": "complete",
        "step": step,
        "attempt_id": attempt_id,
        "node_count": committed_node_count,
        "world_size": world_size,
        "rank_count": world_size,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    manifest_uri = _global_manifest_uri(checkpoint_uri, attempt_id)
    manifest_digest = _put_json_if_absent(store, manifest_uri, global_manifest)
    marker = {
        "version": PROTOCOL_VERSION,
        "kind": "mcore-node-local-checkpoint-commit",
        "status": "complete",
        "step": step,
        "attempt_id": attempt_id,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "file_count": global_manifest["file_count"],
        "total_bytes": global_manifest["total_bytes"],
    }
    try:
        _put_json_if_absent(store, success_uri, marker)
    except CheckpointError:
        if not store.exists(success_uri):
            raise
        marker, _ = _read_success(store, checkpoint_uri)
    _write_latest_candidate(store, checkpoint_uri, step=step)
    return marker


def find_latest_committed(
    store: ObjectStore,
    root_uri: str,
    *,
    verify_objects: bool = True,
) -> str:
    root = S3Location.parse(root_uri)
    candidates: list[tuple[int, str]] = []
    prefix = root.uri.rstrip("/") + "/"
    for uri in store.iter_uris(prefix):
        if not uri.endswith(f"/{SUCCESS_NAME}"):
            continue
        relative = uri[len(prefix) :]
        parts = relative.split("/")
        if len(parts) != 2 or parts[1] != SUCCESS_NAME or CHECKPOINT_RE.fullmatch(parts[0]) is None:
            continue
        candidates.append((_checkpoint_step_from_name(parts[0]), root.join(parts[0]).uri + "/"))
    rejected: list[str] = []
    for _, checkpoint_uri in sorted(candidates, reverse=True):
        try:
            inspect_committed_checkpoint(store, checkpoint_uri, verify_objects=verify_objects)
        except CheckpointError as exc:
            rejected.append(f"{checkpoint_uri}: {exc}")
            continue
        return checkpoint_uri
    suffix = f" Rejected: {' | '.join(rejected[-3:])}" if rejected else ""
    raise CheckpointError(f"No valid committed MCore checkpoint under {root_uri}.{suffix}")


def _local_checkpoint_matches_manifest(
    checkpoint: Path,
    entries: dict[str, dict[str, Any]],
    *,
    step: int,
) -> bool:
    """Return whether a directory is an exact local copy of a committed manifest."""

    if checkpoint.is_symlink():
        raise CheckpointError(f"Refusing restore directory symlink: {checkpoint}")
    if not checkpoint.is_dir():
        return False

    expected_files = set(entries)
    expected_dirs: set[str] = set()
    for relative in expected_files:
        parts = relative.split("/")[:-1]
        expected_dirs.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))

    actual_files: dict[str, Path] = {}
    actual_dirs: set[str] = set()
    for current_root, directory_names, file_names in os.walk(checkpoint, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise CheckpointError(f"Refusing checkpoint symlink: {candidate}")
            relative = candidate.relative_to(checkpoint).as_posix()
            actual_dirs.add(relative)
        for name in file_names:
            candidate = current / name
            if candidate.is_symlink() or not candidate.is_file():
                raise CheckpointError(f"Refusing non-regular checkpoint file: {candidate}")
            relative = candidate.relative_to(checkpoint).as_posix()
            if relative == ".s3_restore.json":
                continue
            actual_files[relative] = candidate

    if actual_dirs != expected_dirs or set(actual_files) != expected_files:
        return False
    for relative, record in entries.items():
        path = actual_files[relative]
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            return False
    validate_local_mcore_checkpoint(checkpoint, expected_step=step)
    return True


def _install_restored_checkpoint(temporary: Path, output_dir: Path) -> None:
    """Replace an existing directory transactionally, restoring it on promotion failure."""

    if output_dir.is_symlink():
        raise CheckpointError(f"Refusing restore directory symlink: {output_dir}")
    if not output_dir.is_dir():
        raise CheckpointError(f"Existing restore path is not a directory: {output_dir}")

    backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
    backup.rmdir()
    try:
        os.replace(output_dir, backup)
    except OSError as exc:
        raise CheckpointError(f"Failed to move existing restore directory to backup: {output_dir}") from exc

    try:
        os.replace(temporary, output_dir)
    except OSError as promotion_error:
        try:
            os.replace(backup, output_dir)
        except OSError as rollback_error:
            raise CheckpointError(
                f"Failed to install restored checkpoint and rollback failed; original remains at {backup}"
            ) from rollback_error
        raise CheckpointError(
            f"Failed to install restored checkpoint; original directory was restored: {output_dir}"
        ) from promotion_error

    try:
        shutil.rmtree(backup)
    except OSError as exc:
        raise CheckpointError(
            f"Restored checkpoint is installed, but old backup cleanup failed: {backup}"
        ) from exc


def _download_checkpoint_entries(
    store: ObjectStore,
    entries: dict[str, dict[str, Any]],
    temporary: Path,
    *,
    download_workers: int,
) -> None:
    """Download manifest entries with a bounded, fail-fast worker pool."""

    def download_one(path: str, record: dict[str, Any]) -> None:
        download_verified(
            store,
            record["source_uris"][0],
            local_path(temporary, path),
            size=record["size"],
            sha256=record["sha256"],
        )

    # Preserve the original single-file-at-a-time path by default. Besides
    # compatibility, this keeps lightweight resume restores from creating any
    # executor threads unless callers explicitly opt in.
    if download_workers == 1:
        for path, record in entries.items():
            download_one(path, record)
        return

    remaining = iter(entries.items())
    executor = ThreadPoolExecutor(
        max_workers=download_workers,
        thread_name_prefix="mcore-s3-restore",
    )
    in_flight: set[Future[None]] = set()

    def submit_next() -> bool:
        try:
            path, record = next(remaining)
        except StopIteration:
            return False
        in_flight.add(executor.submit(download_one, path, record))
        return True

    try:
        for _ in range(min(download_workers, len(entries))):
            submit_next()
        while in_flight:
            completed, pending = wait(in_flight, return_when=FIRST_COMPLETED)
            failure = next(
                (future for future in completed if future.exception() is not None),
                None,
            )
            if failure is not None:
                for future in pending:
                    future.cancel()
                failure.result()

            in_flight = set(pending)
            for _ in completed:
                submit_next()
    finally:
        # A running boto3 transfer cannot be interrupted safely. Stop all work
        # that has not started, then wait for active transfers before the caller
        # removes the temporary restore tree.
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def restore_checkpoint(
    store: ObjectStore,
    checkpoint_uri: str,
    output_dir: Path,
    *,
    replace_existing: bool = False,
    download_workers: int = 1,
) -> dict[str, Any]:
    if not isinstance(download_workers, int) or isinstance(download_workers, bool) or download_workers < 1:
        raise CheckpointError(
            f"MCore restore download_workers must be a positive integer, got {download_workers!r}"
        )
    # ``resolve()`` would follow a final symlink and hide precisely the case we
    # must reject. ``abspath`` gives us a stable absolute sibling path without
    # dereferencing the restore target.
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir.is_symlink():
        raise CheckpointError(f"Refusing restore directory symlink: {output_dir}")
    output_exists = output_dir.exists()
    if output_exists and not replace_existing:
        raise CheckpointError(f"Refusing to overwrite existing restore directory: {output_dir}")
    if output_exists and not output_dir.is_dir():
        raise CheckpointError(f"Existing restore path is not a directory: {output_dir}")

    marker, manifest = _read_success(store, checkpoint_uri)
    entries = _validate_global_manifest(
        manifest,
        source=marker["manifest_uri"],
        checkpoint_uri=checkpoint_uri,
    )
    for record in entries.values():
        for uri in record["source_uris"]:
            verify_object_stat(store, uri, size=record["size"], sha256=record["sha256"])

    restore_metadata = {
        "version": PROTOCOL_VERSION,
        "source_checkpoint": _checkpoint_root(checkpoint_uri).uri + "/",
        "source_attempt": manifest["attempt_id"],
        "manifest_uri": marker["manifest_uri"],
        "manifest_sha256": marker["manifest_sha256"],
    }
    if output_exists and _local_checkpoint_matches_manifest(
        output_dir,
        entries,
        step=manifest["step"],
    ):
        write_json_atomic(output_dir / ".s3_restore.json", restore_metadata)
        return {**restore_metadata, "output_dir": str(output_dir), "reused_existing": True}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.restore-", dir=output_dir.parent))
    try:
        _download_checkpoint_entries(
            store,
            entries,
            temporary,
            download_workers=download_workers,
        )
        validate_local_mcore_checkpoint(temporary, expected_step=manifest["step"])
        if not _local_checkpoint_matches_manifest(temporary, entries, step=manifest["step"]):
            raise CheckpointError(
                f"Downloaded checkpoint does not fully match its manifest: {checkpoint_uri}"
            )
        write_json_atomic(temporary / ".s3_restore.json", restore_metadata)
        if output_exists:
            _install_restored_checkpoint(temporary, output_dir)
        else:
            os.replace(temporary, output_dir)
        return {**restore_metadata, "output_dir": str(output_dir), "reused_existing": False}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish-node", help="Upload one node's local checkpoint shards")
    publish.add_argument("--checkpoint", type=Path, required=True)
    publish.add_argument("--checkpoint-s3", required=True)
    publish.add_argument("--attempt-id", required=True)
    publish.add_argument("--node-rank", type=int, required=True)
    publish.add_argument("--node-count", type=int, required=True)
    publish.add_argument("--world-size", type=int)

    commit = subparsers.add_parser("commit", help="Validate every node and create _SUCCESS atomically")
    commit.add_argument("--checkpoint-s3", required=True)
    commit.add_argument("--attempt-id", required=True)
    commit.add_argument("--node-count", type=int, required=True)
    commit.add_argument("--world-size", type=int)
    commit.add_argument("--wait-seconds", type=float, default=0)
    commit.add_argument("--poll-seconds", type=float, default=5)

    inspect = subparsers.add_parser("inspect", help="Validate a committed checkpoint and all objects")
    inspect.add_argument("--checkpoint-s3", required=True)

    latest = subparsers.add_parser("find-latest", help="Print the newest fully committed checkpoint URI")
    latest.add_argument("--root-s3", required=True)

    restore = subparsers.add_parser("restore", help="Reconstruct a committed checkpoint on node-local disk")
    source = restore.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-s3")
    source.add_argument("--root-s3", help="Find and restore the latest committed checkpoint under this root")
    destination = restore.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument(
        "--output-root",
        type=Path,
        help="Create checkpoint-<step> under this node-local root (convenient with --root-s3)",
    )
    restore.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "After verifying the committed remote checkpoint, reuse an exact local copy or "
            "transactionally replace an incomplete/mismatched directory"
        ),
    )
    restore.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Concurrent manifest-file downloads (default: 1, preserving serial restore)",
    )
    restore.add_argument(
        "--s3-transfer-max-concurrency",
        type=int,
        help=(
            "Boto3 multipart concurrency per file. Concurrent restores default to "
            "max(1, 8 // download_workers)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        transfer_max_concurrency = None
        if args.command == "restore":
            if args.s3_transfer_max_concurrency is not None:
                transfer_max_concurrency = args.s3_transfer_max_concurrency
            elif args.download_workers > 1:
                transfer_max_concurrency = max(1, 8 // args.download_workers)
        store = Boto3ObjectStore(
            profile=args.profile,
            region=args.region,
            max_concurrency=transfer_max_concurrency,
        )
        if args.command == "publish-node":
            result = publish_node(
                store,
                args.checkpoint,
                args.checkpoint_s3,
                attempt_id=args.attempt_id,
                node_rank=args.node_rank,
                node_count=args.node_count,
                world_size=args.world_size,
            )
        elif args.command == "commit":
            result = commit_attempt(
                store,
                args.checkpoint_s3,
                attempt_id=args.attempt_id,
                node_count=args.node_count,
                world_size=args.world_size,
                wait_seconds=args.wait_seconds,
                poll_seconds=args.poll_seconds,
            )
        elif args.command == "inspect":
            result = inspect_committed_checkpoint(store, args.checkpoint_s3)
        elif args.command == "find-latest":
            checkpoint_uri = find_latest_committed(store, args.root_s3)
            result = {"checkpoint_uri": checkpoint_uri}
        else:
            checkpoint_uri = args.checkpoint_s3 or find_latest_committed(store, args.root_s3)
            output_dir = args.output_dir
            if output_dir is None:
                checkpoint_name = _checkpoint_root(checkpoint_uri).key.rsplit("/", 1)[-1]
                output_dir = args.output_root / checkpoint_name
            result = restore_checkpoint(
                store,
                checkpoint_uri,
                output_dir,
                replace_existing=args.replace_existing,
                download_workers=args.download_workers,
            )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
