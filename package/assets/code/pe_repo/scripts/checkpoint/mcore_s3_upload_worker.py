#!/usr/bin/env python3
"""Upload one MCore rank shard and coordinate an atomic S3 checkpoint commit.

This process is intentionally CPU-only and must remain independent of torch/Swift so
training can continue while one worker per global rank streams the completed shard.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.checkpoint.common import (  # noqa: E402
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
)
from scripts.checkpoint.mcore_s3 import (  # noqa: E402
    commit_rank_attempt,
    inspect_committed_checkpoint,
    publish_rank,
)


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def _validate_topology(
    *,
    rank: int,
    world_size: int,
    node_rank: int,
    node_count: int,
    local_rank: int,
) -> None:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise CheckpointError(f"Invalid global rank/world size: rank={rank} world_size={world_size}")
    if node_count <= 0 or node_rank < 0 or node_rank >= node_count:
        raise CheckpointError(f"Invalid node rank/count: rank={node_rank} count={node_count}")
    if world_size % node_count != 0:
        raise CheckpointError(
            f"world_size={world_size} must be divisible by node_count={node_count}"
        )
    ranks_per_node = world_size // node_count
    if local_rank < 0 or local_rank >= ranks_per_node:
        raise CheckpointError(
            f"Invalid local_rank={local_rank}; expected 0 <= local_rank < {ranks_per_node}"
        )
    expected_rank = node_rank * ranks_per_node + local_rank
    if rank != expected_rank:
        raise CheckpointError(
            f"Rank topology mismatch: global={rank} node={node_rank} local={local_rank}; "
            f"expected global rank {expected_rank}"
        )


def wait_for_committed_checkpoint(
    store: ObjectStore,
    checkpoint_uri: str,
    *,
    wait_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Wait for and fully verify the immutable commit visible to this node."""

    if wait_seconds < 0 or poll_seconds <= 0:
        raise CheckpointError("wait_seconds must be non-negative and poll_seconds must be positive")
    success_uri = checkpoint_uri.rstrip("/") + "/_SUCCESS"
    deadline = time.monotonic() + wait_seconds
    while not store.exists(success_uri):
        if time.monotonic() >= deadline:
            raise CheckpointError(f"Timed out waiting for committed checkpoint: {checkpoint_uri}")
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))
    return inspect_committed_checkpoint(store, checkpoint_uri, verify_objects=True)


def prune_committed_local_checkpoints(
    store: ObjectStore,
    root_s3: str,
    output_root: Path,
    *,
    save_total_limit: int,
    protected_checkpoint: Path,
) -> list[Path]:
    """Prune verified old commits in the low-priority worker, never in training.

    A per-node file lock serialises cleanup when uploads from adjacent saves
    overlap. Directories without a valid remote ``_SUCCESS`` are pending or
    corrupt and are retained fail-closed.
    """

    if save_total_limit < 1:
        raise CheckpointError(f"local save_total_limit must be at least 1, got {save_total_limit}")
    output_root = output_root.resolve()
    protected_checkpoint = protected_checkpoint.resolve()
    if protected_checkpoint.parent != output_root:
        raise CheckpointError(
            f"Protected checkpoint is outside output root: {protected_checkpoint} != {output_root}"
        )
    root = S3Location.parse(root_s3)
    lock_path = output_root / ".mcore_s3_retention.lock"
    removed: list[Path] = []
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        candidates: list[tuple[int, Path]] = []
        for path in output_root.iterdir():
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if match is None or not path.is_dir() or path.is_symlink():
                continue
            resolved = path.resolve()
            if resolved.parent != output_root:
                raise CheckpointError(f"Refusing checkpoint outside output root: {path}")
            checkpoint_s3 = root.join(path.name).uri + "/"
            try:
                inspect_committed_checkpoint(store, checkpoint_s3, verify_objects=True)
            except Exception as exc:
                # Absence or a transient verification failure can never justify
                # deleting the only local copy. A later worker retries cleanup.
                print(
                    f"[mcore-s3-worker] retaining unverified local checkpoint "
                    f"{path.name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            candidates.append((int(match.group(1)), resolved))

        candidates.sort()
        # The worker has already closed and uploaded its own shard, so even an
        # old checkpoint whose upload completed out of order needs no special
        # protection. Always retain strictly by training step.
        keep = {path for _, path in candidates[-save_total_limit:]}
        for _, candidate in candidates:
            if candidate in keep:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
            print(
                f"[mcore-s3-worker] pruned committed local checkpoint {candidate}",
                flush=True,
            )
    return removed


def run_worker(
    store: ObjectStore,
    checkpoint: Path,
    checkpoint_uri: str,
    *,
    attempt_id: str,
    rank: int,
    world_size: int,
    node_rank: int,
    node_count: int,
    local_rank: int,
    commit_wait_seconds: float = 7200,
    poll_seconds: float = 5,
    output_root: Path | None = None,
    local_save_total_limit: int = 0,
) -> dict[str, Any]:
    """Run one rank upload; rank 0 commits and every local rank 0 verifies it."""

    _validate_topology(
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        node_count=node_count,
        local_rank=local_rank,
    )
    manifest = publish_rank(
        store,
        checkpoint,
        checkpoint_uri,
        attempt_id=attempt_id,
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        node_count=node_count,
        local_rank=local_rank,
    )
    result: dict[str, Any] = {
        "status": "uploaded",
        "rank": rank,
        "world_size": world_size,
        "checkpoint_uri": checkpoint_uri.rstrip("/") + "/",
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }
    if rank == 0:
        result["commit"] = commit_rank_attempt(
            store,
            checkpoint_uri,
            attempt_id=attempt_id,
            world_size=world_size,
            node_count=node_count,
            wait_seconds=commit_wait_seconds,
            poll_seconds=poll_seconds,
        )
    if local_rank == 0:
        result["verified_commit"] = wait_for_committed_checkpoint(
            store,
            checkpoint_uri,
            wait_seconds=commit_wait_seconds,
            poll_seconds=poll_seconds,
        )
        result["status"] = "committed"
        if local_save_total_limit:
            if output_root is None:
                raise CheckpointError("output_root is required when local_save_total_limit is set")
            removed = prune_committed_local_checkpoints(
                store,
                checkpoint_uri.rsplit("/checkpoint-", 1)[0] + "/",
                output_root,
                save_total_limit=local_save_total_limit,
                protected_checkpoint=checkpoint,
            )
            result["pruned"] = [str(path) for path in removed]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-s3", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--local-save-total-limit", type=int, default=0)
    parser.add_argument("--commit-wait-seconds", type=float, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument(
        "--s3-max-concurrency",
        type=int,
        default=1,
        help="Per-rank boto3 multipart concurrency (one worker already runs per global rank)",
    )
    parser.add_argument("--cpu-nice", type=int, default=10)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    return parser


def _lower_worker_priority(cpu_nice: int) -> None:
    """Best-effort CPU and disk-I/O de-prioritisation for the upload sidecar process."""

    if cpu_nice < 0 or cpu_nice > 19:
        raise CheckpointError(f"cpu_nice must be between 0 and 19, got {cpu_nice}")
    if cpu_nice:
        os.nice(cpu_nice)
    ionice = shutil.which("ionice")
    if ionice:
        subprocess.run(
            [ionice, "-c", "3", "-p", str(os.getpid())],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _lower_worker_priority(args.cpu_nice)
        store = Boto3ObjectStore(
            profile=args.profile,
            region=args.region,
            max_concurrency=args.s3_max_concurrency,
        )
        result = run_worker(
            store,
            args.checkpoint,
            args.checkpoint_s3,
            attempt_id=args.attempt_id,
            rank=args.rank,
            world_size=args.world_size,
            node_rank=args.node_rank,
            node_count=args.node_count,
            local_rank=args.local_rank,
            commit_wait_seconds=args.commit_wait_seconds,
            poll_seconds=args.poll_seconds,
            output_root=args.output_root,
            local_save_total_limit=args.local_save_total_limit,
        )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
