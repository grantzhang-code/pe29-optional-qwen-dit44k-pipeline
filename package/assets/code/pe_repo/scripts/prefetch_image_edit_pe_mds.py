#!/usr/bin/env python3
"""Warm upcoming MDS shards alongside an existing rank-local training loader.

This process follows validated shard-opening records, never consumes samples,
and publishes through the loader's checksum/lock/atomic-rename cache protocol.
Run one instance per node. It requires the single DataLoader worker, TP1/PP1/
CP1 rank layout used by the accompanying held-B300 experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import threading
import time

os.environ["DLMS_IMAGE_EDIT_V6_PE_MDS_LIBRARY_ONLY"] = "true"
from dlms.datasets import image_edit_v6_pe_mds as mds


OPENING = re.compile(
    r"opening MDS shard pass=(\d+) position=(\d+)/(\d+) "
    r"shard_id=(\d+) samples=(\d+) path=(\S+)"
)


class ShardPlanner:
    def __init__(self, shards, config, world_size, ranks, lookahead):
        if world_size < 1 or lookahead < 1 or len(shards) < world_size:
            raise ValueError("invalid world size or lookahead")
        if any(rank not in range(world_size) for rank in ranks):
            raise ValueError("rank outside data-parallel world")
        if [int(s["_shard_id"]) for s in shards] != list(range(len(shards))):
            raise ValueError("metadata shard ids must retain their original index order")
        self.shards, self.config = shards, config
        self.world_size, self.ranks, self.lookahead = world_size, tuple(ranks), lookahead
        self.cursors = {}
        self.invalid_records = 0

    @lru_cache(maxsize=128)
    def order(self, rank, pass_index):
        ordered = list(self.shards[rank::self.world_size])
        stream_seed = sum(int(item["_shard_id"]) for item in ordered)
        if self.config.shuffle_shards:
            random.Random(self.config.seed + stream_seed + pass_index).shuffle(ordered)
        return ordered

    def observe(self, text):
        for record in OPENING.finditer(text):
            pass_index, position, count, shard_id, samples = map(int, record.groups()[:5])
            rank = shard_id % self.world_size
            if rank not in self.ranks:
                continue
            order = self.order(rank, pass_index)
            if not 1 <= position <= len(order) or count != len(order):
                self.invalid_records += 1
                continue
            expected = order[position - 1]
            if (int(expected["_shard_id"]) != shard_id
                    or int(expected["samples"]) != samples
                    or expected["raw_data"]["basename"] != record[6]):
                self.invalid_records += 1
                continue
            # Preprocessing can reopen the first shard. Never rewind the live
            # lookahead when its log is interleaved with the training worker.
            cursor = (pass_index, position)
            if cursor > self.cursors.get(rank, (-1, 0)):
                self.cursors[rank] = cursor

    def wanted(self):
        windows = []
        for rank in self.ranks:
            if rank not in self.cursors:
                continue
            pass_index, position = self.cursors[rank]
            window = []
            for _ in range(self.lookahead + 1):
                ordered = self.order(rank, pass_index)
                if position > len(ordered):
                    if not self.config.repeat:
                        break
                    pass_index, position = pass_index + 1, 1
                    ordered = self.order(rank, pass_index)
                window.append(ordered[position - 1])
                position += 1
            windows.append(window)
        # All ranks' nearest shards precede their more distant shards.
        result, seen = [], set()
        for distance in range(self.lookahead + 1):
            for window in windows:
                if distance < len(window):
                    descriptor = window[distance]
                    shard_id = descriptor["_shard_id"]
                    if shard_id not in seen:
                        result.append(descriptor)
                        seen.add(shard_id)
        return result


class LogFollower:
    def __init__(self, path):
        self.path, self.offset, self.pending = Path(path), 0, ""

    def read(self):
        if self.path.stat().st_size < self.offset:
            self.offset, self.pending = 0, ""
        with self.path.open("rb") as source:
            source.seek(self.offset)
            chunk = source.read(8 * 1024**2)
            self.offset = source.tell()
        text = self.pending + chunk.decode("utf-8", errors="replace")
        last = text.rfind("\n")
        self.pending = text[last + 1:]
        return text[:last + 1] if last >= 0 else ""


def cached(descriptor, config):
    raw = descriptor["raw_data"]
    path = mds._safe_cache_path(Path(config.mds_local_cache_dir), raw["basename"])
    try:
        if path.stat().st_size != int(raw["bytes"]):
            return False
        sidecar = descriptor.get("_target_size_sidecar")
        if sidecar:
            path = mds._safe_cache_path(Path(config.target_size_local_cache_dir), sidecar["relative_key"])
            return path.stat().st_size == int(sidecar["bytes"])
        return True
    except FileNotFoundError:
        return False


def download(descriptor, config):
    started = time.monotonic()
    sidecar = descriptor.get("_target_size_sidecar")
    if sidecar:
        mds._download_target_size_sidecar(sidecar, config)
    mds._download_cached_shard(descriptor, config)
    return {"shard_id": descriptor["_shard_id"], "bytes": int(descriptor["raw_data"]["bytes"]),
            "seconds": round(time.monotonic() - started, 3)}


def atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    document = yaml.safe_load(args.config.read_text())
    env = document["env_vars"]
    if any(int(env[key]) != 1 for key in ("TP", "PP", "CP", "DATALOADER_WORKERS")):
        raise ValueError("prefetch rank mapping requires TP1/PP1/CP1 and one DataLoader worker")
    nodes, local_ranks = int(env["NNODES"]), int(env["NGPUS_PER_NODE"])
    if args.node_rank not in range(nodes) or not 1 <= args.workers <= 8 or not 1 <= args.lookahead <= 32:
        raise ValueError("invalid node rank, concurrency or lookahead")
    if args.poll_seconds < 1:
        raise ValueError("poll interval must be at least one second")
    config = mds.load_runtime_config(document)
    shards = mds._validate_mds_metadata(config)
    # Leave room for training's independent downloads and concurrent partial
    # files. The trainer retains its original 450 GiB cache configuration.
    reserve = 16 * 1024**3
    window_bytes = max(int(s["raw_data"]["bytes"]) for s in shards) * local_ranks * (args.lookahead + 1)
    if window_bytes + reserve >= config.cache_limit_bytes:
        raise ValueError("lookahead window does not fit the shared cache budget")
    download_config = replace(config, cache_limit_bytes=config.cache_limit_bytes - reserve)
    ranks = range(args.node_rank * local_ranks, (args.node_rank + 1) * local_ranks)
    planner = ShardPlanner(shards, config, nodes * local_ranks, ranks, args.lookahead)
    follower = LogFollower(args.training_log)
    cache_root = Path(config.mds_local_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    status = {"pid": os.getpid(), "node_rank": args.node_rank, "state": "starting",
              "started_at": datetime.now(timezone.utc).isoformat(), "lookahead_per_rank": args.lookahead,
              "workers": args.workers, "cache_dir": str(cache_root),
              "prefetch_cache_limit_bytes": download_config.cache_limit_bytes,
              "manifest_sha256": config.mds_manifest_sha256,
              "completed_shards": 0, "completed_bytes": 0, "errors": 0, "recent": []}
    pending, retry_after = {}, {}
    # Node-local singleton lock; a stale PID/status never creates two services.
    with mds._locked_file(cache_root / ".lookahead-prefetch.lock", blocking=False):
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mds-prefetch") as pool:
            while not stop.is_set():
                if args.status.with_suffix(".stop").exists():
                    break
                training = json.loads(args.training_status.read_text())
                if training["state"] in {"completed", "failed"}:
                    break
                planner.observe(follower.read())
                for future, shard_id in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[future]
                    try:
                        result = future.result()
                        status["completed_shards"] += 1
                        status["completed_bytes"] += result["bytes"]
                        status["recent"] = (status["recent"] + [result])[-32:]
                        print(json.dumps({"event": "prefetched", **result}), flush=True)
                        retry_after.pop(shard_id, None)
                    except Exception as exc:
                        status["errors"] += 1
                        status["last_error"] = f"shard {shard_id}: {type(exc).__name__}: {exc}"
                        print(status["last_error"], flush=True)
                        retry_after[shard_id] = time.monotonic() + 60
                wanted = planner.wanted()
                missing = [s for s in wanted if not cached(s, config)]
                free = shutil.disk_usage(cache_root).free
                for descriptor in missing:
                    if len(pending) >= args.workers or free < 32 * 1024**3:
                        break
                    shard_id = descriptor["_shard_id"]
                    if shard_id in pending.values() or retry_after.get(shard_id, 0) > time.monotonic():
                        continue
                    pending[pool.submit(download, descriptor, download_config)] = shard_id
                status.update(state="running", updated_at=datetime.now(timezone.utc).isoformat(),
                              cursors={str(k): list(v) for k, v in planner.cursors.items()},
                              wanted_shards=len(wanted), ready_shards=len(wanted) - len(missing),
                              in_flight=list(pending.values()), free_bytes=free,
                              invalid_log_records=planner.invalid_records, log_offset=follower.offset)
                atomic_json(args.status, status)
                stop.wait(args.poll_seconds)
        status.update(state="stopped", updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.status, status)


if __name__ == "__main__":
    main()
