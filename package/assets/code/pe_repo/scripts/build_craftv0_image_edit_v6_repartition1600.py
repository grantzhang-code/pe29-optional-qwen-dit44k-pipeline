#!/usr/bin/env python3
"""Freeze unfinished craftv0 V6 units and LPT-pack them onto 1,600 RTX GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import image_edit_sidecar24_catalog as source_catalog_module  # noqa: E402
from scripts import image_edit_sidecar24_partition as partition_module  # noqa: E402
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module  # noqa: E402
FORMAT = "image_edit_sidecar24_repartition1600_manifest_v1"
SNAPSHOT_FORMAT = "image_edit_sidecar24_pending_snapshot_v1"
REPARTITION_NODE_ASSIGNMENT_FORMAT = "image_edit_sidecar24_repartition_node_assignment_v1"
REBALANCE_ROW_WEIGHT_BYTES = 64 * 1024 * 1024
REBALANCE_UNIT_OVERHEAD_BYTES = 8 * 1024 * 1024
PHYSICAL_GPU_COUNT = 1600
LOGICAL_WORLD_SIZE = 2040
NODE_COUNT = 592
SOURCE_SHA256 = "3ba196d0d665c2400a1d17c55db7b262fc829adf1e6e7836ce928617b74fffd9"
UNIT_SHA256 = "5197ce481c23b745decf0cca47a17d9579c549691ec7e52fa4e1e070d18c4916"
EXPECTED_UNITS = 372_215
EXPECTED_ROWS = 115_802_213
NODE_LAYOUT = (
    {
        "cohort_id": "u1r8",
        "cluster": "general-aws-use1-prod-0",
        "capacity_reservation_id": "cr-065a4968dd7227922",
        "instance_type": "g7e.48xlarge",
        "pod_count": 27,
        "gpus_per_pod": 8,
    },
    {
        "cohort_id": "u2r8",
        "cluster": "general-aws-usw2-prod-0",
        "capacity_reservation_id": "cr-0d0b136584a3f5cd7",
        "instance_type": "g7e.48xlarge",
        "pod_count": 117,
        "gpus_per_pod": 8,
    },
    {
        "cohort_id": "u1r1",
        "cluster": "general-aws-use1-prod-0",
        "capacity_reservation_id": "cr-08c4d22448740f61b",
        "instance_type": "g7e.2xlarge",
        "pod_count": 192,
        "gpus_per_pod": 1,
    },
    {
        "cohort_id": "u2r1",
        "cluster": "general-aws-usw2-prod-0",
        "capacity_reservation_id": "cr-05dad74029581898c",
        "instance_type": "g7e.2xlarge",
        "pod_count": 256,
        "gpus_per_pod": 1,
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, key


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key, suffix) if value)}"


def node_name(index: int) -> str:
    return f"node-{index:03d}-of-{NODE_COUNT:03d}"


def physical_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    node_index = 0
    rank_offset = 0
    for cohort in NODE_LAYOUT:
        for cohort_pod_index in range(int(cohort["pod_count"])):
            width = int(cohort["gpus_per_pod"])
            nodes.append(
                {
                    **cohort,
                    "node_index": node_index,
                    "cohort_pod_index": cohort_pod_index,
                    "global_rank_offset": rank_offset,
                }
            )
            node_index += 1
            rank_offset += width
    if node_index != NODE_COUNT or rank_offset != PHYSICAL_GPU_COUNT:
        raise AssertionError(
            f"invalid physical topology: nodes={node_index} GPUs={rank_offset}"
        )
    return nodes


def unit_rows(unit: Mapping[str, Any]) -> int:
    return int(unit["global_row_end_exclusive"]) - int(unit["global_row_start"])


def unit_work_estimate(unit: Mapping[str, Any]) -> int:
    return (
        unit_rows(unit) * REBALANCE_ROW_WEIGHT_BYTES
        + int(unit["source"]["bytes"])
        + REBALANCE_UNIT_OVERHEAD_BYTES
    )


def result_keys(unit: Mapping[str, Any], *, result_key_prefix: str) -> tuple[str, str]:
    stem = f"unit-{int(unit['dataset_unit_index']):06d}"
    root = f"{result_key_prefix}/datasets/{unit['dataset_id']}/shards"
    return f"{root}/{stem}.jsonl", f"{root}/{stem}.success.json"


def list_committed_indices(
    client: Any,
    *,
    bucket: str,
    result_key_prefix: str,
    units: Sequence[Mapping[str, Any]],
) -> tuple[set[int], dict[str, int]]:
    expected: dict[str, tuple[int, str]] = {}
    for unit in units:
        index = int(unit["global_unit_index"])
        output_key, marker_key = result_keys(unit, result_key_prefix=result_key_prefix)
        expected[output_key] = (index, "output")
        expected[marker_key] = (index, "marker")
    seen_outputs: set[int] = set()
    seen_markers: set[int] = set()
    unknown_result_objects = 0
    listed_objects = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{result_key_prefix}/datasets/"):
        for item in page.get("Contents", []):
            listed_objects += 1
            key = str(item["Key"])
            identity = expected.get(key)
            if identity is None:
                unknown_result_objects += 1
                continue
            index, kind = identity
            if int(item.get("Size", 0)) <= 0:
                continue
            (seen_outputs if kind == "output" else seen_markers).add(index)
    committed = seen_outputs & seen_markers
    return committed, {
        "listed_objects": listed_objects,
        "recognized_outputs": len(seen_outputs),
        "recognized_markers": len(seen_markers),
        "unknown_result_objects": unknown_result_objects,
        "output_without_marker": len(seen_outputs - seen_markers),
        "marker_without_output": len(seen_markers - seen_outputs),
    }


def lpt_assign(
    units: Sequence[Mapping[str, Any]],
    *,
    rank_count: int = PHYSICAL_GPU_COUNT,
) -> tuple[list[list[int]], list[int]]:
    if len(units) < rank_count:
        raise ValueError(
            f"pending units={len(units)} cannot keep {rank_count} RTX workers non-empty"
        )
    lanes: list[list[int]] = [[] for _ in range(rank_count)]
    loads = [0] * rank_count
    heap = [(0, 0, rank) for rank in range(rank_count)]
    heapq.heapify(heap)
    ordered = sorted(
        units,
        key=lambda unit: (-unit_work_estimate(unit), int(unit["global_unit_index"])),
    )
    for unit in ordered:
        load, count, rank = heapq.heappop(heap)
        index = int(unit["global_unit_index"])
        lanes[rank].append(index)
        loads[rank] = load + unit_work_estimate(unit)
        heapq.heappush(heap, (loads[rank], count + 1, rank))
    for lane in lanes:
        lane.sort()
    assigned = [index for lane in lanes for index in lane]
    expected = sorted(int(unit["global_unit_index"]) for unit in units)
    if sorted(assigned) != expected or len(assigned) != len(set(assigned)):
        raise AssertionError("LPT assignment does not cover pending units exactly once")
    return lanes, loads


def put_json(client: Any, uri: str, value: Any) -> str:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    bucket, key = parse_s3_uri(uri)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if (
        int(head["ContentLength"]) != len(payload)
        or head.get("Metadata", {}).get("sha256") != digest
        or head.get("ContentType") != "application/json"
    ):
        raise IOError(f"uploaded assignment verification failed: {uri}")
    return digest


def build(
    args: argparse.Namespace,
    *,
    client: Any,
) -> dict[str, Any]:
    source = source_catalog_module.load_catalog(args.source_catalog)
    source_sha = source_catalog_module.canonical_catalog_sha256(source)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"source catalog SHA drifted: {source_sha}")
    unit_catalog = unit_catalog_module.load_unit_catalog(args.unit_catalog, source_catalog=source)
    unit_sha = unit_catalog_module.canonical_unit_catalog_sha256(
        unit_catalog, source_catalog=source
    )
    if unit_sha != UNIT_SHA256:
        raise ValueError(f"unit catalog SHA drifted: {unit_sha}")
    units = unit_catalog["units"]
    if len(units) != EXPECTED_UNITS or int(unit_catalog["total_selected_rows"]) != EXPECTED_ROWS:
        raise ValueError("frozen unit catalog totals drifted")
    plan = partition_module.build_partition_plan(
        units,
        partition_count=1,
        partition_index=0,
        global_world_size=LOGICAL_WORLD_SIZE,
        case_limit=None,
    )

    bucket, result_key_prefix = parse_s3_uri(args.output_s3)
    snapshot_started_at = utc_now()
    committed, listing = list_committed_indices(
        client,
        bucket=bucket,
        result_key_prefix=result_key_prefix,
        units=units,
    )
    snapshot_completed_at = utc_now()
    all_indices = {int(unit["global_unit_index"]) for unit in units}
    if not committed <= all_indices:
        raise AssertionError("committed result set contains unknown global units")
    pending = [unit for unit in units if int(unit["global_unit_index"]) not in committed]
    completed_rows = sum(unit_rows(unit) for unit in units if int(unit["global_unit_index"]) in committed)
    pending_rows = sum(unit_rows(unit) for unit in pending)
    pending_indices = [int(unit["global_unit_index"]) for unit in pending]
    completed_indices = sorted(committed)
    snapshot = {
        "format_version": SNAPSHOT_FORMAT,
        "result_root": args.output_s3.rstrip("/"),
        "snapshot_started_at": snapshot_started_at,
        "snapshot_completed_at": snapshot_completed_at,
        "source_catalog_sha256": source_sha,
        "unit_catalog_sha256": unit_sha,
        "logical_world_size": LOGICAL_WORLD_SIZE,
        "selection_sha256": plan.contract["selection_sha256"],
        "total_units": len(units),
        "total_rows": int(unit_catalog["total_selected_rows"]),
        "completed_units": len(committed),
        "completed_rows": completed_rows,
        "pending_units": len(pending),
        "pending_rows": pending_rows,
        "completed_indices_sha256": sha256_bytes(canonical_json_bytes(completed_indices)),
        "pending_indices_sha256": sha256_bytes(canonical_json_bytes(pending_indices)),
        "listing": listing,
    }
    snapshot_sha = sha256_bytes(canonical_json_bytes(snapshot))
    lanes, loads = lpt_assign(pending)
    nodes = physical_nodes()
    prefix = join_s3(args.output_s3, "repartitions", args.repartition_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assignment_records = []
    for node in nodes:
        rank_start = int(node["global_rank_offset"])
        width = int(node["gpus_per_pod"])
        rank_end = rank_start + width
        assignment = {
            "format_version": REPARTITION_NODE_ASSIGNMENT_FORMAT,
            "repartition_id": args.repartition_id,
            "created_at": snapshot_completed_at,
            "snapshot_sha256": snapshot_sha,
            "node_index": int(node["node_index"]),
            "node_count": NODE_COUNT,
            "local_world_size": width,
            "global_rank_offset": rank_start,
            "logical_world_size": LOGICAL_WORLD_SIZE,
            "selection_sha256": plan.contract["selection_sha256"],
            "source_catalog_sha256": source_sha,
            "unit_catalog_sha256": unit_sha,
            "assignments": [
                {
                    "physical_global_rank": rank,
                    "global_unit_indices": lanes[rank],
                }
                for rank in range(rank_start, rank_end)
            ],
        }
        payload = canonical_json_bytes(assignment)
        digest = sha256_bytes(payload)
        filename = f"{node_name(int(node['node_index']))}.json"
        (args.output_dir / filename).write_bytes(payload)
        uri = join_s3(prefix, "assignments", filename)
        if args.upload:
            uploaded = put_json(client, uri, assignment)
            if uploaded != digest:
                raise AssertionError("assignment digest changed during upload")
        assignment_records.append(
            {
                **node,
                "assignment_uri": uri,
                "assignment_sha256": digest,
                "assigned_units": sum(len(lanes[rank]) for rank in range(rank_start, rank_end)),
                "assigned_rows": sum(
                    unit_rows(units[index])
                    for rank in range(rank_start, rank_end)
                    for index in lanes[rank]
                ),
            }
        )
    manifest = {
        "format_version": FORMAT,
        "status": "frozen",
        "repartition_id": args.repartition_id,
        "created_at": snapshot_completed_at,
        "prefix": prefix,
        "snapshot": snapshot,
        "snapshot_sha256": snapshot_sha,
        "physical_gpu_count": PHYSICAL_GPU_COUNT,
        "physical_node_count": NODE_COUNT,
        "logical_world_size": LOGICAL_WORLD_SIZE,
        "accelerator": "rtx-pro-server-6000",
        "h200_gpu_count": 0,
        "priority": "high",
        "load_balance": {
            "algorithm": "deterministic_lpt_rows_source_bytes_v1",
            "row_weight_bytes": REBALANCE_ROW_WEIGHT_BYTES,
            "unit_overhead_bytes": REBALANCE_UNIT_OVERHEAD_BYTES,
            "minimum_rank_units": min(map(len, lanes)),
            "maximum_rank_units": max(map(len, lanes)),
            "minimum_rank_work": min(loads),
            "maximum_rank_work": max(loads),
        },
        "cohorts": [dict(item) for item in NODE_LAYOUT],
        "nodes": assignment_records,
    }
    if sum(item["assigned_units"] for item in assignment_records) != len(pending):
        raise AssertionError("node assignments do not cover every pending unit")
    manifest_payload = canonical_json_bytes(manifest)
    (args.output_dir / "manifest.json").write_bytes(manifest_payload)
    if args.upload:
        put_json(client, join_s3(prefix, "manifest.json"), manifest)
        ready = {
            "format_version": "image_edit_sidecar24_repartition_ready_v1",
            "status": "ready",
            "repartition_id": args.repartition_id,
            "created_at": snapshot_completed_at,
            "manifest_uri": join_s3(prefix, "manifest.json"),
            "manifest_sha256": sha256_bytes(manifest_payload),
            "snapshot_sha256": snapshot_sha,
            "pending_units": len(pending),
            "pending_rows": pending_rows,
            "physical_gpu_count": PHYSICAL_GPU_COUNT,
            "h200_gpu_count": 0,
        }
        put_json(client, join_s3(prefix, "_READY"), ready)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--repartition-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--aws-profile")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parse_s3_uri(args.output_s3)
    except ValueError as exc:
        parser.error(str(exc))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.repartition_id):
        parser.error("--repartition-id must be a lowercase DNS-style identifier")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=args.aws_profile) if args.aws_profile else boto3.Session()
    client = session.client(
        "s3",
        region_name=args.aws_region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    manifest = build(args, client=client)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "repartition_id": manifest["repartition_id"],
                "snapshot_sha256": manifest["snapshot_sha256"],
                "completed_units": manifest["snapshot"]["completed_units"],
                "completed_rows": manifest["snapshot"]["completed_rows"],
                "pending_units": manifest["snapshot"]["pending_units"],
                "pending_rows": manifest["snapshot"]["pending_rows"],
                "physical_gpu_count": manifest["physical_gpu_count"],
                "physical_node_count": manifest["physical_node_count"],
                "prefix": manifest["prefix"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
