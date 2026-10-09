#!/usr/bin/env python3
"""Freeze unfinished chartedit-v3 units and LPT-pack them onto RTX GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import image_edit_sidecar24_catalog as source_catalog_module
from scripts import image_edit_sidecar24_partition as partition_module
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module
from scripts.build_craftv0_image_edit_v6_repartition1600 import (
    canonical_json_bytes,
    join_s3,
    list_committed_indices,
    lpt_assign,
    parse_s3_uri,
    put_json,
    sha256_bytes,
    unit_rows,
    unit_work_estimate,
)


SNAPSHOT_FORMAT = "image_edit_sidecar24_pending_snapshot_v1"
ASSIGNMENT_FORMAT = "image_edit_sidecar24_repartition_node_assignment_v1"
GPUS_PER_POD = 8
SOURCE_SHA256 = "fd0a8c945f78c1e88503f9b8b276d4200b60b237c5e662a387b3223fa9165452"
UNIT_SHA256 = "3499da5c1df5cd5c69ef8ce23db84a6775cccfbca7bb80e7ee27b4b909f304e3"
EXPECTED_UNITS = 104
EXPECTED_ROWS = 42_053
EXPECTED_SELECTION_SHA256 = {
    24: "a51b1ae8f431e99e69d29655236877c4f0bd0642520c6df49a0bf74caface32d",
    64: "a5a4f11585d28182cf5de0f65791bae25510f99e329fb4d7961bdc4503516ba7",
}
CLUSTER = "general-aws-usw2-prod-0"
CAPACITY_RESERVATION_ID = "cr-0d0b136584a3f5cd7"
INSTANCE_TYPE = "g7e.48xlarge"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def node_name(index: int, node_count: int) -> str:
    return f"node-{index:03d}-of-{node_count:03d}"


def build(args: argparse.Namespace, *, client: Any) -> dict[str, Any]:
    physical_gpu_count = int(args.physical_gpu_count)
    node_count = physical_gpu_count // GPUS_PER_POD
    manifest_format = (
        f"image_edit_sidecar24_chartedit_v3_repartition{physical_gpu_count}_manifest_v1"
    )
    source = source_catalog_module.load_catalog(args.source_catalog)
    source_sha = source_catalog_module.canonical_catalog_sha256(source)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"source catalog SHA drifted: {source_sha}")
    unit_catalog = unit_catalog_module.load_unit_catalog(
        args.unit_catalog,
        source_catalog=source,
    )
    unit_sha = unit_catalog_module.canonical_unit_catalog_sha256(
        unit_catalog,
        source_catalog=source,
    )
    if unit_sha != UNIT_SHA256:
        raise ValueError(f"unit catalog SHA drifted: {unit_sha}")
    units = unit_catalog["units"]
    if len(units) != EXPECTED_UNITS or int(unit_catalog["total_selected_rows"]) != EXPECTED_ROWS:
        raise ValueError("frozen chartedit-v3 unit totals drifted")

    plan = partition_module.build_partition_plan(
        units,
        partition_count=1,
        partition_index=0,
        global_world_size=physical_gpu_count,
        case_limit=None,
    )
    selection_sha = str(plan.contract["selection_sha256"])
    if selection_sha != EXPECTED_SELECTION_SHA256[physical_gpu_count]:
        raise ValueError(
            f"{physical_gpu_count}-rank selection contract drifted: {selection_sha}"
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
    units_by_index = {int(unit["global_unit_index"]): unit for unit in units}
    pending = [
        unit for unit in units if int(unit["global_unit_index"]) not in committed
    ]
    if len(pending) < physical_gpu_count:
        raise ValueError(
            f"pending units={len(pending)} cannot keep {physical_gpu_count} RTX ranks non-empty"
        )
    completed_indices = sorted(committed)
    pending_indices = [int(unit["global_unit_index"]) for unit in pending]
    completed_rows = sum(unit_rows(units_by_index[index]) for index in committed)
    pending_rows = sum(unit_rows(unit) for unit in pending)
    snapshot = {
        "format_version": SNAPSHOT_FORMAT,
        "result_root": args.output_s3.rstrip("/"),
        "snapshot_started_at": snapshot_started_at,
        "snapshot_completed_at": snapshot_completed_at,
        "source_catalog_sha256": source_sha,
        "unit_catalog_sha256": unit_sha,
        "logical_world_size": physical_gpu_count,
        "selection_sha256": selection_sha,
        "total_units": EXPECTED_UNITS,
        "total_rows": EXPECTED_ROWS,
        "completed_units": len(committed),
        "completed_rows": completed_rows,
        "pending_units": len(pending),
        "pending_rows": pending_rows,
        "completed_indices_sha256": sha256_bytes(canonical_json_bytes(completed_indices)),
        "pending_indices_sha256": sha256_bytes(canonical_json_bytes(pending_indices)),
        "listing": listing,
    }
    snapshot_sha = sha256_bytes(canonical_json_bytes(snapshot))
    lanes, loads = lpt_assign(pending, rank_count=physical_gpu_count)
    if min(map(len, lanes)) < 1:
        raise AssertionError("every physical rank must receive at least one pending unit")

    prefix = join_s3(args.output_s3, "repartitions", args.repartition_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assignments_dir = args.output_dir / "assignments"
    assignments_dir.mkdir()
    nodes: list[dict[str, Any]] = []
    for node_index in range(node_count):
        rank_start = node_index * GPUS_PER_POD
        rank_end = rank_start + GPUS_PER_POD
        assignment = {
            "format_version": ASSIGNMENT_FORMAT,
            "repartition_id": args.repartition_id,
            "created_at": snapshot_completed_at,
            "snapshot_sha256": snapshot_sha,
            "node_index": node_index,
            "node_count": node_count,
            "local_world_size": GPUS_PER_POD,
            "global_rank_offset": rank_start,
            "logical_world_size": physical_gpu_count,
            "selection_sha256": selection_sha,
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
        filename = f"{node_name(node_index, node_count)}.json"
        (assignments_dir / filename).write_bytes(payload)
        uri = join_s3(prefix, "assignments", filename)
        if args.upload:
            uploaded = put_json(client, uri, assignment)
            if uploaded != digest:
                raise AssertionError("assignment digest changed during upload")
        assigned_indices = [
            index for rank in range(rank_start, rank_end) for index in lanes[rank]
        ]
        nodes.append(
            {
                "node_index": node_index,
                "global_rank_offset": rank_start,
                "gpus_per_pod": GPUS_PER_POD,
                "assignment_uri": uri,
                "assignment_sha256": digest,
                "assigned_units": len(assigned_indices),
                "assigned_rows": sum(unit_rows(units_by_index[index]) for index in assigned_indices),
            }
        )

    cohort = {
        "cohort_id": "u2r8",
        "cluster": CLUSTER,
        "capacity_reservation_id": CAPACITY_RESERVATION_ID,
        "instance_type": INSTANCE_TYPE,
        "accelerator": "rtx-pro-server-6000",
        "cuda_capability": "12.0",
        "pod_count": node_count,
        "gpus_per_pod": GPUS_PER_POD,
        "global_rank_start": 0,
        "node_index_start": 0,
        "priority": "high",
        "torch_cuda_arch_list": "12.0+PTX",
    }
    topology = {
        "format_version": f"image_edit_v6_single_source_{physical_gpu_count}rtx_topology_v1",
        "total_nodes": node_count,
        "total_ranks": physical_gpu_count,
        "rtx_ranks": physical_gpu_count,
        "h200_ranks": 0,
        "selection_policy": "all_rows_no_sampling_resume_committed_units",
        "source_catalog_sha256": source_sha,
        "unit_catalog_sha256": unit_sha,
        "selection_sha256": selection_sha,
        "pools": [dict(cohort)],
        "cohorts": [dict(cohort)],
        "nodes": [
            {
                "node_index": item["node_index"],
                "global_rank_start": item["global_rank_offset"],
                "gpus_per_pod": GPUS_PER_POD,
                "pool_id": "u2r8",
                "accelerator": "rtx-pro-server-6000",
            }
            for item in nodes
        ],
    }
    topology_payload = canonical_json_bytes(topology)
    topology_sha = sha256_bytes(topology_payload)
    (args.output_dir / "topology.json").write_bytes(topology_payload)
    topology_uri = join_s3(prefix, "topology.json")
    if args.upload:
        uploaded = put_json(client, topology_uri, topology)
        if uploaded != topology_sha:
            raise AssertionError("topology digest changed during upload")

    manifest = {
        "format_version": manifest_format,
        "status": "frozen",
        "repartition_id": args.repartition_id,
        "created_at": snapshot_completed_at,
        "result_root": args.output_s3.rstrip("/"),
        "prefix": prefix,
        "snapshot": snapshot,
        "snapshot_sha256": snapshot_sha,
        "physical_gpu_count": physical_gpu_count,
        "physical_node_count": node_count,
        "logical_world_size": physical_gpu_count,
        "accelerator": "rtx-pro-server-6000",
        "h200_gpu_count": 0,
        "priority": "high",
        "topology_uri": topology_uri,
        "topology_sha256": topology_sha,
        "load_balance": {
            "algorithm": "deterministic_lpt_rows_source_bytes_v1",
            "minimum_rank_units": min(map(len, lanes)),
            "maximum_rank_units": max(map(len, lanes)),
            "minimum_rank_work": min(loads),
            "maximum_rank_work": max(loads),
        },
        "cohorts": [cohort],
        "nodes": nodes,
    }
    if sum(item["assigned_units"] for item in nodes) != len(pending):
        raise AssertionError("node assignments do not cover every pending unit")
    if sum(item["assigned_rows"] for item in nodes) != pending_rows:
        raise AssertionError("node assignments do not cover every pending row")
    manifest_payload = canonical_json_bytes(manifest)
    (args.output_dir / "manifest.json").write_bytes(manifest_payload)
    if args.upload:
        manifest_uri = join_s3(prefix, "manifest.json")
        manifest_sha = put_json(client, manifest_uri, manifest)
        ready = {
            "format_version": "image_edit_sidecar24_repartition_ready_v1",
            "status": "ready",
            "repartition_id": args.repartition_id,
            "created_at": snapshot_completed_at,
            "manifest_uri": manifest_uri,
            "manifest_sha256": manifest_sha,
            "snapshot_sha256": snapshot_sha,
            "pending_units": len(pending),
            "pending_rows": pending_rows,
            "physical_gpu_count": physical_gpu_count,
            "h200_gpu_count": 0,
        }
        put_json(client, join_s3(prefix, "_READY"), ready)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--repartition-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--physical-gpu-count",
        type=int,
        choices=sorted(EXPECTED_SELECTION_SHA256),
        default=64,
        help="Total RTX ranks; each pod contains 8 GPUs (default: 64)",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    parse_s3_uri(args.output_s3)
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
                "topology_uri": manifest["topology_uri"],
                "topology_sha256": manifest["topology_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
