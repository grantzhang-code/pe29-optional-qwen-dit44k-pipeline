#!/usr/bin/env python3
"""Regroup an immutable 1,600-rank repartition snapshot into a new RTX pod layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_craftv0_image_edit_v6_repartition1600 as planner  # noqa: E402
from scripts import image_edit_sidecar24_catalog as source_catalog_module  # noqa: E402
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def regroup(args: argparse.Namespace, *, client: Any) -> dict[str, Any]:
    previous = load_json(args.previous_dir / "manifest.json")
    if (
        previous.get("format_version") != planner.FORMAT
        or previous.get("status") != "frozen"
        or previous.get("physical_gpu_count") != planner.PHYSICAL_GPU_COUNT
        or previous.get("logical_world_size") != planner.LOGICAL_WORLD_SIZE
        or previous.get("h200_gpu_count") != 0
    ):
        raise ValueError("previous repartition manifest is incompatible")
    snapshot = previous["snapshot"]
    snapshot_sha = planner.sha256_bytes(planner.canonical_json_bytes(snapshot))
    if snapshot_sha != previous.get("snapshot_sha256"):
        raise ValueError("previous snapshot SHA-256 is invalid")

    lanes: dict[int, list[int]] = {}
    for node in previous["nodes"]:
        filename = str(node["assignment_uri"]).rsplit("/", 1)[-1]
        assignment = load_json(args.previous_dir / filename)
        if (
            assignment.get("snapshot_sha256") != snapshot_sha
            or assignment.get("selection_sha256") != snapshot["selection_sha256"]
            or assignment.get("source_catalog_sha256") != snapshot["source_catalog_sha256"]
            or assignment.get("unit_catalog_sha256") != snapshot["unit_catalog_sha256"]
        ):
            raise ValueError(f"{filename}: assignment binding drifted")
        for lane in assignment["assignments"]:
            rank = int(lane["physical_global_rank"])
            if rank in lanes:
                raise ValueError(f"rank {rank} occurs in multiple previous assignments")
            indices = lane["global_unit_indices"]
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"rank {rank} has an empty/invalid lane")
            lanes[rank] = [int(index) for index in indices]
    if sorted(lanes) != list(range(planner.PHYSICAL_GPU_COUNT)):
        raise ValueError("previous assignments do not cover physical ranks 0..1599")
    assigned_indices = [index for rank in sorted(lanes) for index in lanes[rank]]
    if (
        len(assigned_indices) != snapshot["pending_units"]
        or len(assigned_indices) != len(set(assigned_indices))
        or planner.sha256_bytes(planner.canonical_json_bytes(sorted(assigned_indices)))
        != snapshot["pending_indices_sha256"]
    ):
        raise ValueError("previous assignments do not cover the pending snapshot exactly once")

    source = source_catalog_module.load_catalog(args.source_catalog)
    source_sha = source_catalog_module.canonical_catalog_sha256(source)
    units = unit_catalog_module.load_unit_catalog(args.unit_catalog, source_catalog=source)
    unit_sha = unit_catalog_module.canonical_unit_catalog_sha256(units, source_catalog=source)
    if (source_sha, unit_sha) != (planner.SOURCE_SHA256, planner.UNIT_SHA256):
        raise ValueError("catalog SHA-256 drifted")
    units_by_index = {
        int(unit["global_unit_index"]): unit for unit in units["units"]
    }

    prefix = planner.join_s3(args.output_s3, "repartitions", args.repartition_id)
    created_at = planner.utc_now()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assignment_records = []
    for node in planner.physical_nodes():
        rank_start = int(node["global_rank_offset"])
        width = int(node["gpus_per_pod"])
        rank_end = rank_start + width
        assignment = {
            "format_version": planner.REPARTITION_NODE_ASSIGNMENT_FORMAT,
            "repartition_id": args.repartition_id,
            "created_at": created_at,
            "snapshot_sha256": snapshot_sha,
            "node_index": int(node["node_index"]),
            "node_count": planner.NODE_COUNT,
            "local_world_size": width,
            "global_rank_offset": rank_start,
            "logical_world_size": planner.LOGICAL_WORLD_SIZE,
            "selection_sha256": snapshot["selection_sha256"],
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
        payload = planner.canonical_json_bytes(assignment)
        digest = planner.sha256_bytes(payload)
        filename = f"{planner.node_name(int(node['node_index']))}.json"
        (args.output_dir / filename).write_bytes(payload)
        uri = planner.join_s3(prefix, "assignments", filename)
        if args.upload:
            planner.put_json(client, uri, assignment)
        rank_indices = [
            index for rank in range(rank_start, rank_end) for index in lanes[rank]
        ]
        assignment_records.append(
            {
                **node,
                "assignment_uri": uri,
                "assignment_sha256": digest,
                "assigned_units": len(rank_indices),
                "assigned_rows": sum(
                    planner.unit_rows(units_by_index[index]) for index in rank_indices
                ),
            }
        )

    manifest = {
        **previous,
        "repartition_id": args.repartition_id,
        "created_at": created_at,
        "prefix": prefix,
        "physical_node_count": planner.NODE_COUNT,
        "cohorts": [dict(item) for item in planner.NODE_LAYOUT],
        "nodes": assignment_records,
        "regrouped_from": {
            "repartition_id": previous["repartition_id"],
            "manifest_sha256": planner.sha256_bytes(
                planner.canonical_json_bytes(previous)
            ),
        },
    }
    if (
        sum(item["assigned_units"] for item in assignment_records)
        != snapshot["pending_units"]
        or sum(item["assigned_rows"] for item in assignment_records)
        != snapshot["pending_rows"]
        or max(
            item["global_rank_offset"] + item["gpus_per_pod"]
            for item in assignment_records
        )
        != planner.PHYSICAL_GPU_COUNT
    ):
        raise AssertionError("regrouped node coverage is invalid")
    payload = planner.canonical_json_bytes(manifest)
    (args.output_dir / "manifest.json").write_bytes(payload)
    if args.upload:
        planner.put_json(client, planner.join_s3(prefix, "manifest.json"), manifest)
        planner.put_json(
            client,
            planner.join_s3(prefix, "_READY"),
            {
                "format_version": "image_edit_sidecar24_repartition_ready_v1",
                "status": "ready",
                "repartition_id": args.repartition_id,
                "created_at": created_at,
                "manifest_uri": planner.join_s3(prefix, "manifest.json"),
                "manifest_sha256": planner.sha256_bytes(payload),
                "snapshot_sha256": snapshot_sha,
                "pending_units": snapshot["pending_units"],
                "pending_rows": snapshot["pending_rows"],
                "physical_gpu_count": planner.PHYSICAL_GPU_COUNT,
                "h200_gpu_count": 0,
            },
        )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--repartition-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    planner.parse_s3_uri(args.output_s3)
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        region_name=args.aws_region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    manifest = regroup(args, client=client)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "repartition_id": manifest["repartition_id"],
                "snapshot_sha256": manifest["snapshot_sha256"],
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
