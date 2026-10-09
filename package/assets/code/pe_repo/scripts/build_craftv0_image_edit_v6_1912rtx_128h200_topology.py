#!/usr/bin/env python3
"""Build the frozen 1912-RTX plus 128-H200 craftv0 V6 topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


FORMAT_VERSION = "image_edit_v6_craftv0_1912rtx_128h200_topology_v1"
TOTAL_NODES = 752
TOTAL_RANKS = 2040
RTX_RANKS = 1912
H200_RANKS = 128

POOLS: tuple[dict[str, Any], ...] = (
    {
        "pool_id": "u1m",
        "accelerator": "rtx-pro-server-6000",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-065a4968dd7227922",
        "instance_type": "g7e.48xlarge",
        "pod_count": 27,
        "gpus_per_pod": 8,
        "node_index_start": 0,
        "global_rank_start": 0,
        "cohort_pod_limit": 4,
        "cuda_capability": "12.0",
        "torch_cuda_arch_list": "12.0+PTX",
    },
    {
        "pool_id": "u1sa",
        "accelerator": "rtx-pro-server-6000",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-08c4d22448740f61b",
        "instance_type": "g7e.2xlarge",
        "pod_count": 237,
        "gpus_per_pod": 1,
        "node_index_start": 27,
        "global_rank_start": 216,
        "cohort_pod_limit": 64,
        "cuda_capability": "12.0",
        "torch_cuda_arch_list": "12.0+PTX",
    },
    {
        "pool_id": "u1sb",
        "accelerator": "rtx-pro-server-6000",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-09bb42bed1d457cf6",
        "instance_type": "g7e.2xlarge",
        "pod_count": 17,
        "gpus_per_pod": 1,
        "node_index_start": 264,
        "global_rank_start": 453,
        "cohort_pod_limit": 17,
        "cuda_capability": "12.0",
        "torch_cuda_arch_list": "12.0+PTX",
    },
    {
        "pool_id": "u2m",
        "accelerator": "rtx-pro-server-6000",
        "cluster": "general-aws-usw2-prod-0",
        "region": "usw2",
        "capacity_reservation_id": "cr-0d0b136584a3f5cd7",
        "instance_type": "g7e.48xlarge",
        "pod_count": 141,
        "gpus_per_pod": 8,
        "node_index_start": 281,
        "global_rank_start": 470,
        "cohort_pod_limit": 8,
        "cuda_capability": "12.0",
        "torch_cuda_arch_list": "12.0+PTX",
    },
    {
        "pool_id": "u2s",
        "accelerator": "rtx-pro-server-6000",
        "cluster": "general-aws-usw2-prod-0",
        "region": "usw2",
        "capacity_reservation_id": "cr-05dad74029581898c",
        "instance_type": "g7e.2xlarge",
        "pod_count": 314,
        "gpus_per_pod": 1,
        "node_index_start": 422,
        "global_rank_start": 1598,
        "cohort_pod_limit": 64,
        "cuda_capability": "12.0",
        "torch_cuda_arch_list": "12.0+PTX",
    },
    {
        "pool_id": "u1h",
        "accelerator": "h200",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-00f7965e5d49a2f8a",
        "instance_type": "p5en.48xlarge",
        "pod_count": 16,
        "gpus_per_pod": 8,
        "node_index_start": 736,
        "global_rank_start": 1912,
        "cohort_pod_limit": 4,
        "cuda_capability": "9.0",
        "torch_cuda_arch_list": "9.0+PTX",
    },
)


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


def build_topology() -> dict[str, Any]:
    pools = [dict(pool) for pool in POOLS]
    nodes: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    for pool in pools:
        pod_count = int(pool["pod_count"])
        width = int(pool["gpus_per_pod"])
        node_start = int(pool["node_index_start"])
        rank_start = int(pool["global_rank_start"])
        for local_index in range(pod_count):
            nodes.append(
                {
                    "node_index": node_start + local_index,
                    "global_rank_start": rank_start + local_index * width,
                    "gpus_per_pod": width,
                    "pool_id": pool["pool_id"],
                    "accelerator": pool["accelerator"],
                }
            )
        cohort_limit = int(pool["cohort_pod_limit"])
        for cohort_index, local_start in enumerate(
            range(0, pod_count, cohort_limit)
        ):
            cohort_count = min(cohort_limit, pod_count - local_start)
            cohorts.append(
                {
                    "cohort_id": f"{pool['pool_id']}{cohort_index:02d}",
                    "pool_id": pool["pool_id"],
                    "accelerator": pool["accelerator"],
                    "cluster": pool["cluster"],
                    "capacity_reservation_id": pool[
                        "capacity_reservation_id"
                    ],
                    "instance_type": pool["instance_type"],
                    "pod_count": cohort_count,
                    "gpus_per_pod": width,
                    "node_index_start": node_start + local_start,
                    "global_rank_start": rank_start + local_start * width,
                    "cuda_capability": pool["cuda_capability"],
                    "torch_cuda_arch_list": pool["torch_cuda_arch_list"],
                }
            )
    topology = {
        "format_version": FORMAT_VERSION,
        "selection_commit": "fe469c6bfff4648b5e263099a7d91003149cdee7",
        "selection_sha256": (
            "1fb6fbe3e87f721ccbd6d474b1a84186009dc14ada8bc712e50d31d63ede8c39"
        ),
        "source_catalog_sha256": (
            "3ba196d0d665c2400a1d17c55db7b262fc829adf1e6e7836ce928617b74fffd9"
        ),
        "selection_policy": "all_rows_no_sampling",
        "total_selected_rows": 115802213,
        "priority": "high",
        "preemption_policy": "kueue-lower-priority-only",
        "total_nodes": TOTAL_NODES,
        "total_ranks": TOTAL_RANKS,
        "rtx_ranks": RTX_RANKS,
        "h200_ranks": H200_RANKS,
        "pools": pools,
        "cohorts": cohorts,
        "nodes": nodes,
    }
    validate_topology(topology)
    return topology


def validate_topology(value: dict[str, Any]) -> None:
    if value.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected topology format")
    nodes = value.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != TOTAL_NODES:
        raise ValueError(f"topology must contain {TOTAL_NODES} nodes")
    if [node["node_index"] for node in nodes] != list(range(TOTAL_NODES)):
        raise ValueError("node indexes must be contiguous")
    ranks = [
        rank
        for node in nodes
        for rank in range(
            int(node["global_rank_start"]),
            int(node["global_rank_start"]) + int(node["gpus_per_pod"]),
        )
    ]
    if ranks != list(range(TOTAL_RANKS)):
        raise ValueError(f"logical ranks must cover [0, {TOTAL_RANKS}) exactly")
    accelerator_ranks = {
        accelerator: sum(
            int(node["gpus_per_pod"])
            for node in nodes
            if node["accelerator"] == accelerator
        )
        for accelerator in {"rtx-pro-server-6000", "h200"}
    }
    if accelerator_ranks != {
        "rtx-pro-server-6000": RTX_RANKS,
        "h200": H200_RANKS,
    }:
        raise ValueError(f"accelerator totals drifted: {accelerator_ranks}")
    cohorts = value.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("topology has no cohorts")
    cohort_nodes = [
        index
        for cohort in cohorts
        for index in range(
            int(cohort["node_index_start"]),
            int(cohort["node_index_start"]) + int(cohort["pod_count"]),
        )
    ]
    if cohort_nodes != list(range(TOTAL_NODES)):
        raise ValueError("cohorts must cover all nodes exactly once")


def load_topology(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("topology must be a JSON object")
    validate_topology(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_topology()
    payload = canonical_json_bytes(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "nodes": len(value["nodes"]),
                "ranks": sum(node["gpus_per_pod"] for node in value["nodes"]),
                "rtx_ranks": value["rtx_ranks"],
                "h200_ranks": value["h200_ranks"],
                "cohorts": len(value["cohorts"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
