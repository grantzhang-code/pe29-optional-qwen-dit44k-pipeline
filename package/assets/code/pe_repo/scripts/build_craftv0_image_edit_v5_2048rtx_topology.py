#!/usr/bin/env python3
"""Build and validate the frozen use1/usw2 2048-RTX craftv0 topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


FORMAT_VERSION = "image_edit_v5_craftv0_2048rtx_topology_v1"
TOTAL_NODES = 872
TOTAL_RANKS = 2048
POOLS: tuple[dict[str, Any], ...] = (
    {
        "pool_id": "u1m",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-065a4968dd7227922",
        "instance_type": "g7e.48xlarge",
        "pod_count": 27,
        "gpus_per_pod": 8,
        "node_index_start": 0,
        "global_rank_start": 0,
        # USE1 currently has only eight free plus 48 lower-priority GPUs in
        # this flavor. Four-pod/32-GPU cohorts can be admitted immediately;
        # an atomic eight-pod cohort cannot.
        "cohort_pod_limit": 4,
    },
    {
        "pool_id": "u1sa",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-08c4d22448740f61b",
        "instance_type": "g7e.2xlarge",
        "pod_count": 237,
        "gpus_per_pod": 1,
        "node_index_start": 27,
        "global_rank_start": 216,
        "cohort_pod_limit": 64,
    },
    {
        "pool_id": "u1sb",
        "cluster": "general-aws-use1-prod-0",
        "region": "use1",
        "capacity_reservation_id": "cr-09bb42bed1d457cf6",
        "instance_type": "g7e.2xlarge",
        "pod_count": 17,
        "gpus_per_pod": 1,
        "node_index_start": 264,
        "global_rank_start": 453,
        "cohort_pod_limit": 17,
    },
    {
        "pool_id": "u2m",
        "cluster": "general-aws-usw2-prod-0",
        "region": "usw2",
        "capacity_reservation_id": "cr-0d0b136584a3f5cd7",
        "instance_type": "g7e.48xlarge",
        "pod_count": 141,
        "gpus_per_pod": 8,
        "node_index_start": 281,
        "global_rank_start": 470,
        "cohort_pod_limit": 8,
    },
    {
        "pool_id": "u2s",
        "cluster": "general-aws-usw2-prod-0",
        "region": "usw2",
        "capacity_reservation_id": "cr-05dad74029581898c",
        "instance_type": "g7e.2xlarge",
        "pod_count": 450,
        "gpus_per_pod": 1,
        "node_index_start": 422,
        "global_rank_start": 1598,
        "cohort_pod_limit": 64,
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
        count = int(pool["pod_count"])
        width = int(pool["gpus_per_pod"])
        node_start = int(pool["node_index_start"])
        rank_start = int(pool["global_rank_start"])
        for local_index in range(count):
            nodes.append(
                {
                    "node_index": node_start + local_index,
                    "global_rank_start": rank_start + local_index * width,
                    "gpus_per_pod": width,
                    "pool_id": pool["pool_id"],
                }
            )
        cohort_limit = int(pool["cohort_pod_limit"])
        for cohort_index, local_start in enumerate(range(0, count, cohort_limit)):
            cohort_count = min(cohort_limit, count - local_start)
            cohorts.append(
                {
                    "cohort_id": f"{pool['pool_id']}{cohort_index:02d}",
                    "pool_id": pool["pool_id"],
                    "cluster": pool["cluster"],
                    "capacity_reservation_id": pool["capacity_reservation_id"],
                    "instance_type": pool["instance_type"],
                    "pod_count": cohort_count,
                    "gpus_per_pod": width,
                    "node_index_start": node_start + local_start,
                    "global_rank_start": rank_start + local_start * width,
                }
            )
    topology = {
        "format_version": FORMAT_VERSION,
        "selection_commit": "17064da3709da4ef95e3026f0b817dcc465bb104",
        "source_catalog_sha256": "946a18d0f87afcfb452545e889adf9a38b9bf05b3a3e3a11260d5b298bc6d23e",
        "priority": "high",
        "preemption_policy": "kueue-lower-priority-only",
        "total_nodes": TOTAL_NODES,
        "total_ranks": TOTAL_RANKS,
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
        raise ValueError("logical ranks must cover [0, 2048) exactly once")
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
    if sum(int(cohort["pod_count"]) * int(cohort["gpus_per_pod"]) for cohort in cohorts) != TOTAL_RANKS:
        raise ValueError("cohort GPU total is not 2048")


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
                "cohorts": len(value["cohorts"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
