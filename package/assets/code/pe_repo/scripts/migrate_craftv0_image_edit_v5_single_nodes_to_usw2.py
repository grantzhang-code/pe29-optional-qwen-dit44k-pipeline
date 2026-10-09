#!/usr/bin/env python3
"""Move suspended one-GPU craftv0 nodes from use1 to the usw2 one-GPU pool.

The inference contract does not bind a logical one-GPU node to a physical AWS
region.  This tool therefore moves only JobSets that are still Kueue-suspended,
have never created a Pod, and correspond to width-one entries in the committed
topology.  Logical node/rank identity, model and prompt hashes, output prefix,
and topology SHA remain unchanged.

The operation is deliberately ledgered as delete-source then submit-target so
it can safely resume if submission is interrupted after a source deletion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import rebalance_craftv0_image_edit_v5_pending_cohorts as rebalance


SOURCE_CLUSTER = "general-aws-use1-prod-0"
TARGET_CLUSTER = "general-aws-usw2-prod-0"
TARGET_RESERVATION = "cr-05dad74029581898c"
TARGET_INSTANCE_TYPE = "g7e.2xlarge"
DEFAULT_LIMIT = 59
DEFAULT_LEDGER = (
    rebalance.PLAN_ROOT / "migration-use1-single-to-usw2-20260815023150.jsonl"
)


def topology_nodes(topology: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result = {int(node["node_index"]): dict(node) for node in topology["nodes"]}
    if len(result) != rebalance.EXPECTED_TOPOLOGY["total_nodes"]:
        raise rebalance.RebalanceError("topology contains duplicate node indices")
    return result


def source_records() -> list[dict[str, Any]]:
    records = [
        dict(record)
        for record in rebalance.load_jsonl(
            rebalance.PLAN_ROOT / "rebalance-nodewise-20260815023150.jsonl"
        )
        if record.get("action") == "submit_node"
        and record.get("status") == "submitted"
        and record.get("cluster") == SOURCE_CLUSTER
        and int(record.get("gpus_per_pod", 0)) == 1
    ]
    by_node: dict[int, dict[str, Any]] = {}
    for record in records:
        node_index = int(record["node_index"])
        if node_index in by_node:
            raise rebalance.RebalanceError(
                f"duplicate use1 single-node submission record: node {node_index}"
            )
        by_node[node_index] = record
    return [by_node[index] for index in sorted(by_node)]


def target_cohort(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cohort_id": f"migrate-{source['cohort_id']}",
        "cluster": TARGET_CLUSTER,
        "capacity_reservation_id": TARGET_RESERVATION,
        "instance_type": TARGET_INSTANCE_TYPE,
    }


def prior_state(records: Sequence[Mapping[str, Any]]) -> tuple[set[int], set[int]]:
    deleted = {
        int(record["node_index"])
        for record in records
        if record.get("action") == "delete_source"
        and record.get("status") == "completed"
    }
    submitted = {
        int(record["node_index"])
        for record in records
        if record.get("action") == "submit_target"
        and record.get("status") == "submitted"
    }
    if not submitted <= deleted:
        raise rebalance.RebalanceError(
            f"migration ledger has target submissions without source deletions: "
            f"{sorted(submitted - deleted)}"
        )
    return deleted, submitted


def is_suspended_never_started(source: Mapping[str, Any]) -> bool:
    value = rebalance.kubectl_jobset(SOURCE_CLUSTER, str(source["job_name"]))
    if value is None:
        return False
    conditions = value.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Suspended" and condition.get("status") == "True"
        for condition in conditions
    )


def select_nodes(
    *,
    sources: Sequence[Mapping[str, Any]],
    nodes: Mapping[int, Mapping[str, Any]],
    deleted: set[int],
    submitted: set[int],
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source_value in sources:
        source = dict(source_value)
        node_index = int(source["node_index"])
        node = dict(nodes[node_index])
        if int(node["gpus_per_pod"]) != 1:
            raise rebalance.RebalanceError(
                f"refusing to migrate non-single topology node {node_index}"
            )
        if int(node["global_rank_start"]) != int(source["global_rank_start"]):
            raise rebalance.RebalanceError(f"node {node_index} rank drift")
        if node_index in submitted:
            continue
        if node_index in deleted or is_suspended_never_started(source):
            selected.append((source, node))
        if len(selected) >= limit:
            break
    return selected


def submit_target(source: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, Any]:
    record = rebalance.submit_node(target_cohort(source), node)
    record.update(
        {
            "action": "submit_target",
            "source_cluster": SOURCE_CLUSTER,
            "source_job_name": source["job_name"],
            "original_cluster": SOURCE_CLUSTER,
            "migration_reason": (
                "use usw2 one-GPU free quota and Kueue-native lower-priority preemption"
            ),
        }
    )
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= DEFAULT_LIMIT:
        parser.error(f"--limit must be in [1, {DEFAULT_LIMIT}]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topology = rebalance.load_json(rebalance.TOPOLOGY_PATH)
    if not isinstance(topology, dict):
        raise rebalance.RebalanceError("topology root is not an object")
    rebalance.validate_topology(topology)
    nodes = topology_nodes(topology)
    sources = source_records()
    prior = rebalance.load_jsonl(args.ledger)
    deleted, submitted = prior_state(prior)
    selected = select_nodes(
        sources=sources,
        nodes=nodes,
        deleted=deleted,
        submitted=submitted,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "selected": len(selected),
                "node_indices": [int(node["node_index"]) for _, node in selected],
                "source_cluster": SOURCE_CLUSTER,
                "target_cluster": TARGET_CLUSTER,
                "target_capacity_reservation_id": TARGET_RESERVATION,
                "priority": "high",
                "topology_sha256": rebalance.TOPOLOGY_SHA256,
                "output_s3": rebalance.OUTPUT_S3,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        return 0

    for source, node in selected:
        node_index = int(node["node_index"])
        if node_index not in deleted:
            job_name = str(source["job_name"])
            value = rebalance.kubectl_jobset(SOURCE_CLUSTER, job_name)
            if value is None:
                raise rebalance.RebalanceError(
                    f"source JobSet disappeared without ledger record: {job_name}"
                )
            rebalance.verify_never_started(SOURCE_CLUSTER, job_name, value)
            print(f"[delete-source] node={node_index} job={job_name}", flush=True)
            rebalance.delete_original(SOURCE_CLUSTER, job_name)
            rebalance.append_ledger(
                args.ledger,
                {
                    "action": "delete_source",
                    "status": "completed",
                    "completed_at": rebalance.utc_now(),
                    "node_index": node_index,
                    "global_rank_start": int(node["global_rank_start"]),
                    "gpus_per_pod": 1,
                    "source_cluster": SOURCE_CLUSTER,
                    "source_job_name": job_name,
                },
            )
            deleted.add(node_index)
            rebalance.upload_ledger(args.ledger)

        print(f"[submit-target] node={node_index} cluster={TARGET_CLUSTER}", flush=True)
        record = submit_target(source, node)
        rebalance.append_ledger(args.ledger, record)
        rebalance.upload_ledger(args.ledger)
        submitted.add(node_index)
        print(
            f"[submitted-target] node={node_index} rank={node['global_rank_start']} "
            f"job={record['job_name']}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "complete",
                "deleted_sources": len(deleted),
                "submitted_targets": len(submitted),
                "ledger": str(args.ledger),
                "completed_at": rebalance.utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
