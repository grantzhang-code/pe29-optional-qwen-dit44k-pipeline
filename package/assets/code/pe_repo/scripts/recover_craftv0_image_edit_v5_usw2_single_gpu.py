#!/usr/bin/env python3
"""Recover failed usw2 single-GPU cohorts as independent high-priority JobSets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from typing import Any, Mapping, Sequence

from scripts import rebalance_craftv0_image_edit_v5_pending_cohorts as common
from scripts import recover_craftv0_image_edit_v5_u1sa00 as contract


SOURCES = {
    "u2s00": {
        "jobset": "grantzhang-v5cv0-0815023150-u2s00-103606",
        "nodes": range(422, 486),
        "ranks": range(1598, 1662),
    },
    "u2s06": {
        "jobset": "grantzhang-v5cv0-0815023150-u2s06-103632",
        "nodes": range(806, 870),
        "ranks": range(1982, 2046),
    },
}
LEDGER = common.PLAN_ROOT / "recovery-usw2-single-gpu-20260815023150.jsonl"
ACTUAL_JOB_RE = re.compile(r"--job-name ([^ ]+)")


class RecoveryError(RuntimeError):
    """Fail-closed recovery validation error."""


def _environment(cohort: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, str]:
    env = common.node_environment(cohort, node)
    env.update(
        {
            "INFERENCE_MEMORY": contract.REQUEST_MEMORY,
            "INFERENCE_MEMORY_LIMIT": contract.LIMIT_MEMORY,
        }
    )
    return env


def _submit(
    cohort_id: str,
    source_jobset: str,
    cohort: Mapping[str, Any],
    node: Mapping[str, Any],
) -> dict[str, Any]:
    node_index = int(node["node_index"])
    requested = f"v5cv0-{common.RUN_TAG}-rpn{node_index:03d}"
    command = [
        str(common.REPO_ROOT / ".venv/bin/python"),
        str(common.REPO_ROOT / "scripts/submit_from_job_yaml.py"),
        str(common.CONFIG_PATH),
        requested,
        "--",
        "--no-watch",
        "--cluster",
        str(cohort["cluster"]),
        "--namespace",
        common.NAMESPACE,
        "--priority",
        "high",
        "--no-login",
    ]
    result = common.run(command, env=_environment(cohort, node), check=False)
    matches = ACTUAL_JOB_RE.findall(result.stdout)
    actual = matches[-1] if matches else requested
    if result.returncode:
        raise RecoveryError(
            f"node {node_index} submission failed ({result.returncode}):\n{result.stdout}"
        )
    return {
        "action": "recover_failed_rank",
        "status": "submitted",
        "submitted_at": common.utc_now(),
        "source_jobset": source_jobset,
        "cohort_id": cohort_id,
        "cluster": cohort["cluster"],
        "capacity_reservation_id": cohort["capacity_reservation_id"],
        "instance_type": cohort["instance_type"],
        "requested_job_name": requested,
        "job_name": actual,
        "node_index": node_index,
        "global_rank_start": int(node["global_rank_start"]),
        "gpus_per_pod": 1,
        "memory_request": contract.REQUEST_MEMORY,
        "memory_limit": contract.LIMIT_MEMORY,
        "priority": "high",
        "topology_sha256": common.TOPOLOGY_SHA256,
        "code_overlay_sha256": common.CODE_OVERLAY_SHA256,
        "prompt_contract_sha256": contract.FROZEN_ENV["EXPECTED_PROMPT_CONTRACT_SHA256"],
        "checkpoint_manifest_sha256": contract.FROZEN_ENV["CHECKPOINT_MANIFEST_SHA256"],
    }


def _done_nodes() -> set[int]:
    return {
        int(record["node_index"])
        for record in common.load_jsonl(LEDGER)
        if record.get("action") == "recover_failed_rank"
        and record.get("status") == "submitted"
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--submit-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.submit_workers <= 8:
        parser.error("--submit-workers must be in [1, 8]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topology = common.load_json(common.TOPOLOGY_PATH)
    common.validate_topology(topology)
    cohorts = common.topology_cohorts(topology)
    planned: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    expected_nodes: set[int] = set()
    for cohort_id, expected in SOURCES.items():
        cohort = cohorts[cohort_id]
        source_name = str(expected["jobset"])
        source = common.kubectl_jobset(str(cohort["cluster"]), source_name)
        if source is None:
            raise RecoveryError(f"source JobSet disappeared: {source_name}")
        contract._validate_failed_source(source)
        nodes = common.cohort_nodes(topology, cohort)
        if [int(node["node_index"]) for node in nodes] != list(expected["nodes"]):
            raise RecoveryError(f"{cohort_id} node-index topology drift")
        if [int(node["global_rank_start"]) for node in nodes] != list(expected["ranks"]):
            raise RecoveryError(f"{cohort_id} rank topology drift")
        expected_nodes.update(int(node["node_index"]) for node in nodes)
        planned.extend((cohort_id, source_name, cohort, node) for node in nodes)

    done = _done_nodes()
    if not done <= expected_nodes:
        raise RecoveryError(f"ledger contains unexpected nodes: {sorted(done - expected_nodes)}")
    pending = [item for item in planned if int(item[3]["node_index"]) not in done]
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "cohorts": sorted(SOURCES),
                "pending_jobsets": len(pending),
                "pending_gpus": len(pending),
                "memory_change": {
                    "request": contract.REQUEST_MEMORY,
                    "limit": contract.LIMIT_MEMORY,
                },
                "generation_contract_change": None,
                "output_s3": common.OUTPUT_S3,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute or not pending:
        return 0

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.submit_workers) as pool:
        futures = {pool.submit(_submit, *item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append(f"node {item[3]['node_index']}: {exc}")
                continue
            common.append_ledger(LEDGER, record)
            print(
                f"[submitted] node={record['node_index']} rank={record['global_rank_start']} "
                f"job={record['job_name']}",
                flush=True,
            )
    if errors:
        raise RecoveryError("submission failures:\n" + "\n".join(errors))
    print(
        json.dumps(
            {"status": "submitted", "jobsets": len(_done_nodes()), "ledger": str(LEDGER)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecoveryError, common.RebalanceError, contract.RecoveryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
