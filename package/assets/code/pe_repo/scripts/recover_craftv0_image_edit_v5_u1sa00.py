#!/usr/bin/env python3
"""Recover the failed u1sa00 cohort as topology-identical one-rank JobSets.

The original 64-rank JobSet failed after single-GPU Pods reached their 52Gi
container memory limit.  Recovery keeps the frozen model, prompt, generation
contract, rank topology, code overlay, and output prefix unchanged.  The only
runtime-resource change is a 48Gi request / 58Gi limit, and each logical node
is isolated in its own JobSet so one exhausted retry cannot fail sibling ranks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from scripts import rebalance_craftv0_image_edit_v5_pending_cohorts as common


COHORT_ID = "u1sa00"
FAILED_JOBSET = "grantzhang-v5cv0-0815023150-u1sa00-103429"
LEDGER = common.PLAN_ROOT / "recovery-u1sa00-20260815023150.jsonl"
REQUEST_MEMORY = "48Gi"
LIMIT_MEMORY = "58Gi"
EXPECTED_NODE_RANGE = range(27, 91)
EXPECTED_RANK_RANGE = range(216, 280)
ACTUAL_JOB_RE = re.compile(r"--job-name ([^ ]+)")

FROZEN_ENV = {
    "CODE_OVERLAY_S3": common.CODE_OVERLAY_S3,
    "CODE_OVERLAY_SHA256": common.CODE_OVERLAY_SHA256,
    "SOURCE_CATALOG_SHA256": "946a18d0f87afcfb452545e889adf9a38b9bf05b3a3e3a11260d5b298bc6d23e",
    "INFERENCE_OUTPUT_S3": common.OUTPUT_S3,
    "TOPOLOGY_URI": common.TOPOLOGY_URI,
    "TOPOLOGY_SHA256": common.TOPOLOGY_SHA256,
    "INFERENCE_TOTAL_GPU_COUNT": "2048",
    "INFERENCE_LOCAL_WORLD_SIZE": "1",
    "IMAGE_EDIT_TARGET_SCHEMA": "image_edit_v5_instruction",
    "INFERENCE_TARGET_SCHEMA": "image_edit_v5_instruction",
    "EXPECTED_PROMPT_VERSION": "image_edit_v5_full_json_instruction_v3",
    "EXPECTED_PROMPT_CONTRACT_SHA256": "7f7b4e791ac1632475bf04bf7a45b260d759484bc6fe0cb86bda8a06cedfe442",
    "CHECKPOINT_STEP": "10000",
    "CHECKPOINT_MANIFEST_SHA256": "1335bd055d1e2ca5def207f52c800d73ee5cb5eb9a6d5bb217d8605030551611",
    "HF_ARTIFACT_S3": "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/models/c4b-image-edit-v5-checkpoint-10000-hf-trained-mtp-v1/",
    "INFERENCE_BATCH_SIZE": "32",
    "INFERENCE_REQUEST_WINDOW_SIZE": "128",
    "INFERENCE_PREFETCH_IMAGES": "1",
    "INFERENCE_CROSS_UNIT_STREAMING": "1",
    "INFERENCE_DOWNLOAD_WORKERS": "4",
    "INFERENCE_MAX_PENDING_UNIT_WRITES": "8",
    "INFERENCE_UNIT_UPLOAD_WORKERS": "2",
    "INFERENCE_MAX_MODEL_LEN": "32768",
    "INFERENCE_MAX_NEW_TOKENS": "16384",
    "INFERENCE_MAX_PIXELS": "2359296",
    "INFERENCE_MAX_NUM_BATCHED_TOKENS": "32768",
    "INFERENCE_MM_PROCESSOR_CACHE_GB": "0",
    "INFERENCE_GPU_MEMORY_UTILIZATION": "0.90",
    "INFERENCE_CUDAGRAPH_MODE": "PIECEWISE",
    "INFERENCE_USE_MTP": "1",
    "INFERENCE_NUM_SPECULATIVE_TOKENS": "1",
    "INFERENCE_SEED": "42",
}


class RecoveryError(RuntimeError):
    """Fail-closed recovery validation error."""


def _env_values(jobset: Mapping[str, Any]) -> dict[str, str]:
    raw = jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    return {
        str(item["name"]): str(item["value"])
        for item in raw
        if isinstance(item, Mapping) and "value" in item
    }


def _resources(jobset: Mapping[str, Any]) -> Mapping[str, Any]:
    return jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["resources"]


def _validate_frozen_env(values: Mapping[str, str]) -> None:
    for name, expected in FROZEN_ENV.items():
        if values.get(name) != expected:
            raise RecoveryError(
                f"frozen inference contract drift: {name}={values.get(name)!r}, expected {expected!r}"
            )


def _validate_failed_source(jobset: Mapping[str, Any]) -> None:
    failed = any(
        condition.get("type") == "Failed" and condition.get("status") == "True"
        for condition in jobset.get("status", {}).get("conditions", [])
    )
    if not failed:
        raise RecoveryError(f"source JobSet is not failed: {FAILED_JOBSET}")
    template = jobset["spec"]["replicatedJobs"][0]["template"]["spec"]
    if (template.get("completions"), template.get("parallelism")) != (64, 64):
        raise RecoveryError("source JobSet is not the expected 64-rank cohort")
    _validate_frozen_env(_env_values(jobset))
    resources = _resources(jobset)
    if resources.get("limits", {}).get("memory") != "52Gi":
        raise RecoveryError("source JobSet memory limit drifted from the observed OOM boundary")


def _recovery_environment(cohort: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, str]:
    env = common.node_environment(cohort, node)
    env.update(
        {
            "INFERENCE_MEMORY": REQUEST_MEMORY,
            "INFERENCE_MEMORY_LIMIT": LIMIT_MEMORY,
        }
    )
    return env


def _request_name(node_index: int) -> str:
    return f"v5cv0-{common.RUN_TAG}-rpn{node_index:03d}"


def _submit_node(cohort: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, Any]:
    node_index = int(node["node_index"])
    requested_name = _request_name(node_index)
    command = [
        str(common.REPO_ROOT / ".venv/bin/python"),
        str(common.REPO_ROOT / "scripts/submit_from_job_yaml.py"),
        str(common.CONFIG_PATH),
        requested_name,
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
    result = common.run(command, env=_recovery_environment(cohort, node), check=False)
    matches = ACTUAL_JOB_RE.findall(result.stdout)
    actual_name = matches[-1] if matches else requested_name
    if result.returncode != 0:
        raise RecoveryError(
            f"node {node_index} submission failed ({result.returncode}):\n{result.stdout}"
        )
    return {
        "action": "recover_failed_rank",
        "status": "submitted",
        "submitted_at": common.utc_now(),
        "source_jobset": FAILED_JOBSET,
        "cohort_id": COHORT_ID,
        "cluster": cohort["cluster"],
        "capacity_reservation_id": cohort["capacity_reservation_id"],
        "instance_type": cohort["instance_type"],
        "requested_job_name": requested_name,
        "job_name": actual_name,
        "node_index": node_index,
        "global_rank_start": int(node["global_rank_start"]),
        "gpus_per_pod": 1,
        "memory_request": REQUEST_MEMORY,
        "memory_limit": LIMIT_MEMORY,
        "priority": "high",
        "topology_sha256": common.TOPOLOGY_SHA256,
        "code_overlay_sha256": common.CODE_OVERLAY_SHA256,
        "prompt_contract_sha256": FROZEN_ENV["EXPECTED_PROMPT_CONTRACT_SHA256"],
        "checkpoint_manifest_sha256": FROZEN_ENV["CHECKPOINT_MANIFEST_SHA256"],
    }


def _prior_nodes() -> set[int]:
    active: set[int] = set()
    for record in common.load_jsonl(LEDGER):
        action = record.get("action")
        status = record.get("status")
        if action == "recover_failed_rank" and status == "submitted":
            active.add(int(record["node_index"]))
        elif action == "retract_unschedulable_recovery" and status == "completed":
            active.discard(int(record["node_index"]))
    return active


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--submit-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.submit_workers <= 8:
        parser.error("--submit-workers must be in [1, 8]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topology = common.load_json(common.TOPOLOGY_PATH)
    common.validate_topology(topology)
    cohort = common.topology_cohorts(topology)[COHORT_ID]
    nodes = common.cohort_nodes(topology, cohort)
    if [int(node["node_index"]) for node in nodes] != list(EXPECTED_NODE_RANGE):
        raise RecoveryError("u1sa00 node-index topology drift")
    if [int(node["global_rank_start"]) for node in nodes] != list(EXPECTED_RANK_RANGE):
        raise RecoveryError("u1sa00 global-rank topology drift")
    if any(int(node["gpus_per_pod"]) != 1 for node in nodes):
        raise RecoveryError("u1sa00 is no longer a single-GPU cohort")

    source = common.kubectl_jobset(str(cohort["cluster"]), FAILED_JOBSET)
    if source is None:
        raise RecoveryError(f"source JobSet disappeared: {FAILED_JOBSET}")
    _validate_failed_source(source)

    done = _prior_nodes()
    unknown = done - set(EXPECTED_NODE_RANGE)
    if unknown:
        raise RecoveryError(f"recovery ledger contains unexpected nodes: {sorted(unknown)}")
    pending = [node for node in nodes if int(node["node_index"]) not in done]
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "source_jobset": FAILED_JOBSET,
                "cohort_id": COHORT_ID,
                "pending_jobsets": len(pending),
                "pending_gpus": len(pending),
                "node_range": [27, 90],
                "rank_range": [216, 279],
                "memory_change": {"request": REQUEST_MEMORY, "limit": LIMIT_MEMORY},
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
        futures = {pool.submit(_submit_node, cohort, node): node for node in pending}
        for future in as_completed(futures):
            node = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append(f"node {node['node_index']}: {exc}")
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
            {
                "status": "submitted",
                "jobsets": len(_prior_nodes()),
                "gpus": len(_prior_nodes()),
                "ledger": str(LEDGER),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecoveryError, common.RebalanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
