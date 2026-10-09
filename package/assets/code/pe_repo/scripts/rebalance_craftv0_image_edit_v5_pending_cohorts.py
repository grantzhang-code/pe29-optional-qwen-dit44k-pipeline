#!/usr/bin/env python3
"""Replace suspended craftv0 cohorts with topology-identical one-pod JobSets.

Kueue admits a JobSet atomically.  The original full-run cohorts intentionally
bounded each workload at 32--64 GPUs, but that can strand smaller holes in a
capacity reservation.  This operational tool only touches original cohorts
that are still suspended and have never created a Pod.  It deletes them
through Arnold, then resubmits every existing logical topology node as its own
high-priority JobSet.  Node indices, global ranks, model/prompt/runtime hashes,
and the S3 output namespace remain unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = REPO_ROOT / "artifacts/craftv0_image_edit_v5_full_17064da_plan"
TOPOLOGY_PATH = PLAN_ROOT / "topology.json"
ORIGINAL_LEDGER = PLAN_ROOT / "submission-20260815023150.jsonl"
CONFIG_PATH = REPO_ROOT / "job_config/c4b_image_edit_v5_craftv0_full_2048rtx_inference.yaml"

NAMESPACE = "b-core-cn-prod"
APPLICATION_ID = "core-cn"
RUN_TAG = "0815023150"
RUN_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v5/craftv0/full/"
    "checkpoint-10000/2048x-rtx-pro-6000-pr36/20260815023150"
)
OUTPUT_S3 = f"{RUN_ROOT}/results/"
TOPOLOGY_URI = f"{OUTPUT_S3}_inputs/topology.json"
TOPOLOGY_SHA256 = "d4249020bdcce078d788c37c98aad77a3f82481de7ca02d87d046e991bb97321"
CODE_OVERLAY_S3 = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/"
    "code_overlays/craftv0-v5-full-2048rtx-"
    "caffa9fe6dd8ce96352a43b1e7b401ee8a2599be5ad9e974fb33188104f0579d.tar.gz"
)
CODE_OVERLAY_SHA256 = "caffa9fe6dd8ce96352a43b1e7b401ee8a2599be5ad9e974fb33188104f0579d"
AWS_PROFILE = "prod_core-cn-readwrite"

EXPECTED_TOPOLOGY = {
    "format_version": "image_edit_v5_craftv0_2048rtx_topology_v1",
    "total_nodes": 872,
    "total_ranks": 2048,
    "priority": "high",
}
ACTUAL_JOB_RE = re.compile(r"--job-name ([^ ]+)")


class RebalanceError(RuntimeError):
    """Fail-closed operational validation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        raise RebalanceError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RebalanceError(f"{path}:{line_number}: ledger row is not an object")
            values.append(value)
    return values


def append_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_topology(value: Mapping[str, Any]) -> None:
    for key, expected in EXPECTED_TOPOLOGY.items():
        if value.get(key) != expected:
            raise RebalanceError(
                f"topology contract drift: {key}={value.get(key)!r}, expected {expected!r}"
            )
    nodes = value.get("nodes")
    cohorts = value.get("cohorts")
    if not isinstance(nodes, list) or len(nodes) != 872:
        raise RebalanceError("topology does not contain exactly 872 nodes")
    if not isinstance(cohorts, list) or len(cohorts) != 38:
        raise RebalanceError("topology does not contain exactly 38 cohorts")
    if sum(int(node["gpus_per_pod"]) for node in nodes) != 2048:
        raise RebalanceError("topology node widths do not sum to 2048")


def original_records() -> dict[str, dict[str, Any]]:
    records = load_jsonl(ORIGINAL_LEDGER)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        cohort_id = str(record.get("cohort_id", ""))
        if not cohort_id or record.get("status") != "submitted":
            continue
        if cohort_id in result:
            raise RebalanceError(f"duplicate original ledger cohort: {cohort_id}")
        result[cohort_id] = record
    if len(result) != 38:
        raise RebalanceError(f"expected 38 original ledger cohorts, got {len(result)}")
    return result


def topology_cohorts(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {str(item["cohort_id"]): dict(item) for item in value["cohorts"]}
    if len(result) != 38:
        raise RebalanceError("duplicate topology cohort IDs")
    return result


def cohort_nodes(
    topology: Mapping[str, Any], cohort: Mapping[str, Any]
) -> list[dict[str, Any]]:
    start = int(cohort["node_index_start"])
    count = int(cohort["pod_count"])
    width = int(cohort["gpus_per_pod"])
    rank_start = int(cohort["global_rank_start"])
    nodes = [dict(node) for node in topology["nodes"][start : start + count]]
    if len(nodes) != count:
        raise RebalanceError(f"cohort {cohort['cohort_id']} node slice is truncated")
    for offset, node in enumerate(nodes):
        expected_node = start + offset
        expected_rank = rank_start + offset * width
        if int(node["node_index"]) != expected_node:
            raise RebalanceError(f"cohort {cohort['cohort_id']} node index drift")
        if int(node["global_rank_start"]) != expected_rank:
            raise RebalanceError(f"cohort {cohort['cohort_id']} global rank drift")
        if int(node["gpus_per_pod"]) != width:
            raise RebalanceError(f"cohort {cohort['cohort_id']} node width drift")
        if node["pool_id"] != cohort["pool_id"]:
            raise RebalanceError(f"cohort {cohort['cohort_id']} pool drift")
    return nodes


def kubectl_jobset(cluster: str, job_name: str) -> dict[str, Any] | None:
    result = run(
        [
            "kubectl",
            "--context",
            cluster,
            "-n",
            NAMESPACE,
            "get",
            "jobset",
            job_name,
            "-o",
            "json",
        ],
        check=False,
    )
    if result.returncode != 0:
        if "NotFound" in result.stdout or "not found" in result.stdout:
            return None
        raise RebalanceError(f"cannot read JobSet {job_name}:\n{result.stdout}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RebalanceError(f"JobSet {job_name} response is not an object")
    return value


def verify_never_started(cluster: str, job_name: str, value: Mapping[str, Any]) -> None:
    conditions = value.get("status", {}).get("conditions", [])
    suspended = any(
        condition.get("type") == "Suspended" and condition.get("status") == "True"
        for condition in conditions
    )
    if not suspended:
        raise RebalanceError(f"refusing to split non-suspended JobSet: {job_name}")
    statuses = value.get("status", {}).get("replicatedJobsStatus", [])
    for status in statuses:
        for field in ("active", "ready", "succeeded", "failed"):
            if int(status.get(field, 0) or 0) != 0:
                raise RebalanceError(
                    f"refusing to split started JobSet {job_name}: {field}={status.get(field)}"
                )
    pods = run(
        [
            "kubectl",
            "--context",
            cluster,
            "-n",
            NAMESPACE,
            "get",
            "pods",
            "-l",
            f"jobset.sigs.k8s.io/jobset-name={job_name}",
            "-o",
            "json",
        ]
    )
    items = json.loads(pods.stdout).get("items", [])
    if items:
        raise RebalanceError(f"refusing to split JobSet {job_name}: it has {len(items)} Pods")


def delete_original(cluster: str, job_name: str) -> None:
    # Arnold --no-login consumes the current context from KUBECONFIG even when
    # --cluster is present.  Use an isolated context-selected kubeconfig just
    # like submit_from_job_yaml.py does, so a use1 shell cannot delete/query a
    # same-named workload in usw2 (or vice versa).
    raw_config = run(["kubectl", "config", "view", "--raw"]).stdout
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="craftv0-rebalance-kubeconfig-", delete=False
    )
    kubeconfig_path = Path(handle.name)
    try:
        handle.write(raw_config)
        handle.close()
        run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig_path),
                "config",
                "use-context",
                cluster,
            ]
        )
        delete_env = dict(os.environ)
        delete_env["KUBECONFIG"] = str(kubeconfig_path)
        result = run(
            [
                "arnold",
                "delete",
                "--application-id",
                APPLICATION_ID,
                "--cluster",
                cluster,
                "--namespace",
                NAMESPACE,
                "--job-name",
                job_name,
                "--no-login",
            ],
            env=delete_env,
        )
        print(result.stdout, end="", flush=True)
    finally:
        try:
            kubeconfig_path.unlink()
        except FileNotFoundError:
            pass
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if kubectl_jobset(cluster, job_name) is None:
            return
        time.sleep(2)
    raise RebalanceError(f"timed out waiting for deleted JobSet: {job_name}")


def node_job_request_name(node_index: int) -> str:
    return f"v5cv0-{RUN_TAG}-spn{node_index:03d}"


def node_environment(cohort: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, str]:
    width = int(node["gpus_per_pod"])
    env = dict(os.environ)
    env.update(
        {
            "KUBE_CONTEXT": str(cohort["cluster"]),
            "RTX_PRO_6000_CAPACITY_RESERVATION_ID": str(
                cohort["capacity_reservation_id"]
            ),
            "RTX_INSTANCE_TYPE": str(cohort["instance_type"]),
            "JOB_COMPLETION_COUNT": "1",
            "INFERENCE_LOCAL_WORLD_SIZE": str(width),
            "NODE_INDEX_OFFSET": str(node["node_index"]),
            "GLOBAL_RANK_OFFSET": str(node["global_rank_start"]),
            "CODE_OVERLAY_S3": CODE_OVERLAY_S3,
            "CODE_OVERLAY_SHA256": CODE_OVERLAY_SHA256,
            "INFERENCE_OUTPUT_S3": OUTPUT_S3,
            "TOPOLOGY_URI": TOPOLOGY_URI,
            "TOPOLOGY_SHA256": TOPOLOGY_SHA256,
        }
    )
    if width == 8:
        env.update(
            {
                "INFERENCE_GPUS": "0,1,2,3,4,5,6,7",
                "INFERENCE_CPU": "26",
                "INFERENCE_MEMORY": "340Gi",
                "INFERENCE_MEMORY_LIMIT": "420Gi",
                "INFERENCE_SHM_SIZE": "32Gi",
            }
        )
    elif width == 1:
        env.update(
            {
                "INFERENCE_GPUS": "0",
                "INFERENCE_CPU": "4",
                "INFERENCE_MEMORY": "40Gi",
                "INFERENCE_MEMORY_LIMIT": "52Gi",
                "INFERENCE_SHM_SIZE": "16Gi",
            }
        )
    else:
        raise RebalanceError(f"unsupported topology node width: {width}")
    return env


def submit_node(cohort: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, Any]:
    node_index = int(node["node_index"])
    requested_name = node_job_request_name(node_index)
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/submit_from_job_yaml.py"),
        str(CONFIG_PATH),
        requested_name,
        "--",
        "--no-watch",
        "--cluster",
        str(cohort["cluster"]),
        "--namespace",
        NAMESPACE,
        "--priority",
        "high",
        "--no-login",
    ]
    result = run(command, env=node_environment(cohort, node), check=False)
    matches = ACTUAL_JOB_RE.findall(result.stdout)
    actual_name = matches[-1] if matches else requested_name
    if result.returncode != 0:
        raise RebalanceError(
            f"node {node_index} submission failed ({result.returncode}):\n{result.stdout}"
        )
    return {
        "action": "submit_node",
        "status": "submitted",
        "submitted_at": utc_now(),
        "cohort_id": cohort["cohort_id"],
        "original_cluster": cohort["cluster"],
        "cluster": cohort["cluster"],
        "capacity_reservation_id": cohort["capacity_reservation_id"],
        "instance_type": cohort["instance_type"],
        "requested_job_name": requested_name,
        "job_name": actual_name,
        "node_index": node_index,
        "global_rank_start": int(node["global_rank_start"]),
        "gpus_per_pod": int(node["gpus_per_pod"]),
        "priority": "high",
        "topology_sha256": TOPOLOGY_SHA256,
        "code_overlay_sha256": CODE_OVERLAY_SHA256,
    }


def upload_ledger(path: Path) -> None:
    result = run(
        [
            "aws",
            "--profile",
            AWS_PROFILE,
            "s3",
            "cp",
            str(path),
            f"{RUN_ROOT}/submission/{path.name}",
            "--region",
            "us-east-1",
            "--only-show-errors",
        ]
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)


def previously_deleted(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(record["cohort_id"])
        for record in records
        if record.get("action") == "delete_cohort" and record.get("status") == "completed"
    }


def submitted_nodes(records: Iterable[Mapping[str, Any]]) -> set[int]:
    return {
        int(record["node_index"])
        for record in records
        if record.get("action") == "submit_node" and record.get("status") == "submitted"
    }


def candidate_order(cohort: Mapping[str, Any]) -> tuple[int, int]:
    cluster = str(cohort["cluster"])
    width = int(cohort["gpus_per_pod"])
    # First fill the observed use1 single-GPU and usw2 eight-GPU quota holes.
    if cluster == "general-aws-use1-prod-0" and width == 1:
        group = 0
    elif cluster == "general-aws-usw2-prod-0" and width == 8:
        group = 1
    else:
        group = 2
    return group, int(cohort["node_index_start"])


def select_candidates(
    topology: Mapping[str, Any],
    originals: Mapping[str, Mapping[str, Any]],
    requested: set[str],
    already_deleted: set[str],
) -> list[dict[str, Any]]:
    cohorts = topology_cohorts(topology)
    selected: list[dict[str, Any]] = []
    missing = requested - set(cohorts)
    if missing:
        raise RebalanceError(f"unknown requested cohort IDs: {sorted(missing)}")
    for cohort_id, cohort in cohorts.items():
        if requested and cohort_id not in requested:
            continue
        original = originals[cohort_id]
        for field in (
            "cluster",
            "capacity_reservation_id",
            "instance_type",
            "pod_count",
            "gpus_per_pod",
            "node_index_start",
            "global_rank_start",
        ):
            if original[field] != cohort[field]:
                raise RebalanceError(f"cohort {cohort_id} ledger/topology drift in {field}")
        if cohort_id in already_deleted:
            selected.append(cohort)
            continue
        jobset = kubectl_jobset(str(cohort["cluster"]), str(original["job_name"]))
        if jobset is None:
            continue
        conditions = jobset.get("status", {}).get("conditions", [])
        if any(
            condition.get("type") == "Suspended" and condition.get("status") == "True"
            for condition in conditions
        ):
            selected.append(cohort)
    return sorted(selected, key=candidate_order)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform Arnold delete and high-priority one-node resubmissions",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Restrict to a cohort ID; repeat as needed. Default: every suspended cohort.",
    )
    parser.add_argument("--submit-workers", type=int, default=4)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=PLAN_ROOT / "rebalance-nodewise-20260815023150.jsonl",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.submit_workers <= 8:
        parser.error("--submit-workers must be in [1, 8]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topology = load_json(TOPOLOGY_PATH)
    if not isinstance(topology, dict):
        raise RebalanceError("topology root is not an object")
    validate_topology(topology)
    originals = original_records()
    prior = load_jsonl(args.ledger)
    deleted = previously_deleted(prior)
    done_nodes = submitted_nodes(prior)
    candidates = select_candidates(topology, originals, set(args.cohort), deleted)
    planned_nodes = [
        node
        for cohort in candidates
        for node in cohort_nodes(topology, cohort)
        if int(node["node_index"]) not in done_nodes
    ]
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "cohorts": [cohort["cohort_id"] for cohort in candidates],
                "cohort_count": len(candidates),
                "remaining_node_jobs": len(planned_nodes),
                "remaining_gpus": sum(int(node["gpus_per_pod"]) for node in planned_nodes),
                "topology_sha256": TOPOLOGY_SHA256,
                "output_s3": OUTPUT_S3,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        return 0

    for cohort in candidates:
        cohort_id = str(cohort["cohort_id"])
        original = originals[cohort_id]
        nodes = [
            node
            for node in cohort_nodes(topology, cohort)
            if int(node["node_index"]) not in done_nodes
        ]
        if cohort_id not in deleted:
            job_name = str(original["job_name"])
            jobset = kubectl_jobset(str(cohort["cluster"]), job_name)
            if jobset is None:
                raise RebalanceError(
                    f"original JobSet disappeared without a deletion ledger record: {job_name}"
                )
            verify_never_started(str(cohort["cluster"]), job_name, jobset)
            print(f"[delete] cohort={cohort_id} job={job_name}", flush=True)
            delete_original(str(cohort["cluster"]), job_name)
            append_ledger(
                args.ledger,
                {
                    "action": "delete_cohort",
                    "status": "completed",
                    "completed_at": utc_now(),
                    "cohort_id": cohort_id,
                    "job_name": job_name,
                    "cluster": cohort["cluster"],
                    "pod_count": cohort["pod_count"],
                    "gpus_per_pod": cohort["gpus_per_pod"],
                    "reason": "replace suspended atomic cohort with topology-identical one-pod JobSets",
                },
            )
            deleted.add(cohort_id)
            upload_ledger(args.ledger)
        if not nodes:
            continue
        print(
            f"[submit] cohort={cohort_id} node_jobs={len(nodes)} "
            f"gpus={sum(int(node['gpus_per_pod']) for node in nodes)}",
            flush=True,
        )
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=args.submit_workers) as pool:
            futures = {pool.submit(submit_node, cohort, node): node for node in nodes}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # continue recording sibling successes
                    errors.append(f"node {node['node_index']}: {exc}")
                    continue
                append_ledger(args.ledger, record)
                done_nodes.add(int(record["node_index"]))
                print(
                    f"[submitted] node={record['node_index']} ranks="
                    f"{record['global_rank_start']}+{record['gpus_per_pod']} "
                    f"job={record['job_name']}",
                    flush=True,
                )
        upload_ledger(args.ledger)
        if errors:
            raise RebalanceError("node submissions failed:\n" + "\n".join(errors))
    print(
        json.dumps(
            {
                "status": "complete",
                "deleted_cohorts": len(deleted),
                "submitted_node_jobs": len(done_nodes),
                "ledger": str(args.ledger),
                "completed_at": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RebalanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
