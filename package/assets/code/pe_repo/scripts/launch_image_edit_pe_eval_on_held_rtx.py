#!/usr/bin/env python3
"""Launch the frozen PE benchmark on the ten already-held RTX PRO 6000 Pods."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import boto3


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_pe_mds import (  # noqa: E402
    LABEL_PROJECTION_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    prompt_contract_sha256,
)
from scripts.checkpoint.common import Boto3ObjectStore  # noqa: E402
from scripts.checkpoint.mcore_s3 import inspect_committed_checkpoint  # noqa: E402


BENCHMARK_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/evaluation/"
    "iev6-full72-pe64-dedup-merged-20260827-v1-train5m-complement-"
    "tasktype40-seed20260828-v1/"
)
BENCHMARK_MANIFEST_SHA256 = "1c1c46e660bded12b42decad2ee3d6febd64acfbd5b9dd81d6a5ec17cd28ec57"
MCORE_S3_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/runs/"
    "c4b-iev6-pe5m-targetsize-v4-schemafix-s0-20k-b300x32/20260827174415/mcore/"
)
INITIAL_ARTIFACT_S3 = (
    "s3://core-cn-oss.canva.com/usr/joeyzhang/design-layout-megatron-swift/models/"
    "c4b-long-mtp3-checkpoint-10000-hf-trained-mtp-v1/"
)
INITIAL_ARTIFACT_MANIFEST_SHA256 = "818a699117a17e7bab71d221757a6c9f7a3acfb6d5fb9816c056911a3afb13fe"
HOLD_IMAGE_URI = (
    "699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@"
    "sha256:3da1b7636d5345e9b4bd3c1a7016c810bf2277e910110055b4c045d7c34d9dc2"
)
NAMESPACE = "b-core-cn-prod"
HOLD_JOBSETS = (
    ("general-aws-use1-prod-0", "grantzhang-rtx1-hold-u1-5-0827-224954"),
    ("general-aws-usw2-prod-0", "grantzhang-rtx1-hold-w2-5-0827-224954"),
)
_CHECKPOINT_MARKER_RE = re.compile(r"checkpoint-([1-9][0-9]*)/_SUCCESS$")


def parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"invalid S3 URI: {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return bucket, key


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key.rstrip('/'), suffix) if value)}"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: Sequence[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout if capture else ""


def s3_client(profile: str) -> Any:
    return boto3.Session(profile_name=profile).client("s3", region_name="us-east-1")


def latest_complete_checkpoint(profile: str) -> dict[str, Any]:
    client = s3_client(profile)
    bucket, prefix = parse_s3(MCORE_S3_ROOT)
    steps: list[int] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            match = _CHECKPOINT_MARKER_RE.search(str(item.get("Key") or ""))
            if match:
                steps.append(int(match.group(1)))
    if not steps:
        raise RuntimeError(f"no committed checkpoint under {MCORE_S3_ROOT}")
    step = max(steps)
    checkpoint_s3 = join_s3(MCORE_S3_ROOT, f"checkpoint-{step}") + "/"
    inspected = inspect_committed_checkpoint(
        Boto3ObjectStore(profile=profile, region="us-east-1"),
        checkpoint_s3,
        verify_objects=True,
    )
    marker_response = client.get_object(
        Bucket=bucket,
        Key=f"{prefix.rstrip('/')}/checkpoint-{step}/_SUCCESS",
    )
    marker = json.loads(marker_response["Body"].read())
    if (
        marker.get("status") != "complete"
        or marker.get("step") != step
        or marker.get("manifest_uri") != inspected.get("manifest_uri")
    ):
        raise RuntimeError("latest checkpoint marker and manifest disagree")
    return {
        "step": step,
        "checkpoint_s3": checkpoint_s3,
        "manifest_sha256": marker["manifest_sha256"],
        "marker": marker,
        "inspection": inspected,
    }


def held_pods() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for context, jobset in HOLD_JOBSETS:
        raw = command(
            [
                "kubectl",
                "--context",
                context,
                "-n",
                NAMESPACE,
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={jobset}",
                "-o",
                "json",
            ],
            capture=True,
        )
        document = json.loads(raw)
        pods = sorted(document.get("items", []), key=lambda item: item["metadata"]["name"])
        if len(pods) != 5:
            raise RuntimeError(f"held JobSet {context}/{jobset} has {len(pods)} Pods, expected 5")
        for pod in pods:
            if pod.get("status", {}).get("phase") != "Running":
                raise RuntimeError(f"held Pod is not Running: {pod['metadata']['name']}")
            containers = pod.get("spec", {}).get("containers", [])
            gpu = containers[0].get("resources", {}).get("requests", {}).get("nvidia.com/gpu")
            if str(gpu) != "1":
                raise RuntimeError(f"held Pod does not request exactly one GPU: {pod['metadata']['name']}")
            result.append(
                {
                    "context": context,
                    "jobset": jobset,
                    "pod": pod["metadata"]["name"],
                    "node": str(pod.get("spec", {}).get("nodeName") or ""),
                }
            )
    if len(result) != 10:
        raise AssertionError("held RTX topology is not exactly ten GPUs")
    return result


def download_benchmark(client: Any, destination: Path) -> dict[str, Any]:
    bucket, prefix = parse_s3(BENCHMARK_ROOT)
    manifest_key = f"{prefix.rstrip('/')}/evaluation_manifest.jsonl"
    response = client.get_object(Bucket=bucket, Key=manifest_key)
    payload = response["Body"].read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != BENCHMARK_MANIFEST_SHA256:
        raise RuntimeError(f"benchmark manifest changed: {digest} != {BENCHMARK_MANIFEST_SHA256}")
    destination.write_bytes(payload)
    success = client.get_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/_SUCCESS")
    success_payload = success["Body"].read()
    success_value = json.loads(success_payload)
    if (
        success_value.get("complete") is not True
        or success_value.get("rows") != 1160
        or success_value.get("task_types") != 29
        or success_value.get("training_intersection_rows") != 0
    ):
        raise RuntimeError("benchmark publication is not the frozen complete complement")
    return {
        "root": BENCHMARK_ROOT,
        "manifest_s3": f"s3://{bucket}/{manifest_key}",
        "manifest_sha256": digest,
        "manifest_etag": response.get("ETag", "").strip('"'),
        "manifest_version_id": response.get("VersionId"),
        "success": success_value,
        "success_sha256": hashlib.sha256(success_payload).hexdigest(),
    }


def create_overlay(destination: Path) -> str:
    members = [REPO_ROOT / "dlms", REPO_ROOT / "scripts", REPO_ROOT / "src"]
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for member in members:
            if not member.exists():
                continue
            for path in sorted(member.rglob("*")):
                relative = path.relative_to(REPO_ROOT)
                if "__pycache__" in relative.parts or path.suffix == ".pyc":
                    continue
                info = archive.gettarinfo(str(path), arcname=str(relative))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
                else:
                    archive.addfile(info)
    return sha256_file(destination)


def upload_file(client: Any, source: Path, destination: str) -> None:
    bucket, key = parse_s3(destination)
    client.upload_file(str(source), bucket, key)


def put_json(client: Any, destination: str, value: Mapping[str, Any]) -> str:
    payload = canonical_bytes(value)
    bucket, key = parse_s3(destination)
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
    return hashlib.sha256(payload).hexdigest()


def require_empty_prefix(client: Any, root: str) -> None:
    bucket, key = parse_s3(root)
    result = client.list_objects_v2(Bucket=bucket, Prefix=key.rstrip("/") + "/", MaxKeys=1)
    if int(result.get("KeyCount", 0)) != 0:
        raise RuntimeError(f"refusing non-empty run root: {root}")


def bootstrap_text(
    *,
    node_index: int,
    run_id: str,
    run_root: str,
    overlay_s3: str,
    overlay_sha256: str,
    partition_index_sha256: str,
    checkpoint: Mapping[str, Any],
    artifact_s3: str,
    artifact_attempt_id: str,
) -> str:
    values = {
        "PE_NODE_INDEX": str(node_index),
        "PE_NODE_COUNT": "10",
        "PE_RUN_ID": run_id,
        "PE_RUN_ROOT_S3": run_root,
        "PE_INPUT_PARTS_S3": join_s3(run_root, "input", "parts") + "/",
        "PE_PARTITION_INDEX_SHA256": partition_index_sha256,
        "PE_CODE_OVERLAY_S3": overlay_s3,
        "PE_CODE_OVERLAY_SHA256": overlay_sha256,
        "PE_MCORE_S3_ROOT": MCORE_S3_ROOT,
        "PE_CHECKPOINT_S3": str(checkpoint["checkpoint_s3"]),
        "PE_CHECKPOINT_STEP": str(checkpoint["step"]),
        "PE_CHECKPOINT_MANIFEST_SHA256": str(checkpoint["manifest_sha256"]),
        "PE_INITIAL_ARTIFACT_S3": INITIAL_ARTIFACT_S3,
        "PE_INITIAL_ARTIFACT_MANIFEST_SHA256": INITIAL_ARTIFACT_MANIFEST_SHA256,
        "PE_HF_ARTIFACT_S3": artifact_s3,
        "PE_HF_ARTIFACT_ATTEMPT_ID": artifact_attempt_id,
        "PE_HOLD_IMAGE_URI": HOLD_IMAGE_URI,
    }
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())
    return f"""#!/usr/bin/env bash
set -euo pipefail
{exports}
bootstrap_root=/workspace/pe-eval/{shlex.quote(run_id)}/bootstrap-{node_index:03d}
mkdir -p "$bootstrap_root/repo"
aws s3 cp "$PE_CODE_OVERLAY_S3" "$bootstrap_root/code-overlay.tar.gz" --only-show-errors
test "$(sha256sum "$bootstrap_root/code-overlay.tar.gz" | awk '{{print $1}}')" = "$PE_CODE_OVERLAY_SHA256"
tar -xzf "$bootstrap_root/code-overlay.tar.gz" -C "$bootstrap_root/repo"
exec bash "$bootstrap_root/repo/scripts/run_image_edit_pe_eval_held_rtx_node.sh"
"""


def launch_remote(pod: Mapping[str, str], *, bootstrap_s3: str, run_id: str, node_index: int) -> str:
    remote_root = f"/workspace/pe-eval/{run_id}/launcher-{node_index:03d}"
    remote_script = f"{remote_root}/bootstrap.sh"
    remote_log = f"{remote_root}/bootstrap.log"
    remote_command = (
        f"mkdir -p {shlex.quote(remote_root)} && "
        f"aws s3 cp {shlex.quote(bootstrap_s3)} {shlex.quote(remote_script)} --only-show-errors && "
        f"chmod 700 {shlex.quote(remote_script)} && "
        f"setsid -f bash {shlex.quote(remote_script)} "
        f"> {shlex.quote(remote_log)} 2>&1 < /dev/null && "
        f"echo started"
    )
    return command(
        [
            "kubectl",
            "--context",
            pod["context"],
            "-n",
            NAMESPACE,
            "exec",
            pod["pod"],
            "--",
            "bash",
            "-lc",
            remote_command,
        ],
        capture=True,
    ).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if prompt_contract_sha256() != PROMPT_CONTRACT_SHA256:
        raise RuntimeError("local PE prompt contract hash drifted")
    pods = held_pods()
    checkpoint = latest_complete_checkpoint(args.aws_profile)
    timestamp = args.submit_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"pe-c{checkpoint['step']}-{timestamp}"
    run_root = (
        "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/evaluations/"
        f"pe-complement-tasktype40-v1/checkpoint-{checkpoint['step']}/"
        f"10x-rtx-pro-6000/{timestamp}/"
    )
    artifact_s3 = (
        "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/models/"
        f"c4b-iev6-pe5m-targetsize-v4-schemafix-checkpoint-{checkpoint['step']}-"
        "hf-trained-mtp-v1/"
    )
    artifact_attempt_id = (
        f"pe-c{checkpoint['step']}-{checkpoint['manifest_sha256'][:12]}-{timestamp}"
    )
    client = s3_client(args.aws_profile)
    require_empty_prefix(client, run_root)

    temporary_root = Path(tempfile.mkdtemp(prefix="pe-eval-held-rtx-"))
    benchmark_path = temporary_root / "evaluation_manifest.jsonl"
    benchmark = download_benchmark(client, benchmark_path)
    prepared_root = temporary_root / "prepared"
    command(
        [
            sys.executable,
            "scripts/prepare_image_edit_pe_eval_inference.py",
            "--benchmark-manifest",
            str(benchmark_path),
            "--benchmark-manifest-sha256",
            BENCHMARK_MANIFEST_SHA256,
            "--output-dir",
            str(prepared_root),
            "--expected-rows",
            "1160",
            "--expected-task-types",
            "29",
            "--expected-per-task",
            "40",
            "--part-count",
            "10",
        ],
        capture=True,
    )
    prepare_summary = json.loads((prepared_root / "prepare_summary.json").read_text(encoding="utf-8"))
    partition_sha256 = prepare_summary["partition_index"]["sha256"]
    overlay_path = temporary_root / "code-overlay.tar.gz"
    overlay_sha256 = create_overlay(overlay_path)
    overlay_s3 = (
        "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/"
        f"code_overlays/pe-eval-c{checkpoint['step']}-10rtx-{overlay_sha256}.tar.gz"
    )
    contract = {
        "complete": False,
        "status": "prepared" if args.dry_run else "launching",
        "format_version": "image-edit-pe-held-10rtx-run-contract-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_root": run_root,
        "benchmark": benchmark,
        "checkpoint": checkpoint,
        "initial_artifact": {
            "s3": INITIAL_ARTIFACT_S3,
            "manifest_sha256": INITIAL_ARTIFACT_MANIFEST_SHA256,
        },
        "hf_artifact": {
            "s3": artifact_s3,
            "attempt_id": artifact_attempt_id,
            "logical_mtp_layers": 3,
            "shared_weights": True,
            "physical_mtp_layers": 1,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
        },
        "sampling": {
            "strategy": "deterministic_greedy_v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 42,
            "max_model_len": 32768,
            "max_new_tokens": 27000,
            "thinking": False,
            "mtp_method": "mtp",
            "num_speculative_tokens": 1,
        },
        "topology": {
            "gpu_count": 10,
            "gpu_type": "RTX PRO 6000",
            "pods": pods,
            "held_jobsets_preserved": [jobset for _context, jobset in HOLD_JOBSETS],
        },
        "input": prepare_summary,
        "overlay": {"s3": overlay_s3, "sha256": overlay_sha256},
    }
    (temporary_root / "run_contract.json").write_bytes(canonical_bytes(contract))
    if args.dry_run:
        return {**contract, "local_temporary_root": str(temporary_root)}

    upload_file(client, overlay_path, overlay_s3)
    upload_file(client, prepared_root / "inference_manifest.jsonl", join_s3(run_root, "input", "inference_manifest.jsonl"))
    upload_file(client, prepared_root / "ground_truth.jsonl", join_s3(run_root, "ground_truth", "ground_truth.jsonl"))
    upload_file(client, prepared_root / "prepare_summary.json", join_s3(run_root, "input", "prepare_summary.json"))
    upload_file(client, benchmark_path, join_s3(run_root, "benchmark", "evaluation_manifest.jsonl"))
    for path in sorted((prepared_root / "parts").iterdir()):
        upload_file(client, path, join_s3(run_root, "input", "parts", path.name))
    put_json(client, join_s3(run_root, "run_contract.json"), contract)

    launches: list[dict[str, Any]] = []
    for node_index, pod in enumerate(pods):
        bootstrap = bootstrap_text(
            node_index=node_index,
            run_id=run_id,
            run_root=run_root,
            overlay_s3=overlay_s3,
            overlay_sha256=overlay_sha256,
            partition_index_sha256=partition_sha256,
            checkpoint=checkpoint,
            artifact_s3=artifact_s3,
            artifact_attempt_id=artifact_attempt_id,
        )
        bootstrap_path = temporary_root / f"bootstrap-{node_index:03d}.sh"
        bootstrap_path.write_text(bootstrap, encoding="utf-8")
        bootstrap_s3 = join_s3(run_root, "bootstrap", bootstrap_path.name)
        upload_file(client, bootstrap_path, bootstrap_s3)
        pid = launch_remote(
            pod,
            bootstrap_s3=bootstrap_s3,
            run_id=run_id,
            node_index=node_index,
        )
        launches.append({"node_index": node_index, **pod, "pid": pid, "bootstrap_s3": bootstrap_s3})
    launch_status = {
        "status": "launched",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_root": run_root,
        "launches": launches,
    }
    put_json(client, join_s3(run_root, "launch_status.json"), launch_status)
    return {**contract, **launch_status, "local_temporary_root": str(temporary_root)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--submit-timestamp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.submit_timestamp and re.fullmatch(r"[0-9]{14}", args.submit_timestamp) is None:
        parser.error("--submit-timestamp must be YYYYmmddHHMMSS")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
