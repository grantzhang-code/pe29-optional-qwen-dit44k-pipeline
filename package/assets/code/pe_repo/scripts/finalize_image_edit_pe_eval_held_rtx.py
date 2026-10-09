#!/usr/bin/env python3
"""Wait for ten held-RTX shards, validate prompt parity, merge, and commit results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_pe_mds import (  # noqa: E402
    LABEL_PROJECTION_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)


EXPECTED_ROWS = 1160
EXPECTED_NODES = 10
EXPECTED_SAMPLING = {
    "strategy": "deterministic_greedy_v1",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "seed": 42,
    "max_model_len": 32768,
    "max_new_tokens": 27000,
    "num_speculative_tokens": 1,
    "thinking": False,
}


def parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"invalid S3 URI: {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return bucket, key.rstrip("/")


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key, suffix) if value)}"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def get_bytes(client: Any, uri: str) -> bytes:
    bucket, key = parse_s3(uri)
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def maybe_get_bytes(client: Any, uri: str) -> bytes | None:
    try:
        return get_bytes(client, uri)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def put_bytes(client: Any, uri: str, payload: bytes, *, content_type: str) -> None:
    bucket, key = parse_s3(uri)
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def wait_for_nodes(client: Any, run_root: str, *, wait_seconds: int, poll_seconds: int) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        complete: list[int] = []
        failed: list[int] = []
        for node_index in range(EXPECTED_NODES):
            node_name = f"node-{node_index:03d}-of-{EXPECTED_NODES:03d}"
            node_root = join_s3(run_root, "results", "nodes", node_name)
            if maybe_get_bytes(client, join_s3(node_root, "_SUCCESS")) is not None:
                complete.append(node_index)
            elif maybe_get_bytes(client, join_s3(node_root, "_FAILED.json")) is not None:
                failed.append(node_index)
        print(json.dumps({"complete_nodes": complete, "failed_nodes": failed}), flush=True)
        if len(complete) == EXPECTED_NODES:
            return
        if failed:
            raise RuntimeError(f"held RTX inference failed on nodes {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for inference nodes; complete={complete}")
        time.sleep(poll_seconds)


def validate_prompt_row(row: Mapping[str, Any]) -> None:
    if row.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("prediction prompt version mismatch")
    if row.get("prompt_contract_sha256") != PROMPT_CONTRACT_SHA256:
        raise ValueError("prediction prompt SHA mismatch")
    if row.get("label_projection_version") != LABEL_PROJECTION_VERSION:
        raise ValueError("prediction label projection mismatch")
    instruction = row.get("instruction")
    size = row.get("target_canvas_size")
    if not isinstance(instruction, str) or not isinstance(size, Mapping):
        raise ValueError("prediction lacks prompt inputs")
    expected_user = build_user_prompt(instruction, (int(size["width"]), int(size["height"])))
    messages = row.get("messages")
    if row.get("inference_error"):
        if messages is not None:
            raise ValueError("inference-error row unexpectedly exposes a conversation")
        return
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or messages[0] != {"role": "system", "content": SYSTEM_PROMPT}
        or messages[1] != {"role": "user", "content": expected_user}
        or messages[2] != {"role": "assistant", "content": row.get("response")}
    ):
        raise ValueError("prediction conversation differs from the exact training prompt")
    if row.get("images") != [row.get("source_s3_uri")]:
        raise ValueError("prediction did not use exactly the one Source image")
    if row.get("sampling") != EXPECTED_SAMPLING:
        raise ValueError("prediction sampling contract mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = boto3.Session(profile_name=args.aws_profile).client("s3", region_name="us-east-1")
    success_uri = join_s3(args.run_root, "_SUCCESS")
    existing = maybe_get_bytes(client, success_uri)
    if existing is not None:
        value = json.loads(existing)
        if value.get("complete") is not True:
            raise RuntimeError("run root has a non-complete _SUCCESS marker")
        return value
    wait_for_nodes(
        client,
        args.run_root,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )
    contract = json.loads(get_bytes(client, join_s3(args.run_root, "run_contract.json")))
    if (
        contract.get("prompt", {}).get("version") != PROMPT_VERSION
        or contract.get("prompt", {}).get("contract_sha256") != PROMPT_CONTRACT_SHA256
        or contract.get("topology", {}).get("gpu_count") != EXPECTED_NODES
    ):
        raise ValueError("run contract prompt/topology mismatch")
    rows: list[dict[str, Any]] = []
    node_summaries: list[dict[str, Any]] = []
    for node_index in range(EXPECTED_NODES):
        node_name = f"node-{node_index:03d}-of-{EXPECTED_NODES:03d}"
        node_root = join_s3(args.run_root, "results", "nodes", node_name)
        marker = json.loads(get_bytes(client, join_s3(node_root, "_SUCCESS")))
        predictions_payload = get_bytes(client, join_s3(node_root, "predictions.jsonl"))
        predictions_sha256 = hashlib.sha256(predictions_payload).hexdigest()
        if marker.get("predictions_sha256") != predictions_sha256:
            raise ValueError(f"node {node_index} prediction SHA mismatch")
        node_rows = [json.loads(line) for line in predictions_payload.splitlines() if line.strip()]
        if len(node_rows) != marker.get("rows"):
            raise ValueError(f"node {node_index} row count mismatch")
        for row in node_rows:
            if not isinstance(row, dict):
                raise ValueError(f"node {node_index} emitted a non-object row")
            validate_prompt_row(row)
        rows.extend(node_rows)
        node_summaries.append(json.loads(get_bytes(client, join_s3(node_root, "summary.json"))))
    rows.sort(key=lambda row: int(row["input_index"]))
    if [int(row["input_index"]) for row in rows] != list(range(EXPECTED_ROWS)):
        raise ValueError("merged predictions do not cover every input index exactly once")
    if len({str(row["sample_id"]) for row in rows}) != EXPECTED_ROWS:
        raise ValueError("merged predictions contain duplicate sample IDs")

    ground_truth_payload = get_bytes(client, join_s3(args.run_root, "ground_truth", "ground_truth.jsonl"))
    ground_truth = [json.loads(line) for line in ground_truth_payload.splitlines() if line.strip()]
    if len(ground_truth) != EXPECTED_ROWS:
        raise ValueError("ground truth row count changed")
    for prediction, target in zip(rows, ground_truth, strict=True):
        if (
            prediction["input_index"] != target["input_index"]
            or prediction["sample_id"] != target["sample_id"]
        ):
            raise ValueError("prediction/ground-truth row alignment changed")

    merged_payload = b"".join(canonical_bytes(row) for row in rows)
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counter = per_task[str(row["task_type"])]
        counter["rows"] += 1
        counter["json_valid"] += int(bool(row.get("json_valid")))
        counter["schema_valid"] += int(bool(row.get("schema_valid")))
        counter["inference_errors"] += int(bool(row.get("inference_error")))
    artifact_hashes = {
        summary.get("artifact", {}).get("manifest_sha256") for summary in node_summaries
    }
    if len(artifact_hashes) != 1 or None in artifact_hashes:
        raise ValueError("nodes did not use one identical HF artifact")
    summary = {
        "complete": True,
        "status": "complete",
        "format_version": "image-edit-pe-held-10rtx-evaluation-result-v1",
        "run_root": args.run_root,
        "checkpoint": contract["checkpoint"],
        "hf_artifact": {
            **contract["hf_artifact"],
            "manifest_sha256": next(iter(artifact_hashes)),
        },
        "benchmark": contract["benchmark"],
        "prompt": contract["prompt"],
        "sampling": contract["sampling"],
        "topology": contract["topology"],
        "results": {
            "rows": len(rows),
            "unique_sample_ids": len({row["sample_id"] for row in rows}),
            "json_valid": sum(int(bool(row.get("json_valid"))) for row in rows),
            "schema_valid": sum(int(bool(row.get("schema_valid"))) for row in rows),
            "inference_errors": sum(int(bool(row.get("inference_error"))) for row in rows),
            "predictions_sha256": hashlib.sha256(merged_payload).hexdigest(),
            "predictions_bytes": len(merged_payload),
            "ground_truth_sha256": hashlib.sha256(ground_truth_payload).hexdigest(),
            "task_types": len(per_task),
            "per_task": {key: dict(value) for key, value in sorted(per_task.items())},
        },
        "node_summaries": node_summaries,
    }
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    put_bytes(
        client,
        join_s3(args.run_root, "results", "predictions.jsonl"),
        merged_payload,
        content_type="application/x-ndjson",
    )
    put_bytes(
        client,
        join_s3(args.run_root, "results", "summary.json"),
        summary_payload,
        content_type="application/json",
    )
    success = {
        "complete": True,
        "status": "complete",
        "rows": EXPECTED_ROWS,
        "task_types": 29,
        "checkpoint_step": contract["checkpoint"]["step"],
        "checkpoint_manifest_sha256": contract["checkpoint"]["manifest_sha256"],
        "hf_artifact_manifest_sha256": next(iter(artifact_hashes)),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "sampling": EXPECTED_SAMPLING,
        "predictions_sha256": summary["results"]["predictions_sha256"],
        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
    }
    put_bytes(client, success_uri, canonical_bytes(success), content_type="application/json")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--wait-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.wait_seconds < 0 or args.poll_seconds <= 0:
        parser.error("wait/poll seconds are invalid")
    parse_s3(args.run_root)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
