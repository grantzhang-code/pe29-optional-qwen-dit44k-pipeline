#!/usr/bin/env python3
"""Run one 8-GPU node of object-sharded image-edit full-JSON inference.

The source Parquet inventory is immutable for one run.  Source objects are
assigned by ``object_index % global_world_size``; each GPU owns one global rank
and publishes one JSONL plus a success marker per source object.  A restarted
JobSet therefore resumes at the source-object boundary instead of discarding a
whole node's long-running inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_full_json import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
)
from scripts.prepare_image_edit_full_json_shard import (  # noqa: E402
    build_source_inventory,
    parse_s3_uri as parse_inventory_s3_uri,
)


NODE_FORMAT = "image_edit_full_json_parquet_node_v1"
RESULT_FORMAT = "image_edit_full_json_parquet_results_v1"
ROOT_SUCCESS_FORMAT = "image_edit_full_json_parquet_results_commit_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def normalize_s3_root(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key}/" if key else f"s3://{bucket}/"


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    combined = "/".join(value for value in (key, suffix) if value)
    return f"s3://{bucket}/{combined}"


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=128,
        ),
    )


def _missing_s3_error(error: BaseException) -> bool:
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def read_s3_json(client: Any, uri: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
    bucket, key = parse_s3_uri(uri)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except BaseException as exc:
        if missing_ok and _missing_s3_error(exc):
            return None
        raise
    payload = response["Body"].read()
    metadata = response.get("Metadata")
    if not isinstance(metadata, Mapping) or metadata.get("sha256") != sha256_bytes(payload):
        raise ValueError(f"JSON object SHA-256 metadata is missing or wrong: {uri}")
    if response.get("ContentType") != "application/json":
        raise ValueError(f"JSON object ContentType is invalid: {uri}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return value


def upload_bytes(client: Any, payload: bytes, uri: str, *, content_type: str) -> str:
    bucket, key = parse_s3_uri(uri)
    digest = sha256_bytes(payload)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != len(payload) or head.get("Metadata", {}).get("sha256") != digest:
        raise IOError(f"uploaded object verification failed: {uri}")
    return digest


def upload_json(client: Any, value: Any, uri: str) -> str:
    return upload_bytes(client, canonical_json_bytes(value), uri, content_type="application/json")


def upload_file(client: Any, source: Path, uri: str, *, content_type: str) -> None:
    bucket, key = parse_s3_uri(uri)
    digest = file_sha256(source)
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type, "Metadata": {"sha256": digest}},
    )


def node_name(node_index: int, node_count: int) -> str:
    return f"node-{node_index:03d}-of-{node_count:03d}"


def node_prefix(output_s3: str, node_index: int, node_count: int) -> str:
    return join_s3(output_s3, "nodes", node_name(node_index, node_count))


def generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attention_backend": args.attention_backend,
        "batch_size_per_gpu": args.batch_size,
        "continue_on_error": True,
        "download_workers": args.download_workers,
        "enforce_eager": bool(args.enforce_eager),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "include_conversation": False,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "num_speculative_tokens": args.num_speculative_tokens,
        "seed": args.seed,
        "temperature": 0.0,
        "tensor_parallel_size_per_worker": 1,
        "use_mtp": bool(args.use_mtp),
    }


def worker_generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Mirror the output-affecting contract sealed by each GPU worker."""

    return {
        "temperature": 0.0,
        "thinking": False,
        "include_conversation": False,
        "continue_on_error": True,
        "image_order": ["reference frame_0", "target"],
        "batch_size": args.batch_size,
        "download_workers": args.download_workers,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": args.attention_backend or None,
        "enforce_eager": bool(args.enforce_eager),
        "use_mtp": bool(args.use_mtp),
        "num_speculative_tokens": args.num_speculative_tokens,
        "seed": args.seed,
    }


def static_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_dataset_s3": normalize_s3_root(args.source_dataset_s3),
        "expected_count": args.expected_count,
        "expected_objects": args.expected_objects,
        "checkpoint_s3": normalize_s3_root(args.checkpoint_s3),
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "hf_artifact_s3": normalize_s3_root(args.hf_artifact_s3),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "node_count": args.node_count,
        "local_world_size": args.local_world_size,
        "global_world_size": args.node_count * args.local_world_size,
        "generation": generation_contract(args),
    }


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: actual={actual!r} expected={expected!r}")


def validate_node_marker(
    marker: Mapping[str, Any], args: argparse.Namespace, *, require_runtime: bool
) -> None:
    _require_equal(marker.get("format_version"), NODE_FORMAT, "node marker format")
    _require_equal(marker.get("status"), "complete", "node marker status")
    _require_equal(marker.get("node_index"), args.node_index, "node marker node_index")
    _require_equal(marker.get("node_count"), args.node_count, "node marker node_count")
    contract = marker.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("node marker lacks contract")
    for key, expected in static_contract(args).items():
        _require_equal(contract.get(key), expected, f"node marker contract.{key}")
    if require_runtime:
        _require_equal(
            contract.get("hf_artifact_manifest_sha256"),
            args.hf_artifact_manifest_sha256,
            "node marker HF artifact digest",
        )
        digest = contract.get("source_inventory_sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError("node marker has invalid source inventory digest")


def check_node_complete(client: Any, args: argparse.Namespace) -> bool:
    uri = join_s3(node_prefix(args.output_s3, args.node_index, args.node_count), "_SUCCESS")
    marker = read_s3_json(client, uri, missing_ok=True)
    if marker is None:
        return False
    validate_node_marker(marker, args, require_runtime=False)
    return True


def build_inventory(client: Any, args: argparse.Namespace) -> tuple[dict[str, Any], bytes, str]:
    inventory = build_source_inventory(
        parse_inventory_s3_uri(args.source_dataset_s3.rstrip("/")),
        s3_client=client,
        workers=args.inventory_workers,
    )
    _require_equal(inventory.get("object_count"), args.expected_objects, "source object count")
    _require_equal(inventory.get("total_rows"), args.expected_count, "source row count")
    payload = canonical_json_bytes(inventory)
    return inventory, payload, sha256_bytes(payload)


def expected_objects_for_rank(
    inventory: Mapping[str, Any], global_rank: int, global_world_size: int
) -> list[dict[str, Any]]:
    values = inventory.get("objects")
    if not isinstance(values, list):
        raise ValueError("source inventory has no objects list")
    return [
        dict(item)
        for item in values
        if int(item["object_index"]) % global_world_size == global_rank
    ]


def worker_command(
    args: argparse.Namespace,
    *,
    inventory_path: Path,
    inventory_sha256: str,
    global_rank: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/infer_image_edit_full_json_parquet_rank.py"),
        "--model",
        str(args.model),
        "--inventory",
        str(inventory_path),
        "--inventory-sha256",
        inventory_sha256,
        "--source-dataset-s3",
        args.source_dataset_s3,
        "--output-s3",
        args.output_s3,
        "--work-dir",
        str(args.work_dir),
        "--global-rank",
        str(global_rank),
        "--global-world-size",
        str(args.node_count * args.local_world_size),
        "--checkpoint-step",
        str(args.checkpoint_step),
        "--checkpoint-manifest-sha256",
        args.checkpoint_manifest_sha256,
        "--hf-artifact-manifest-sha256",
        args.hf_artifact_manifest_sha256,
        "--batch-size",
        str(args.batch_size),
        "--download-workers",
        str(args.download_workers),
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-pixels",
        str(args.max_pixels),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--attention-backend",
        args.attention_backend,
        "--progress-every",
        str(args.progress_every),
        "--seed",
        str(args.seed),
        "--num-speculative-tokens",
        str(args.num_speculative_tokens),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.use_mtp:
        command.append("--use-mtp")
    return command


def _terminate_processes(processes: Iterable[subprocess.Popen[Any]]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()
    deadline = time.monotonic() + 60
    for process in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def _load_rank_summary(
    path: Path,
    *,
    global_rank: int,
    expected_objects: Sequence[Mapping[str, Any]],
    inventory_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"rank {global_rank} summary is not an object")
    _require_equal(value.get("status"), "complete", f"rank {global_rank} status")
    _require_equal(value.get("global_rank"), global_rank, f"rank {global_rank} identity")
    _require_equal(
        value.get("global_world_size"),
        args.node_count * args.local_world_size,
        f"rank {global_rank} world size",
    )
    _require_equal(
        value.get("format_version"),
        "image_edit_full_json_parquet_rank_summary_v1",
        f"rank {global_rank} summary format",
    )
    _require_equal(
        value.get("source_dataset_s3"),
        args.source_dataset_s3.rstrip("/"),
        f"rank {global_rank} source dataset",
    )
    inventory_contract = value.get("inventory")
    if not isinstance(inventory_contract, Mapping):
        raise ValueError(f"rank {global_rank} summary has no inventory contract")
    _require_equal(
        inventory_contract.get("sha256"),
        inventory_sha256,
        f"rank {global_rank} inventory digest",
    )
    _require_equal(
        inventory_contract.get("total_objects"),
        args.expected_objects,
        f"rank {global_rank} inventory object count",
    )
    _require_equal(
        inventory_contract.get("total_rows"),
        args.expected_count,
        f"rank {global_rank} inventory row count",
    )
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"rank {global_rank} summary has no checkpoint contract")
    _require_equal(
        checkpoint.get("step"), args.checkpoint_step, f"rank {global_rank} checkpoint step"
    )
    _require_equal(
        checkpoint.get("manifest_sha256"),
        args.checkpoint_manifest_sha256,
        f"rank {global_rank} checkpoint digest",
    )
    hf_artifact = value.get("hf_artifact")
    if not isinstance(hf_artifact, Mapping):
        raise ValueError(f"rank {global_rank} summary has no HF artifact contract")
    _require_equal(
        hf_artifact.get("manifest_sha256"),
        args.hf_artifact_manifest_sha256,
        f"rank {global_rank} HF artifact digest",
    )
    _require_equal(
        value.get("prompt"),
        {"version": PROMPT_VERSION, "contract_sha256": PROMPT_CONTRACT_SHA256},
        f"rank {global_rank} prompt contract",
    )
    _require_equal(
        value.get("generation"),
        worker_generation_contract(args),
        f"rank {global_rank} generation contract",
    )
    shards = value.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"rank {global_rank} summary has no shards list")
    actual_indices = sorted(int(item["object_index"]) for item in shards)
    expected_indices = [int(item["object_index"]) for item in expected_objects]
    _require_equal(actual_indices, expected_indices, f"rank {global_rank} object coverage")
    expected_by_index = {int(item["object_index"]): item for item in expected_objects}
    for shard in shards:
        _validate_shard_against_inventory(
            shard,
            expected_by_index[int(shard["object_index"])],
            output_s3=args.output_s3,
        )
    expected_totals = _aggregate_shards(shards)
    expected_totals.pop("objects")
    _require_equal(value.get("totals"), expected_totals, f"rank {global_rank} totals")
    _require_equal(
        value.get("assigned_objects"),
        len(expected_objects),
        f"rank {global_rank} assigned object count",
    )
    generated = value.get("generated_objects")
    resumed = value.get("resumed_objects")
    if not isinstance(generated, int) or isinstance(generated, bool) or generated < 0:
        raise ValueError(f"rank {global_rank} generated object count is invalid")
    if not isinstance(resumed, int) or isinstance(resumed, bool) or resumed < 0:
        raise ValueError(f"rank {global_rank} resumed object count is invalid")
    _require_equal(
        generated + resumed,
        len(expected_objects),
        f"rank {global_rank} generated/resumed object count",
    )
    _require_equal(
        value.get("summary_s3_uri"),
        join_s3(args.output_s3, "ranks", f"rank-{global_rank:03d}", "summary.json"),
        f"rank {global_rank} summary URI",
    )
    return value


def build_node_marker(
    args: argparse.Namespace,
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    rank_summaries: Sequence[Mapping[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    shards = sorted(
        (dict(shard) for summary in rank_summaries for shard in summary["shards"]),
        key=lambda item: int(item["object_index"]),
    )
    global_world_size = args.node_count * args.local_world_size
    local_ranks = list(
        range(
            args.node_index * args.local_world_size,
            (args.node_index + 1) * args.local_world_size,
        )
    )
    expected = sorted(
        int(item["object_index"])
        for rank in local_ranks
        for item in expected_objects_for_rank(inventory, rank, global_world_size)
    )
    _require_equal(
        [int(item["object_index"]) for item in shards], expected, "node shard coverage"
    )
    expected_rows = sum(int(inventory["objects"][index]["rows"]) for index in expected)
    actual_rows = sum(int(item["output"]["rows"]) for item in shards)
    _require_equal(actual_rows, expected_rows, "node output row count")
    contract = {
        **static_contract(args),
        "source_inventory_sha256": inventory_sha256,
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
    }
    totals = _aggregate_shards(shards)
    return {
        "format_version": NODE_FORMAT,
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "node_index": args.node_index,
        "node_count": args.node_count,
        "global_ranks": local_ranks,
        "contract": contract,
        "totals": totals,
        "rank_summaries": [
            {
                key: summary[key]
                for key in (
                    "global_rank",
                    "summary_s3_uri",
                    "assigned_objects",
                    "generated_objects",
                    "resumed_objects",
                    "totals",
                )
                if key in summary
            }
            for summary in rank_summaries
        ],
        "shards": shards,
    }


def _validate_output_counts(output: Mapping[str, Any]) -> dict[str, int]:
    counts = output.get("counts")
    keys = {"rows", "json_valid", "schema_valid", "failed", "inference_errors"}
    if not isinstance(counts, Mapping) or set(counts) != keys:
        raise ValueError("shard output counts have an unexpected schema")
    normalized: dict[str, int] = {}
    for key in sorted(keys):
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"shard output counts.{key} is invalid")
        normalized[key] = value
    _require_equal(output.get("rows"), normalized["rows"], "shard output/count rows")
    if not normalized["schema_valid"] <= normalized["json_valid"] <= normalized["rows"]:
        raise ValueError("shard JSON/schema counts are inconsistent")
    if normalized["failed"] != normalized["rows"] - normalized["schema_valid"]:
        raise ValueError("shard failure count is inconsistent")
    if normalized["inference_errors"] > normalized["failed"]:
        raise ValueError("shard inference error count is inconsistent")
    return normalized


def _aggregate_shards(shards: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "objects": len(shards),
        "rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "failed": 0,
        "inference_errors": 0,
        "bytes": 0,
    }
    for shard in shards:
        output = shard.get("output")
        if not isinstance(output, Mapping):
            raise ValueError("shard lacks output identity")
        counts = _validate_output_counts(output)
        for key in ("rows", "json_valid", "schema_valid", "failed", "inference_errors"):
            totals[key] += counts[key]
        size = output.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("shard output byte count is invalid")
        totals["bytes"] += size
    return totals


def _validate_shard_against_inventory(
    shard: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    output_s3: str | None = None,
) -> None:
    _require_equal(shard.get("object_index"), item.get("object_index"), "shard object index")
    source = shard.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("shard lacks source identity")
    for key in (
        "bucket",
        "key",
        "etag",
        "bytes",
        "rows",
        "global_row_start",
        "global_row_end_exclusive",
    ):
        _require_equal(source.get(key), item.get(key), f"shard source.{key}")
    output = shard.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("shard lacks output identity")
    _require_equal(output.get("rows"), item.get("rows"), "shard output rows")
    if output_s3 is not None:
        _require_equal(
            output.get("uri"),
            join_s3(output_s3, "shards", f"part-{int(item['object_index']):05d}.jsonl"),
            "shard output URI",
        )
    if not isinstance(output.get("uri"), str) or SHA256_RE.fullmatch(
        str(output.get("sha256", ""))
    ) is None:
        raise ValueError("shard output lacks URI/SHA-256")
    _validate_output_counts(output)
    marker_uri = shard.get("success_marker_uri")
    if output_s3 is not None:
        _require_equal(
            marker_uri,
            join_s3(
                output_s3,
                "shards",
                f"part-{int(item['object_index']):05d}.success.json",
            ),
            "shard success marker URI",
        )
    if shard.get("disposition") not in {"generated", "resumed"}:
        raise ValueError("shard disposition is invalid")


def maybe_finalize(
    client: Any,
    args: argparse.Namespace,
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
) -> dict[str, Any] | None:
    node_markers: list[dict[str, Any]] = []
    for index in range(args.node_count):
        uri = join_s3(node_prefix(args.output_s3, index, args.node_count), "_SUCCESS")
        marker = read_s3_json(client, uri, missing_ok=True)
        if marker is None:
            return None
        marker_args = argparse.Namespace(**{**vars(args), "node_index": index})
        validate_node_marker(marker, marker_args, require_runtime=True)
        _require_equal(
            marker["contract"].get("source_inventory_sha256"),
            inventory_sha256,
            f"node {index} inventory digest",
        )
        marker_shards = marker.get("shards")
        if not isinstance(marker_shards, list):
            raise ValueError(f"node {index} marker has no shards list")
        expected_ranks = list(
            range(index * args.local_world_size, (index + 1) * args.local_world_size)
        )
        _require_equal(marker.get("global_ranks"), expected_ranks, f"node {index} ranks")
        expected_indices = sorted(
            int(item["object_index"])
            for rank in expected_ranks
            for item in expected_objects_for_rank(
                inventory,
                rank,
                args.node_count * args.local_world_size,
            )
        )
        _require_equal(
            sorted(int(shard["object_index"]) for shard in marker_shards),
            expected_indices,
            f"node {index} shard coverage",
        )
        _require_equal(
            marker.get("totals"),
            _aggregate_shards(marker_shards),
            f"node {index} totals",
        )
        node_markers.append(marker)

    shards = sorted(
        (dict(shard) for marker in node_markers for shard in marker["shards"]),
        key=lambda item: int(item["object_index"]),
    )
    _require_equal(len(shards), args.expected_objects, "global shard count")
    _require_equal(
        [int(item["object_index"]) for item in shards],
        list(range(args.expected_objects)),
        "global shard coverage",
    )
    for shard, item in zip(shards, inventory["objects"], strict=True):
        _validate_shard_against_inventory(shard, item, output_s3=args.output_s3)
    totals = _aggregate_shards(shards)
    _require_equal(totals["rows"], args.expected_count, "global result row count")
    contract = {
        **static_contract(args),
        "source_inventory_sha256": inventory_sha256,
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
    }
    completed_at = max(str(marker["completed_at"]) for marker in node_markers)
    manifest = {
        "format_version": RESULT_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "contract": contract,
        "totals": totals,
        "nodes": [
            {
                "node_index": int(marker["node_index"]),
                "success_uri": join_s3(
                    node_prefix(args.output_s3, int(marker["node_index"]), args.node_count),
                    "_SUCCESS",
                ),
                "totals": marker["totals"],
            }
            for marker in node_markers
        ],
        "shards": shards,
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_uri = join_s3(args.output_s3, "manifest.json")
    manifest_digest = upload_bytes(
        client, manifest_payload, manifest_uri, content_type="application/json"
    )
    success = {
        "format_version": ROOT_SUCCESS_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "objects": totals["objects"],
        "rows": totals["rows"],
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
    }
    upload_json(client, success, join_s3(args.output_s3, "_SUCCESS"))
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    client = s3_client()
    if check_node_complete(client, args):
        # A previous attempt may have crashed after committing the last node
        # marker but before publishing the root manifest. Rebuild the frozen
        # inventory and make finalization idempotently recover that small
        # commit window instead of declaring success with no root _SUCCESS.
        inventory, _, inventory_sha256 = build_inventory(client, args)
        manifest = maybe_finalize(
            client,
            args,
            inventory=inventory,
            inventory_sha256=inventory_sha256,
        )
        return {
            "status": "already_complete",
            "node_index": args.node_index,
            "root_finalized": manifest is not None,
        }

    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix = node_prefix(args.output_s3, args.node_index, args.node_count)
    status_uri = join_s3(prefix, "status.json")
    upload_json(
        client,
        {"status": "preparing", "started_at": started_at, "node_index": args.node_index},
        status_uri,
    )
    inventory, inventory_payload, inventory_sha256 = build_inventory(client, args)
    inventory_path = args.work_dir / "source_inventory.json"
    inventory_path.write_bytes(inventory_payload)
    upload_bytes(
        client,
        inventory_payload,
        join_s3(prefix, "source_inventory.json"),
        content_type="application/json",
    )

    gpu_tokens = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_tokens) != args.local_world_size:
        raise ValueError(f"--gpus must contain exactly {args.local_world_size} values")
    global_rank_offset = args.node_index * args.local_world_size
    processes: dict[int, subprocess.Popen[Any]] = {}
    handles: dict[int, Any] = {}
    for local_rank, gpu in enumerate(gpu_tokens):
        global_rank = global_rank_offset + local_rank
        rank_dir = args.work_dir / f"rank-{global_rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (rank_dir / "worker.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            }
        )
        processes[global_rank] = subprocess.Popen(
            worker_command(
                args,
                inventory_path=inventory_path,
                inventory_sha256=inventory_sha256,
                global_rank=global_rank,
            ),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        handles[global_rank] = log_handle

    upload_json(
        client,
        {
            "status": "running",
            "started_at": started_at,
            "node_index": args.node_index,
            "global_ranks": sorted(processes),
            "source_inventory_sha256": inventory_sha256,
        },
        status_uri,
    )
    remaining = set(processes)
    failure: RuntimeError | None = None
    while remaining:
        for rank in list(remaining):
            returncode = processes[rank].poll()
            if returncode is None:
                continue
            remaining.remove(rank)
            handles[rank].close()
            rank_dir = args.work_dir / f"rank-{rank:03d}"
            upload_file(
                client,
                rank_dir / "worker.log",
                join_s3(args.output_s3, "ranks", f"rank-{rank:03d}", "worker.log"),
                content_type="text/plain",
            )
            if returncode != 0 and failure is None:
                failure = RuntimeError(f"global rank {rank} exited with code {returncode}")
        if failure is not None:
            _terminate_processes(processes.values())
            for rank in list(remaining):
                handles[rank].close()
            raise failure
        if remaining:
            time.sleep(2)

    global_world_size = args.node_count * args.local_world_size
    rank_summaries: list[dict[str, Any]] = []
    for global_rank in sorted(processes):
        rank_dir = args.work_dir / f"rank-{global_rank:03d}"
        rank_summaries.append(
            _load_rank_summary(
                rank_dir / "summary.json",
                global_rank=global_rank,
                expected_objects=expected_objects_for_rank(
                    inventory, global_rank, global_world_size
                ),
                inventory_sha256=inventory_sha256,
                args=args,
            )
        )

    marker = build_node_marker(
        args,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        rank_summaries=rank_summaries,
        started_at=started_at,
    )
    upload_json(client, marker, join_s3(prefix, "summary.json"))
    upload_json(client, marker, join_s3(prefix, "_SUCCESS"))
    maybe_finalize(
        client,
        args,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
    )
    upload_json(
        client,
        {"status": "complete", "completed_at": marker["completed_at"]},
        status_uri,
    )
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-complete", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-dataset-s3", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-objects", type=int, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/image-edit-parquet"))
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--checkpoint-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--hf-artifact-s3", required=True)
    parser.add_argument("--hf-artifact-manifest-sha256")
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1536 * 1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    parse_s3_uri(args.source_dataset_s3)
    parse_s3_uri(args.output_s3)
    parse_s3_uri(args.checkpoint_s3)
    parse_s3_uri(args.hf_artifact_s3)
    for name in (
        "expected_count",
        "expected_objects",
        "node_count",
        "local_world_size",
        "checkpoint_step",
        "inventory_workers",
        "batch_size",
        "download_workers",
        "max_model_len",
        "max_new_tokens",
        "max_pixels",
        "progress_every",
        "num_speculative_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.node_index < 0 or args.node_index >= args.node_count:
        parser.error("--node-index must be in [0, --node-count)")
    if SHA256_RE.fullmatch(args.checkpoint_manifest_sha256) is None:
        parser.error("--checkpoint-manifest-sha256 must be 64 lowercase hex characters")
    if not args.check_complete:
        if args.model is None:
            parser.error("--model is required unless --check-complete is used")
        if not args.hf_artifact_manifest_sha256 or SHA256_RE.fullmatch(
            args.hf_artifact_manifest_sha256
        ) is None:
            parser.error("--hf-artifact-manifest-sha256 must be 64 lowercase hex characters")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    client = s3_client()
    if args.check_complete:
        complete = check_node_complete(client, args)
        print(json.dumps({"complete": complete, "node_index": args.node_index}, sort_keys=True))
        return 0 if complete else 3
    try:
        result = run(args)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "node_index": args.node_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            upload_json(
                client,
                failure,
                join_s3(node_prefix(args.output_s3, args.node_index, args.node_count), "status.json"),
            )
        except BaseException as upload_error:
            print(f"failed to upload node failure status: {upload_error}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
