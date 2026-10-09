#!/usr/bin/env python3
"""Run and finalize one 8-GPU node of sidecar24 unit-sharded inference.

Every local GPU owns one independent TP=1 worker.  Units first enter a
deterministic data part, then part-local ordinal assigns ranks; resume remains
at the (possibly canary-clipped) unit boundary.  A
node commits only after all of its rank summaries and unit shard bindings have
been validated.  Any node that observes all node commits may idempotently
publish the root manifest and ``_SUCCESS`` marker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

from dlms.datasets.image_edit_inference_contract import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SUPPORTED_TARGET_SCHEMAS,
    TARGET_SCHEMA,
    require_target_schema,
)
from scripts import image_edit_sidecar24_catalog as source_catalog_module  # noqa: E402
from scripts import image_edit_sidecar24_partition as partition_module  # noqa: E402
from scripts import infer_image_edit_full_json_sidecar24_rank as rank_worker  # noqa: E402
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module  # noqa: E402


NODE_FORMAT = "image_edit_full_json_sidecar24_node_v1"
RESULT_FORMAT = "image_edit_full_json_sidecar24_results_v1"
ROOT_SUCCESS_FORMAT = "image_edit_full_json_sidecar24_results_commit_v1"
NODE_MARKER_FIELDS = frozenset(
    {
        "format_version",
        "status",
        "started_at",
        "completed_at",
        "node_index",
        "node_count",
        "global_ranks",
        "contract",
        "totals",
        "rank_summaries",
        "shards",
    }
)
EXPECTED_SELECTED_ROWS = unit_catalog_module.EXPECTED_TOTAL_SELECTED_ROWS
EXPECTED_DATASETS = source_catalog_module.EXPECTED_DATASET_COUNT
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VLLM_PORT_BASE_START = 20000
VLLM_PORT_WINDOW_SIZE = 1024
MAX_TCP_PORT = 65535

OUTPUT_COUNT_FIELDS = frozenset({"rows", "json_valid", "schema_valid", "failed", "inference_errors"})
REBALANCE_ROW_WEIGHT_BYTES = 64 * 1024 * 1024
REBALANCE_UNIT_OVERHEAD_BYTES = 8 * 1024 * 1024
DEFAULT_REBALANCE_SCAN_WORKERS = 64
REPARTITION_NODE_ASSIGNMENT_FORMAT = "image_edit_sidecar24_repartition_node_assignment_v1"
REPARTITION_NODE_RESULT_FORMAT = "image_edit_sidecar24_repartition_node_result_v1"


def expected_dataset_count(source_catalog: Mapping[str, Any]) -> int:
    if source_catalog.get("schema_version") == source_catalog_module.CRAFTV0_SCHEMA_VERSION:
        return source_catalog_module.expected_dataset_count(source_catalog)
    return EXPECTED_DATASETS


def expected_selected_rows(source_catalog: Mapping[str, Any]) -> int:
    if source_catalog.get("schema_version") == source_catalog_module.CRAFTV0_SCHEMA_VERSION:
        return source_catalog_module.expected_total_selected_rows(source_catalog)
    return EXPECTED_SELECTED_ROWS


@dataclass(frozen=True)
class BoundCatalogs:
    source: dict[str, Any]
    units: dict[str, Any]
    source_sha256: str
    unit_sha256: str
    source_path: Path
    unit_path: Path


def vllm_port_base(local_rank: int, local_world_size: int) -> int:
    """Return a deterministic, non-overlapping EngineCore port-window base."""

    if local_world_size <= 0 or local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(f"invalid local rank/world size: rank={local_rank} world={local_world_size}")
    last_port = VLLM_PORT_BASE_START + local_world_size * VLLM_PORT_WINDOW_SIZE - 1
    if last_port > MAX_TCP_PORT:
        raise ValueError(f"{local_world_size} vLLM port windows exceed TCP port {MAX_TCP_PORT}")
    return VLLM_PORT_BASE_START + local_rank * VLLM_PORT_WINDOW_SIZE


def global_world_size(args: argparse.Namespace) -> int:
    explicit = getattr(args, "global_world_size", None)
    return int(explicit) if explicit is not None else args.node_count * args.local_world_size


def node_global_rank_offset(args: argparse.Namespace, node_index: int) -> int:
    explicit = getattr(args, "global_rank_offset", None)
    if explicit is not None:
        if node_index != args.node_index:
            raise ValueError(
                "an explicit global rank offset can validate only the current node; "
                "use the topology-aware global finalizer for other nodes"
            )
        return int(explicit)
    return node_index * args.local_world_size


def node_global_ranks(args: argparse.Namespace, node_index: int) -> list[int]:
    start = node_global_rank_offset(args, node_index)
    return list(range(start, start + args.local_world_size))


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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, key


def normalize_s3_root(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key}/"


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


def read_s3_json(
    client: Any,
    uri: str,
    *,
    missing_ok: bool = False,
) -> dict[str, Any] | None:
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
    if (
        int(head["ContentLength"]) != len(payload)
        or head.get("ContentType") != content_type
        or head.get("Metadata", {}).get("sha256") != digest
    ):
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
    head = client.head_object(Bucket=bucket, Key=key)
    if (
        int(head["ContentLength"]) != source.stat().st_size
        or head.get("ContentType") != content_type
        or head.get("Metadata", {}).get("sha256") != digest
    ):
        raise IOError(f"uploaded file verification failed: {uri}")


def node_name(node_index: int, node_count: int) -> str:
    return f"node-{node_index:03d}-of-{node_count:03d}"


def node_prefix(output_s3: str, node_index: int, node_count: int) -> str:
    return join_s3(output_s3, "nodes", node_name(node_index, node_count))


def external_repartition_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "external_repartition_prefix", None))


def external_repartition_node_prefix(args: argparse.Namespace) -> str:
    prefix = getattr(args, "external_repartition_prefix", None)
    if not prefix:
        raise ValueError("external repartition prefix is not configured")
    return join_s3(prefix, "nodes", node_name(args.node_index, args.node_count))


def runtime_node_prefix(args: argparse.Namespace) -> str:
    if external_repartition_enabled(args):
        return external_repartition_node_prefix(args)
    return node_prefix(configured_output_s3(args), args.node_index, args.node_count)


def _rank_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        attention_backend=args.attention_backend,
        target_schema=TARGET_SCHEMA,
        batch_size=args.batch_size,
        continue_on_error=True,
        cross_unit_streaming=bool(getattr(args, "cross_unit_streaming", False)),
        cudagraph_mm_encoder=bool(getattr(args, "cudagraph_mm_encoder", False)),
        cudagraph_mode=getattr(args, "cudagraph_mode", None),
        cudagraph_specialize_lora=getattr(args, "cudagraph_specialize_lora", None),
        download_workers=args.download_workers,
        enable_prefix_caching=getattr(args, "enable_prefix_caching", None),
        enable_tower_connector_lora=getattr(args, "enable_tower_connector_lora", None),
        enforce_eager=bool(args.enforce_eager),
        expected_cuda_version=getattr(args, "expected_cuda_version", None),
        expected_flashinfer_cubin_version=getattr(
            args, "expected_flashinfer_cubin_version", None
        ),
        expected_flashinfer_jit_cache_version=getattr(
            args, "expected_flashinfer_jit_cache_version", None
        ),
        expected_flashinfer_version=getattr(args, "expected_flashinfer_version", None),
        expected_torch_version=getattr(args, "expected_torch_version", None),
        expected_transformers_version=getattr(args, "expected_transformers_version", None),
        expected_triton_version=getattr(args, "expected_triton_version", None),
        expected_vllm_version=getattr(args, "expected_vllm_version", None),
        gdn_prefill_backend=getattr(args, "gdn_prefill_backend", None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        include_conversation=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
        max_cudagraph_capture_size=getattr(args, "max_cudagraph_capture_size", None),
        max_pending_unit_writes=getattr(
            args,
            "max_pending_unit_writes",
            rank_worker.DEFAULT_MAX_PENDING_UNIT_WRITES,
        ),
        max_pixels=args.max_pixels,
        max_tokens=args.max_new_tokens,
        mm_processor_cache_gb=getattr(args, "mm_processor_cache_gb", None),
        mm_encoder_attn_backend=getattr(args, "mm_encoder_attn_backend", None),
        num_speculative_tokens=args.num_speculative_tokens,
        prefetch_images=bool(args.prefetch_images),
        request_window_size=args.request_window_size,
        seed=args.seed,
        speculative_attention_backend=getattr(args, "speculative_attention_backend", None),
        tensor_parallel_size=1,
        use_mtp=bool(args.use_mtp),
        unit_upload_workers=getattr(
            args,
            "unit_upload_workers",
            rank_worker.DEFAULT_UNIT_UPLOAD_WORKERS,
        ),
        world_size=global_world_size(args),
        expected_hld_postprocess_version=getattr(
            args, "expected_hld_postprocess_version", None
        ),
        package_v6_bboxes_yxyx=bool(
            getattr(args, "package_v6_bboxes_yxyx", False)
        ),
        partition_count=int(getattr(args, "partition_count", 1)),
        partition_index=int(getattr(args, "partition_index", 0)),
        case_limit=getattr(args, "case_limit", None),
    )


def worker_generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    return rank_worker.generation_contract(_rank_generation_args(args))


def generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **worker_generation_contract(args),
        "tensor_parallel_size_per_worker": 1,
        "batch_size_per_gpu": args.batch_size,
    }


def static_contract(args: argparse.Namespace, catalogs: BoundCatalogs) -> dict[str, Any]:
    plan = build_partition_plan(args, catalogs)
    contract = {
        "source_catalog_uri": args.source_catalog_uri,
        "source_catalog_sha256": catalogs.source_sha256,
        "source_catalog_format_version": catalogs.source["schema_version"],
        "source_catalog_rows": catalogs.source["total_rows"],
        "unit_catalog_uri": args.unit_catalog_uri,
        "unit_catalog_sha256": catalogs.unit_sha256,
        "unit_catalog_format_version": unit_catalog_module.FORMAT_VERSION,
        "expected_units": plan.contract["selected_units"],
        "expected_rows": plan.contract["selected_rows"],
        "dataset_count": len(catalogs.units["datasets"]),
        "data_partition": plan.contract,
        "checkpoint_s3": normalize_s3_root(args.checkpoint_s3),
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "hf_artifact_s3": normalize_s3_root(args.hf_artifact_s3),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "node_count": args.node_count,
        "local_world_size": args.local_world_size,
        "global_world_size": global_world_size(args),
        "generation": generation_contract(args),
    }
    topology_uri = getattr(args, "topology_uri", None)
    topology_sha256 = getattr(args, "topology_sha256", None)
    if topology_uri is not None or topology_sha256 is not None:
        if not topology_uri or not topology_sha256:
            raise ValueError("topology URI and SHA-256 must be supplied together")
        contract["topology_uri"] = topology_uri
        contract["topology_sha256"] = topology_sha256
    code_overlay_sha256 = getattr(args, "code_overlay_sha256", None)
    if code_overlay_sha256 is not None:
        contract["code_overlay_sha256"] = code_overlay_sha256
    return contract


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: actual={actual!r} expected={expected!r}")


def load_bound_catalogs(args: argparse.Namespace) -> BoundCatalogs:
    source = source_catalog_module.load_catalog(args.source_catalog)
    source_digest = source_catalog_module.canonical_sha256(source)
    _require_equal(source_digest, args.source_catalog_sha256, "source catalog SHA-256")

    units = unit_catalog_module.load_unit_catalog(
        args.unit_catalog,
        source_catalog=source,
    )
    unit_digest = unit_catalog_module.canonical_unit_catalog_sha256(
        units,
        source_catalog=source,
    )
    _require_equal(unit_digest, args.unit_catalog_sha256, "unit catalog SHA-256")
    expected_datasets = expected_dataset_count(source)
    expected_rows = expected_selected_rows(source)
    _require_equal(len(source["datasets"]), expected_datasets, "source dataset count")
    _require_equal(len(units["datasets"]), expected_datasets, "unit dataset count")
    _require_equal(
        units["total_selected_rows"],
        expected_rows,
        "unit catalog selected rows",
    )
    return BoundCatalogs(
        source=source,
        units=units,
        source_sha256=source_digest,
        unit_sha256=unit_digest,
        source_path=Path(args.source_catalog),
        unit_path=Path(args.unit_catalog),
    )


def write_canonical_catalogs(catalogs: BoundCatalogs, work_dir: Path) -> BoundCatalogs:
    root = work_dir / "catalogs"
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source_catalog.json"
    unit_path = root / "unit_catalog.json"
    source_path.write_bytes(source_catalog_module.canonical_catalog_bytes(catalogs.source))
    unit_path.write_bytes(
        unit_catalog_module.canonical_unit_catalog_bytes(
            catalogs.units,
            source_catalog=catalogs.source,
        )
    )
    return BoundCatalogs(
        source=catalogs.source,
        units=catalogs.units,
        source_sha256=catalogs.source_sha256,
        unit_sha256=catalogs.unit_sha256,
        source_path=source_path,
        unit_path=unit_path,
    )


def expected_units_for_rank(
    unit_catalog: Mapping[str, Any],
    global_rank: int,
    global_world_size: int,
    partition_count: int = 1,
    partition_index: int = 0,
    case_limit: int | None = None,
) -> list[dict[str, Any]]:
    units = unit_catalog.get("units")
    if not isinstance(units, list):
        raise ValueError("unit catalog has no units list")
    plan = partition_module.build_partition_plan(
        units,
        partition_count=partition_count,
        partition_index=partition_index,
        global_world_size=global_world_size,
        case_limit=case_limit,
    )
    if global_rank < 0 or global_rank >= global_world_size:
        raise ValueError(
            f"invalid global rank/world size: rank={global_rank} world={global_world_size}"
        )
    return [dict(unit) for unit in plan.units_by_rank[global_rank]]


def build_partition_plan(
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
) -> partition_module.PartitionPlan:
    key = (
        id(catalogs.units),
        int(getattr(args, "partition_count", 1)),
        int(getattr(args, "partition_index", 0)),
        global_world_size(args),
        getattr(args, "case_limit", None),
    )
    cached = getattr(args, "_partition_plan_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    plan = partition_module.build_partition_plan(
        catalogs.units["units"],
        partition_count=key[1],
        partition_index=key[2],
        global_world_size=key[3],
        case_limit=key[4],
    )
    setattr(args, "_partition_plan_cache", (key, plan))
    return plan


def unit_work_estimate(unit: Mapping[str, Any]) -> int:
    """Estimate combined generation and Parquet-read cost for deterministic LPT."""

    identity = rank_worker.unit_identity(unit)
    rows = int(identity["selected_rows"])
    source_bytes = int(identity["source"]["bytes"])
    return rows * REBALANCE_ROW_WEIGHT_BYTES + source_bytes + REBALANCE_UNIT_OVERHEAD_BYTES


def balance_pending_units(
    units: Sequence[Mapping[str, Any]],
    physical_ranks: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    """Assign pending units to local GPUs with deterministic longest-processing-time packing."""

    if not physical_ranks or len(set(physical_ranks)) != len(physical_ranks):
        raise ValueError("physical ranks must be a non-empty unique sequence")
    lanes = {rank: [] for rank in physical_ranks}
    loads = {rank: 0 for rank in physical_ranks}
    seen: set[int] = set()
    ordered = sorted(
        (dict(unit) for unit in units),
        key=lambda unit: (-unit_work_estimate(unit), int(unit["global_unit_index"])),
    )
    for unit in ordered:
        index = int(unit["global_unit_index"])
        if index in seen:
            raise ValueError(f"duplicate pending global_unit_index: {index}")
        seen.add(index)
        owner = min(
            physical_ranks,
            key=lambda rank: (loads[rank], len(lanes[rank]), rank),
        )
        lanes[owner].append(unit)
        loads[owner] += unit_work_estimate(unit)
    for lane in lanes.values():
        lane.sort(key=lambda unit: int(unit["global_unit_index"]))
    return lanes


def _unit_binding(
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    return rank_worker.unit_binding(
        unit,
        source_catalog_sha256=catalogs.source_sha256,
        unit_catalog_sha256=catalogs.unit_sha256,
        checkpoint_step=args.checkpoint_step,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        hf_artifact_manifest_sha256=args.hf_artifact_manifest_sha256,
        generation=worker_generation_contract(args),
    )


def pending_node_units(
    client: Any,
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
) -> list[dict[str, Any]]:
    """Validate committed markers concurrently and return only unfinished node units."""

    units = _node_expected_units(args, catalogs, node_index=args.node_index)
    selection_root = effective_output_s3(args, catalogs)

    def is_pending(unit: Mapping[str, Any]) -> bool:
        identity = rank_worker.unit_identity(unit)
        output_uri, marker_uri = rank_worker.output_uris(
            selection_root,
            identity["dataset_id"],
            identity["dataset_unit_index"],
        )
        resumed = rank_worker.load_existing_marker(
            client,
            marker_uri=marker_uri,
            expected_binding=_unit_binding(args, catalogs, unit),
            expected_output_uri=output_uri,
        )
        return resumed is None

    workers = min(
        int(getattr(args, "rebalance_scan_workers", DEFAULT_REBALANCE_SCAN_WORKERS)),
        max(1, len(units)),
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sidecar24-resume-scan") as executor:
        pending_flags = list(executor.map(is_pending, units))
    return [dict(unit) for unit, pending in zip(units, pending_flags, strict=True) if pending]


def write_rebalanced_assignments(
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    pending: Sequence[Mapping[str, Any]],
) -> dict[int, Path]:
    ranks = node_global_ranks(args, args.node_index)
    lanes = balance_pending_units(pending, ranks)
    plan = build_partition_plan(args, catalogs)
    root = args.work_dir / "rebalanced-assignments"
    paths: dict[int, Path] = {}
    for rank in ranks:
        units = lanes[rank]
        if not units:
            continue
        path = root / f"physical-rank-{rank:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes(
                {
                    "format_version": rank_worker.REBALANCED_ASSIGNMENT_FORMAT,
                    "physical_global_rank": rank,
                    "selection_sha256": plan.contract["selection_sha256"],
                    "global_unit_indices": [
                        int(unit["global_unit_index"]) for unit in units
                    ],
                }
            )
        )
        paths[rank] = path
    return paths


def load_external_repartition_assignments(
    client: Any,
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
) -> tuple[dict[str, Any], dict[int, Path], str]:
    """Load one immutable physical-node assignment and render rank-worker inputs."""

    assignment_uri = join_s3(
        args.external_repartition_prefix,
        "assignments",
        f"{node_name(args.node_index, args.node_count)}.json",
    )
    value = read_s3_json(client, assignment_uri)
    if value is None:  # pragma: no cover - read_s3_json only returns None with missing_ok
        raise FileNotFoundError(assignment_uri)
    expected_fields = {
        "format_version",
        "repartition_id",
        "created_at",
        "snapshot_sha256",
        "node_index",
        "node_count",
        "local_world_size",
        "global_rank_offset",
        "logical_world_size",
        "selection_sha256",
        "source_catalog_sha256",
        "unit_catalog_sha256",
        "assignments",
    }
    _require_equal(set(value), expected_fields, "external repartition assignment fields")
    _require_equal(
        value["format_version"],
        REPARTITION_NODE_ASSIGNMENT_FORMAT,
        "external repartition assignment format",
    )
    _require_equal(value["repartition_id"], args.repartition_id, "repartition ID")
    if not isinstance(value["created_at"], str) or not value["created_at"]:
        raise ValueError("external repartition assignment has no created_at")
    if SHA256_RE.fullmatch(str(value["snapshot_sha256"])) is None:
        raise ValueError("external repartition snapshot SHA-256 is invalid")
    _require_equal(value["node_index"], args.node_index, "repartition node index")
    _require_equal(value["node_count"], args.node_count, "repartition node count")
    _require_equal(
        value["local_world_size"], args.local_world_size, "repartition local world size"
    )
    _require_equal(
        value["global_rank_offset"],
        node_global_rank_offset(args, args.node_index),
        "repartition global rank offset",
    )
    _require_equal(
        value["logical_world_size"], global_world_size(args), "repartition logical world size"
    )
    plan = build_partition_plan(args, catalogs)
    _require_equal(
        value["selection_sha256"],
        plan.contract["selection_sha256"],
        "repartition selection SHA-256",
    )
    _require_equal(
        value["source_catalog_sha256"], catalogs.source_sha256, "repartition source catalog"
    )
    _require_equal(
        value["unit_catalog_sha256"], catalogs.unit_sha256, "repartition unit catalog"
    )

    assignments = value["assignments"]
    expected_ranks = node_global_ranks(args, args.node_index)
    if not isinstance(assignments, list) or len(assignments) != len(expected_ranks):
        raise ValueError("external repartition assignments do not cover every local GPU")
    unit_count = len(catalogs.units["units"])
    seen: set[int] = set()
    root = args.work_dir / "external-repartition-assignments"
    paths: dict[int, Path] = {}
    for lane, expected_rank in zip(assignments, expected_ranks, strict=True):
        if not isinstance(lane, dict) or set(lane) != {
            "physical_global_rank",
            "global_unit_indices",
        }:
            raise ValueError("external repartition rank assignment has an unexpected schema")
        _require_equal(
            lane["physical_global_rank"], expected_rank, "repartition physical rank"
        )
        indices = lane["global_unit_indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(index) is not int or index < 0 or index >= unit_count for index in indices)
            or len(indices) != len(set(indices))
        ):
            raise ValueError(f"external repartition rank {expected_rank} indices are invalid")
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(
                f"external repartition node repeats unit indices: {sorted(overlap)[:8]}"
            )
        seen.update(indices)
        path = root / f"physical-rank-{expected_rank:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes(
                {
                    "format_version": rank_worker.REBALANCED_ASSIGNMENT_FORMAT,
                    "physical_global_rank": expected_rank,
                    "selection_sha256": value["selection_sha256"],
                    "global_unit_indices": indices,
                }
            )
        )
        paths[expected_rank] = path
    return value, paths, assignment_uri


def _load_operational_worker_summary(
    path: Path,
    *,
    rank: int,
    assigned_indices: Sequence[int],
    selection_sha256: str,
    logical_world_size: int,
    units_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "format_version",
        "status",
        "started_at",
        "completed_at",
        "physical_global_rank",
        "global_world_size",
        "selection_sha256",
        "assigned_units",
        "generated_units",
        "resumed_units",
        "global_unit_indices",
        "totals",
    }
    _require_equal(set(value), expected_fields, f"repartition rank {rank} summary fields")
    _require_equal(
        value["format_version"],
        rank_worker.REBALANCED_WORKER_SUMMARY_FORMAT,
        f"repartition rank {rank} summary format",
    )
    _require_equal(value["status"], "complete", f"repartition rank {rank} status")
    _require_equal(value["physical_global_rank"], rank, "repartition summary rank")
    _require_equal(
        value["global_world_size"], logical_world_size, "repartition summary logical world"
    )
    _require_equal(value["selection_sha256"], selection_sha256, "repartition summary selection")
    _require_equal(
        value["global_unit_indices"], list(assigned_indices), "repartition summary unit coverage"
    )
    _require_equal(value["assigned_units"], len(assigned_indices), "repartition assigned units")
    generated = value["generated_units"]
    resumed = value["resumed_units"]
    if type(generated) is not int or generated < 0 or type(resumed) is not int or resumed < 0:
        raise ValueError(f"repartition rank {rank} generated/resumed counts are invalid")
    _require_equal(generated + resumed, len(assigned_indices), "repartition unit dispositions")
    totals = value["totals"]
    if not isinstance(totals, dict) or set(totals) != OUTPUT_COUNT_FIELDS | {"bytes"}:
        raise ValueError(f"repartition rank {rank} totals have an unexpected schema")
    for key, count in totals.items():
        if type(count) is not int or count < 0:
            raise ValueError(f"repartition rank {rank} totals.{key} is invalid")
    expected_rows = sum(
        int(units_by_index[index]["global_row_end_exclusive"])
        - int(units_by_index[index]["global_row_start"])
        for index in assigned_indices
    )
    _require_equal(totals["rows"], expected_rows, "repartition summary rows")
    _require_equal(totals["inference_errors"], 0, "repartition inference errors")
    return value


def build_repartition_node_result(
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    *,
    assignment: Mapping[str, Any],
    assignment_uri: str,
    worker_summaries: Sequence[Mapping[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    totals = {key: 0 for key in sorted(OUTPUT_COUNT_FIELDS | {"bytes"})}
    for summary in worker_summaries:
        for key in totals:
            totals[key] += int(summary["totals"][key])
    return {
        "format_version": REPARTITION_NODE_RESULT_FORMAT,
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "repartition_id": args.repartition_id,
        "snapshot_sha256": assignment["snapshot_sha256"],
        "node_index": args.node_index,
        "node_count": args.node_count,
        "global_ranks": node_global_ranks(args, args.node_index),
        "logical_world_size": global_world_size(args),
        "selection_sha256": assignment["selection_sha256"],
        "source_catalog_sha256": catalogs.source_sha256,
        "unit_catalog_sha256": catalogs.unit_sha256,
        "assignment_uri": assignment_uri,
        "assignment_sha256": sha256_bytes(canonical_json_bytes(assignment)),
        "assigned_units": sum(int(summary["assigned_units"]) for summary in worker_summaries),
        "generated_units": sum(int(summary["generated_units"]) for summary in worker_summaries),
        "resumed_units": sum(int(summary["resumed_units"]) for summary in worker_summaries),
        "totals": totals,
        "worker_summaries": [dict(summary) for summary in worker_summaries],
    }


def effective_output_s3(args: argparse.Namespace, catalogs: BoundCatalogs) -> str:
    # Resolve the full plan as an early selection-capacity guard, then use the
    # basic contract for the stable namespace (the digest remains in markers).
    # A few process-launch unit tests intentionally use an empty catalog seam.
    if catalogs.units.get("units"):
        build_partition_plan(args, catalogs)
    return configured_output_s3(args)


def configured_output_s3(args: argparse.Namespace) -> str:
    contract = partition_module.basic_contract(
        partition_count=int(getattr(args, "partition_count", 1)),
        partition_index=int(getattr(args, "partition_index", 0)),
        global_world_size=global_world_size(args),
        case_limit=getattr(args, "case_limit", None),
    )
    return partition_module.selection_output_root(
        args.output_s3,
        contract,
    )


def _catalog_summary_contract(
    catalogs: BoundCatalogs,
    *,
    source_uri: str,
    unit_uri: str,
) -> dict[str, Any]:
    return rank_worker.catalog_bindings(
        source_catalog_uri=source_uri,
        source_catalog=catalogs.source,
        source_catalog_sha256=catalogs.source_sha256,
        unit_catalog_uri=unit_uri,
        unit_catalog=catalogs.units,
        unit_catalog_sha256=catalogs.unit_sha256,
    )


def worker_command(
    args: argparse.Namespace,
    *,
    catalogs: BoundCatalogs,
    global_rank: int,
    assigned_unit_indices: Path | None = None,
    defer_rank_summary: bool = False,
) -> list[str]:
    if (assigned_unit_indices is None) != (not defer_rank_summary):
        raise ValueError(
            "assigned_unit_indices and defer_rank_summary must be supplied together"
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/infer_image_edit_full_json_sidecar24_rank.py"),
        "--model",
        str(args.model),
        "--target-schema",
        TARGET_SCHEMA,
        "--source-catalog",
        str(catalogs.source_path),
        "--source-catalog-uri",
        args.source_catalog_uri,
        "--source-catalog-sha256",
        catalogs.source_sha256,
        "--unit-catalog",
        str(catalogs.unit_path),
        "--unit-catalog-uri",
        args.unit_catalog_uri,
        "--unit-catalog-sha256",
        catalogs.unit_sha256,
        "--output-s3",
        args.output_s3,
        "--work-dir",
        str(args.work_dir),
        "--global-rank",
        str(global_rank),
        "--global-world-size",
        str(global_world_size(args)),
        "--checkpoint-step",
        str(args.checkpoint_step),
        "--checkpoint-manifest-sha256",
        args.checkpoint_manifest_sha256,
        "--hf-artifact-manifest-sha256",
        args.hf_artifact_manifest_sha256,
        "--partition-count",
        str(int(getattr(args, "partition_count", 1))),
        "--partition-index",
        str(int(getattr(args, "partition_index", 0))),
        "--tensor-parallel-size",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--request-window-size",
        str(args.request_window_size),
        "--download-workers",
        str(args.download_workers),
        "--unit-read-workers",
        str(getattr(args, "unit_read_workers", rank_worker.DEFAULT_UNIT_READ_WORKERS)),
        "--unit-prefetch-depth",
        str(getattr(args, "unit_prefetch_depth", rank_worker.DEFAULT_UNIT_PREFETCH_DEPTH)),
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
        "--aws-region",
        args.aws_region,
    ]
    if getattr(args, "case_limit", None) is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    if assigned_unit_indices is not None:
        command.extend(["--assigned-unit-indices", str(assigned_unit_indices)])
    if defer_rank_summary:
        command.append("--defer-rank-summary")
    if getattr(args, "package_v6_bboxes_yxyx", False):
        command.extend(
            [
                "--package-v6-bboxes-yxyx",
                "--expected-hld-postprocess-version",
                str(args.expected_hld_postprocess_version),
            ]
        )
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    if max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    mm_processor_cache_gb = getattr(args, "mm_processor_cache_gb", None)
    if mm_processor_cache_gb is not None:
        command.extend(["--mm-processor-cache-gb", str(mm_processor_cache_gb)])
    max_cudagraph_capture_size = getattr(args, "max_cudagraph_capture_size", None)
    if max_cudagraph_capture_size is not None:
        command.extend(["--max-cudagraph-capture-size", str(max_cudagraph_capture_size)])
    command.extend(
        [
            "--max-pending-unit-writes",
            str(
                getattr(
                    args,
                    "max_pending_unit_writes",
                    rank_worker.DEFAULT_MAX_PENDING_UNIT_WRITES,
                )
            ),
        ]
    )
    command.extend(
        [
            "--unit-upload-workers",
            str(
                getattr(
                    args,
                    "unit_upload_workers",
                    rank_worker.DEFAULT_UNIT_UPLOAD_WORKERS,
                )
            ),
        ]
    )
    for option, value in (
        ("--gdn-prefill-backend", getattr(args, "gdn_prefill_backend", None)),
        ("--mm-encoder-attn-backend", getattr(args, "mm_encoder_attn_backend", None)),
        ("--cudagraph-mode", getattr(args, "cudagraph_mode", None)),
        (
            "--speculative-attention-backend",
            getattr(args, "speculative_attention_backend", None),
        ),
        ("--expected-torch-version", getattr(args, "expected_torch_version", None)),
        (
            "--expected-transformers-version",
            getattr(args, "expected_transformers_version", None),
        ),
        ("--expected-vllm-version", getattr(args, "expected_vllm_version", None)),
        (
            "--expected-flashinfer-version",
            getattr(args, "expected_flashinfer_version", None),
        ),
        (
            "--expected-flashinfer-cubin-version",
            getattr(args, "expected_flashinfer_cubin_version", None),
        ),
        (
            "--expected-flashinfer-jit-cache-version",
            getattr(args, "expected_flashinfer_jit_cache_version", None),
        ),
        ("--expected-triton-version", getattr(args, "expected_triton_version", None)),
        ("--expected-cuda-version", getattr(args, "expected_cuda_version", None)),
    ):
        if value:
            command.extend([option, value])
    if args.aws_profile:
        command.extend(["--aws-profile", args.aws_profile])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.use_mtp:
        command.append("--use-mtp")
    if args.prefetch_images:
        command.append("--prefetch-images")
    if getattr(args, "cross_unit_streaming", False):
        command.append("--cross-unit-streaming")
    if getattr(args, "cudagraph_mm_encoder", False):
        command.append("--cudagraph-mm-encoder")
    if getattr(args, "cudagraph_specialize_lora", None) is True:
        command.append("--cudagraph-specialize-lora")
    if getattr(args, "cudagraph_specialize_lora", None) is False:
        command.append("--no-cudagraph-specialize-lora")
    if getattr(args, "enable_tower_connector_lora", None) is True:
        command.append("--enable-tower-connector-lora")
    if getattr(args, "enable_tower_connector_lora", None) is False:
        command.append("--no-enable-tower-connector-lora")
    if getattr(args, "enable_prefix_caching", None) is True:
        command.append("--enable-prefix-caching")
    if getattr(args, "enable_prefix_caching", None) is False:
        command.append("--no-enable-prefix-caching")
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
            process.wait()


def _validate_output_counts(output: Mapping[str, Any], *, expected_rows: int) -> dict[str, int]:
    counts = output.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != OUTPUT_COUNT_FIELDS:
        raise ValueError("shard output counts have an unexpected schema")
    normalized: dict[str, int] = {}
    for key in sorted(OUTPUT_COUNT_FIELDS):
        value = counts[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"shard output counts.{key} is invalid")
        normalized[key] = value
    _require_equal(output.get("rows"), expected_rows, "shard output rows")
    _require_equal(normalized["rows"], expected_rows, "shard output/count rows")
    if not normalized["schema_valid"] <= normalized["json_valid"] <= normalized["rows"]:
        raise ValueError("shard JSON/schema counts are inconsistent")
    if normalized["failed"] != normalized["rows"] - normalized["schema_valid"]:
        raise ValueError("shard failure count is inconsistent")
    if normalized["inference_errors"] > normalized["failed"]:
        raise ValueError("shard inference error count is inconsistent")
    if normalized["inference_errors"] != 0:
        raise ValueError("a successful shard cannot contain inference errors")
    return normalized


def _aggregate_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    include_units: bool,
) -> dict[str, int]:
    totals = {
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
        rows = shard.get("global_row_end_exclusive", -1) - shard.get("global_row_start", 0)
        counts = _validate_output_counts(output, expected_rows=rows)
        for key in ("rows", "json_valid", "schema_valid", "failed", "inference_errors"):
            totals[key] += counts[key]
        size = output.get("bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("shard output byte count is invalid")
        totals["bytes"] += size
    if include_units:
        return {"units": len(shards), **totals}
    return totals


def _validate_shard_against_unit(
    shard: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    output_s3: str,
) -> None:
    expected_identity = rank_worker.unit_identity(unit)
    _require_equal(
        set(shard),
        set(rank_worker.COMPACT_SHARD_FIELDS),
        "compact shard fields",
    )
    for field in (
        "global_unit_index",
        "dataset_id",
        "dataset_unit_index",
        "source",
        "local_row_ranges",
        "dataset_row_start",
        "dataset_row_end_exclusive",
        "global_row_start",
        "global_row_end_exclusive",
    ):
        _require_equal(shard.get(field), expected_identity[field], f"shard {field}")

    output = shard.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "uri",
        "sha256",
        "bytes",
        "rows",
        "counts",
    }:
        raise ValueError("shard output has an unexpected schema")
    output_uri, marker_uri = rank_worker.output_uris(
        output_s3,
        expected_identity["dataset_id"],
        expected_identity["dataset_unit_index"],
    )
    _require_equal(output.get("uri"), output_uri, "shard output URI")
    _require_equal(shard.get("success_marker_uri"), marker_uri, "shard marker URI")
    if SHA256_RE.fullmatch(str(output.get("sha256", ""))) is None:
        raise ValueError("shard output SHA-256 is invalid")
    expected_rows = expected_identity["selected_rows"]
    _validate_output_counts(output, expected_rows=expected_rows)
    if type(output.get("bytes")) is not int or output["bytes"] <= 0:
        raise ValueError("shard output byte count is invalid")
    if shard.get("disposition") not in {"generated", "resumed"}:
        raise ValueError("shard disposition is invalid")


def _load_rank_summary(
    path: Path,
    *,
    global_rank: int,
    expected_units: Sequence[Mapping[str, Any]],
    catalogs: BoundCatalogs,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selection_root = effective_output_s3(args, catalogs)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"rank {global_rank} summary is not an object")
    _require_equal(
        set(value),
        rank_worker.RANK_SUMMARY_FIELDS,
        f"rank {global_rank} summary fields",
    )
    _require_equal(
        value.get("format_version"),
        rank_worker.RANK_SUMMARY_FORMAT,
        f"rank {global_rank} summary format",
    )
    _require_equal(value.get("status"), "complete", f"rank {global_rank} status")
    for field in ("started_at", "completed_at"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"rank {global_rank} summary has no {field}")
    _require_equal(value.get("global_rank"), global_rank, f"rank {global_rank} identity")
    _require_equal(
        value.get("global_world_size"),
        global_world_size(args),
        f"rank {global_rank} world size",
    )
    _require_equal(
        value.get("catalogs"),
        _catalog_summary_contract(
            catalogs,
            source_uri=args.source_catalog_uri,
            unit_uri=args.unit_catalog_uri,
        ),
        f"rank {global_rank} catalogs",
    )
    _require_equal(
        value.get("checkpoint"),
        {
            "step": args.checkpoint_step,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        f"rank {global_rank} checkpoint",
    )
    _require_equal(
        value.get("hf_artifact"),
        {"manifest_sha256": args.hf_artifact_manifest_sha256},
        f"rank {global_rank} HF artifact",
    )
    _require_equal(
        value.get("prompt"),
        {"version": PROMPT_VERSION, "contract_sha256": PROMPT_CONTRACT_SHA256},
        f"rank {global_rank} prompt",
    )
    _require_equal(
        value.get("generation"),
        worker_generation_contract(args),
        f"rank {global_rank} generation",
    )
    shards = value.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"rank {global_rank} summary has no shards list")
    expected_indices = [int(unit["global_unit_index"]) for unit in expected_units]
    _require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        expected_indices,
        f"rank {global_rank} unit coverage",
    )
    expected_by_index = {int(unit["global_unit_index"]): unit for unit in expected_units}
    for shard in shards:
        _validate_shard_against_unit(
            shard,
            expected_by_index[int(shard["global_unit_index"])],
            output_s3=selection_root,
        )
    _require_equal(
        value.get("totals"),
        _aggregate_shards(shards, include_units=False),
        f"rank {global_rank} totals",
    )
    _require_equal(value.get("assigned_units"), len(expected_units), "rank assigned units")
    generated = value.get("generated_units")
    resumed = value.get("resumed_units")
    if type(generated) is not int or generated < 0:
        raise ValueError(f"rank {global_rank} generated_units is invalid")
    if type(resumed) is not int or resumed < 0:
        raise ValueError(f"rank {global_rank} resumed_units is invalid")
    _require_equal(generated + resumed, len(expected_units), "rank generated/resumed units")
    _require_equal(
        sum(shard["disposition"] == "generated" for shard in shards),
        generated,
        f"rank {global_rank} generated dispositions",
    )
    _require_equal(
        sum(shard["disposition"] == "resumed" for shard in shards),
        resumed,
        f"rank {global_rank} resumed dispositions",
    )
    if type(value.get("runtime_initialized")) is not bool:
        raise ValueError(f"rank {global_rank} runtime_initialized is invalid")
    _require_equal(
        value["runtime_initialized"],
        generated > 0,
        f"rank {global_rank} runtime initialization",
    )
    _require_equal(
        value.get("summary_s3_uri"),
        rank_worker.rank_summary_uri(selection_root, global_rank),
        f"rank {global_rank} summary URI",
    )
    return value


def _node_expected_units(
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    *,
    node_index: int,
) -> list[dict[str, Any]]:
    ranks = node_global_ranks(args, node_index)
    plan = build_partition_plan(args, catalogs)
    values = [
        dict(unit)
        for rank in ranks
        for unit in plan.units_by_rank[rank]
    ]
    return sorted(values, key=lambda unit: int(unit["global_unit_index"]))


def build_node_marker(
    args: argparse.Namespace,
    *,
    catalogs: BoundCatalogs,
    rank_summaries: Sequence[Mapping[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    shards = sorted(
        (dict(shard) for summary in rank_summaries for shard in summary["shards"]),
        key=lambda shard: int(shard["global_unit_index"]),
    )
    expected_units = _node_expected_units(
        args,
        catalogs,
        node_index=args.node_index,
    )
    expected_indices = [int(unit["global_unit_index"]) for unit in expected_units]
    _require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        expected_indices,
        "node unit coverage",
    )
    selection_root = effective_output_s3(args, catalogs)
    for shard, unit in zip(shards, expected_units, strict=True):
        _validate_shard_against_unit(shard, unit, output_s3=selection_root)
    local_ranks = node_global_ranks(args, args.node_index)
    return {
        "format_version": NODE_FORMAT,
        "status": "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "node_index": args.node_index,
        "node_count": args.node_count,
        "global_ranks": local_ranks,
        "contract": {
            **static_contract(args, catalogs),
            "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
        },
        "totals": _aggregate_shards(shards, include_units=True),
        "rank_summaries": [
            {
                key: summary[key]
                for key in (
                    "global_rank",
                    "summary_s3_uri",
                    "assigned_units",
                    "generated_units",
                    "resumed_units",
                    "totals",
                )
            }
            for summary in rank_summaries
        ],
        "shards": shards,
    }


def validate_node_marker(
    marker: Mapping[str, Any],
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    *,
    node_index: int,
) -> None:
    _require_equal(set(marker), NODE_MARKER_FIELDS, "node marker fields")
    _require_equal(marker.get("format_version"), NODE_FORMAT, "node marker format")
    _require_equal(marker.get("status"), "complete", "node marker status")
    _require_equal(marker.get("node_index"), node_index, "node marker node_index")
    _require_equal(marker.get("node_count"), args.node_count, "node marker node_count")
    for field in ("started_at", "completed_at"):
        if not isinstance(marker.get(field), str) or not marker[field]:
            raise ValueError(f"node marker has no {field}")
    expected_ranks = node_global_ranks(args, node_index)
    _require_equal(marker.get("global_ranks"), expected_ranks, "node marker ranks")
    expected_contract = {
        **static_contract(args, catalogs),
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
    }
    _require_equal(marker.get("contract"), expected_contract, "node marker contract")
    shards = marker.get("shards")
    if not isinstance(shards, list):
        raise ValueError("node marker has no shards list")
    expected_units = _node_expected_units(args, catalogs, node_index=node_index)
    _require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        [int(unit["global_unit_index"]) for unit in expected_units],
        "node marker unit coverage",
    )
    selection_root = effective_output_s3(args, catalogs)
    for shard, unit in zip(shards, expected_units, strict=True):
        _validate_shard_against_unit(shard, unit, output_s3=selection_root)
    _require_equal(
        marker.get("totals"),
        _aggregate_shards(shards, include_units=True),
        "node marker totals",
    )
    summaries = marker.get("rank_summaries")
    if not isinstance(summaries, list) or len(summaries) != args.local_world_size:
        raise ValueError("node marker rank_summaries is incomplete")
    _require_equal(
        [summary.get("global_rank") for summary in summaries],
        expected_ranks,
        "node marker rank summary identities",
    )
    plan = build_partition_plan(args, catalogs)
    for summary, rank in zip(summaries, expected_ranks, strict=True):
        if set(summary) != {
            "global_rank",
            "summary_s3_uri",
            "assigned_units",
            "generated_units",
            "resumed_units",
            "totals",
        }:
            raise ValueError(f"node marker rank {rank} summary has unexpected fields")
        rank_units = [dict(unit) for unit in plan.units_by_rank[rank]]
        rank_unit_indices = {int(unit["global_unit_index"]) for unit in rank_units}
        rank_shards = [
            shard
            for shard in shards
            if int(shard["global_unit_index"]) in rank_unit_indices
        ]
        _require_equal(summary["assigned_units"], len(rank_units), f"rank {rank} assigned units")
        generated = summary["generated_units"]
        resumed = summary["resumed_units"]
        if type(generated) is not int or generated < 0:
            raise ValueError(f"node marker rank {rank} generated_units is invalid")
        if type(resumed) is not int or resumed < 0:
            raise ValueError(f"node marker rank {rank} resumed_units is invalid")
        _require_equal(generated + resumed, len(rank_units), f"rank {rank} unit dispositions")
        _require_equal(
            sum(shard["disposition"] == "generated" for shard in rank_shards),
            generated,
            f"rank {rank} generated dispositions",
        )
        _require_equal(
            sum(shard["disposition"] == "resumed" for shard in rank_shards),
            resumed,
            f"rank {rank} resumed dispositions",
        )
        _require_equal(
            summary["summary_s3_uri"],
            rank_worker.rank_summary_uri(selection_root, rank),
            f"rank {rank} summary URI",
        )
        _require_equal(
            summary["totals"],
            _aggregate_shards(rank_shards, include_units=False),
            f"rank {rank} compact totals",
        )


def check_node_complete(
    client: Any,
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
) -> bool:
    selection_root = effective_output_s3(args, catalogs)
    uri = join_s3(node_prefix(selection_root, args.node_index, args.node_count), "_SUCCESS")
    marker = read_s3_json(client, uri, missing_ok=True)
    if marker is None:
        return False
    validate_node_marker(
        marker,
        args,
        catalogs,
        node_index=args.node_index,
    )
    return True


def _dataset_result_summaries(
    unit_catalog: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    expected_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = {
        str(summary["dataset_id"]): [] for summary in unit_catalog["datasets"]
    }
    for shard in shards:
        by_dataset[str(shard["dataset_id"])].append(shard)
    expected_by_dataset: dict[str, list[Mapping[str, Any]]] = {
        str(summary["dataset_id"]): [] for summary in unit_catalog["datasets"]
    }
    for unit in expected_units:
        expected_by_dataset[str(unit["dataset_id"])].append(unit)

    results = []
    for source_summary in unit_catalog["datasets"]:
        dataset_id = str(source_summary["dataset_id"])
        dataset_shards = by_dataset[dataset_id]
        dataset_units = expected_by_dataset[dataset_id]
        totals = _aggregate_shards(dataset_shards, include_units=True)
        selected_rows = sum(
            int(unit["global_row_end_exclusive"]) - int(unit["global_row_start"])
            for unit in dataset_units
        )
        _require_equal(totals["units"], len(dataset_units), f"{dataset_id} units")
        _require_equal(totals["rows"], selected_rows, f"{dataset_id} rows")
        results.append(
            {
                "dataset_id": dataset_id,
                "dataset_ordinal": source_summary["dataset_ordinal"],
                "family": source_summary["family"],
                "adapter": source_summary["adapter"],
                "selected_rows": selected_rows,
                "unit_count": len(dataset_units),
                "main_inventory_sha256": source_summary["main_inventory_sha256"],
                "sidecar_inventory_sha256": source_summary["sidecar_inventory_sha256"],
                "totals": totals,
            }
        )
    return results


def maybe_finalize(
    client: Any,
    args: argparse.Namespace,
    *,
    catalogs: BoundCatalogs,
) -> dict[str, Any] | None:
    selection_root = effective_output_s3(args, catalogs)
    plan = build_partition_plan(args, catalogs)
    node_markers: list[dict[str, Any]] = []
    for index in range(args.node_count):
        uri = join_s3(node_prefix(selection_root, index, args.node_count), "_SUCCESS")
        marker = read_s3_json(client, uri, missing_ok=True)
        if marker is None:
            return None
        validate_node_marker(marker, args, catalogs, node_index=index)
        node_markers.append(marker)

    shards = sorted(
        (dict(shard) for marker in node_markers for shard in marker["shards"]),
        key=lambda shard: int(shard["global_unit_index"]),
    )
    units = plan.units
    _require_equal(len(shards), len(units), "partition shard count")
    _require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        [int(unit["global_unit_index"]) for unit in units],
        "partition unit coverage",
    )
    for shard, unit in zip(shards, units, strict=True):
        _validate_shard_against_unit(shard, unit, output_s3=selection_root)

    totals = _aggregate_shards(shards, include_units=True)
    _require_equal(totals["rows"], plan.contract["selected_rows"], "partition result rows")
    dataset_summaries = _dataset_result_summaries(catalogs.units, shards, units)
    expected_datasets = expected_dataset_count(catalogs.source)
    _require_equal(len(dataset_summaries), expected_datasets, "result dataset count")
    _require_equal(
        sum(summary["selected_rows"] for summary in dataset_summaries),
        plan.contract["selected_rows"],
        "dataset result row total",
    )
    _require_equal(
        sum(summary["unit_count"] for summary in dataset_summaries),
        len(units),
        "dataset result unit total",
    )

    completed_at = max(str(marker["completed_at"]) for marker in node_markers)
    manifest = {
        "format_version": RESULT_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "contract": {
            **static_contract(args, catalogs),
            "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
        },
        "totals": totals,
        "datasets": dataset_summaries,
        "nodes": [
            {
                "node_index": int(marker["node_index"]),
                "success_uri": join_s3(
                    node_prefix(
                        selection_root,
                        int(marker["node_index"]),
                        args.node_count,
                    ),
                    "_SUCCESS",
                ),
                "totals": marker["totals"],
            }
            for marker in node_markers
        ],
        "shards": shards,
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_uri = join_s3(selection_root, "manifest.json")
    manifest_digest = upload_bytes(
        client,
        manifest_payload,
        manifest_uri,
        content_type="application/json",
    )
    success = {
        "format_version": ROOT_SUCCESS_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "contract": {
            **static_contract(args, catalogs),
            "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
        },
        "source_catalog_uri": args.source_catalog_uri,
        "source_catalog_sha256": catalogs.source_sha256,
        "unit_catalog_uri": args.unit_catalog_uri,
        "unit_catalog_sha256": catalogs.unit_sha256,
        "datasets": expected_datasets,
        "units": totals["units"],
        "rows": totals["rows"],
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
    }
    upload_json(client, success, join_s3(selection_root, "_SUCCESS"))
    return manifest


def _upload_rank_log(
    client: Any,
    args: argparse.Namespace,
    global_rank: int,
    *,
    filename: str = "worker.log",
) -> None:
    log_path = args.work_dir / f"rank-{global_rank:03d}" / filename
    if not log_path.is_file():
        raise FileNotFoundError(f"rank {global_rank} worker log is missing: {log_path}")
    if external_repartition_enabled(args):
        destination = join_s3(
            external_repartition_node_prefix(args),
            "ranks",
            f"rank-{global_rank:03d}",
            filename,
        )
    else:
        destination = join_s3(
            configured_output_s3(args), "ranks", f"rank-{global_rank:03d}", filename
        )
    upload_file(
        client,
        log_path,
        destination,
        content_type="text/plain",
    )


def _worker_environment(
    args: argparse.Namespace,
    *,
    gpu: str,
    local_rank: int,
    global_rank: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "VLLM_PORT": str(vllm_port_base(local_rank, args.local_world_size)),
        }
    )
    flashinfer_workspace_base = environment.get("FLASHINFER_WORKSPACE_BASE")
    if flashinfer_workspace_base:
        flashinfer_workspace = Path(flashinfer_workspace_base) / f"rank-{global_rank:03d}"
        flashinfer_workspace.mkdir(parents=True, exist_ok=True)
        environment["FLASHINFER_WORKSPACE_BASE"] = str(flashinfer_workspace)
    return environment


def _run_worker_phase(
    client: Any,
    args: argparse.Namespace,
    catalogs: BoundCatalogs,
    gpu_tokens: Sequence[str],
    *,
    assignment_paths: Mapping[int, Path] | None,
    log_filename: str,
) -> list[int]:
    """Run one fail-fast local worker phase and durably publish every log."""

    global_rank_offset = node_global_rank_offset(args, args.node_index)
    processes: dict[int, subprocess.Popen[Any]] = {}
    handles: dict[int, Any] = {}
    uploaded_logs: set[int] = set()
    for local_rank, gpu in enumerate(gpu_tokens):
        global_rank = global_rank_offset + local_rank
        if assignment_paths is not None and global_rank not in assignment_paths:
            continue
        rank_dir = args.work_dir / f"rank-{global_rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (rank_dir / log_filename).open("w", encoding="utf-8")
        assignment_path = (
            assignment_paths.get(global_rank) if assignment_paths is not None else None
        )
        processes[global_rank] = subprocess.Popen(
            worker_command(
                args,
                catalogs=catalogs,
                global_rank=global_rank,
                assigned_unit_indices=assignment_path,
                defer_rank_summary=assignment_path is not None,
            ),
            cwd=REPO_ROOT,
            env=_worker_environment(
                args,
                gpu=gpu,
                local_rank=local_rank,
                global_rank=global_rank,
            ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        handles[global_rank] = log_handle

    remaining = set(processes)
    failure: RuntimeError | None = None
    try:
        while remaining:
            for rank in sorted(remaining):
                returncode = processes[rank].poll()
                if returncode is None:
                    continue
                remaining.remove(rank)
                handles[rank].close()
                _upload_rank_log(client, args, rank, filename=log_filename)
                uploaded_logs.add(rank)
                if returncode != 0 and failure is None:
                    failure = RuntimeError(f"global rank {rank} exited with code {returncode}")
            if failure is not None:
                _terminate_processes(processes.values())
                break
            if remaining:
                time.sleep(2)
    finally:
        for rank, process in processes.items():
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not handles[rank].closed:
                handles[rank].close()
            if rank not in uploaded_logs:
                _upload_rank_log(client, args, rank, filename=log_filename)
                uploaded_logs.add(rank)
    if failure is not None:
        raise failure
    return sorted(processes)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    client = s3_client()
    catalogs = load_bound_catalogs(args)
    selection_root = effective_output_s3(args, catalogs)
    repartition_mode = external_repartition_enabled(args)
    if not repartition_mode and check_node_complete(client, args, catalogs):
        manifest = (
            None
            if getattr(args, "defer_global_finalize", False)
            else maybe_finalize(client, args, catalogs=catalogs)
        )
        return {
            "status": "already_complete",
            "node_index": args.node_index,
            "root_finalized": manifest is not None,
        }

    args.work_dir.mkdir(parents=True, exist_ok=True)
    catalogs = write_canonical_catalogs(catalogs, args.work_dir)
    prefix = (
        external_repartition_node_prefix(args)
        if repartition_mode
        else node_prefix(selection_root, args.node_index, args.node_count)
    )
    status_uri = join_s3(prefix, "status.json")
    upload_json(
        client,
        {"status": "preparing", "started_at": started_at, "node_index": args.node_index},
        status_uri,
    )
    if not repartition_mode:
        upload_bytes(
            client,
            catalogs.source_path.read_bytes(),
            join_s3(prefix, "source_catalog.json"),
            content_type="application/json",
        )
        upload_bytes(
            client,
            catalogs.unit_path.read_bytes(),
            join_s3(prefix, "unit_catalog.json"),
            content_type="application/json",
        )

    gpu_tokens = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_tokens) != args.local_world_size or len(set(gpu_tokens)) != len(gpu_tokens):
        raise ValueError(f"--gpus must contain exactly {args.local_world_size} unique values")

    if repartition_mode:
        assignment, assignment_paths, assignment_uri = load_external_repartition_assignments(
            client, args, catalogs
        )
        upload_json(
            client,
            {
                "status": "running",
                "started_at": started_at,
                "repartition_id": args.repartition_id,
                "snapshot_sha256": assignment["snapshot_sha256"],
                "node_index": args.node_index,
                "global_ranks": node_global_ranks(args, args.node_index),
                "assigned_units": sum(
                    len(lane["global_unit_indices"])
                    for lane in assignment["assignments"]
                ),
                "assignment_uri": assignment_uri,
            },
            status_uri,
        )
        completed_ranks = _run_worker_phase(
            client,
            args,
            catalogs,
            gpu_tokens,
            assignment_paths=assignment_paths,
            log_filename="repartition-worker.log",
        )
        _require_equal(
            completed_ranks,
            node_global_ranks(args, args.node_index),
            "repartition worker ranks",
        )
        units_by_index = {
            int(unit["global_unit_index"]): unit for unit in catalogs.units["units"]
        }
        summaries = []
        for lane in assignment["assignments"]:
            rank = int(lane["physical_global_rank"])
            summaries.append(
                _load_operational_worker_summary(
                    args.work_dir / f"rank-{rank:03d}" / "operational-summary.json",
                    rank=rank,
                    assigned_indices=lane["global_unit_indices"],
                    selection_sha256=assignment["selection_sha256"],
                    logical_world_size=global_world_size(args),
                    units_by_index=units_by_index,
                )
            )
        result = build_repartition_node_result(
            args,
            catalogs,
            assignment=assignment,
            assignment_uri=assignment_uri,
            worker_summaries=summaries,
            started_at=started_at,
        )
        upload_json(client, result, join_s3(prefix, "summary.json"))
        upload_json(client, result, join_s3(prefix, "_SUCCESS"))
        upload_json(
            client,
            {
                "status": "complete",
                "completed_at": result["completed_at"],
                "repartition_id": args.repartition_id,
                "assigned_units": result["assigned_units"],
                "generated_units": result["generated_units"],
                "resumed_units": result["resumed_units"],
            },
            status_uri,
        )
        return result

    rebalance_enabled = (
        bool(getattr(args, "rebalance_local_ranks", False))
        and args.local_world_size > 1
    )
    pending: list[dict[str, Any]] = []
    assignment_paths: dict[int, Path] = {}
    if rebalance_enabled:
        upload_json(
            client,
            {
                "status": "scanning_pending_units",
                "started_at": started_at,
                "node_index": args.node_index,
                "global_ranks": node_global_ranks(args, args.node_index),
            },
            status_uri,
        )
        pending = pending_node_units(client, args, catalogs)
        assignment_paths = write_rebalanced_assignments(args, catalogs, pending)

    upload_json(
        client,
        {
            "status": "running",
            "started_at": started_at,
            "node_index": args.node_index,
            "global_ranks": node_global_ranks(args, args.node_index),
            "rebalance_local_ranks": rebalance_enabled,
            "pending_units_at_scan": len(pending) if rebalance_enabled else None,
            "active_rebalanced_workers": len(assignment_paths) if rebalance_enabled else None,
            "source_catalog_sha256": catalogs.source_sha256,
            "unit_catalog_sha256": catalogs.unit_sha256,
        },
        status_uri,
    )
    if assignment_paths:
        _run_worker_phase(
            client,
            args,
            catalogs,
            gpu_tokens,
            assignment_paths=assignment_paths,
            log_filename="rebalance-worker.log",
        )

    # Re-run the immutable modulo rank lanes in resume-only mode. Since the
    # rebalanced phase committed every missing unit first, these workers never
    # initialize vLLM; they only validate markers and publish canonical rank
    # summaries required by the original node/finalizer contract.
    canonical_ranks = _run_worker_phase(
        client,
        args,
        catalogs,
        gpu_tokens,
        assignment_paths=None,
        log_filename="worker.log",
    )
    _require_equal(
        canonical_ranks,
        node_global_ranks(args, args.node_index),
        "canonical worker ranks",
    )

    plan = build_partition_plan(args, catalogs)
    rank_summaries = []
    for global_rank in canonical_ranks:
        rank_summaries.append(
            _load_rank_summary(
                args.work_dir / f"rank-{global_rank:03d}" / "summary.json",
                global_rank=global_rank,
                expected_units=[
                    dict(unit) for unit in plan.units_by_rank[global_rank]
                ],
                catalogs=catalogs,
                args=args,
            )
        )

    marker = build_node_marker(
        args,
        catalogs=catalogs,
        rank_summaries=rank_summaries,
        started_at=started_at,
    )
    validate_node_marker(marker, args, catalogs, node_index=args.node_index)
    upload_json(client, marker, join_s3(prefix, "summary.json"))
    upload_json(client, marker, join_s3(prefix, "_SUCCESS"))
    manifest = (
        None
        if getattr(args, "defer_global_finalize", False)
        else maybe_finalize(client, args, catalogs=catalogs)
    )
    upload_json(
        client,
        {
            "status": "complete",
            "completed_at": marker["completed_at"],
            "root_finalized": manifest is not None,
        },
        status_uri,
    )
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-schema",
        choices=SUPPORTED_TARGET_SCHEMAS,
        default=TARGET_SCHEMA,
    )
    parser.add_argument("--check-complete", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument(
        "--source-catalog-uri",
        required=True,
        help="Stable s3:// identity of the coordinator-committed source catalog",
    )
    parser.add_argument("--source-catalog-sha256", required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument(
        "--unit-catalog-uri",
        required=True,
        help="Stable s3:// identity of the coordinator-committed unit catalog",
    )
    parser.add_argument("--unit-catalog-sha256", required=True)
    parser.add_argument("--topology-uri")
    parser.add_argument("--topology-sha256")
    parser.add_argument("--code-overlay-sha256")
    parser.add_argument("--output-s3", required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/image-edit-sidecar24"),
    )
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument(
        "--rebalance-local-ranks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LPT-balance unfinished units across GPUs, then publish canonical summaries",
    )
    parser.add_argument(
        "--rebalance-scan-workers",
        type=int,
        default=DEFAULT_REBALANCE_SCAN_WORKERS,
    )
    parser.add_argument(
        "--global-world-size",
        type=int,
        help="Explicit logical rank count for mixed-width cross-JobSet execution",
    )
    parser.add_argument(
        "--global-rank-offset",
        type=int,
        help="First logical rank owned by this node for mixed-width execution",
    )
    parser.add_argument(
        "--defer-global-finalize",
        action="store_true",
        help="Commit this node only; a topology-aware finalizer publishes the root",
    )
    parser.add_argument(
        "--external-repartition-prefix",
        help="Immutable S3 root containing assignments/ and operational node results",
    )
    parser.add_argument("--repartition-id")
    parser.add_argument(
        "--repartition-only",
        action="store_true",
        help="Run only externally assigned unfinished units; do not publish canonical summaries",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--checkpoint-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--hf-artifact-s3", required=True)
    parser.add_argument("--hf-artifact-manifest-sha256", required=True)
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Number of mutually exclusive data parts",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Zero-based part to process",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Global canary row limit across every node/GPU rank",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--request-window-size",
        type=int,
        default=rank_worker.DEFAULT_REQUEST_WINDOW_SIZE,
        help="Requests queued per blocking vLLM call; active batch remains --batch-size",
    )
    parser.add_argument(
        "--prefetch-images",
        action="store_true",
        help="Load/decode the next request window while the current window generates",
    )
    parser.add_argument("--cross-unit-streaming", action="store_true")
    parser.add_argument(
        "--max-pending-unit-writes",
        type=int,
        default=rank_worker.DEFAULT_MAX_PENDING_UNIT_WRITES,
    )
    parser.add_argument(
        "--unit-upload-workers",
        type=int,
        default=rank_worker.DEFAULT_UNIT_UPLOAD_WORKERS,
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--unit-read-workers",
        type=int,
        default=rank_worker.DEFAULT_UNIT_READ_WORKERS,
    )
    parser.add_argument(
        "--unit-prefetch-depth",
        type=int,
        default=rank_worker.DEFAULT_UNIT_PREFETCH_DEPTH,
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1536 * 1536)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mm-processor-cache-gb", type=float)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--gdn-prefill-backend")
    parser.add_argument("--mm-encoder-attn-backend")
    parser.add_argument("--cudagraph-mode")
    parser.add_argument("--max-cudagraph-capture-size", type=int)
    parser.add_argument(
        "--cudagraph-mm-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cudagraph-specialize-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-tower-connector-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--speculative-attention-backend")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--expected-torch-version")
    parser.add_argument("--expected-transformers-version")
    parser.add_argument("--expected-vllm-version")
    parser.add_argument("--expected-flashinfer-version")
    parser.add_argument("--expected-flashinfer-cubin-version")
    parser.add_argument("--expected-flashinfer-jit-cache-version")
    parser.add_argument("--expected-triton-version")
    parser.add_argument("--expected-cuda-version")
    parser.add_argument("--expected-hld-postprocess-version")
    parser.add_argument("--package-v6-bboxes-yxyx", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_target_schema(args.target_schema)
    except ValueError as exc:
        parser.error(str(exc))
    for uri in (
        args.output_s3,
        args.source_catalog_uri,
        args.unit_catalog_uri,
        args.checkpoint_s3,
        args.hf_artifact_s3,
    ):
        try:
            parse_s3_uri(uri)
        except ValueError as exc:
            parser.error(str(exc))
    if args.topology_uri is not None:
        try:
            parse_s3_uri(args.topology_uri)
        except ValueError as exc:
            parser.error(str(exc))
    if (args.topology_uri is None) != (args.topology_sha256 is None):
        parser.error("--topology-uri and --topology-sha256 must be supplied together")
    repartition_values = (
        args.external_repartition_prefix,
        args.repartition_id,
        args.repartition_only,
    )
    if any(repartition_values) and not all(repartition_values):
        parser.error(
            "--external-repartition-prefix, --repartition-id, and --repartition-only "
            "must be supplied together"
        )
    if args.repartition_only:
        try:
            parse_s3_uri(args.external_repartition_prefix)
        except ValueError as exc:
            parser.error(str(exc))
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.repartition_id):
            parser.error("--repartition-id must be a lowercase DNS-style identifier")
        if not args.defer_global_finalize:
            parser.error("--repartition-only requires --defer-global-finalize")
    if args.package_v6_bboxes_yxyx:
        if TARGET_SCHEMA != "image_edit_v6_direct_root_xyxy":
            parser.error("--package-v6-bboxes-yxyx requires the V6 XYXY schema")
        if not args.expected_hld_postprocess_version:
            parser.error(
                "--package-v6-bboxes-yxyx requires "
                "--expected-hld-postprocess-version"
            )
    for name in (
        "node_count",
        "local_world_size",
        "rebalance_scan_workers",
        "checkpoint_step",
        "batch_size",
        "request_window_size",
        "max_pending_unit_writes",
        "unit_upload_workers",
        "download_workers",
        "unit_read_workers",
        "unit_prefetch_depth",
        "max_model_len",
        "max_new_tokens",
        "max_pixels",
        "progress_every",
        "num_speculative_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.request_window_size < args.batch_size:
        parser.error("--request-window-size must be at least --batch-size")
    if args.unit_upload_workers > args.max_pending_unit_writes:
        parser.error("--unit-upload-workers must not exceed --max-pending-unit-writes")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.max_cudagraph_capture_size is not None:
        if args.max_cudagraph_capture_size <= 0:
            parser.error("--max-cudagraph-capture-size must be positive")
        if args.max_cudagraph_capture_size > args.batch_size:
            parser.error("--max-cudagraph-capture-size must not exceed --batch-size")
    if args.cudagraph_mm_encoder:
        parser.error("--cudagraph-mm-encoder is unsafe with the pinned vLLM 0.24 runtime")
    if args.enforce_eager and (
        args.cudagraph_mode
        or args.max_cudagraph_capture_size is not None
        or args.cudagraph_specialize_lora is not None
    ):
        parser.error("CUDA graph options cannot be combined with --enforce-eager")
    if args.mm_processor_cache_gb is not None and args.mm_processor_cache_gb < 0:
        parser.error("--mm-processor-cache-gb must be non-negative")
    if args.node_index < 0 or args.node_index >= args.node_count:
        parser.error("--node-index must be in [0, --node-count)")
    if args.global_world_size is not None and args.global_world_size <= 0:
        parser.error("--global-world-size must be positive")
    if args.global_rank_offset is not None:
        if args.global_rank_offset < 0:
            parser.error("--global-rank-offset must be non-negative")
        if args.global_world_size is None:
            parser.error("--global-rank-offset requires --global-world-size")
        if args.global_rank_offset + args.local_world_size > args.global_world_size:
            parser.error("the node's explicit logical ranks exceed --global-world-size")
        default_offset = args.node_index * args.local_world_size
        default_world = args.node_count * args.local_world_size
        if (
            (args.global_rank_offset != default_offset or args.global_world_size != default_world)
            and not args.defer_global_finalize
        ):
            parser.error("mixed-width rank mapping requires --defer-global-finalize")
    if args.partition_count <= 0:
        parser.error("--partition-count must be positive")
    if args.partition_index < 0 or args.partition_index >= args.partition_count:
        parser.error("--partition-index must be in [0, --partition-count)")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("--case-limit must be positive")
    for name in (
        "source_catalog_sha256",
        "unit_catalog_sha256",
        "checkpoint_manifest_sha256",
        "hf_artifact_manifest_sha256",
    ):
        if SHA256_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase 64-hex")
    for name in ("topology_sha256", "code_overlay_sha256"):
        value = getattr(args, name)
        if value is not None and SHA256_RE.fullmatch(value) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase 64-hex")
    if not args.check_complete and args.model is None:
        parser.error("--model is required unless --check-complete is used")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    gpu_tokens = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_tokens) != args.local_world_size or len(set(gpu_tokens)) != len(gpu_tokens):
        parser.error(f"--gpus must contain exactly {args.local_world_size} unique values")
    args.output_s3 = args.output_s3.rstrip("/")
    args.source_catalog_uri = args.source_catalog_uri.rstrip("/")
    args.unit_catalog_uri = args.unit_catalog_uri.rstrip("/")
    if args.topology_uri is not None:
        args.topology_uri = args.topology_uri.rstrip("/")
    args.checkpoint_s3 = args.checkpoint_s3.rstrip("/")
    args.hf_artifact_s3 = args.hf_artifact_s3.rstrip("/")
    if args.external_repartition_prefix is not None:
        args.external_repartition_prefix = args.external_repartition_prefix.rstrip("/")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    client = s3_client()
    if args.check_complete:
        catalogs = load_bound_catalogs(args)
        complete = check_node_complete(client, args, catalogs)
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
                join_s3(
                    runtime_node_prefix(args),
                    "status.json",
                ),
            )
        except BaseException as upload_error:
            print(f"failed to upload node failure status: {upload_error}", file=sys.stderr)
        raise
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "node-summary.json").write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
