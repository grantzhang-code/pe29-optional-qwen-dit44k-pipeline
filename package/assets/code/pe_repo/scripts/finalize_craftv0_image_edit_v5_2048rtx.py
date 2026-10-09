#!/usr/bin/env python3
"""Validate all mixed-width craftv0 node commits and publish the root commit."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOPOLOGY_VARIANT = os.environ.get("CRAFTV0_TOPOLOGY_VARIANT", "v5_2048rtx")
if TOPOLOGY_VARIANT == "v6_1912rtx_128h200":
    os.environ.setdefault("IMAGE_EDIT_TARGET_SCHEMA", "image_edit_v6_direct_root_xyxy")
    os.environ.setdefault("IMAGE_EDIT_V6_PROMPT_VARIANT", "training_v7")
else:
    os.environ.setdefault("IMAGE_EDIT_TARGET_SCHEMA", "image_edit_v5_instruction")

from dlms.datasets.image_edit_inference_contract import (
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    TARGET_SCHEMA,
)
if TOPOLOGY_VARIANT == "v6_1912rtx_128h200":
    from scripts import (  # type: ignore[no-redef]
        build_craftv0_image_edit_v6_1912rtx_128h200_topology as topology_module,
    )
elif TOPOLOGY_VARIANT == "v5_2048rtx":
    from scripts import (  # type: ignore[no-redef]
        build_craftv0_image_edit_v5_2048rtx_topology as topology_module,
    )
else:
    raise RuntimeError(f"unsupported CRAFTV0_TOPOLOGY_VARIANT={TOPOLOGY_VARIANT!r}")
from scripts import image_edit_sidecar24_catalog as source_catalog_module
from scripts import image_edit_sidecar24_partition as partition_module
from scripts import infer_image_edit_full_json_sidecar24_rank as rank_worker
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module
from scripts import run_image_edit_full_json_sidecar24_node as node_runner


if TOPOLOGY_VARIANT == "v6_1912rtx_128h200":
    MANIFEST_FORMAT = "image_edit_v6_craftv0_mixed_accelerator_results_v1"
    SUCCESS_FORMAT = "image_edit_v6_craftv0_mixed_accelerator_results_commit_v1"
else:
    MANIFEST_FORMAT = "image_edit_v5_craftv0_mixed_rtx_results_v1"
    SUCCESS_FORMAT = "image_edit_v5_craftv0_mixed_rtx_results_commit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def common_args(args: argparse.Namespace, node: Mapping[str, Any]) -> argparse.Namespace:
    width = int(node["gpus_per_pod"])
    return argparse.Namespace(
        target_schema=TARGET_SCHEMA,
        source_catalog=args.source_catalog,
        source_catalog_uri=args.source_catalog_uri,
        source_catalog_sha256=args.source_catalog_sha256,
        unit_catalog=args.unit_catalog,
        unit_catalog_uri=args.unit_catalog_uri,
        unit_catalog_sha256=args.unit_catalog_sha256,
        topology_uri=args.topology_uri,
        topology_sha256=args.topology_sha256,
        code_overlay_sha256=args.code_overlay_sha256,
        output_s3=args.output_s3,
        work_dir=Path("/tmp/craftv0-finalizer-unused"),
        node_index=int(node["node_index"]),
        node_count=topology_module.TOTAL_NODES,
        local_world_size=width,
        global_world_size=topology_module.TOTAL_RANKS,
        global_rank_offset=int(node["global_rank_start"]),
        defer_global_finalize=True,
        gpus=",".join(str(index) for index in range(width)),
        checkpoint_s3=args.checkpoint_s3,
        checkpoint_step=getattr(args, "checkpoint_step", 10_000),
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        hf_artifact_s3=args.hf_artifact_s3,
        hf_artifact_manifest_sha256=args.hf_artifact_manifest_sha256,
        partition_count=1,
        partition_index=0,
        case_limit=None,
        batch_size=32,
        request_window_size=128,
        prefetch_images=True,
        cross_unit_streaming=True,
        max_pending_unit_writes=8,
        unit_upload_workers=2,
        download_workers=4,
        max_model_len=32_768,
        max_new_tokens=getattr(args, "max_new_tokens", 16_384),
        max_pixels=2_359_296,
        max_num_batched_tokens=32_768,
        mm_processor_cache_gb=0.0,
        gpu_memory_utilization=0.90,
        attention_backend="FLASHINFER",
        gdn_prefill_backend="triton",
        mm_encoder_attn_backend="FLASH_ATTN",
        cudagraph_mode="PIECEWISE",
        max_cudagraph_capture_size=32,
        cudagraph_mm_encoder=False,
        cudagraph_specialize_lora=True,
        enable_tower_connector_lora=True,
        enable_prefix_caching=False,
        speculative_attention_backend="auto",
        progress_every=25,
        enforce_eager=False,
        use_mtp=True,
        num_speculative_tokens=1,
        seed=42,
        aws_profile=args.aws_profile,
        aws_region="us-east-1",
        expected_torch_version="2.11.0",
        expected_transformers_version="5.8.0",
        expected_vllm_version="0.26.0",
        expected_flashinfer_version="0.6.14",
        expected_flashinfer_cubin_version=None,
        expected_flashinfer_jit_cache_version=None,
        expected_triton_version="3.6.0",
        expected_cuda_version="13.0",
        expected_hld_postprocess_version=(
            getattr(args, "expected_hld_postprocess_version", None)
        ),
        package_v6_bboxes_yxyx=(
            TOPOLOGY_VARIANT == "v6_1912rtx_128h200"
        ),
    )


def expected_node_contract(
    args: argparse.Namespace,
    node: Mapping[str, Any],
    catalogs: node_runner.BoundCatalogs,
    plan: partition_module.PartitionPlan,
) -> dict[str, Any]:
    node_args = common_args(args, node)
    return {
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
        "checkpoint_s3": node_runner.normalize_s3_root(args.checkpoint_s3),
        "checkpoint_step": getattr(args, "checkpoint_step", 10_000),
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "hf_artifact_s3": node_runner.normalize_s3_root(args.hf_artifact_s3),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "node_count": topology_module.TOTAL_NODES,
        "local_world_size": int(node["gpus_per_pod"]),
        "global_world_size": topology_module.TOTAL_RANKS,
        "generation": node_runner.generation_contract(node_args),
        "topology_uri": args.topology_uri,
        "topology_sha256": args.topology_sha256,
        "code_overlay_sha256": args.code_overlay_sha256,
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
    }


def validate_node_marker(
    marker: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    node: Mapping[str, Any],
    catalogs: node_runner.BoundCatalogs,
    plan: partition_module.PartitionPlan,
) -> None:
    index = int(node["node_index"])
    width = int(node["gpus_per_pod"])
    rank_start = int(node["global_rank_start"])
    ranks = list(range(rank_start, rank_start + width))
    node_runner._require_equal(set(marker), node_runner.NODE_MARKER_FIELDS, "node marker fields")
    node_runner._require_equal(marker.get("format_version"), node_runner.NODE_FORMAT, "node format")
    node_runner._require_equal(marker.get("status"), "complete", "node status")
    node_runner._require_equal(marker.get("node_index"), index, "node index")
    node_runner._require_equal(marker.get("node_count"), topology_module.TOTAL_NODES, "node count")
    node_runner._require_equal(marker.get("global_ranks"), ranks, "node ranks")
    node_runner._require_equal(
        marker.get("contract"),
        expected_node_contract(args, node, catalogs, plan),
        f"node {index} contract",
    )
    for field in ("started_at", "completed_at"):
        if not isinstance(marker.get(field), str) or not marker[field]:
            raise ValueError(f"node {index} has no {field}")

    expected_units = sorted(
        (unit for rank in ranks for unit in plan.units_by_rank[rank]),
        key=lambda unit: int(unit["global_unit_index"]),
    )
    shards = marker.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"node {index} has no shard list")
    node_runner._require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        [int(unit["global_unit_index"]) for unit in expected_units],
        f"node {index} unit coverage",
    )
    for shard, unit in zip(shards, expected_units, strict=True):
        node_runner._validate_shard_against_unit(shard, unit, output_s3=args.output_s3)
    node_runner._require_equal(
        marker.get("totals"),
        node_runner._aggregate_shards(shards, include_units=True),
        f"node {index} totals",
    )

    summaries = marker.get("rank_summaries")
    if not isinstance(summaries, list) or len(summaries) != width:
        raise ValueError(f"node {index} rank summaries are incomplete")
    node_runner._require_equal(
        [summary.get("global_rank") for summary in summaries], ranks, f"node {index} rank identities"
    )
    shard_by_unit = {int(shard["global_unit_index"]): shard for shard in shards}
    summary_fields = {
        "global_rank",
        "summary_s3_uri",
        "assigned_units",
        "generated_units",
        "resumed_units",
        "totals",
    }
    for summary, rank in zip(summaries, ranks, strict=True):
        if set(summary) != summary_fields:
            raise ValueError(f"rank {rank} compact summary has unexpected fields")
        rank_units = plan.units_by_rank[rank]
        rank_shards = [shard_by_unit[int(unit["global_unit_index"])] for unit in rank_units]
        node_runner._require_equal(summary["assigned_units"], len(rank_units), f"rank {rank} units")
        generated = summary["generated_units"]
        resumed = summary["resumed_units"]
        if type(generated) is not int or generated < 0 or type(resumed) is not int or resumed < 0:
            raise ValueError(f"rank {rank} dispositions are invalid")
        node_runner._require_equal(generated + resumed, len(rank_units), f"rank {rank} dispositions")
        node_runner._require_equal(
            sum(shard["disposition"] == "generated" for shard in rank_shards),
            generated,
            f"rank {rank} generated units",
        )
        node_runner._require_equal(
            sum(shard["disposition"] == "resumed" for shard in rank_shards),
            resumed,
            f"rank {rank} resumed units",
        )
        node_runner._require_equal(
            summary["summary_s3_uri"],
            rank_worker.rank_summary_uri(args.output_s3, rank),
            f"rank {rank} summary URI",
        )
        node_runner._require_equal(
            summary["totals"],
            node_runner._aggregate_shards(rank_shards, include_units=False),
            f"rank {rank} totals",
        )


def root_contract(
    args: argparse.Namespace,
    catalogs: node_runner.BoundCatalogs,
    plan: partition_module.PartitionPlan,
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    generation = node_runner.generation_contract(common_args(args, topology["nodes"][0]))
    return {
        "source_catalog_uri": args.source_catalog_uri,
        "source_catalog_sha256": catalogs.source_sha256,
        "source_catalog_format_version": catalogs.source["schema_version"],
        "source_catalog_rows": catalogs.source["total_rows"],
        "unit_catalog_uri": args.unit_catalog_uri,
        "unit_catalog_sha256": catalogs.unit_sha256,
        "unit_catalog_format_version": unit_catalog_module.FORMAT_VERSION,
        "topology_uri": args.topology_uri,
        "topology_sha256": args.topology_sha256,
        "code_overlay_sha256": args.code_overlay_sha256,
        "expected_units": plan.contract["selected_units"],
        "expected_rows": plan.contract["selected_rows"],
        "dataset_count": len(catalogs.units["datasets"]),
        "data_partition": plan.contract,
        "checkpoint_s3": node_runner.normalize_s3_root(args.checkpoint_s3),
        "checkpoint_step": getattr(args, "checkpoint_step", 10_000),
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "hf_artifact_s3": node_runner.normalize_s3_root(args.hf_artifact_s3),
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "node_count": topology_module.TOTAL_NODES,
        "global_world_size": topology_module.TOTAL_RANKS,
        "pod_topology": dict(
            sorted(
                Counter(
                    f"{node.get('accelerator', 'rtx-pro-server-6000')}_"
                    f"{int(node['gpus_per_pod'])}gpu_pods"
                    for node in topology["nodes"]
                ).items()
            )
        ),
        "generation": generation,
    }


def fetch_marker(client: Any, uri: str) -> dict[str, Any] | None:
    return node_runner.read_s3_json(client, uri, missing_ok=True)


def wait_for_nodes(
    client: Any,
    *,
    args: argparse.Namespace,
    topology: Mapping[str, Any],
    catalogs: node_runner.BoundCatalogs,
    plan: partition_module.PartitionPlan,
) -> list[dict[str, Any]]:
    pending = {int(node["node_index"]): node for node in topology["nodes"]}
    complete: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + args.wait_seconds
    while pending:
        futures = {}
        with ThreadPoolExecutor(max_workers=min(args.read_workers, len(pending))) as executor:
            for index, node in pending.items():
                uri = node_runner.join_s3(
                    node_runner.node_prefix(args.output_s3, index, topology_module.TOTAL_NODES),
                    "_SUCCESS",
                )
                futures[executor.submit(fetch_marker, client, uri)] = (index, node)
            for future in as_completed(futures):
                index, node = futures[future]
                marker = future.result()
                if marker is None:
                    continue
                validate_node_marker(
                    marker,
                    args=args,
                    node=node,
                    catalogs=catalogs,
                    plan=plan,
                )
                complete[index] = marker
        for index in complete:
            pending.pop(index, None)
        print(
            json.dumps(
                {"complete_nodes": len(complete), "pending_nodes": len(pending), "checked_at": utc_now()},
                sort_keys=True,
            ),
            flush=True,
        )
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out with {len(pending)} node commits missing")
        time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))
    return [complete[index] for index in range(topology_module.TOTAL_NODES)]


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    topology = topology_module.load_topology(args.topology)
    topology_payload = topology_module.canonical_json_bytes(topology)
    actual_topology_sha = node_runner.sha256_bytes(topology_payload)
    node_runner._require_equal(actual_topology_sha, args.topology_sha256, "topology SHA-256")

    session = boto3.Session(profile_name=args.aws_profile or None, region_name="us-east-1")
    client = session.client(
        "s3",
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, max_pool_connections=128),
    )
    remote_topology = node_runner.read_s3_json(client, args.topology_uri)
    node_runner._require_equal(remote_topology, topology, "remote topology")

    load_args = argparse.Namespace(
        source_catalog=args.source_catalog,
        source_catalog_sha256=args.source_catalog_sha256,
        unit_catalog=args.unit_catalog,
        unit_catalog_sha256=args.unit_catalog_sha256,
    )
    catalogs = node_runner.load_bound_catalogs(load_args)
    plan = partition_module.build_partition_plan(
        catalogs.units["units"],
        partition_count=1,
        partition_index=0,
        global_world_size=topology_module.TOTAL_RANKS,
        case_limit=None,
    )
    markers = wait_for_nodes(
        client,
        args=args,
        topology=topology,
        catalogs=catalogs,
        plan=plan,
    )
    shards = sorted(
        (dict(shard) for marker in markers for shard in marker["shards"]),
        key=lambda shard: int(shard["global_unit_index"]),
    )
    node_runner._require_equal(len(shards), len(plan.units), "root shard count")
    node_runner._require_equal(
        [int(shard["global_unit_index"]) for shard in shards],
        [int(unit["global_unit_index"]) for unit in plan.units],
        "root unit coverage",
    )
    totals = node_runner._aggregate_shards(shards, include_units=True)
    node_runner._require_equal(totals["rows"], plan.contract["selected_rows"], "root rows")
    datasets = node_runner._dataset_result_summaries(catalogs.units, shards, plan.units)
    contract = root_contract(args, catalogs, plan, topology)
    completed_at = max(str(marker["completed_at"]) for marker in markers)
    manifest = {
        "format_version": MANIFEST_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "contract": contract,
        "totals": totals,
        "datasets": datasets,
        "nodes": [
            {
                "node_index": node["node_index"],
                "global_rank_start": node["global_rank_start"],
                "gpus_per_pod": node["gpus_per_pod"],
                "pool_id": node["pool_id"],
                "success_uri": node_runner.join_s3(
                    node_runner.node_prefix(
                        args.output_s3,
                        int(node["node_index"]),
                        topology_module.TOTAL_NODES,
                    ),
                    "_SUCCESS",
                ),
                "totals": markers[int(node["node_index"])]["totals"],
            }
            for node in topology["nodes"]
        ],
        "shards": shards,
    }
    manifest_uri = node_runner.join_s3(args.output_s3, "manifest.json")
    manifest_sha = node_runner.upload_bytes(
        client,
        node_runner.canonical_json_bytes(manifest),
        manifest_uri,
        content_type="application/json",
    )
    success = {
        "format_version": SUCCESS_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_sha,
        "contract": contract,
        "datasets": len(datasets),
        "units": totals["units"],
        "rows": totals["rows"],
        "json_valid": totals["json_valid"],
        "schema_valid": totals["schema_valid"],
        "failed": totals["failed"],
        "inference_errors": totals["inference_errors"],
    }
    node_runner.upload_json(client, success, node_runner.join_s3(args.output_s3, "_SUCCESS"))
    return success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--source-catalog-uri", required=True)
    parser.add_argument("--source-catalog-sha256", required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog-uri", required=True)
    parser.add_argument("--unit-catalog-sha256", required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-uri", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--code-overlay-sha256", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument(
        "--checkpoint-s3",
        default="s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/runs/c4b-v5json-v3-4n-r3/20260807162025/mcore/checkpoint-10000/",
    )
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        default="1335bd055d1e2ca5def207f52c800d73ee5cb5eb9a6d5bb217d8605030551611",
    )
    parser.add_argument("--checkpoint-step", type=int, default=10_000)
    parser.add_argument(
        "--hf-artifact-s3",
        default="s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/models/c4b-image-edit-v5-checkpoint-10000-hf-trained-mtp-v1/",
    )
    parser.add_argument(
        "--hf-artifact-manifest-sha256",
        default="c9c22265b52a0cbd7ab2d71b18d0c507b9e5ecbd75c758dd5ca6a6f3c396722d",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16_384)
    parser.add_argument(
        "--expected-hld-postprocess-version",
        default=(
            "image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7"
            if TOPOLOGY_VARIANT == "v6_1912rtx_128h200"
            else None
        ),
    )
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--wait-seconds", type=int, default=259_200)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--read-workers", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_schema = (
        "image_edit_v6_direct_root_xyxy"
        if TOPOLOGY_VARIANT == "v6_1912rtx_128h200"
        else "image_edit_v5_instruction"
    )
    if TARGET_SCHEMA != expected_schema:
        raise SystemExit(
            f"target schema drifted: {TARGET_SCHEMA!r} != {expected_schema!r}"
        )
    for value in (
        args.source_catalog_sha256,
        args.unit_catalog_sha256,
        args.topology_sha256,
        args.code_overlay_sha256,
        args.checkpoint_manifest_sha256,
        args.hf_artifact_manifest_sha256,
    ):
        if node_runner.SHA256_RE.fullmatch(value) is None:
            raise SystemExit(f"invalid SHA-256: {value!r}")
    for uri in (
        args.source_catalog_uri,
        args.unit_catalog_uri,
        args.topology_uri,
        args.output_s3,
        args.checkpoint_s3,
        args.hf_artifact_s3,
    ):
        node_runner.parse_s3_uri(uri)
    if args.wait_seconds <= 0 or args.poll_seconds <= 0 or args.read_workers <= 0:
        raise SystemExit("wait/poll/read worker values must be positive")
    args.output_s3 = args.output_s3.rstrip("/")
    args.source_catalog_uri = args.source_catalog_uri.rstrip("/")
    args.unit_catalog_uri = args.unit_catalog_uri.rstrip("/")
    args.topology_uri = args.topology_uri.rstrip("/")
    result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
