#!/usr/bin/env python3
"""Run eight independent Qwen image-edit JSON workers, merge, and upload to S3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.gcp_wif import download_gcs_file
from dlms.datasets.image_edit_inference_contract import (
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SUPPORTED_TARGET_SCHEMAS,
    TARGET_SCHEMA,
    require_target_schema,
)
from scripts.infer_image_edit_full_json import normalize_input_row


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key, suffix) if value)}"


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=16,
        ),
    )


def upload_file(client: Any, source: Path, destination: str) -> None:
    bucket, key = parse_s3_uri(destination)
    client.upload_file(str(source), bucket, key)


def upload_if_exists(client: Any, source: Path, destination: str) -> None:
    if source.is_file():
        upload_file(client, source, destination)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_manifest(
    uri: str,
    destination: Path,
    expected_count: int,
    expected_sha256: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("gs://"):
        download_gcs_file(uri, str(destination))
    elif uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        if not key:
            raise ValueError(f"S3 manifest URI must identify an object: {uri}")
        s3_client().download_file(bucket, key, str(destination))
    else:
        raise ValueError("input manifest must be an explicit gs:// or s3:// URI")
    actual_sha256 = file_sha256(destination)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "input manifest SHA-256 mismatch: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    count = 0
    sample_ids: set[str] = set()
    with destination.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest line {line_number} is not an object")
            record = normalize_input_row(row, count)
            if record.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id in input manifest: {record.sample_id}")
            sample_ids.add(record.sample_id)
            remote_prefixes = ("gs://", "s3://", "parquet+s3://")
            if not record.reference_image.startswith(
                remote_prefixes
            ) or not record.target_image.startswith(remote_prefixes):
                raise ValueError(
                    f"sample {record.sample_id} does not resolve to two S3/GCS images"
                )
            count += 1
    if count != expected_count:
        raise ValueError(f"input count mismatch: actual={count} expected={expected_count}")
    return {
        "uri": uri,
        "local_path": str(destination),
        "rows": count,
        "sha256": actual_sha256,
        "unique_sample_ids": len(sample_ids),
    }


def worker_command(args: argparse.Namespace, manifest: Path, rank: int) -> list[str]:
    rank_dir = args.output_dir / "ranks" / f"rank-{rank:03d}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/infer_image_edit_full_json.py"),
        "--model",
        str(args.model),
        "--target-schema",
        getattr(args, "target_schema", TARGET_SCHEMA),
        "--input-jsonl",
        str(manifest),
        "--output-jsonl",
        str(rank_dir / "predictions.jsonl"),
        "--max-samples",
        str(args.expected_count),
        "--rank",
        str(rank),
        "--world-size",
        str(args.world_size),
        "--batch-size",
        str(args.batch_size),
        "--request-window-size",
        str(getattr(args, "request_window_size", args.batch_size)),
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
        "--seed",
        str(getattr(args, "seed", 42)),
        "--attention-backend",
        args.attention_backend,
        "--progress-every",
        str(args.progress_every),
        "--continue-on-error",
        "--include-conversation",
        "--no-tqdm",
    ]
    command.append(
        "--prefetch-images" if bool(getattr(args, "prefetch_images", True)) else "--no-prefetch-images"
    )
    optional_value_args = (
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("mm_processor_cache_gb", "--mm-processor-cache-gb"),
        ("gdn_prefill_backend", "--gdn-prefill-backend"),
        ("mm_encoder_attn_backend", "--mm-encoder-attn-backend"),
        ("cudagraph_mode", "--cudagraph-mode"),
        ("max_cudagraph_capture_size", "--max-cudagraph-capture-size"),
        ("expected_torch_version", "--expected-torch-version"),
        ("expected_transformers_version", "--expected-transformers-version"),
        ("expected_vllm_version", "--expected-vllm-version"),
        ("expected_flashinfer_version", "--expected-flashinfer-version"),
        ("expected_flashinfer_cubin_version", "--expected-flashinfer-cubin-version"),
        ("expected_flashinfer_jit_cache_version", "--expected-flashinfer-jit-cache-version"),
        ("expected_triton_version", "--expected-triton-version"),
        ("expected_cuda_version", "--expected-cuda-version"),
    )
    for attribute, flag in optional_value_args:
        value = getattr(args, attribute, None)
        if value is not None:
            command.extend([flag, str(value)])
    boolean_option_args = (
        ("cudagraph_specialize_lora", "--cudagraph-specialize-lora"),
        ("enable_tower_connector_lora", "--enable-tower-connector-lora"),
        ("enable_prefix_caching", "--enable-prefix-caching"),
    )
    for attribute, flag in boolean_option_args:
        value = getattr(args, attribute, None)
        if value is not None:
            command.append(flag if bool(value) else f"--no-{flag[2:]}")
    # Multimodal CUDA graphs are deliberately disabled: the PR #36 fast path
    # graphs the decoder while leaving the vision encoder eager so concurrent
    # requests cannot accidentally reuse image semantics.
    command.append("--no-cudagraph-mm-encoder")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.use_mtp:
        command.extend(
            [
                "--use-mtp",
                "--num-speculative-tokens",
                str(args.num_speculative_tokens),
            ]
        )
        if args.speculative_attention_backend:
            command.extend(
                [
                    "--speculative-attention-backend",
                    args.speculative_attention_backend,
                ]
            )
    return command


def upload_rank(client: Any, output_dir: Path, output_s3: str, rank: int) -> None:
    rank_dir = output_dir / "ranks" / f"rank-{rank:03d}"
    for filename in ("predictions.jsonl", "worker.log", "status.json"):
        upload_if_exists(
            client,
            rank_dir / filename,
            join_s3(output_s3, "ranks", f"rank-{rank:03d}", filename),
        )


def merge_results(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rank_counts: dict[str, int] = {}
    for rank in range(args.world_size):
        path = args.output_dir / "ranks" / f"rank-{rank:03d}" / "predictions.jsonl"
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"rank {rank} emitted a non-object row")
                rows.append(row)
                count += 1
        rank_counts[str(rank)] = count
    rows.sort(key=lambda row: int(row["input_index"]))
    actual_indices = [int(row["input_index"]) for row in rows]
    if actual_indices != list(range(args.expected_count)):
        raise ValueError("merged rank outputs do not cover every input_index exactly once")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != args.expected_count:
        raise ValueError("merged rank outputs contain duplicate sample IDs")
    merged_path = args.output_dir / "predictions.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    return {
        "rows": len(rows),
        "json_valid": sum(bool(row.get("json_valid")) for row in rows),
        "schema_valid": sum(bool(row.get("schema_valid")) for row in rows),
        "failed": sum(not bool(row.get("schema_valid")) for row in rows),
        "inference_errors": sum(bool(row.get("inference_error")) for row in rows),
        "rank_rows": rank_counts,
        "predictions_sha256": file_sha256(merged_path),
        "predictions_bytes": merged_path.stat().st_size,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "input" / "manifest.jsonl"
    client = s3_client()
    status_path = args.output_dir / "status.json"
    summary_path = args.output_dir / "summary.json"
    status = {
        "status": "preparing",
        "started_at": utc_now(),
        "output_s3": args.output_s3,
        "source_dataset_s3": args.source_dataset_s3,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "target_schema": args.target_schema,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
    }
    atomic_json(status_path, status)
    upload_file(client, status_path, join_s3(args.output_s3, "status.json"))
    manifest = prepare_manifest(
        args.input_manifest,
        manifest_path,
        args.expected_count,
        args.input_manifest_sha256,
    )
    upload_file(client, manifest_path, join_s3(args.output_s3, "input_manifest.jsonl"))
    worker_manifest_path = manifest_path
    input_materialization: dict[str, Any] | None = None
    if args.materialize_embedded_parquet:
        from scripts.materialize_embedded_parquet_inference_manifest import (
            materialize_manifest,
        )

        worker_manifest_path = args.output_dir / "input" / "runtime_manifest.jsonl"
        input_materialization = materialize_manifest(
            manifest_path,
            worker_manifest_path,
            work_dir=args.output_dir / "input" / "materialized_media",
            source_dataset_s3=args.source_dataset_s3,
            expected_count=args.expected_count,
            s3_client=client,
        )
        materialization_path = args.output_dir / "input" / "materialization_summary.json"
        atomic_json(materialization_path, input_materialization)
        upload_file(
            client,
            materialization_path,
            join_s3(args.output_s3, "input_materialization.json"),
        )

    gpu_tokens = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_tokens) != args.world_size:
        raise ValueError(f"--gpus must contain exactly {args.world_size} device tokens")
    processes: dict[int, subprocess.Popen[Any]] = {}
    log_handles: dict[int, Any] = {}
    for rank, gpu in enumerate(gpu_tokens):
        rank_dir = args.output_dir / "ranks" / f"rank-{rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            rank_dir / "status.json",
            {"status": "running", "rank": rank, "gpu": gpu, "started_at": utc_now()},
        )
        log_handle = (rank_dir / "worker.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                # Give each independent TP1 vLLM engine a disjoint port range.
                "VLLM_PORT": str(20000 + rank * 1024),
            }
        )
        flashinfer_workspace_base = environment.get("FLASHINFER_WORKSPACE_BASE")
        if flashinfer_workspace_base:
            rank_workspace = Path(flashinfer_workspace_base) / f"rank-{rank:03d}"
            rank_workspace.mkdir(parents=True, exist_ok=True)
            environment["FLASHINFER_WORKSPACE_BASE"] = str(rank_workspace)
        processes[rank] = subprocess.Popen(
            worker_command(args, worker_manifest_path, rank),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handles[rank] = log_handle
    status["status"] = "running"
    status["manifest"] = manifest
    status["input_materialization"] = input_materialization
    status["workers"] = args.world_size
    atomic_json(status_path, status)
    upload_file(client, status_path, join_s3(args.output_s3, "status.json"))

    remaining = set(processes)
    failure: RuntimeError | None = None
    while remaining:
        for rank in list(remaining):
            returncode = processes[rank].poll()
            if returncode is None:
                continue
            remaining.remove(rank)
            log_handles[rank].close()
            rank_status = {
                "status": "complete" if returncode == 0 else "failed",
                "rank": rank,
                "returncode": returncode,
                "completed_at": utc_now(),
            }
            atomic_json(args.output_dir / "ranks" / f"rank-{rank:03d}" / "status.json", rank_status)
            upload_rank(client, args.output_dir, args.output_s3, rank)
            print(json.dumps({"event": "worker_exit", **rank_status}, sort_keys=True), flush=True)
            if returncode != 0 and failure is None:
                failure = RuntimeError(f"inference rank {rank} exited with code {returncode}")
        if failure is not None:
            for rank in remaining:
                processes[rank].terminate()
            for rank in list(remaining):
                processes[rank].wait(timeout=60)
                log_handles[rank].close()
                upload_rank(client, args.output_dir, args.output_s3, rank)
            raise failure
        if remaining:
            time.sleep(2)

    aggregate = merge_results(args)
    conversion = json.loads(args.conversion_provenance.read_text(encoding="utf-8"))
    summary = {
        "status": "complete",
        "started_at": status["started_at"],
        "completed_at": utc_now(),
        "source_dataset_s3": args.source_dataset_s3,
        "input_manifest": manifest,
        "input_materialization": input_materialization,
        "output_s3": args.output_s3,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "target_schema": args.target_schema,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "conversion": conversion,
        "world_size": args.world_size,
        "generation": {
            "batch_size_per_gpu": args.batch_size,
            "request_window_size_per_gpu": getattr(
                args, "request_window_size", args.batch_size
            ),
            "prefetch_images": bool(getattr(args, "prefetch_images", True)),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
            "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
            "mm_processor_cache_gb": getattr(args, "mm_processor_cache_gb", None),
            "thinking": False,
            "seed": args.seed,
            "tensor_parallel_size_per_worker": 1,
            "dtype": "bfloat16",
            "attention_backend": args.attention_backend,
            "gdn_prefill_backend": getattr(args, "gdn_prefill_backend", None),
            "mm_encoder_attn_backend": getattr(args, "mm_encoder_attn_backend", None),
            "cudagraph_mode": getattr(args, "cudagraph_mode", None),
            "max_cudagraph_capture_size": getattr(
                args, "max_cudagraph_capture_size", None
            ),
            "cudagraph_mm_encoder": False,
            "cudagraph_specialize_lora": getattr(
                args, "cudagraph_specialize_lora", None
            ),
            "enable_tower_connector_lora": getattr(
                args, "enable_tower_connector_lora", None
            ),
            "enable_prefix_caching": getattr(args, "enable_prefix_caching", None),
            "use_mtp": args.use_mtp,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.use_mtp else 0
            ),
        },
        "results": aggregate,
    }
    atomic_json(summary_path, summary)
    atomic_json(status_path, {"status": "complete", "completed_at": summary["completed_at"]})
    upload_file(client, args.output_dir / "predictions.jsonl", join_s3(args.output_s3, "predictions.jsonl"))
    upload_file(client, summary_path, join_s3(args.output_s3, "summary.json"))
    upload_file(client, status_path, join_s3(args.output_s3, "status.json"))
    upload_file(
        client,
        args.conversion_provenance,
        join_s3(args.output_s3, "conversion_provenance.json"),
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-schema",
        choices=SUPPORTED_TARGET_SCHEMAS,
        default=TARGET_SCHEMA,
        help="Must match IMAGE_EDIT_TARGET_SCHEMA selected before process import",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conversion-provenance", type=Path, required=True)
    parser.add_argument("--source-dataset-s3", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument(
        "--materialize-embedded-parquet",
        action="store_true",
        help=(
            "Decode one contiguous embedded-Parquet cohort once and run workers from "
            "local paths while preserving the frozen parquet+s3 identities."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=6304)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--request-window-size", type=int)
    parser.add_argument(
        "--prefetch-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1536 * 1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mm-processor-cache-gb", type=float)
    parser.add_argument("--gdn-prefill-backend")
    parser.add_argument("--mm-encoder-attn-backend")
    parser.add_argument("--cudagraph-mode")
    parser.add_argument("--max-cudagraph-capture-size", type=int)
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
    parser.add_argument("--expected-torch-version")
    parser.add_argument("--expected-transformers-version")
    parser.add_argument("--expected-vllm-version")
    parser.add_argument("--expected-flashinfer-version")
    parser.add_argument("--expected-flashinfer-cubin-version")
    parser.add_argument("--expected-flashinfer-jit-cache-version")
    parser.add_argument("--expected-triton-version")
    parser.add_argument("--expected-cuda-version")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--speculative-attention-backend")
    args = parser.parse_args(argv)
    if args.request_window_size is None:
        args.request_window_size = args.batch_size
    try:
        require_target_schema(args.target_schema)
    except ValueError as exc:
        parser.error(str(exc))
    parse_s3_uri(args.source_dataset_s3)
    parse_s3_uri(args.output_s3)
    for name in (
        "checkpoint_step",
        "expected_count",
        "world_size",
        "batch_size",
        "request_window_size",
        "download_workers",
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
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.mm_processor_cache_gb is not None and args.mm_processor_cache_gb < 0:
        parser.error("--mm-processor-cache-gb must be non-negative")
    if args.max_cudagraph_capture_size is not None:
        if args.max_cudagraph_capture_size <= 0:
            parser.error("--max-cudagraph-capture-size must be positive")
        if args.max_cudagraph_capture_size > args.batch_size:
            parser.error("--max-cudagraph-capture-size must not exceed --batch-size")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except BaseException as exc:  # persist terminal failure state before propagating
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        failure_path = args.output_dir / "status.json"
        atomic_json(failure_path, failure)
        try:
            upload_file(s3_client(), failure_path, join_s3(args.output_s3, "status.json"))
        except BaseException as upload_error:
            print(f"failed to upload terminal status: {upload_error}", file=sys.stderr)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
