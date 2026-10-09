#!/usr/bin/env python3
"""Restore one immutable Qwen3.5 MCore checkpoint and export a local HF model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.checkpoint.common import (  # noqa: E402
    Boto3ObjectStore,
    CheckpointError,
    S3Location,
    sha256_file,
)
from scripts.checkpoint.hf_artifact_s3 import (  # noqa: E402
    MANIFEST_NAME as GENERIC_HF_MANIFEST_NAME,
    restore_artifact as restore_generic_hf_artifact,
    validate_local_artifact as validate_generic_hf_artifact,
)
from scripts.checkpoint.mcore_s3 import (  # noqa: E402
    inspect_committed_checkpoint,
    restore_checkpoint,
)
from scripts.mtp_eval.hf_artifact import (  # noqa: E402
    COMMIT_KIND as MCORE_HF_COMMIT_KIND,
    MANIFEST_NAME as MCORE_HF_MANIFEST_NAME,
    prepare_trained_artifact,
    publish_artifact,
    restore_artifact as restore_mcore_hf_artifact,
    validate_local_artifact as validate_mcore_hf_artifact,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_EXPECTED_TENSORS = 1199
DEFAULT_EXPECTED_MTP_TENSORS = 15
DEFAULT_MTP_NUM_LAYERS = 1


def _success_uri(root: str) -> str:
    location = S3Location.parse(root)
    key = location.key.rstrip("/")
    return S3Location(location.bucket, f"{key}/_SUCCESS").uri


def _read_marker(store: Boto3ObjectStore, root: str) -> dict[str, Any]:
    marker = json.loads(store.read_bytes(_success_uri(root)))
    if not isinstance(marker, dict) or marker.get("status") != "complete":
        raise CheckpointError(f"incomplete _SUCCESS marker under {root}: {marker!r}")
    return marker


def _check_digest(value: str, *, name: str) -> None:
    if SHA256_RE.fullmatch(value) is None:
        raise CheckpointError(f"{name} must be a lowercase 64-hex SHA-256")


def _restore_initial_artifact(
    store: Boto3ObjectStore,
    artifact_s3: str,
    output_dir: Path,
    *,
    marker: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Restore either historical FSDP-HF or native MCore-HF bootstrap artifacts."""

    if marker.get("kind") == MCORE_HF_COMMIT_KIND:
        if output_dir.exists():
            _manifest, entries = validate_mcore_hf_artifact(output_dir)
            local_digest = sha256_file(output_dir / MCORE_HF_MANIFEST_NAME)
            if local_digest != expected_manifest_sha256:
                raise CheckpointError(
                    "existing initial MCore-HF artifact differs from the pinned manifest: "
                    f"local={local_digest} expected={expected_manifest_sha256}"
                )
            return {
                "artifact_s3": artifact_s3,
                "output_dir": str(output_dir),
                "manifest_sha256": local_digest,
                "file_count": len(entries),
                "reused_existing": True,
                "verified": True,
            }
        return restore_mcore_hf_artifact(
            store,
            artifact_s3,
            output_dir,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    if output_dir.exists():
        _manifest, entries = validate_generic_hf_artifact(output_dir)
        local_digest = sha256_file(output_dir / GENERIC_HF_MANIFEST_NAME)
        if local_digest != expected_manifest_sha256:
            raise CheckpointError(
                "existing initial HF artifact differs from the pinned manifest: "
                f"local={local_digest} expected={expected_manifest_sha256}"
            )
        return {
            "artifact_s3": artifact_s3,
            "output_dir": str(output_dir),
            "manifest_sha256": local_digest,
            "file_count": len(entries),
            "reused_existing": True,
            "verified": True,
        }
    return restore_generic_hf_artifact(store, artifact_s3, output_dir)


def _validate_export(
    model_dir: Path,
    *,
    expected_tensors: int,
    expected_mtp_tensors: int,
) -> dict[str, Any]:
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise CheckpointError(f"HF export is missing required files: {missing}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise CheckpointError("HF export has no weight_map")
    mtp_keys = [key for key in weight_map if str(key).startswith("mtp.")]
    if not weight_map:
        raise CheckpointError("HF export weight_map is empty")
    if (
        (expected_tensors and len(weight_map) != expected_tensors)
        or (expected_mtp_tensors and len(mtp_keys) != expected_mtp_tensors)
    ):
        raise CheckpointError(
            "HF export tensor coverage mismatch: "
            f"total={len(weight_map)}/{expected_tensors or 'auto'} "
            f"mtp={len(mtp_keys)}/{expected_mtp_tensors or 'auto'}"
        )
    shard_names = sorted(set(str(name) for name in weight_map.values()))
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    if missing_shards:
        raise CheckpointError(f"HF export index references missing shards: {missing_shards}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    mtp_layers = {
        text_config.get("mtp_num_hidden_layers"),
        text_config.get("num_nextn_predict_layers"),
    }
    if 1 not in mtp_layers:
        raise CheckpointError("HF export config does not declare the trained one-layer MTP head")
    return {
        "tensor_count": len(weight_map),
        "mtp_tensor_count": len(mtp_keys),
        "expected_tensor_count": expected_tensors or None,
        "expected_mtp_tensor_count": expected_mtp_tensors or None,
        "weight_shards": shard_names,
        "total_weight_bytes": sum((model_dir / name).stat().st_size for name in shard_names),
    }


def _validate_restored_mtp(
    checkpoint_dir: Path,
    *,
    expected_mtp_num_layers: int,
    expected_mtp_shared_weights: bool,
) -> dict[str, Any]:
    args_path = checkpoint_dir / "args.json"
    if not args_path.is_file():
        raise CheckpointError(f"restored MCore checkpoint lacks {args_path}")
    checkpoint_args = json.loads(args_path.read_text(encoding="utf-8"))
    actual_layers = checkpoint_args.get("mtp_num_layers")
    actual_shared = bool(checkpoint_args.get("mtp_shared_weights", False))
    if actual_layers != expected_mtp_num_layers:
        raise CheckpointError(
            f"MCore checkpoint mtp_num_layers is {actual_layers!r}, "
            f"expected {expected_mtp_num_layers}"
        )
    if actual_shared != expected_mtp_shared_weights:
        raise CheckpointError(
            f"MCore checkpoint mtp_shared_weights is {actual_shared!r}, "
            f"expected {expected_mtp_shared_weights!r}"
        )
    return {
        "mtp_num_layers": actual_layers,
        "mtp_shared_weights": actual_shared,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _check_digest(args.checkpoint_manifest_sha256, name="checkpoint_manifest_sha256")
    _check_digest(args.initial_artifact_manifest_sha256, name="initial_artifact_manifest_sha256")
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    initial_dir = work_root / "initial-hf"
    checkpoint_dir = work_root / f"checkpoint-{args.checkpoint_step}"
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise CheckpointError(f"refusing to overwrite existing HF output: {output_dir}")
    if output_dir.parent != work_root:
        raise CheckpointError("--output-dir must be a direct child of --work-root")

    store = Boto3ObjectStore(region=args.aws_region, max_concurrency=args.s3_transfer_max_concurrency)
    checkpoint_marker = _read_marker(store, args.checkpoint_s3)
    if checkpoint_marker.get("step") != args.checkpoint_step:
        raise CheckpointError(
            f"checkpoint marker step={checkpoint_marker.get('step')!r}, expected={args.checkpoint_step}"
        )
    if checkpoint_marker.get("manifest_sha256") != args.checkpoint_manifest_sha256:
        raise CheckpointError("checkpoint manifest SHA-256 does not match the pinned value")
    checkpoint_inspection = inspect_committed_checkpoint(store, args.checkpoint_s3, verify_objects=True)

    initial_marker = _read_marker(store, args.initial_artifact_s3)
    if initial_marker.get("manifest_sha256") != args.initial_artifact_manifest_sha256:
        raise CheckpointError("initial HF artifact manifest SHA-256 does not match the pinned value")
    initial_restore = _restore_initial_artifact(
        store,
        args.initial_artifact_s3,
        initial_dir,
        marker=initial_marker,
        expected_manifest_sha256=args.initial_artifact_manifest_sha256,
    )
    checkpoint_restore = restore_checkpoint(
        store,
        args.checkpoint_s3,
        checkpoint_dir,
        replace_existing=True,
        download_workers=args.mcore_download_workers,
    )
    mtp_num_layers = getattr(args, "mtp_num_layers", DEFAULT_MTP_NUM_LAYERS)
    mtp_shared_weights = getattr(args, "mtp_shared_weights", False)
    checkpoint_mtp = _validate_restored_mtp(
        checkpoint_dir,
        expected_mtp_num_layers=mtp_num_layers,
        expected_mtp_shared_weights=mtp_shared_weights,
    )

    command = [
        args.megatron_bin,
        "export",
        "--to-hf",
        "true",
        "--output-dir",
        str(output_dir),
        "--use-hf",
        "true",
        "--mcore-model",
        str(checkpoint_dir),
        "--model",
        str(initial_dir),
        "--model-type",
        "qwen3_5",
        "--template",
        "qwen3_5",
        "--torch-dtype",
        "bfloat16",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--tuner-type",
        "full",
        "--mtp-num-layers",
        str(mtp_num_layers),
    ]
    if mtp_shared_weights:
        command.extend(["--mtp_shared_weights", "true"])
    environment = os.environ.copy()
    # Held UTP Pods default to the AWS OFI NCCL plugin. This single-rank export
    # uses the venv's NCCL build, whose plugin ABI may differ from the image's
    # OFI plugin, and needs no cross-node transport. Force the built-in socket
    # transport so even the world-size-one checkpoint all-reduce is deterministic.
    environment.pop("NCCL_NET_PLUGIN", None)
    environment.update(
        {
            "NPROC_PER_NODE": "1",
            "NNODES": "1",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29671",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_VISIBLE_DEVICES": args.conversion_gpu,
            "NCCL_IB_DISABLE": "1",
            "NCCL_NET": "Socket",
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    subprocess.run(command, check=True, env=environment)
    export_validation = _validate_export(
        output_dir,
        expected_tensors=args.expected_tensors,
        expected_mtp_tensors=args.expected_mtp_tensors,
    )
    artifact_commit: dict[str, Any] | None = None
    if args.artifact_s3:
        if args.expected_tensors != 738 or args.expected_mtp_tensors != 15:
            raise CheckpointError(
                "publishing through the trained MCore-to-HF protocol requires "
                "--expected-tensors 738 and --expected-mtp-tensors 15"
            )
        prepare_trained_artifact(
            output_dir,
            initial_dir,
            source_checkpoint_s3=args.checkpoint_s3,
            source_checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            source_step=args.checkpoint_step,
            initial_artifact_s3=args.initial_artifact_s3,
            initial_artifact_manifest_sha256=args.initial_artifact_manifest_sha256,
            export_command=command,
            mtp_num_layers=mtp_num_layers,
            mtp_shared_weights=mtp_shared_weights,
        )
        artifact_commit = publish_artifact(
            store,
            output_dir,
            args.artifact_s3,
            attempt_id=args.artifact_attempt_id,
        )
    result = {
        "status": "complete",
        "checkpoint": {
            **checkpoint_inspection,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        "checkpoint_restore": checkpoint_restore,
        "checkpoint_mtp": checkpoint_mtp,
        "initial_artifact": {
            **initial_restore,
            "manifest_sha256": args.initial_artifact_manifest_sha256,
        },
        "export_dir": str(output_dir),
        "export_command": command,
        "export_validation": export_validation,
        "artifact_s3": args.artifact_s3,
        "artifact_commit": artifact_commit,
    }
    (work_root / "conversion_provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.cleanup_sources:
        shutil.rmtree(initial_dir)
        shutil.rmtree(checkpoint_dir)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--initial-artifact-s3", required=True)
    parser.add_argument("--initial-artifact-manifest-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--mcore-download-workers", type=int, default=3)
    parser.add_argument("--s3-transfer-max-concurrency", type=int, default=6)
    parser.add_argument("--conversion-gpu", default="0")
    parser.add_argument("--megatron-bin", default="megatron")
    parser.add_argument("--mtp-num-layers", type=int, default=DEFAULT_MTP_NUM_LAYERS)
    parser.add_argument("--mtp-shared-weights", action="store_true")
    parser.add_argument(
        "--expected-tensors",
        type=int,
        default=DEFAULT_EXPECTED_TENSORS,
        help="Exact HF weight-map size; use 0 to accept any positive count",
    )
    parser.add_argument(
        "--expected-mtp-tensors",
        type=int,
        default=DEFAULT_EXPECTED_MTP_TENSORS,
        help="Exact MTP weight-map size; use 0 to accept any count",
    )
    parser.add_argument(
        "--artifact-s3",
        help="Optional immutable S3 destination for the sealed trained HF artifact",
    )
    parser.add_argument(
        "--artifact-attempt-id",
        help="Immutable publication attempt ID; required together with --artifact-s3",
    )
    parser.add_argument("--cleanup-sources", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint_step <= 0:
        parser.error("--checkpoint-step must be positive")
    if args.mtp_num_layers <= 0:
        parser.error("--mtp-num-layers must be positive")
    if args.mtp_num_layers != DEFAULT_MTP_NUM_LAYERS and not args.mtp_shared_weights:
        parser.error("multiple logical MTP steps require --mtp-shared-weights")
    if args.mcore_download_workers <= 0 or args.s3_transfer_max_concurrency <= 0:
        parser.error("download concurrency values must be positive")
    if args.expected_tensors < 0 or args.expected_mtp_tensors < 0:
        parser.error("expected tensor counts must be non-negative")
    if bool(args.artifact_s3) != bool(args.artifact_attempt_id):
        parser.error("--artifact-s3 and --artifact-attempt-id must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
