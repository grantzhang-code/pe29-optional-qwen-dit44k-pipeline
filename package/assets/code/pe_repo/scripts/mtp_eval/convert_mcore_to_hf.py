#!/usr/bin/env python3
"""Restore an explicitly pinned complete MCore checkpoint and publish trained HF weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    parse_json_bytes,
    sha256_bytes,
    sha256_file,
)
from scripts.checkpoint.hf_artifact_s3 import (
    MANIFEST_NAME as LEGACY_MANIFEST_NAME,
    restore_artifact as restore_legacy_artifact,
    validate_local_artifact as validate_legacy_local_artifact,
)
from scripts.checkpoint.mcore_s3 import inspect_committed_checkpoint, restore_checkpoint
from scripts.mtp_eval.hf_artifact import (
    INITIAL_REFERENCE_MODE,
    InitialReference,
    MTP_NUM_LAYERS,
    legacy_artifact_lock,
    prepare_trained_artifact,
    publish_artifact,
    restore_initial_metadata_mtp_reference,
    validate_local_artifact,
    validate_initial_metadata_mtp_reference,
)


DEFAULT_WAIT_SECONDS = 60 * 60
DEFAULT_POLL_SECONDS = 30.0
FULL_INITIAL_RESTORE_MODE = "full"
VALID_INITIAL_RESTORE_MODES = {FULL_INITIAL_RESTORE_MODE, INITIAL_REFERENCE_MODE}
SELECTIVE_RUNTIME_VERSIONS = {
    "ms-swift": "4.4.2",
    "mcore-bridge": "1.6.1",
}
SELECTIVE_RUNTIME_IMAGE_DIGEST = "sha256:3da1b7636d5345e9b4bd3c1a7016c810bf2277e910110055b4c045d7c34d9dc2"


def _normalise_root(uri: str) -> S3Location:
    location = S3Location.parse(uri)
    return S3Location(location.bucket, location.key.rstrip("/"))


def exact_checkpoint_uri(root_s3: str, step: int) -> str:
    if step <= 0:
        raise CheckpointError(f"Checkpoint step must be positive, got {step}")
    return _normalise_root(root_s3).join(f"checkpoint-{step}").uri + "/"


def lock_checkpoint(
    store: ObjectStore,
    root_s3: str,
    *,
    step: int,
    expected_manifest_sha256: str | None = None,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait for, validate, and lock the exact checkpoint commit.

    A higher directory name, tracker file, or partial upload is never accepted;
    the exact step must have a valid immutable ``_SUCCESS`` and global manifest.
    """

    if wait_seconds < 0 or poll_seconds <= 0:
        raise CheckpointError("wait_seconds must be non-negative and poll_seconds must be positive")
    checkpoint_uri = exact_checkpoint_uri(root_s3, step)
    success_uri = _normalise_root(checkpoint_uri).join("_SUCCESS").uri
    deadline = monotonic() + wait_seconds
    while not store.exists(success_uri):
        if monotonic() >= deadline:
            raise CheckpointError(
                f"Timed out waiting for committed checkpoint-{step} _SUCCESS: {success_uri}"
            )
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))

    # Once _SUCCESS exists it is immutable. Any validation failure is terminal,
    # rather than silently waiting on a corrupt commit that cannot be replaced.
    inspected = inspect_committed_checkpoint(store, checkpoint_uri, verify_objects=True)
    success_data = store.read_bytes(success_uri)
    success_digest = sha256_bytes(success_data)
    success_stat = store.stat(success_uri)
    if success_stat.size != len(success_data) or success_stat.metadata.get("sha256") != success_digest:
        raise CheckpointError(f"Checkpoint _SUCCESS envelope mismatch: {success_uri}")
    marker = parse_json_bytes(success_data, source=success_uri)
    manifest_sha256 = marker.get("manifest_sha256")
    if marker.get("status") != "complete" or marker.get("step") != step:
        raise CheckpointError(f"Checkpoint marker is not the complete pinned step {step}: {marker!r}")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise CheckpointError(f"Checkpoint marker has no valid manifest digest: {success_uri}")
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise CheckpointError(
            "Pinned checkpoint manifest changed: "
            f"expected={expected_manifest_sha256} actual={manifest_sha256}"
        )
    if inspected["step"] != step or inspected["manifest_uri"] != marker.get("manifest_uri"):
        raise CheckpointError("Checkpoint inspection and _SUCCESS marker disagree")
    return {
        **inspected,
        "manifest_sha256": manifest_sha256,
        "success_uri": success_uri,
        "success_sha256": success_digest,
    }


def build_export_command(
    mcore_checkpoint: Path,
    initial_model: Path,
    output_dir: Path,
    *,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    mtp_num_layers: int = MTP_NUM_LAYERS,
    mtp_shared_weights: bool = False,
    megatron_bin: str = "megatron",
) -> list[str]:
    if min(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size) <= 0:
        raise CheckpointError("Megatron parallel sizes must be positive")
    if mtp_num_layers <= 0:
        raise CheckpointError("MTP layer count must be positive")
    if mtp_num_layers != MTP_NUM_LAYERS and not mtp_shared_weights:
        raise CheckpointError(
            "Multiple logical MTP steps require shared weights for the one-head HF contract"
        )
    command = [
        megatron_bin,
        "export",
        "--to-hf",
        "true",
        "--output-dir",
        str(output_dir),
        "--use-hf",
        "true",
        "--mcore-model",
        str(mcore_checkpoint),
        "--model",
        str(initial_model),
        "--model-type",
        "qwen3_5",
        "--template",
        "qwen3_5",
        "--torch-dtype",
        "bfloat16",
        "--tensor-model-parallel-size",
        str(tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_model_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
        "--tuner-type",
        "full",
        "--mtp-num-layers",
        str(mtp_num_layers),
    ]
    if mtp_shared_weights:
        command.extend(["--mtp_shared_weights", "true"])
    return command


def export_environment(
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    required = {
        "NPROC_PER_NODE": str(world_size),
        "NNODES": "1",
        "NODE_RANK": "0",
    }
    for key, value in required.items():
        current = env.get(key)
        if current is not None and current != value:
            raise CheckpointError(f"Export environment {key} must be {value}, got {current}")
        env[key] = value
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29671")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    return env


def _restore_initial_artifact(
    store: ObjectStore,
    artifact_s3: str,
    output_dir: Path,
    *,
    manifest_sha256: str,
    restore_mode: str,
) -> tuple[dict[str, Any], Path, InitialReference | None]:
    if restore_mode not in VALID_INITIAL_RESTORE_MODES:
        raise CheckpointError(f"Unsupported initial artifact restore mode: {restore_mode!r}")
    if restore_mode == INITIAL_REFERENCE_MODE:
        if output_dir.exists():
            reference = validate_initial_metadata_mtp_reference(
                output_dir,
                expected_artifact_s3=artifact_s3,
                expected_manifest_sha256=manifest_sha256,
            )
            reused_existing = True
        else:
            reference = restore_initial_metadata_mtp_reference(
                store,
                artifact_s3,
                output_dir,
                expected_manifest_sha256=manifest_sha256,
            )
            reused_existing = False
        return (
            {
                "output_dir": str(output_dir),
                "model_dir": str(reference.metadata_dir),
                "reference_dir": str(reference.reference_dir),
                "mode": restore_mode,
                "manifest_sha256": reference.manifest_sha256,
                "downloaded_files": list(reference.downloaded_files),
                "skipped_weight_files": list(reference.skipped_weight_files),
                "downloaded_bytes": reference.downloaded_bytes,
                "skipped_bytes": reference.skipped_bytes,
                "reused_existing": reused_existing,
            },
            reference.metadata_dir,
            reference,
        )

    if output_dir.exists():
        validate_legacy_local_artifact(output_dir)
        local_digest = sha256_file(output_dir / LEGACY_MANIFEST_NAME)
        if local_digest != manifest_sha256:
            raise CheckpointError(
                f"Existing initial artifact does not match the remote lock: "
                f"local={local_digest} remote={manifest_sha256}"
            )
        return (
            {
                "output_dir": str(output_dir),
                "model_dir": str(output_dir),
                "mode": restore_mode,
                "reused_existing": True,
            },
            output_dir,
            None,
        )
    result = restore_legacy_artifact(store, artifact_s3, output_dir)
    local_digest = sha256_file(output_dir / LEGACY_MANIFEST_NAME)
    if local_digest != manifest_sha256:
        raise CheckpointError(
            f"Restored initial artifact manifest mismatch: local={local_digest} remote={manifest_sha256}"
        )
    return (
        {
            **result,
            "model_dir": str(output_dir),
            "mode": restore_mode,
            "reused_existing": False,
        },
        output_dir,
        None,
    )


def validate_selective_runtime(
    restore_mode: str,
    *,
    runtime_image_uri: str | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    if restore_mode not in VALID_INITIAL_RESTORE_MODES:
        raise CheckpointError(f"Unsupported initial artifact restore mode: {restore_mode!r}")
    if restore_mode == FULL_INITIAL_RESTORE_MODE:
        return {}
    if (
        not isinstance(runtime_image_uri, str)
        or runtime_image_uri.rpartition("@")[2] != SELECTIVE_RUNTIME_IMAGE_DIGEST
    ):
        raise CheckpointError(
            "Selective initial restore is only audited for the pinned converter image: "
            f"expected_digest={SELECTIVE_RUNTIME_IMAGE_DIGEST} actual={runtime_image_uri!r}"
        )
    versions: dict[str, str] = {}
    for distribution, expected in SELECTIVE_RUNTIME_VERSIONS.items():
        try:
            actual = version_getter(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CheckpointError(
                f"Selective initial restore requires {distribution}=={expected}, but it is absent"
            ) from exc
        if actual != expected:
            raise CheckpointError(
                "Selective initial restore has not been audited for this runtime: "
                f"{distribution} expected={expected} actual={actual}"
            )
        versions[distribution] = actual
    return versions


def _validate_restored_mcore(
    path: Path,
    checkpoint_lock: dict[str, Any],
    *,
    expected_mtp_num_layers: int = MTP_NUM_LAYERS,
    expected_mtp_shared_weights: bool = False,
) -> None:
    restore_metadata_path = path / ".s3_restore.json"
    if not restore_metadata_path.is_file():
        raise CheckpointError(f"Restored MCore checkpoint lacks {restore_metadata_path}")
    restore_metadata = json.loads(restore_metadata_path.read_text(encoding="utf-8"))
    if (
        restore_metadata.get("manifest_sha256") != checkpoint_lock["manifest_sha256"]
        or restore_metadata.get("source_checkpoint") != checkpoint_lock["checkpoint_uri"]
    ):
        raise CheckpointError("Restored MCore checkpoint does not match the immutable checkpoint lock")
    args_path = path / "args.json"
    if not args_path.is_file():
        raise CheckpointError(f"Restored MCore checkpoint lacks {args_path}")
    checkpoint_args = json.loads(args_path.read_text(encoding="utf-8"))
    if checkpoint_args.get("mtp_num_layers") != expected_mtp_num_layers:
        raise CheckpointError(
            f"MCore checkpoint mtp_num_layers is {checkpoint_args.get('mtp_num_layers')!r}, "
            f"expected {expected_mtp_num_layers}"
        )
    if bool(checkpoint_args.get("mtp_shared_weights", False)) != expected_mtp_shared_weights:
        raise CheckpointError(
            "MCore checkpoint mtp_shared_weights is "
            f"{checkpoint_args.get('mtp_shared_weights', False)!r}, "
            f"expected {expected_mtp_shared_weights!r}"
        )


def run_conversion(
    store: ObjectStore,
    *,
    mcore_s3_root: str,
    checkpoint_step: int,
    initial_artifact_s3: str,
    initial_artifact_manifest_sha256: str | None = None,
    initial_restore_mode: str = FULL_INITIAL_RESTORE_MODE,
    artifact_s3: str,
    attempt_id: str,
    work_root: Path,
    expected_checkpoint_manifest_sha256: str | None = None,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    mtp_num_layers: int = MTP_NUM_LAYERS,
    mtp_shared_weights: bool = False,
    mcore_download_workers: int = 1,
    megatron_bin: str = "megatron",
    runner: Callable[..., Any] = subprocess.run,
    runtime_image_uri: str | None = None,
    runtime_version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    if (
        not isinstance(mcore_download_workers, int)
        or isinstance(mcore_download_workers, bool)
        or mcore_download_workers < 1
    ):
        raise CheckpointError(
            f"mcore_download_workers must be a positive integer, got {mcore_download_workers!r}"
        )
    runtime_versions = validate_selective_runtime(
        initial_restore_mode,
        runtime_image_uri=runtime_image_uri,
        version_getter=runtime_version_getter,
    )
    if initial_restore_mode == INITIAL_REFERENCE_MODE and initial_artifact_manifest_sha256 is None:
        raise CheckpointError("metadata-mtp initial restore requires --initial-artifact-manifest-sha256")
    checkpoint_lock = lock_checkpoint(
        store,
        mcore_s3_root,
        step=checkpoint_step,
        expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
    )
    initial_lock = legacy_artifact_lock(store, initial_artifact_s3)
    if (
        initial_artifact_manifest_sha256 is not None
        and initial_lock["manifest_sha256"] != initial_artifact_manifest_sha256
    ):
        raise CheckpointError(
            "Pinned initial artifact manifest changed: "
            f"expected={initial_artifact_manifest_sha256} "
            f"actual={initial_lock['manifest_sha256']}"
        )

    work_root = Path(os.path.abspath(os.fspath(work_root)))
    if work_root.is_symlink():
        raise CheckpointError(f"Refusing symlink work root: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    initial_dir = work_root / "initial-hf"
    checkpoint_dir = work_root / f"checkpoint-{checkpoint_step}"
    output_dir = work_root / f"checkpoint-{checkpoint_step}-trained-hf"

    initial_restore, initial_model_dir, initial_reference = _restore_initial_artifact(
        store,
        initial_artifact_s3,
        initial_dir,
        manifest_sha256=initial_lock["manifest_sha256"],
        restore_mode=initial_restore_mode,
    )
    checkpoint_restore = restore_checkpoint(
        store,
        checkpoint_lock["checkpoint_uri"],
        checkpoint_dir,
        replace_existing=True,
        download_workers=mcore_download_workers,
    )
    _validate_restored_mcore(
        checkpoint_dir,
        checkpoint_lock,
        expected_mtp_num_layers=mtp_num_layers,
        expected_mtp_shared_weights=mtp_shared_weights,
    )

    command = build_export_command(
        checkpoint_dir,
        initial_model_dir,
        output_dir,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        mtp_num_layers=mtp_num_layers,
        mtp_shared_weights=mtp_shared_weights,
        megatron_bin=megatron_bin,
    )
    if output_dir.exists():
        manifest, _ = validate_local_artifact(output_dir)
        source = manifest["source_mcore_checkpoint"]
        export = manifest["export"]
        if (
            source["step"] != checkpoint_step
            or source["manifest_sha256"] != checkpoint_lock["manifest_sha256"]
            or export["mtp_num_layers"] != mtp_num_layers
            or export.get("mtp_shared_weights", False) != mtp_shared_weights
        ):
            raise CheckpointError(f"Existing sealed export has different provenance: {output_dir}")
        reused_export = True
    else:
        env = export_environment(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )
        runner(command, check=True, env=env)
        if not output_dir.is_dir():
            raise CheckpointError(f"Megatron export produced no output directory: {output_dir}")
        prepare_trained_artifact(
            output_dir,
            initial_model_dir,
            initial_reference=initial_reference,
            source_checkpoint_s3=checkpoint_lock["checkpoint_uri"],
            source_checkpoint_manifest_sha256=checkpoint_lock["manifest_sha256"],
            source_step=checkpoint_step,
            initial_artifact_s3=initial_artifact_s3,
            initial_artifact_manifest_sha256=initial_lock["manifest_sha256"],
            export_command=command,
            mtp_num_layers=mtp_num_layers,
            mtp_shared_weights=mtp_shared_weights,
        )
        reused_export = False
    artifact_marker = publish_artifact(
        store,
        output_dir,
        artifact_s3,
        attempt_id=attempt_id,
    )
    return {
        "status": "complete",
        "source_step": checkpoint_step,
        "checkpoint_lock": checkpoint_lock,
        "initial_artifact_lock": initial_lock,
        "initial_restore": initial_restore,
        "initial_restore_mode": initial_restore_mode,
        "selective_runtime_versions": runtime_versions,
        "selective_runtime_image_uri": (
            runtime_image_uri if initial_restore_mode == INITIAL_REFERENCE_MODE else None
        ),
        "checkpoint_restore": checkpoint_restore,
        "export_command": command,
        "export_dir": str(output_dir),
        "reused_export": reused_export,
        "artifact_s3": artifact_s3,
        "artifact_commit": artifact_marker,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--mcore-s3-root", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--initial-artifact-s3", required=True)
    parser.add_argument("--initial-artifact-manifest-sha256")
    parser.add_argument(
        "--initial-restore-mode",
        choices=sorted(VALID_INITIAL_RESTORE_MODES),
        default=FULL_INITIAL_RESTORE_MODE,
    )
    parser.add_argument("--runtime-image-uri")
    parser.add_argument("--artifact-s3", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--mtp-num-layers", type=int, default=MTP_NUM_LAYERS)
    parser.add_argument("--mtp-shared-weights", action="store_true")
    parser.add_argument(
        "--mcore-download-workers",
        type=int,
        default=1,
        help="Concurrent MCore manifest-file downloads (default: 1)",
    )
    parser.add_argument(
        "--s3-transfer-max-concurrency",
        type=int,
        default=8,
        help=(
            "Maximum boto3 multipart concurrency per file. Its product with "
            "--mcore-download-workers bounds aggregate MCore download concurrency."
        ),
    )
    parser.add_argument("--megatron-bin", default="megatron")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = Boto3ObjectStore(
            profile=args.profile,
            region=args.region,
            max_concurrency=args.s3_transfer_max_concurrency,
        )
        result = run_conversion(
            store,
            mcore_s3_root=args.mcore_s3_root,
            checkpoint_step=args.checkpoint_step,
            initial_artifact_s3=args.initial_artifact_s3,
            initial_artifact_manifest_sha256=args.initial_artifact_manifest_sha256,
            initial_restore_mode=args.initial_restore_mode,
            runtime_image_uri=args.runtime_image_uri,
            artifact_s3=args.artifact_s3,
            attempt_id=args.attempt_id,
            work_root=args.work_root,
            expected_checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            mtp_num_layers=args.mtp_num_layers,
            mtp_shared_weights=args.mtp_shared_weights,
            mcore_download_workers=args.mcore_download_workers,
            megatron_bin=args.megatron_bin,
        )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
