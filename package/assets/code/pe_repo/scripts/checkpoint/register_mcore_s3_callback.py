"""Register asynchronous native MCore -> S3 checkpoint publication.

MCore's async request owns the local checkpoint lifetime.  This module appends a
ready hook to that request *after* MCore's metadata finalizers.  The hook writes
the tiny streaming-resume contract when enabled, then starts a CPU-only uploader;
it never hashes or uploads checkpoint tensors on the training process.  Training
shutdown first drains MCore and then waits for every uploader.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable


# custom_register_path adds this file's directory, not necessarily the repository
# root, to sys.path. Make the protocol package importable in either launch mode.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.checkpoint.common import Boto3ObjectStore, CheckpointError, S3Location  # noqa: E402
from scripts.checkpoint.mcore_s3 import inspect_committed_checkpoint  # noqa: E402
from dlms.checkpoint.streaming_resume import (  # noqa: E402
    streaming_resume_enabled,
    write_checkpoint_metadata,
)
from swift.megatron.callbacks import megatron_callbacks_map  # noqa: E402
from swift.megatron.callbacks.base import MegatronCallback  # noqa: E402


CALLBACK_NAME = "mcore_s3_checkpoint"
_CHECKPOINT_DIR_RE = re.compile(r"checkpoint-(\d+)$")
_ITERATION_DIR_RE = re.compile(r"iter_\d+$")
_DIST_CALLBACK_ATTR = "_dlms_mcore_s3_active_callback"
_TRAINER_CALLBACK_ATTR = "_dlms_mcore_s3_callback"
_WRAPPER_ATTR = "_dlms_mcore_s3_wrapper"


class _UploadProcess:
    def __init__(self, checkpoint: Path, process: Any) -> None:
        self.checkpoint = checkpoint
        self.process = process
        self.returncode: int | None = None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise CheckpointError(f"{name} is required by {CALLBACK_NAME}")
    return value


def _required_int_env(name: str) -> int:
    raw = _required_env(name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise CheckpointError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise CheckpointError(f"{name} must be non-negative, got {value}")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise CheckpointError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise CheckpointError(f"{name} must be positive, got {value}")
    return value


def _checkpoint_uri(root_uri: str, output_dir: Path) -> str:
    if _CHECKPOINT_DIR_RE.fullmatch(output_dir.name) is None:
        raise CheckpointError(f"Unexpected Megatron on_save directory: {output_dir}")
    root = S3Location.parse(root_uri)
    return root.join(output_dir.name).uri + "/"


def _checkpoint_from_dist_save(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path:
    checkpoint_dir = kwargs.get("checkpoint_dir")
    if checkpoint_dir is None and len(args) >= 2:
        checkpoint_dir = args[1]
    if checkpoint_dir is None:
        raise CheckpointError("Could not determine checkpoint_dir from dist_checkpointing.save")
    path = Path(checkpoint_dir).absolute()
    checkpoint = path.parent if _ITERATION_DIR_RE.fullmatch(path.name) else path
    if _CHECKPOINT_DIR_RE.fullmatch(checkpoint.name) is None:
        raise CheckpointError(
            f"Unexpected MCore checkpoint directory passed to dist_checkpointing.save: {path}"
        )
    return checkpoint


def _dist() -> Any | None:
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    return dist if dist.is_available() and dist.is_initialized() else None


def _invocation_attempt_id(base_attempt_id: str) -> str:
    """Give every distributed callback invocation a shared, fresh namespace."""
    dist = _dist()
    if dist is None:
        # Keep non-distributed unit tests and administrative use deterministic.
        return base_attempt_id

    token_holder: list[str | None] = [uuid.uuid4().hex[:12] if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(token_holder, src=0)
    token = token_holder[0]
    if not isinstance(token, str) or re.fullmatch(r"[a-f0-9]{12}", token) is None:
        raise CheckpointError(f"Invalid invocation token broadcast by rank 0: {token!r}")
    attempt_id = f"{base_attempt_id}-{token}"
    if len(attempt_id) > 128:
        raise CheckpointError(
            "MCORE_S3_ATTEMPT_ID is too long to append the distributed invocation token: "
            f"base_length={len(base_attempt_id)} maximum_base_length=115"
        )
    return attempt_id


def _collective_stage(label: str, action: Callable[[], None] | None) -> None:
    """Run an elected-rank action and propagate its exception to every rank."""
    error = ""
    if action is not None:
        try:
            action()
        except Exception:
            error = traceback.format_exc()
    dist = _dist()
    if dist is None:
        errors = [error]
    else:
        errors = [""] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [(rank, message) for rank, message in enumerate(errors) if message]
    if failures:
        rank, message = failures[0]
        raise CheckpointError(f"{label} failed on global rank {rank}:\n{message}")


def _install_runtime_hooks(callback: "MCoreS3CheckpointCallback") -> None:
    """Install one process-wide DCP hook and one trainer shutdown hook."""

    megatron_lm_utils = importlib.import_module("swift.megatron.utils.megatron_lm_utils")
    dist_checkpointing = megatron_lm_utils.dist_checkpointing
    setattr(dist_checkpointing, _DIST_CALLBACK_ATTR, callback)

    original_save = dist_checkpointing.save
    if not getattr(original_save, _WRAPPER_ATTR, False):

        @functools.wraps(original_save)
        def save_with_s3_ready_hook(*args: Any, **kwargs: Any) -> Any:
            request = original_save(*args, **kwargs)
            active = getattr(dist_checkpointing, _DIST_CALLBACK_ATTR, None)
            if request is not None and active is not None:
                checkpoint = _checkpoint_from_dist_save(args, kwargs)
                request.add_finalize_fn(
                    functools.partial(active._on_checkpoint_ready, checkpoint)
                )
            return request

        setattr(save_with_s3_ready_hook, _WRAPPER_ATTR, True)
        dist_checkpointing.save = save_with_s3_ready_hook

    trainer_module = importlib.import_module("swift.megatron.trainers.base")
    trainer_class = trainer_module.BaseMegatronTrainer
    original_finalize = trainer_class.finalize_training
    if not getattr(original_finalize, _WRAPPER_ATTR, False):

        @functools.wraps(original_finalize)
        def finalize_with_s3_uploads(trainer: Any, *args: Any, **kwargs: Any) -> Any:
            # The original call performs blocking DCP finalization.  Only after
            # every ready hook has run is it safe to wait for upload children.
            result = original_finalize(trainer, *args, **kwargs)
            active = getattr(trainer, _TRAINER_CALLBACK_ATTR, None)
            if active is not None:
                active._finalize_uploads()
            return result

        setattr(finalize_with_s3_uploads, _WRAPPER_ATTR, True)
        trainer_class.finalize_training = finalize_with_s3_uploads

    setattr(callback.trainer, _TRAINER_CALLBACK_ATTR, callback)


class MCoreS3CheckpointCallback(MegatronCallback):
    """Start one rank-local uploader after each native async DCP save."""

    def __init__(self, trainer: Any):
        super().__init__(trainer)
        if not bool(getattr(self.args, "async_save", False)):
            raise CheckpointError(
                f"{CALLBACK_NAME} requires --async_save true"
            )
        if bool(getattr(self.args, "save_safetensors", True)):
            raise CheckpointError(f"{CALLBACK_NAME} requires --save_safetensors false")
        self.root_s3 = _required_env("MCORE_S3_ROOT")
        self.attempt_id = _invocation_attempt_id(_required_env("MCORE_S3_ATTEMPT_ID"))
        self.node_count = _required_int_env("NNODES")
        self.node_rank = _required_int_env("NODE_RANK")
        self.local_rank = int(os.environ.get("LOCAL_RANK", getattr(self.args, "local_rank", -1)))
        self.global_rank = int(os.environ.get("RANK", getattr(self.args, "rank", -1)))
        self.world_size = int(os.environ.get("WORLD_SIZE", getattr(self.args, "global_world_size", 0)))
        self.save_total_limit = int(getattr(self.args, "save_total_limit", 0) or 0)
        self.local_save_total_limit = int(
            os.environ.get("MCORE_LOCAL_SAVE_TOTAL_LIMIT") or self.save_total_limit
        )
        self.commit_wait_seconds = _positive_float_env("MCORE_S3_COMMIT_WAIT_SECONDS", 7200)
        self.poll_seconds = _positive_float_env("MCORE_S3_POLL_SECONDS", 5)
        if self.node_count <= 0 or not 0 <= self.node_rank < self.node_count:
            raise CheckpointError(f"Invalid NODE_RANK/NNODES: {self.node_rank}/{self.node_count}")
        if self.local_rank < 0 or self.global_rank < 0 or self.world_size <= 0:
            raise CheckpointError(
                "LOCAL_RANK, RANK and WORLD_SIZE must be set by torchrun before callback construction"
            )
        self.store = Boto3ObjectStore(
            profile=os.environ.get("AWS_PROFILE") or None,
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self._expected: set[Path] = set()
        self._uploads: dict[Path, _UploadProcess] = {}
        self._spawn_errors: dict[Path, str] = {}
        _install_runtime_hooks(self)

    def on_save(self, output_dir: str) -> None:
        # Deliberately bookkeeping-only: async DCP files are not complete here.
        checkpoint = Path(output_dir).absolute()
        _checkpoint_uri(self.root_s3, checkpoint)
        self._expected.add(checkpoint)

    def on_step_begin(self) -> None:
        self._poll_uploads(wait=False)
        failure = self._failed_upload_message(require_all_ready=False)
        if failure:
            # Fail fast.  Let the job launcher terminate peers instead of
            # allowing rank 0/local-rank 0 workers to wait for missing
            # manifests until the commit timeout.
            raise CheckpointError(failure)

    def _worker_command(self, checkpoint: Path) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "scripts.checkpoint.mcore_s3_upload_worker",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-s3",
            _checkpoint_uri(self.root_s3, checkpoint),
            "--attempt-id",
            self.attempt_id,
            "--rank",
            str(self.global_rank),
            "--world-size",
            str(self.world_size),
            "--node-rank",
            str(self.node_rank),
            "--node-count",
            str(self.node_count),
            "--local-rank",
            str(self.local_rank),
            "--commit-wait-seconds",
            str(self.commit_wait_seconds),
            "--poll-seconds",
            str(self.poll_seconds),
            "--region",
            os.environ.get("AWS_REGION", "us-east-1"),
        ]
        profile = os.environ.get("AWS_PROFILE", "").strip()
        if profile:
            command.extend(("--profile", profile))
        if self.local_save_total_limit:
            command.extend(
                (
                    "--output-root",
                    str(Path(self.args.output_dir).absolute()),
                    "--local-save-total-limit",
                    str(self.local_save_total_limit),
                )
            )
        return command

    def _on_checkpoint_ready(self, checkpoint: Path) -> None:
        """MCore finalizer: seal resume metadata, spawn uploader, and return."""

        checkpoint = checkpoint.absolute()
        self._expected.add(checkpoint)
        if checkpoint in self._uploads or checkpoint in self._spawn_errors:
            return
        if streaming_resume_enabled() and self.global_rank == 0:
            try:
                from megatron.core import parallel_state as mpu

                metadata_path = write_checkpoint_metadata(
                    checkpoint,
                    self.args,
                    data_parallel_world_size=mpu.get_data_parallel_world_size(),
                )
                print(
                    f"[mcore-s3] wrote streaming resume contract: {metadata_path}",
                    flush=True,
                )
            except Exception:
                # Do not publish a checkpoint that would silently restart the
                # iterable stream.  Surface this through the normal fail-fast
                # uploader path on the next step/finalization.
                self._spawn_errors[checkpoint] = traceback.format_exc()
                return
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                self._worker_command(checkpoint),
                cwd=str(_REPO_ROOT),
                env=environment,
                # Inherit stdout/stderr so uploader progress and failures stay
                # visible in the rank's normal pod logs without pipe deadlocks.
            )
        except Exception:
            self._spawn_errors[checkpoint] = traceback.format_exc()
            return
        self._uploads[checkpoint] = _UploadProcess(checkpoint, process)
        print(
            f"[mcore-s3] spawned CPU uploader pid={process.pid} for {checkpoint.name}",
            flush=True,
        )

    def _poll_uploads(self, *, wait: bool) -> None:
        for upload in self._uploads.values():
            if upload.returncode is not None:
                continue
            returncode = upload.process.wait() if wait else upload.process.poll()
            if returncode is None:
                continue
            upload.returncode = int(returncode)
            print(
                f"[mcore-s3] uploader pid={upload.process.pid} checkpoint="
                f"{upload.checkpoint.name} exited rc={upload.returncode}",
                flush=True,
            )

    def _failed_upload_message(self, *, require_all_ready: bool) -> str:
        failures = [
            f"{checkpoint}: failed to spawn\n{message}"
            for checkpoint, message in sorted(self._spawn_errors.items())
        ]
        failures.extend(
            f"{checkpoint}: uploader exited with status {upload.returncode}"
            for checkpoint, upload in sorted(self._uploads.items())
            if upload.returncode not in {None, 0}
        )
        if require_all_ready:
            missing = sorted(self._expected - self._uploads.keys() - self._spawn_errors.keys())
            failures.extend(f"{checkpoint}: DCP ready hook did not start an uploader" for checkpoint in missing)
        return "\n".join(failures)

    @staticmethod
    def _checkpoint_step(checkpoint: Path) -> int:
        match = _CHECKPOINT_DIR_RE.fullmatch(checkpoint.name)
        if match is None:
            raise CheckpointError(f"Unexpected Megatron checkpoint directory: {checkpoint}")
        return int(match.group(1))

    def _latest_checkpoint(self) -> Path | None:
        return max(self._expected, key=self._checkpoint_step) if self._expected else None

    def _wait_for_local_uploads(self) -> None:
        self._poll_uploads(wait=True)
        failure = self._failed_upload_message(require_all_ready=True)
        if failure:
            raise CheckpointError(failure)

    def _verify_latest_commit(self) -> None:
        latest = self._latest_checkpoint()
        if latest is None:
            return
        inspect_committed_checkpoint(
            self.store,
            _checkpoint_uri(self.root_s3, latest),
            verify_objects=True,
        )

    def _finalize_uploads(self) -> None:
        _collective_stage("async checkpoint upload", self._wait_for_local_uploads)
        _collective_stage(
            "latest checkpoint commit verification",
            self._verify_latest_commit if self.global_rank == 0 else None,
        )


existing = megatron_callbacks_map.get(CALLBACK_NAME)
if existing not in {None, MCoreS3CheckpointCallback}:
    raise RuntimeError(f"Megatron callback name collision: {CALLBACK_NAME}")
megatron_callbacks_map[CALLBACK_NAME] = MCoreS3CheckpointCallback
print(f"[mcore-s3] registered Megatron callback: {CALLBACK_NAME}", flush=True)
