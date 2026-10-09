"""Fail-closed resume contract for rank-local streaming DataLoaders.

Native MCore checkpoints restore the optimizer iteration and
``consumed_train_samples``, but ms-swift's streaming DataLoader ignores that
counter and starts an iterable dataset from the beginning.  The rank-local
loader uses the helpers here to replay and discard exactly the *collated
microbatches* committed by the checkpoint.  Replaying after packing is
important: one packed sequence contains a variable number of source rows.

The producer cursor is deliberately not checkpointed.  PyTorch DataLoader
workers prefetch beyond the last batch consumed by the optimizer, so that
cursor would skip unseen data after a restart.  The logical committed counter
is the only safe recovery boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_VERSION = 1
METADATA_NAME = "dlms_streaming_resume.json"
ENABLED_ENV = "DLMS_STREAMING_RESUME_ENABLED"
DATASET_CONTRACT_ENV = "DLMS_STREAM_DATASET_CONTRACT_SHA256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


class StreamingResumeError(RuntimeError):
    """The checkpoint cannot safely recover the streaming sample position."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise StreamingResumeError(f"{name} must be a boolean, got {raw!r}")


def streaming_resume_enabled() -> bool:
    return _env_bool(ENABLED_ENV, False)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def install_dataset_contract(contract: Mapping[str, Any]) -> str:
    """Publish an immutable dataset-stream fingerprint for the loader/callback."""

    digest = canonical_sha256(contract)
    existing = os.environ.get(DATASET_CONTRACT_ENV, "").strip()
    if existing and existing != digest:
        raise StreamingResumeError(
            f"{DATASET_CONTRACT_ENV} already has a conflicting value: "
            f"{existing} != {digest}"
        )
    os.environ[DATASET_CONTRACT_ENV] = digest
    return digest


def _required_dataset_contract() -> str:
    value = os.environ.get(DATASET_CONTRACT_ENV, "").strip()
    if _SHA256_RE.fullmatch(value) is None:
        raise StreamingResumeError(
            f"{DATASET_CONTRACT_ENV} must be installed by the dataset loader before "
            "building a resumable streaming DataLoader"
        )
    return value


def _plain_arg(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, name, default)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    return str(value)


def build_replay_contract(args: Any, *, data_parallel_world_size: int) -> dict[str, Any]:
    """Capture every setting that can change rank-local packed sample order."""

    if data_parallel_world_size <= 0:
        raise StreamingResumeError(
            f"data_parallel_world_size must be positive, got {data_parallel_world_size}"
        )
    micro_batch_size = int(_plain_arg(args, "micro_batch_size", 0) or 0)
    global_batch_size = int(_plain_arg(args, "global_batch_size", 0) or 0)
    local_step_samples = data_parallel_world_size * micro_batch_size
    if micro_batch_size <= 0 or global_batch_size <= 0:
        raise StreamingResumeError(
            f"invalid batch sizes: MBS={micro_batch_size} GBS={global_batch_size}"
        )
    if global_batch_size % local_step_samples:
        raise StreamingResumeError(
            "GBS must be divisible by DP*MBS for deterministic streaming replay: "
            f"GBS={global_batch_size} DP={data_parallel_world_size} MBS={micro_batch_size}"
        )

    return {
        "version": PROTOCOL_VERSION,
        "kind": "dlms-rank-local-streaming-replay-contract",
        "replay_unit": "post-packing-collated-microbatch",
        "dataset_contract_sha256": _required_dataset_contract(),
        "data_parallel_world_size": data_parallel_world_size,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "num_microbatches": global_batch_size // local_step_samples,
        # Dataset/tokenization settings can change filtering, lengths, or order.
        "dataset": _plain_arg(args, "dataset"),
        "seed": _plain_arg(args, "seed"),
        "data_seed": _plain_arg(args, "data_seed"),
        "streaming": _plain_arg(args, "streaming"),
        "dataset_shuffle": _plain_arg(args, "dataset_shuffle"),
        "split_dataset_ratio": _plain_arg(args, "split_dataset_ratio"),
        "eval_iters": _plain_arg(args, "eval_iters"),
        "dataset_num_proc": _plain_arg(args, "dataset_num_proc"),
        "dataloader_num_workers": _plain_arg(args, "dataloader_num_workers"),
        "packing": _plain_arg(args, "packing"),
        "packing_length": _plain_arg(args, "packing_length"),
        "packing_strategy": _plain_arg(args, "packing_strategy"),
        "max_length": _plain_arg(args, "max_length"),
        "truncation_strategy": _plain_arg(args, "truncation_strategy"),
        "template": _plain_arg(args, "template"),
        "model_type": _plain_arg(args, "model_type"),
        "add_non_thinking_prefix": _plain_arg(args, "add_non_thinking_prefix"),
        "response_prefix": _plain_arg(args, "response_prefix"),
        "max_pixels": _plain_arg(args, "max_pixels"),
        "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM", ""),
        # A DP-layout change reassigns physical MDS shards, even if world size is equal.
        "tensor_model_parallel_size": _plain_arg(args, "tensor_model_parallel_size"),
        "pipeline_model_parallel_size": _plain_arg(args, "pipeline_model_parallel_size"),
        "context_parallel_size": _plain_arg(args, "context_parallel_size"),
    }


def local_microbatches_to_skip(
    consumed_train_samples: int,
    *,
    data_parallel_world_size: int,
    micro_batch_size: int,
    global_batch_size: int,
) -> int:
    """Convert MCore's global committed sample counter to this DP rank's batches."""

    values = {
        "consumed_train_samples": consumed_train_samples,
        "data_parallel_world_size": data_parallel_world_size,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
    }
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise StreamingResumeError(f"resume counters must be non-negative integers: {values}")
    if data_parallel_world_size == 0 or micro_batch_size == 0 or global_batch_size == 0:
        raise StreamingResumeError(f"resume batch dimensions must be positive: {values}")
    local_batch_span = data_parallel_world_size * micro_batch_size
    if global_batch_size % local_batch_span:
        raise StreamingResumeError(
            f"GBS={global_batch_size} is not divisible by DP*MBS={local_batch_span}"
        )
    if consumed_train_samples % global_batch_size:
        raise StreamingResumeError(
            "checkpoint consumed_train_samples is not an optimizer-step boundary: "
            f"consumed={consumed_train_samples} GBS={global_batch_size}"
        )
    if consumed_train_samples % local_batch_span:
        raise StreamingResumeError(
            "checkpoint consumed_train_samples cannot be divided across DP ranks: "
            f"consumed={consumed_train_samples} DP*MBS={local_batch_span}"
        )
    return consumed_train_samples // local_batch_span


def build_checkpoint_metadata(
    args: Any,
    *,
    data_parallel_world_size: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    contract = build_replay_contract(
        args,
        data_parallel_world_size=data_parallel_world_size,
    )
    global_batch_size = int(contract["global_batch_size"])
    consumed_train_samples = checkpoint_step * global_batch_size
    return {
        "version": PROTOCOL_VERSION,
        "kind": "dlms-rank-local-streaming-resume",
        "status": "complete",
        "checkpoint_step": checkpoint_step,
        "consumed_train_samples": consumed_train_samples,
        "local_microbatches_to_skip": local_microbatches_to_skip(
            consumed_train_samples,
            data_parallel_world_size=data_parallel_world_size,
            micro_batch_size=int(contract["micro_batch_size"]),
            global_batch_size=global_batch_size,
        ),
        "contract_sha256": canonical_sha256(contract),
        "contract": contract,
    }


def write_checkpoint_metadata(
    checkpoint: Path,
    args: Any,
    *,
    data_parallel_world_size: int,
) -> Path | None:
    """Atomically add the resume contract before rank 0 publishes a checkpoint."""

    if not streaming_resume_enabled():
        return None
    checkpoint = checkpoint.absolute()
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        raise StreamingResumeError(f"unexpected checkpoint directory: {checkpoint}")
    metadata = build_checkpoint_metadata(
        args,
        data_parallel_world_size=data_parallel_world_size,
        checkpoint_step=int(match.group(1)),
    )
    destination = checkpoint / METADATA_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{METADATA_NAME}.", suffix=".tmp", dir=checkpoint
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_resume_checkpoint(
    checkpoint: Path,
    args: Any,
    *,
    data_parallel_world_size: int,
    consumed_train_samples: int,
) -> dict[str, Any]:
    """Validate a restored checkpoint and return its replay plan."""

    checkpoint = checkpoint.absolute()
    metadata_path = checkpoint / METADATA_NAME
    if not metadata_path.is_file():
        raise StreamingResumeError(
            f"resumable streaming checkpoint is missing {metadata_path}; refusing to "
            "restart the iterable dataset from the beginning"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StreamingResumeError(f"invalid streaming resume metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise StreamingResumeError(f"streaming resume metadata must be an object: {metadata_path}")

    expected_contract = build_replay_contract(
        args,
        data_parallel_world_size=data_parallel_world_size,
    )
    expected_digest = canonical_sha256(expected_contract)
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": "dlms-rank-local-streaming-resume",
        "status": "complete",
        "consumed_train_samples": consumed_train_samples,
        "contract_sha256": expected_digest,
        "contract": expected_contract,
    }
    mismatches = {
        key: {"checkpoint": metadata.get(key), "current": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    checkpoint_step = int(match.group(1)) if match else -1
    if metadata.get("checkpoint_step") != checkpoint_step:
        mismatches["checkpoint_step"] = {
            "checkpoint": metadata.get("checkpoint_step"),
            "current": checkpoint_step,
        }
    batches = local_microbatches_to_skip(
        consumed_train_samples,
        data_parallel_world_size=data_parallel_world_size,
        micro_batch_size=int(expected_contract["micro_batch_size"]),
        global_batch_size=int(expected_contract["global_batch_size"]),
    )
    if metadata.get("local_microbatches_to_skip") != batches:
        mismatches["local_microbatches_to_skip"] = {
            "checkpoint": metadata.get("local_microbatches_to_skip"),
            "current": batches,
        }
    if mismatches:
        raise StreamingResumeError(
            "streaming resume contract mismatch; refusing a silent sample-order change: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True, default=str)
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "consumed_train_samples": consumed_train_samples,
        "local_microbatches_to_skip": batches,
        "contract_sha256": expected_digest,
    }


def resume_plan(args: Any, *, data_parallel_world_size: int) -> dict[str, Any]:
    """Build and, for a nonzero native resume, authenticate the replay plan."""

    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    micro_batch_size = int(getattr(args, "micro_batch_size", 0) or 0)
    global_batch_size = int(getattr(args, "global_batch_size", 0) or 0)
    local_microbatches_to_skip(
        consumed,
        data_parallel_world_size=data_parallel_world_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
    )
    if consumed == 0:
        # Build the contract even for step zero, so a missing dataset fingerprint
        # fails before the first optimizer step rather than at checkpoint time.
        contract = build_replay_contract(args, data_parallel_world_size=data_parallel_world_size)
        return {
            "checkpoint": None,
            "checkpoint_step": 0,
            "consumed_train_samples": 0,
            "local_microbatches_to_skip": 0,
            "contract_sha256": canonical_sha256(contract),
        }
    checkpoint_raw = getattr(args, "mcore_model", None)
    if not checkpoint_raw:
        raise StreamingResumeError(
            "consumed_train_samples is nonzero but args.mcore_model is unset"
        )
    return validate_resume_checkpoint(
        Path(str(checkpoint_raw)),
        args,
        data_parallel_world_size=data_parallel_world_size,
        consumed_train_samples=consumed,
    )
