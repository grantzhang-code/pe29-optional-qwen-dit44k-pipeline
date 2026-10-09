"""Rank-local streaming dataloader: one reader per DP replica instead of two per job.

PROBLEM
    ms-swift routes every streaming run through ``MegatronDataLoaderDispatcher``
    (``swift/megatron/trainers/utils.py``), whose ``group`` is
    ``mpu.get_data_parallel_group()``. Only DP-group rank 0 reads; it pulls
    ``dp_world_size`` micro-batches **serially** with ``next(base_iter)`` and
    ``scatter_object_list``s them to the rest of the group.

    With TP=2 there are exactly two DP groups, so exactly two *global* ranks --
    0 and 1, both on node 0 -- perform all data loading for the whole job, and
    every batch is pickled and scattered over the network. The per-step cost is
    O(dp_world_size) serial encodes on a single node's CPU, which is why step
    time grows with cluster size even though per-GPU work is constant:

        1 node (DP=4)   7.9 s/step     MFU 0.149
        2 nodes (DP=8)  11.1 s/step    MFU 0.106
        4 nodes (DP=16) 74.5 s/step    MFU 0.016

FIX
    Drop the dispatcher. Every rank iterates its own DataLoader over its own
    slice of the S3 stream, exactly like the non-streaming path already does via
    ``MegatronPretrainingRandomSampler(data_parallel_rank=...)``.

    ``dlms/datasets/craft_mixed4.py`` already implements the sharding
    (``shard_paths`` assigns whole JSONL shards round-robin to a data rank, so
    there is no read amplification within a DP replica) but gates it behind
    ``SWIFT_RANK_LOCAL_ITERABLE_DATASET``; ``data_stream_shard_context()``
    otherwise returns ``(0, 1)`` precisely because the dispatcher is in charge.

    Those two halves MUST flip together:
      - dispatcher removed but sharding off  -> all ranks read the same stream
        (duplicate samples, duplicate gradients)
      - sharding on but dispatcher left in   -> rank 0 reads 1/dp of the data
        and scatters it to everyone (silent 15/16 data loss)
    So this module sets the env var itself rather than trusting the job config.

TP CORRECTNESS
    Ranks in the same DP replica but different TP ranks must see *identical*
    samples. They do, by construction: the shard key is
    ``mpu.get_data_parallel_rank()``, which is equal across a TP group, and
    every RNG in the source stream is seeded from ``seed``/``pass_index``/
    ``data_rank`` -- never from the global rank. So TP peers replay the same
    shard list in the same order.

    The one way they could still diverge is a non-deterministic *drop*:
    ``IterablePackingDataset._processor`` swallows ``template.encode``
    exceptions (``strict=False``), so a transient S3 image fetch failure on one
    TP peer and not the other would silently change pack composition and thus
    ``seq_lens``. That corrupts the TP all-reduces rather than raising. The
    fingerprint guard below turns that into a loud error; keep it on.

COST
    TP peers read the same shards, so S3 bytes and tokenization CPU are
    duplicated ``tp_size`` times. That is the price of needing no
    communication. The alternative -- load only on the TP source rank and
    broadcast the batch within the TP group -- halves the CPU but needs a
    shape-aware broadcast of a ragged dict; see the module docstring note in
    docs/ before reaching for it.

    Sizing: previously only ranks 0/1 iterated, so ``DATASET_NUM_PROC`` could be
    large without cost. Now every rank's encode pool is live, so a node runs
    ``ranks_per_node * DATASET_NUM_PROC`` encode processes. On a 96-core B300
    node with 8 ranks, DATASET_NUM_PROC=8..12 is the sane band; 32 oversubscribes
    by ~2.7x.

Inject via ``--custom_register_path .../dlms/patches/rank_local_streaming.py``
(self-applies on import).
"""
from __future__ import annotations

import os

from dlms.checkpoint.streaming_resume import resume_plan, streaming_resume_enabled

_APPLIED = False

# Must be set before dlms.datasets.craft_mixed4 resolves its shard context. That
# happens at dataset *build* time (inside CraftMixed4Loader.load), long after all
# --custom_register_path modules are imported, so import order here is irrelevant.
os.environ["SWIFT_RANK_LOCAL_ITERABLE_DATASET"] = "true"

_GUARD = os.environ.get("STREAM_TP_CONSISTENCY_GUARD", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _dp_context():
    """(dp_rank, dp_world, tp_rank, tp_world) or None when mpu is not up yet."""
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return None
        return (
            mpu.get_data_parallel_rank(),
            mpu.get_data_parallel_world_size(),
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_tensor_model_parallel_world_size(),
        )
    except (ImportError, RuntimeError, AssertionError):
        return None


def _fingerprint(batch):
    """Cheap order-sensitive signature of a collated micro-batch."""
    import torch

    ids = batch.get("input_ids") if isinstance(batch, dict) else None
    if not isinstance(ids, torch.Tensor):
        return None
    flat = ids.reshape(-1)
    # numel pins the pack layout; the weighted sum catches same-length-different-content.
    idx = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.int64)
    checksum = int((flat.to(torch.int64) * idx).sum().item())
    return flat.numel(), checksum


class _RankLocalStreamingLoader:
    """Iterate this rank's own shard of the stream. No cross-rank data movement."""

    def __init__(
        self,
        base_dataloader,
        *,
        tp_group=None,
        tp_world=1,
        resume_batches=0,
        resume_info=None,
    ):
        self.base_dataloader = base_dataloader
        self._tp_group = tp_group
        self._tp_world = tp_world
        self._resume_batches = int(resume_batches)
        self._resume_info = resume_info or {}
        self._step = 0

    def _check_tp_consistency(self, batch):
        if self._tp_world <= 1 or self._tp_group is None:
            return
        import torch
        import torch.distributed as dist

        fp = _fingerprint(batch)
        if fp is None:
            return
        local = torch.tensor(fp, dtype=torch.int64, device=torch.cuda.current_device())
        gathered = torch.empty(
            (self._tp_world, local.numel()), dtype=torch.int64, device=local.device
        )
        dist.all_gather_into_tensor(gathered, local, group=self._tp_group)
        if not bool((gathered == gathered[0]).all().item()):
            raise RuntimeError(
                "rank_local_streaming: TP peers disagree on micro-batch "
                f"{self._step} -- (numel, checksum) per TP rank = {gathered.tolist()}. "
                "A sample was dropped on some TP ranks but not others (most likely a "
                "transient S3 image fetch failure swallowed by "
                "IterablePackingDataset._processor). Training would silently corrupt "
                "the TP all-reduces, so it is aborted here."
            )

    def __iter__(self):
        iterator = iter(self.base_dataloader)
        if self._resume_batches:
            self._replay_committed_microbatches(iterator)
        for batch in iterator:
            self._step += 1
            if _GUARD:
                self._check_tp_consistency(batch)
            yield batch

    @staticmethod
    def _is_global_rank_zero():
        try:
            import torch.distributed as dist

            return not dist.is_initialized() or dist.get_rank() == 0
        except (ImportError, RuntimeError):
            return True

    def _replay_committed_microbatches(self, iterator):
        """Discard exactly the packed batches represented by the checkpoint."""

        total = self._resume_batches
        if self._is_global_rank_zero():
            print(
                "[dlms-resume] deterministic packed-batch replay starting: "
                f"checkpoint_step={self._resume_info.get('checkpoint_step')} "
                f"consumed_train_samples={self._resume_info.get('consumed_train_samples')} "
                f"local_microbatches={total} "
                f"contract={self._resume_info.get('contract_sha256')}",
                flush=True,
            )
        progress_interval = max(100, min(1000, total // 20 or 1))
        for index in range(total):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "rank_local_streaming: stream ended while replaying committed "
                    f"microbatches ({index}/{total}); checkpoint cannot be resumed safely"
                ) from exc
            self._step += 1
            if _GUARD:
                self._check_tp_consistency(batch)
            if (
                self._is_global_rank_zero()
                and (index + 1) < total
                and (index + 1) % progress_interval == 0
            ):
                print(
                    f"[dlms-resume] replayed {index + 1}/{total} committed local microbatches",
                    flush=True,
                )
        # Every rank must finish reconstructing the same logical checkpoint
        # boundary before any rank enters the next model forward pass.
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("rank_local_streaming: resume barrier failed") from exc
        if self._is_global_rank_zero():
            print(
                f"[dlms-resume] replay complete; next local microbatch is {total + 1}",
                flush=True,
            )


def _dataset_resume_role(dataset):
    """Find the role marker through Swift's packing/preprocessing wrappers."""

    current = dataset
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        role = getattr(current, "_dlms_streaming_resume_role", None)
        if role is not None:
            return role
        current = getattr(current, "dataset", None)
    return None


def _resolved_dataset_resume_role(args, dataset):
    role = _dataset_resume_role(dataset)
    if role is not None:
        return role
    # Hugging Face may rebuild an IterableDataset during preprocessing and
    # discard custom Python attributes.  This job has no validation stream;
    # BaseMegatronTrainer therefore calls the builder only for train.  Use that
    # explicit trainer contract instead of turning a harmless lost attribute
    # into a startup failure.
    eval_iters = int(getattr(args, "eval_iters", 0) or 0)
    split_ratio = float(getattr(args, "split_dataset_ratio", 0.0) or 0.0)
    if eval_iters == 0 and split_ratio == 0.0:
        return "train"
    return None


def _build_rank_local_streaming_dataloader(args, dataset, collate_fn):
    import torch

    base = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        persistent_workers=args.dataloader_persistent_workers if args.dataloader_num_workers > 0 else False,
    )

    tp_group, tp_world = None, 1
    resume_info = None
    resume_batches = 0
    ctx = _dp_context()
    if ctx is not None:
        from megatron.core import parallel_state as mpu

        _, dp_world, _, tp_world = ctx
        if tp_world > 1:
            tp_group = mpu.get_tensor_model_parallel_group()
        if streaming_resume_enabled():
            role = _resolved_dataset_resume_role(args, dataset)
            if role not in {"train", "validation"}:
                raise RuntimeError(
                    "rank_local_streaming: resumable stream has no train/validation role marker"
                )
            if role == "train":
                resume_info = resume_plan(args, data_parallel_world_size=dp_world)
                resume_batches = int(resume_info["local_microbatches_to_skip"])
    elif streaming_resume_enabled():
        raise RuntimeError(
            "rank_local_streaming: model-parallel state is required for resumable streaming"
        )
    return _RankLocalStreamingLoader(
        base,
        tp_group=tp_group,
        tp_world=tp_world,
        resume_batches=resume_batches,
        resume_info=resume_info,
    )


def apply() -> list[str]:
    """Replace build_streaming_dataloader everywhere it is bound. Idempotent."""
    global _APPLIED
    patched: list[str] = []
    from swift.megatron.trainers import base as _base
    from swift.megatron.trainers import utils as _utils

    # base.py does `from .utils import build_streaming_dataloader`, so the name is
    # bound in both modules and both must be replaced.
    for mod, mod_name in ((_utils, "swift.megatron.trainers.utils"), (_base, "swift.megatron.trainers.base")):
        cur = getattr(mod, "build_streaming_dataloader", None)
        if cur is None:
            continue
        if getattr(cur, "_dlms_rank_local", False):
            patched.append(f"{mod_name} (already)")
            continue
        _build_rank_local_streaming_dataloader._dlms_rank_local = True
        mod.build_streaming_dataloader = _build_rank_local_streaming_dataloader
        patched.append(mod_name)

    _APPLIED = True
    ctx = _dp_context()
    where = (
        f"dp_rank={ctx[0]}/{ctx[1]} tp_rank={ctx[2]}/{ctx[3]}" if ctx else "mpu not initialized yet"
    )
    print(
        f"[dlms] rank-local streaming dataloader applied ({where}); "
        f"SWIFT_RANK_LOCAL_ITERABLE_DATASET=true, tp_consistency_guard={_GUARD}; "
        f"streaming_resume={streaming_resume_enabled()}; "
        f"patched={patched}",
        flush=True,
    )
    return patched


apply()
