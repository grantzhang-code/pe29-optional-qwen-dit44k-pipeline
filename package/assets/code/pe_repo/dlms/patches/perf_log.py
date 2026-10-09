"""Log measured Megatron training throughput instead of only a nominal token budget.

The stock callback has no step timing.  The previous DLMS patch estimated throughput as
``packing_length * global_batch_size / elapsed``.  That is useful for kernel utilization,
but it silently treats padding as training data.  Packing experiments need both numbers:

* ``nonpad_tokens_per_second`` counts tokens represented by the packed samples.
* ``label_tokens_per_second`` counts supervised labels (``labels != -100``).
* ``nominal_tokens_per_second`` counts the configured packing-capacity upper bound.
* ``packing_fill_ratio`` is non-padding tokens divided by that configured capacity.

The patch times ``next(data_iterator)`` separately, counts each micro-batch, and aggregates
over the data-parallel group once per optimizer step.  TP peers are therefore not counted
twice.  The target sweep is TP=1/PP=1/CP=1; other topologies retain useful nominal metrics,
but true-token counts on pipeline stages should be interpreted with care.

Inject through ``--custom_register_path dlms/patches/perf_log.py``.  Set
``DLMS_PERF_LOG_NO_AUTO_APPLY=1`` only when importing its pure helpers in unit tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

_APPLIED = False


def _sum_seq_lens(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if hasattr(value, "sum") and hasattr(value, "numel"):
            return int(value.sum().item())
        return sum(int(v.item() if hasattr(v, "item") else v) for v in value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _count_tokens(batch: Any) -> tuple[int, int]:
    """Return ``(non-padding input tokens, supervised label tokens)`` for a batch.

    ``seq_lens`` is captured before ``prepare_batch`` removes it.  It remains exact with
    either dynamic production packing or the optional fixed-length padding patch.
    """
    if not isinstance(batch, dict):
        return 0, 0

    nonpad = _sum_seq_lens(batch.get("seq_lens"))
    if nonpad is None:
        mask = batch.get("attention_mask_2d", batch.get("attention_mask"))
        try:
            # A 2-D token mask is safe; a square causal mask is not a token count.
            if mask is not None and getattr(mask, "ndim", 0) <= 2:
                nonpad = int(mask.sum().item())
        except (TypeError, RuntimeError):
            nonpad = None
    if nonpad is None:
        packed = batch.get("packed_seq_params")
        cu = getattr(packed, "cu_seqlens_q", None)
        try:
            nonpad = int(cu[-1].item()) if cu is not None else None
        except (IndexError, TypeError, RuntimeError):
            nonpad = None
    if nonpad is None:
        ids = batch.get("input_ids")
        try:
            nonpad = int(ids.numel()) if ids is not None else 0
        except (TypeError, RuntimeError):
            nonpad = 0

    labels = batch.get("labels")
    try:
        label_tokens = int((labels != -100).sum().item()) if labels is not None else 0
    except (TypeError, RuntimeError):
        label_tokens = 0
    return max(nonpad, 0), max(label_tokens, 0)


@dataclass
class _Step:
    iteration: int
    started: float
    nonpad_tokens: int = 0
    label_tokens: int = 0
    data_wait_seconds: float = 0.0
    batch_prepare_seconds: float = 0.0


@dataclass
class _Tracker:
    current: _Step | None = None
    completed: dict[int, dict[str, float]] = field(default_factory=dict)
    emitted: set[int] = field(default_factory=set)

    def begin(self, iteration: int) -> None:
        if self.current is not None and self.current.iteration == iteration:
            return
        self.current = _Step(iteration=iteration, started=time.perf_counter())
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def batch(self, iteration: int, batch: Any, wait: float, prepare: float) -> None:
        current = self.current
        if current is None or current.iteration != iteration:
            return
        nonpad, labels = _count_tokens(batch)
        current.nonpad_tokens += nonpad
        current.label_tokens += labels
        current.data_wait_seconds += max(wait, 0.0)
        current.batch_prepare_seconds += max(prepare, 0.0)

    def end(self, completed_iteration: int) -> None:
        current = self.current
        if current is None or current.iteration + 1 != completed_iteration:
            return
        peak_allocated = peak_reserved = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                peak_allocated = float(torch.cuda.max_memory_allocated())
                peak_reserved = float(torch.cuda.max_memory_reserved())
        except Exception:
            pass
        self.completed[completed_iteration] = {
            "step_seconds": max(time.perf_counter() - current.started, 1e-12),
            "nonpad_tokens": float(current.nonpad_tokens),
            "label_tokens": float(current.label_tokens),
            "data_wait_seconds": current.data_wait_seconds,
            "batch_prepare_seconds": current.batch_prepare_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
        self.current = None
        # Logging every step is expected for a sweep.  Bound memory if a caller does not.
        if len(self.completed) > 128:
            for old in sorted(self.completed)[:-64]:
                self.completed.pop(old, None)


_TRACKER = _Tracker()


def _consts(args: Any) -> tuple[float, float, int, int, int]:
    peak = float(os.environ.get("DEVICE_TFLOPS", "0")) or 2.25e15
    if peak < 1e6:  # also accept TFLOP/s, e.g. DEVICE_TFLOPS=2250
        peak *= 1e12
    params = float(os.environ.get("MODEL_PARAMS", "0")) or 27e9
    try:
        import torch.distributed as dist

        ngpu = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        ngpu = 1
    gbs = int(getattr(args, "global_batch_size", 1) or 1)
    plen = int(getattr(args, "packing_length", 0) or getattr(args, "max_length", 0) or 0)
    return peak, params, ngpu, gbs, plen


def _aggregate_dp(record: dict[str, float]) -> dict[str, float]:
    """SUM token counts and MAX critical-path timing/memory across the DP group."""
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return record
        try:
            from megatron.core import parallel_state as mpu

            try:
                group = mpu.get_data_parallel_group(with_context_parallel=True)
            except TypeError:
                group = mpu.get_data_parallel_group()
        except Exception:
            group = None

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        counts = torch.tensor(
            [record["nonpad_tokens"], record["label_tokens"]], dtype=torch.float64, device=device
        )
        maxima = torch.tensor(
            [
                record["step_seconds"],
                record["data_wait_seconds"],
                record["batch_prepare_seconds"],
                record["peak_allocated_bytes"],
                record["peak_reserved_bytes"],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
        values = counts.tolist() + maxima.tolist()
        return {
            "nonpad_tokens": values[0],
            "label_tokens": values[1],
            "step_seconds": values[2],
            "data_wait_seconds": values[3],
            "batch_prepare_seconds": values[4],
            "peak_allocated_bytes": values[5],
            "peak_reserved_bytes": values[6],
        }
    except Exception:
        # Metrics must never abort a long training run.  A sweep will make the missing
        # fields obvious and its analyzer will mark that candidate incomplete.
        return record


def _derive_metrics(
    record: dict[str, float], *, peak: float, params: float, ngpu: int, gbs: int, packing_length: int
) -> dict[str, float]:
    seconds = max(float(record["step_seconds"]), 1e-12)
    nonpad = float(record["nonpad_tokens"])
    labels = float(record["label_tokens"])
    nominal = float(max(packing_length, 0) * max(gbs, 0))
    nonpad_tps = nonpad / seconds
    label_tps = labels / seconds
    nominal_tps = nominal / seconds if nominal else 0.0
    denom = max(peak * max(ngpu, 1), 1e-12)
    out = {
        "step_ms": round(seconds * 1000.0, 3),
        "samples_per_second": round(gbs / seconds, 6),
        "nonpad_tokens": int(nonpad),
        "label_tokens": int(labels),
        "nonpad_tokens_per_second": round(nonpad_tps, 3),
        "label_tokens_per_second": round(label_tps, 3),
        "nominal_tokens_per_second": round(nominal_tps, 3),
        "packing_fill_ratio": round(nonpad / nominal, 8) if nominal else 0.0,
        "data_wait_ms": round(float(record["data_wait_seconds"]) * 1000.0, 3),
        "batch_prepare_ms": round(float(record["batch_prepare_seconds"]) * 1000.0, 3),
        "data_wait_fraction": round(float(record["data_wait_seconds"]) / seconds, 8),
        "peak_memory_allocated_gib": round(float(record["peak_allocated_bytes"]) / 1024**3, 4),
        "peak_memory_reserved_gib": round(float(record["peak_reserved_bytes"]) / 1024**3, 4),
        "mfu_nonpad": round(6.0 * params * nonpad_tps / denom, 6),
        "mfu_nominal": round(6.0 * params * nominal_tps / denom, 6),
    }
    # Compatibility keys now intentionally mean true non-padding throughput.  Callers
    # needing the old padded-budget estimate should use the explicit nominal keys.
    out["tokens_per_second"] = out["nonpad_tokens_per_second"]
    out["mfu"] = out["mfu_nonpad"]
    return out


def _perf(state: Any, args: Any) -> dict[str, float]:
    iteration = int(getattr(state, "iteration", 0))
    if iteration in _TRACKER.emitted:
        return {}
    record = _TRACKER.completed.get(iteration)
    if record is None:
        return {}
    record = _aggregate_dp(record)
    peak, params, ngpu, gbs, plen = _consts(args)
    out = _derive_metrics(record, peak=peak, params=params, ngpu=ngpu, gbs=gbs, packing_length=plen)
    _TRACKER.emitted.add(iteration)
    _TRACKER.completed.pop(iteration, None)
    return out


def _patch_get_batch() -> None:
    from swift.megatron.trainers.base import BaseMegatronTrainer

    if getattr(BaseMegatronTrainer.get_batch, "_dlms_perf", False):
        return

    def get_batch(self, data_iterator, vp_stage=None):
        iteration = int(getattr(self.state, "iteration", 0))
        started = time.perf_counter()
        raw_batch = next(data_iterator)
        waited = time.perf_counter() - started
        started = time.perf_counter()
        prepared = self._prepare_batch(raw_batch, vp_stage)
        prepare = time.perf_counter() - started
        _TRACKER.batch(iteration, raw_batch, waited, prepare)
        return prepared

    get_batch._dlms_perf = True
    BaseMegatronTrainer.get_batch = get_batch


def _patch_callback(cls: type) -> None:
    if not getattr(cls.on_step_begin, "_dlms_perf", False):
        original_begin = cls.on_step_begin

        def on_step_begin(self):
            _TRACKER.begin(int(getattr(self.state, "iteration", 0)))
            return original_begin(self)

        on_step_begin._dlms_perf = True
        cls.on_step_begin = on_step_begin

    if not getattr(cls.on_step_end, "_dlms_perf", False):
        original_end = cls.on_step_end

        def on_step_end(self):
            _TRACKER.end(int(getattr(self.state, "iteration", 0)))
            return original_end(self)

        on_step_end._dlms_perf = True
        cls.on_step_end = on_step_end

    if not getattr(cls.on_log, "_dlms_perf", False):
        original_log = cls.on_log

        def on_log(self, logs):
            for key, value in _perf(self.state, self.args).items():
                logs.setdefault(key, value)
            return original_log(self, logs)

        on_log._dlms_perf = True
        cls.on_log = on_log


def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    from swift.megatron.callbacks import print as _print
    from swift.megatron.callbacks import wandb as _wandb

    _patch_get_batch()
    for cls in (_print.PrintCallback, _wandb.WandbCallback):
        _patch_callback(cls)
    _APPLIED = True
    print(
        "[dlms] perf_log applied (true/label/nominal tokens/s, fill, data wait, memory, MFU -> jsonl + W&B)",
        flush=True,
    )
    return True


if os.environ.get("DLMS_PERF_LOG_NO_AUTO_APPLY") != "1":
    apply()
