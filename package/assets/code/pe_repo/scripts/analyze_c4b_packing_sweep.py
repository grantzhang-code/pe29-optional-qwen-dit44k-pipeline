#!/usr/bin/env python3
"""Summarize a C4B packing-length sweep from Megatron ``logging.jsonl`` files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


REQUIRED_METRICS = (
    "step_ms",
    "nonpad_tokens",
    "label_tokens",
    "packing_fill_ratio",
    "data_wait_ms",
    "peak_memory_reserved_gib",
    "mfu_nonpad",
)
LOSS_KEYS = ("loss", "mtp_1_loss")


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _iteration(record: dict[str, Any]) -> int | None:
    value = record.get("iteration")
    if isinstance(value, str):
        value = value.split("/", 1)[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_measurements(path: Path) -> list[dict[str, Any]]:
    """Read and de-duplicate complete per-step metric records by iteration."""
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A CUDA OOM/SIGKILL can interrupt the async JSONL writer mid-line. Keep
                # all committed measurements so the failed candidate can still be reported.
                continue
            iteration = _iteration(record)
            if iteration is None or any(key not in record for key in REQUIRED_METRICS):
                continue
            records[iteration] = record
    return [records[key] for key in sorted(records)]


def _read_status(directory: Path) -> dict[str, Any]:
    path = directory / "status.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "invalid_status_file"}


def discover_logging_jsonl(directory: Path) -> Path:
    """Resolve Swift's log, which may live below its ``v0-*`` run directory."""
    direct = directory / "logging.jsonl"
    if direct.is_file():
        return direct
    versioned = sorted(path for path in directory.glob("*/logging.jsonl") if path.is_file())
    if versioned:
        # Swift's version directory embeds a sortable timestamp.  If a candidate was
        # retried in place, use the last run rather than silently mixing measurements.
        return versioned[-1]
    return direct


def summarize_candidate(
    directory: Path,
    packing_length: int,
    *,
    warmup_steps: int,
    measure_steps: int,
    global_batch_size: int,
    gpu_memory_gib: float | None,
) -> dict[str, Any]:
    logging_jsonl = discover_logging_jsonl(directory)
    records = read_measurements(logging_jsonl)
    measured = records[warmup_steps : warmup_steps + measure_steps]
    status = _read_status(directory)
    loss_values = {
        key: [float(record[key]) for record in measured if key in record]
        for key in LOSS_KEYS
    }
    losses_finite = {
        key: len(values) == len(measured) and all(math.isfinite(value) for value in values)
        for key, values in loss_values.items()
    }
    result: dict[str, Any] = {
        "packing_length": packing_length,
        "run_status": status.get("status", "unknown"),
        "exit_code": status.get("exit_code"),
        "warmup_records": min(len(records), warmup_steps),
        "measured_records": len(measured),
        "required_measured_records": measure_steps,
        "losses_finite": losses_finite,
        "complete": (
            len(measured) == measure_steps
            and status.get("exit_code", 0) == 0
            and all(losses_finite.values())
        ),
        "logging_jsonl": str(logging_jsonl),
    }
    if not measured:
        return result

    step_ms = [float(record["step_ms"]) for record in measured]
    total_seconds = sum(step_ms) / 1000.0
    nonpad_tokens = sum(int(record["nonpad_tokens"]) for record in measured)
    label_tokens = sum(int(record["label_tokens"]) for record in measured)
    nominal_tokens = packing_length * global_batch_size * len(measured)
    wait_ms = sum(float(record["data_wait_ms"]) for record in measured)
    peak_allocated = max(float(record.get("peak_memory_allocated_gib", 0.0)) for record in measured)
    peak_reserved = max(float(record["peak_memory_reserved_gib"]) for record in measured)
    result["metrics"] = {
        "loss_mean": (
            round(statistics.fmean(loss_values["loss"]), 8) if loss_values["loss"] else None
        ),
        "mtp_1_loss_mean": (
            round(statistics.fmean(loss_values["mtp_1_loss"]), 8)
            if loss_values["mtp_1_loss"]
            else None
        ),
        "nonpad_tokens_per_second": round(nonpad_tokens / total_seconds, 3),
        "label_tokens_per_second": round(label_tokens / total_seconds, 3),
        "packing_fill_ratio": round(nonpad_tokens / nominal_tokens, 8),
        "step_ms_p50": round(percentile(step_ms, 0.50) or 0.0, 3),
        "step_ms_p95": round(percentile(step_ms, 0.95) or 0.0, 3),
        "step_ms_mean": round(statistics.fmean(step_ms), 3),
        "data_wait_ms_mean": round(wait_ms / len(measured), 3),
        "data_wait_fraction": round(wait_ms / sum(step_ms), 8),
        "peak_memory_allocated_gib": round(peak_allocated, 4),
        "peak_memory_reserved_gib": round(peak_reserved, 4),
        "mfu_nonpad_mean": round(statistics.fmean(float(record["mfu_nonpad"]) for record in measured), 6),
        "mfu_nominal_mean": round(
            statistics.fmean(float(record.get("mfu_nominal", 0.0)) for record in measured), 6
        ),
    }
    if gpu_memory_gib and gpu_memory_gib > 0:
        result["metrics"]["peak_memory_fraction"] = round(peak_reserved / gpu_memory_gib, 6)
    return result


def discover_candidates(root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for directory in root.glob("packing-*"):
        if not directory.is_dir():
            continue
        try:
            packing_length = int(directory.name.removeprefix("packing-"))
        except ValueError:
            continue
        found.append((packing_length, directory))
    return sorted(found)


def build_summary(
    root: Path,
    *,
    warmup_steps: int,
    measure_steps: int,
    global_batch_size: int,
    gpu_memory_gib: float | None,
    max_memory_fraction: float = 0.85,
) -> dict[str, Any]:
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("max_memory_fraction must be in (0, 1]")
    candidates = [
        summarize_candidate(
            directory,
            packing_length,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            global_batch_size=global_batch_size,
            gpu_memory_gib=gpu_memory_gib,
        )
        for packing_length, directory in discover_candidates(root)
    ]
    rankable = [candidate for candidate in candidates if candidate["complete"] and candidate.get("metrics")]
    rankable.sort(key=lambda item: item["metrics"]["nonpad_tokens_per_second"], reverse=True)
    for candidate in candidates:
        fraction = candidate.get("metrics", {}).get("peak_memory_fraction")
        candidate["production_safe"] = bool(
            candidate["complete"] and fraction is not None and fraction <= max_memory_fraction
        )
    production_rankable = [candidate for candidate in rankable if candidate["production_safe"]]
    largest_complete = max(
        (candidate["packing_length"] for candidate in rankable), default=None
    )
    largest_production_safe = max(
        (candidate["packing_length"] for candidate in production_rankable), default=None
    )
    return {
        "schema_version": 1,
        "sweep_root": str(root),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "global_batch_size": global_batch_size,
        "gpu_memory_gib": gpu_memory_gib,
        "max_memory_fraction": max_memory_fraction,
        "ranking_metric": "nonpad_tokens_per_second",
        "throughput_winner_packing_length": rankable[0]["packing_length"] if rankable else None,
        "winner_packing_length": (
            production_rankable[0]["packing_length"] if production_rankable else None
        ),
        "largest_complete_packing_length": largest_complete,
        "largest_production_safe_packing_length": largest_production_safe,
        "candidates": candidates,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C4B packing efficiency sweep",
        "",
        f"Warmup: {summary['warmup_steps']} steps; measured: {summary['measure_steps']} steps; "
        f"GBS: {summary['global_batch_size']}.",
        "",
        "| packing | status | production safe | measured | non-pad tok/s | label tok/s | fill | step p50/p95 ms | "
        "loss / MTP loss | data wait | peak reserved GiB | MFU (non-pad) |",
        "|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in summary["candidates"]:
        metrics = candidate.get("metrics", {})
        status = "complete" if candidate["complete"] else candidate["run_status"]
        lines.append(
            "| {packing_length} | {status} | {safe} | {measured_records}/{required_measured_records} | "
            "{tps} | {label_tps} | {fill} | {p50}/{p95} | {loss}/{mtp_loss} | "
            "{wait} | {memory} | {mfu} |".format(
                **candidate,
                status=status,
                safe="yes" if candidate.get("production_safe") else "no",
                tps=metrics.get("nonpad_tokens_per_second", ""),
                label_tps=metrics.get("label_tokens_per_second", ""),
                fill=metrics.get("packing_fill_ratio", ""),
                p50=metrics.get("step_ms_p50", ""),
                p95=metrics.get("step_ms_p95", ""),
                loss=metrics.get("loss_mean", ""),
                mtp_loss=metrics.get("mtp_1_loss_mean", ""),
                wait=metrics.get("data_wait_fraction", ""),
                memory=metrics.get("peak_memory_reserved_gib", ""),
                mfu=metrics.get("mfu_nonpad_mean", ""),
            )
        )
    lines.extend(
        [
            "",
            "Raw throughput winner: "
            f"**{summary['throughput_winner_packing_length']}**. Production-safe winner "
            f"(peak reserved <= {summary['max_memory_fraction']:.0%}): "
            f"**{summary['winner_packing_length']}**.",
            "Largest complete packing length: "
            f"**{summary['largest_complete_packing_length']}**; largest production-safe "
            f"packing length: **{summary['largest_production_safe_packing_length']}**.",
            "",
            "> `nominal packing_length × GBS` is deliberately not the ranking metric.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=30)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-gib", type=float)
    parser.add_argument("--max-memory-fraction", type=float, default=0.85)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(
        args.root,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        global_batch_size=args.global_batch_size,
        gpu_memory_gib=args.gpu_memory_gib,
        max_memory_fraction=args.max_memory_fraction,
    )
    json_output = args.json_output or args.root / "summary.json"
    markdown_output = args.markdown_output or args.root / "summary.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"winner_packing_length": summary["winner_packing_length"], "json": str(json_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
