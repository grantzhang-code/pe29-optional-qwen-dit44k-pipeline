#!/usr/bin/env python3
"""Summarize the reproducible metrics and checkpoints from the F1 LR study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUNS = (
    "qwen35-f1-cosine-2ep-lr5e6-20260724",
    "qwen35-f1-linear-2ep-lr5e6-20260724",
    "qwen35-f1-cosine-1ep-lr5e6-20260724",
)


def _iteration(record: dict[str, Any]) -> int:
    return int(str(record["iteration"]).split("/", 1)[0])


def _latest_version(run_root: Path) -> Path:
    versions = sorted(path for path in run_root.glob("v*-*") if path.is_dir())
    if not versions:
        raise FileNotFoundError(f"No version directory under {run_root}")
    return versions[-1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run_id: str, output_root: Path, metadata_root: Path) -> dict[str, Any]:
    version_dir = _latest_version(output_root / run_id)
    args = _read_json(version_dir / "args.json")
    records = [
        json.loads(line)
        for line in (version_dir / "logging.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [record for record in records if "loss" in record]
    validation = [
        {"step": _iteration(record), "loss": float(record["eval_loss"])}
        for record in records
        if "eval_loss" in record
    ]
    finite = all(
        math.isfinite(float(record[key]))
        for record in train
        for key in ("loss", "grad_norm")
        if key in record
    )
    checkpoints = []
    for path in sorted(version_dir.glob("checkpoint-*")):
        tracker = path / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            checkpoints.append({"step": int(tracker.read_text().strip()), "path": str(path)})
    status_path = metadata_root / run_id / "status.json"
    status = _read_json(status_path) if status_path.is_file() else {"status": "missing"}
    best = min(validation, key=lambda item: item["loss"]) if validation else None
    return {
        "run_id": run_id,
        "version_dir": str(version_dir),
        "schedule": args.get("lr_decay_style"),
        "epochs": args.get("num_train_epochs"),
        "peak_lr": args.get("lr"),
        "min_lr": args.get("min_lr"),
        "status": status,
        "train_steps_logged": len(train),
        "finite_loss_and_grad_norm": finite,
        "initial_validation_loss": validation[0]["loss"] if validation else None,
        "final_validation_loss": validation[-1]["loss"] if validation else None,
        "best_validation": best,
        "validation_curve": validation,
        "checkpoints": checkpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", dest="runs", help="Run ID; may be repeated")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/agent_sft"),
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    summaries = []
    for run_id in args.runs or DEFAULT_RUNS:
        try:
            summaries.append(summarize(run_id, args.output_root, args.metadata_root))
        except FileNotFoundError as exc:
            summaries.append({"run_id": run_id, "status": {"status": "not_started"}, "error": str(exc)})
    result = {"runs": summaries}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
