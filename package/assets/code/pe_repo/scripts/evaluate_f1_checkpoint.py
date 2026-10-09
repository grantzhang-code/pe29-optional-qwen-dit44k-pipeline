#!/usr/bin/env python3
"""Run exact teacher-forced evaluation of an F1 test file from an MCore checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--canonical-test-file", type=Path)
    parser.add_argument(
        "--grouped-test-file",
        type=Path,
        default=Path("agent_space/f1_test_grouped.swift.jsonl"),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--scratch-config", type=Path, default=Path("agent_space/f1_test_eval.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/f1-selected-test"),
    )
    parser.add_argument("--eval-iters", type=int, default=5)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    test_file = args.test_file.resolve()
    source_config = checkpoint.parent / "megatron_config.yaml"
    if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
        parser.error(f"incomplete checkpoint: {checkpoint}")
    if not test_file.is_file():
        parser.error(f"missing test file: {test_file}")
    if not source_config.is_file():
        parser.error(f"missing source config: {source_config}")
    if args.eval_iters <= 0:
        parser.error("--eval-iters must be positive")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    group_names = []
    global_batch_size = None
    if args.canonical_test_file:
        canonical_rows = [
            json.loads(line)
            for line in args.canonical_test_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        swift_rows = [
            json.loads(line)
            for line in test_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(canonical_rows) != len(swift_rows):
            parser.error("canonical and SWIFT test files have different row counts")
        grouped = {}
        for canonical, swift in zip(canonical_rows, swift_rows, strict=True):
            name = canonical["metadata"]["subcategory"]
            grouped.setdefault(name, []).append(swift)
        counts = {name: len(rows) for name, rows in grouped.items()}
        if len(set(counts.values())) != 1:
            parser.error(f"category counts are not balanced: {counts}")
        group_names = sorted(grouped)
        global_batch_size = next(iter(counts.values()))
        args.grouped_test_file.parent.mkdir(parents=True, exist_ok=True)
        args.grouped_test_file.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for name in group_names
                for row in grouped[name]
            ),
            encoding="utf-8",
        )
        test_file = args.grouped_test_file.resolve()

    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config_env = dict(config.get("ENV", {}))
    for key in tuple(config_env):
        if key.startswith("WANDB_") or key.startswith("AGENT_SFT_"):
            config_env.pop(key)
    config_env.update(
        {
            "AGENT_SFT_EVAL_ONLY": "1",
            "AGENT_SFT_EVAL_ONLY_PATH": str(args.artifact.resolve()),
        }
    )
    if group_names:
        config_env["AGENT_SFT_EVAL_GROUP_NAMES"] = json.dumps(group_names)
    config["ENV"] = config_env
    for key in ("num_train_epochs", "report_to", "wandb_project", "wandb_exp_name"):
        config.pop(key, None)
    config.update(
        {
            "dataset": [str(test_file)],
            "val_dataset": [str(test_file)],
            "packing": False,
            "eval_iters": args.eval_iters,
            "eval_steps": 1_000_000,
            "external_plugins": [
                str(
                    (
                        Path(__file__).parents[1]
                        / "src/agent_sft/training/plugins/eval_only.py"
                    ).resolve()
                )
            ],
            "mcore_model": str(checkpoint),
            "finetune": False,
            "no_load_optim": True,
            "no_load_rng": True,
            "train_iters": int(
                (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
            ),
            "async_save": False,
            "save_strategy": "steps",
            "save_steps": 1_000_000,
            "save_safetensors": False,
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    if global_batch_size is not None:
        config["global_batch_size"] = global_batch_size
        config["eval_iters"] = 1
    args.scratch_config.parent.mkdir(parents=True, exist_ok=True)
    args.scratch_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    megatron = shutil.which("megatron")
    if megatron is None:
        parser.error("megatron is not on PATH")
    env = os.environ.copy()
    env["AGENT_SFT_EVAL_ONLY"] = "1"
    env["AGENT_SFT_EVAL_ONLY_PATH"] = str(args.artifact.resolve())
    completed = subprocess.run(
        [megatron, "sft", str(args.scratch_config.resolve())],
        env=env,
        check=False,
    )
    if completed.returncode:
        return completed.returncode
    if not args.artifact.is_file():
        raise RuntimeError(f"evaluation completed without artifact: {args.artifact}")
    print(json.dumps(json.loads(args.artifact.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
