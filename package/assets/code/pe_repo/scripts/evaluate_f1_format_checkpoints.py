#!/usr/bin/env python3
"""Run real-tool F1 format rollouts for every scheduled validation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_sft.evaluation.format_compliance import aggregate_trajectory_results


F1_CATEGORIES = (
    "F1a_weather",
    "F1b_events_schedule",
    "F1c_sports",
    "F1d_news_topical",
    "F1e_entity_current",
    "F1f_stats_data",
    "F1g_place_details",
    "F1h_cultural_historical",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _wait_for_training(run_dir: Path, poll_seconds: int) -> None:
    status_path = run_dir / "status.json"
    while True:
        if status_path.is_file():
            status = _read_json(status_path)
            state = status.get("status")
            if state == "completed":
                return
            if state in {"failed", "launch_error", "trained_export_failed"}:
                raise RuntimeError(f"training ended with status {state!r}: {status}")
        print(f"waiting for training completion: {status_path}", flush=True)
        time.sleep(poll_seconds)


def _resolve_version_dir(output_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in output_root.glob("v*")
            if path.is_dir() and (path / "args.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no Megatron version directory found under {output_root}")
    return candidates[-1]


def _scheduled_iterations(megatron_args: dict[str, Any]) -> list[int]:
    train_iters = int(megatron_args["train_iters"])
    eval_steps = int(megatron_args["eval_steps"])
    epochs = int(megatron_args["num_train_epochs"])
    epoch_stride = max(train_iters // epochs, 1)
    iterations = {0, train_iters}
    iterations.update(range(eval_steps, train_iters + 1, eval_steps))
    iterations.update(
        iteration
        for iteration in range(epoch_stride, train_iters + 1, epoch_stride)
        if iteration <= train_iters
    )
    return sorted(iterations)


def _event_iterations(path: Path) -> list[int]:
    iterations = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                iterations.add(int(json.loads(line)["iteration"]))
    if not iterations or 0 not in iterations:
        raise ValueError(f"validation event log has no step-zero baseline: {path}")
    return sorted(iterations)


def _export_checkpoint(
    *,
    checkpoint: Path,
    output_dir: Path,
    megatron_args: dict[str, Any],
    model: str,
    revision: str,
) -> None:
    if (output_dir / "config.json").is_file() and list(output_dir.glob("*.safetensors")):
        return
    megatron = shutil.which("megatron")
    if megatron is None:
        raise RuntimeError("megatron is not on PATH")
    command = [
        megatron,
        "export",
        "--to-hf",
        "true",
        "--output-dir",
        str(output_dir),
        "--exist-ok",
        "true",
        "--use-hf",
        "true",
        "--mcore-model",
        str(checkpoint),
        "--model",
        model,
        "--model-revision",
        revision,
        "--template",
        str(megatron_args["template"]),
        "--agent-template",
        str(megatron_args["agent_template"]),
        "--torch-dtype",
        "bfloat16",
        "--tensor-model-parallel-size",
        str(megatron_args["tensor_model_parallel_size"]),
        "--pipeline-model-parallel-size",
        str(megatron_args["pipeline_model_parallel_size"]),
        "--context-parallel-size",
        str(megatron_args["context_parallel_size"]),
        "--tuner-type",
        "full",
    ]
    subprocess.run(command, check=True)
    if not (output_dir / "config.json").is_file():
        raise RuntimeError(f"checkpoint export produced no config.json: {output_dir}")


def _evaluate_all_categories(
    *,
    model: str,
    revision: str | None,
    validation_file: Path,
    artifact: Path,
    samples_per_category: int,
    search_model: str,
    cuda_devices: list[str],
    max_new_tokens: int,
) -> None:
    if artifact.is_file():
        prior = _read_json(artifact)
        expected = samples_per_category * len(F1_CATEGORIES)
        if (
            prior.get("status") == "completed"
            and prior.get("summary", {}).get("overall", {}).get("sample_count") == expected
        ):
            return
    if len(cuda_devices) != len(F1_CATEGORIES):
        raise ValueError(
            f"exactly {len(F1_CATEGORIES)} CUDA devices are required, got {cuda_devices}"
        )
    evaluator = Path(__file__).with_name("evaluate_f1_format_rollout.py")
    shard_dir = artifact.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, Path, subprocess.Popen]] = []
    for category, cuda_device in zip(F1_CATEGORIES, cuda_devices, strict=True):
        shard = shard_dir / f"{category}.json"
        if shard.is_file():
            value = _read_json(shard)
            if (
                value.get("status") == "completed"
                and value.get("summary", {}).get("overall", {}).get("sample_count")
                == samples_per_category
            ):
                continue
        command = [
            sys.executable,
            str(evaluator),
            "--model",
            model,
            "--validation-file",
            str(validation_file),
            "--artifact",
            str(shard),
            "--samples-per-category",
            str(samples_per_category),
            "--categories",
            category,
            "--max-new-tokens",
            str(max_new_tokens),
            "--device-map",
            "cuda:0",
            "--attn-impl",
            "eager",
            "--search-model",
            search_model,
            "--overwrite",
        ]
        if revision is not None:
            command.extend(["--revision", revision])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        processes.append((category, shard, subprocess.Popen(command, env=env)))

    failures = []
    for category, _, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((category, returncode))
    if failures:
        raise RuntimeError(f"format evaluation shards failed: {failures}")

    shard_values = [_read_json(shard_dir / f"{category}.json") for category in F1_CATEGORIES]
    results = [result for value in shard_values for result in value["results"]]
    combined = {
        "schema_version": 1,
        "status": "completed",
        "model": model,
        "search_model": search_model,
        "validation_file": str(validation_file.resolve()),
        "samples_per_category": samples_per_category,
        "selected_sample_ids": [
            sample_id for value in shard_values for sample_id in value["selected_sample_ids"]
        ],
        "summary": aggregate_trajectory_results(results),
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(artifact, combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--search-model", default=os.getenv("AGENT_SFT_SEARCH_MODEL", "gpt-5.6"))
    parser.add_argument("--samples-per-category", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--cuda-devices",
        nargs="+",
        default=[str(index) for index in range(8)],
        help="One CUDA device per F1 category; all eight categories run in parallel.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error("--poll-seconds must be in [1, 60]")
    if args.wait:
        _wait_for_training(args.run_dir, args.poll_seconds)
    elif _read_json(args.run_dir / "status.json").get("status") != "completed":
        parser.error("training is not complete; pass --wait to monitor it")

    version_dir = _resolve_version_dir(args.output_root)
    megatron_args = _read_json(version_dir / "args.json")
    iterations = _event_iterations(args.run_dir / "validation_events.jsonl")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.artifact_dir / "index.json"
    index = {
        "schema_version": 1,
        "status": "running",
        "run_dir": str(args.run_dir.resolve()),
        "version_dir": str(version_dir.resolve()),
        "validation_file": str(args.validation_file.resolve()),
        "iterations": iterations,
        "samples_per_category": args.samples_per_category,
        "search_model": args.search_model,
        "results": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if index_path.is_file():
        prior = _read_json(index_path)
        index["results"].update(prior.get("results", {}))

    for iteration in iterations:
        artifact = args.artifact_dir / f"iteration-{iteration:06d}.json"
        if iteration == 0:
            eval_model = args.model
            eval_revision: str | None = args.revision
        else:
            checkpoint = version_dir / f"checkpoint-{iteration}"
            if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
                raise FileNotFoundError(f"missing checkpoint for validation iteration {iteration}")
            eval_model = str((args.artifact_dir / "exports" / f"checkpoint-{iteration}-hf").resolve())
            _export_checkpoint(
                checkpoint=checkpoint,
                output_dir=Path(eval_model),
                megatron_args=megatron_args,
                model=args.model,
                revision=args.revision,
            )
            eval_revision = None

        print(f"evaluating format compliance at iteration {iteration}: {eval_model}", flush=True)
        _evaluate_all_categories(
            model=eval_model,
            revision=eval_revision,
            validation_file=args.validation_file,
            artifact=artifact,
            samples_per_category=args.samples_per_category,
            search_model=args.search_model,
            cuda_devices=args.cuda_devices,
            max_new_tokens=args.max_new_tokens,
        )
        result = _read_json(artifact)
        index["results"][str(iteration)] = {
            "artifact": str(artifact.resolve()),
            "summary": result["summary"],
        }
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(index_path, index)

    index["status"] = "completed"
    index["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(index_path, index)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
