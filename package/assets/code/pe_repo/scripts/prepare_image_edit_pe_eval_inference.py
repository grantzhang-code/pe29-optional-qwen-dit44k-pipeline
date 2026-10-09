#!/usr/bin/env python3
"""Validate the frozen PE benchmark and create label-free 10-way inference parts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_pe_mds import (  # noqa: E402
    LABEL_PROJECTION_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
)


FORMAT_VERSION = "image-edit-pe-eval-inference-input-v1"
BENCHMARK_FORMAT_VERSION = "image-edit-pe-task-type-eval-benchmark-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return {
        "filename": path.name,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"benchmark line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_sparse_label(label: Mapping[str, Any], label_text: str, *, row_index: int) -> None:
    if canonical_json(label) != label_text:
        raise ValueError(f"row {row_index}: assistant target is not canonical JSON")
    elements = label.get("elements")
    layout = elements.get("layout") if isinstance(elements, Mapping) else None
    if not isinstance(layout, list):
        raise ValueError(f"row {row_index}: sparse label has no elements.layout list")
    labels = {
        item.get("label")
        for item in layout
        if isinstance(item, Mapping) and isinstance(item.get("label"), str)
    }
    if len(labels) != len(layout):
        raise ValueError(f"row {row_index}: sparse label has invalid/duplicate region labels")
    for item in layout:
        edit = item.get("edit")
        instruction = edit.get("instruction") if isinstance(edit, Mapping) else None
        if not isinstance(instruction, str) or not instruction or instruction == "keep":
            raise ValueError(f"row {row_index}: sparse label contains a keep/invalid region")
        parent = item.get("parent")
        if parent != "ROOT" and parent not in labels:
            raise ValueError(f"row {row_index}: sparse label contains a dangling parent")


def validate_and_project(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    expected_task_types: int,
    expected_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if len(rows) != expected_rows:
        raise ValueError(f"benchmark rows changed: {len(rows)} != {expected_rows}")
    sample_ids: set[str] = set()
    global_rows: set[int] = set()
    task_counts: Counter[str] = Counter()
    inference: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    for input_index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        instruction = row.get("instruction")
        source = row.get("source_s3_uri")
        target = row.get("target_s3_uri")
        task_type = row.get("task_type")
        target_size = row.get("target_canvas_size")
        selection = row.get("selection")
        label_text = row.get("assistant_target_json")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(f"row {input_index}: invalid/duplicate sample_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"row {input_index}: instruction is missing")
        if (
            not isinstance(source, str)
            or not source.startswith(("s3://", "parquet+s3://"))
            or not isinstance(target, str)
            or not target.startswith(("s3://", "parquet+s3://"))
            or source == target
        ):
            raise ValueError(f"row {input_index}: media binding is invalid")
        if not isinstance(task_type, str) or not task_type:
            raise ValueError(f"row {input_index}: task_type is missing")
        if (
            not isinstance(target_size, Mapping)
            or type(target_size.get("width")) is not int
            or type(target_size.get("height")) is not int
            or target_size["width"] <= 0
            or target_size["height"] <= 0
        ):
            raise ValueError(f"row {input_index}: target size is invalid")
        if not isinstance(selection, Mapping) or selection.get("training_bitset_member") is not False:
            raise ValueError(f"row {input_index}: row is not a proven training-set complement")
        global_row = selection.get("global_row_index")
        if type(global_row) is not int or global_row < 0 or global_row in global_rows:
            raise ValueError(f"row {input_index}: invalid/duplicate global row")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"row {input_index}: prompt version mismatch")
        if row.get("prompt_contract_sha256") != PROMPT_CONTRACT_SHA256:
            raise ValueError(f"row {input_index}: prompt contract mismatch")
        if row.get("label_projection_version") != LABEL_PROJECTION_VERSION:
            raise ValueError(f"row {input_index}: label projection mismatch")
        if not isinstance(label_text, str) or not label_text:
            raise ValueError(f"row {input_index}: assistant target is missing")
        label = json.loads(label_text)
        if not isinstance(label, Mapping):
            raise ValueError(f"row {input_index}: assistant target is not an object")
        validate_sparse_label(label, label_text, row_index=input_index)

        sample_ids.add(sample_id)
        global_rows.add(global_row)
        task_counts[task_type] += 1
        common = {
            "input_index": input_index,
            "sample_id": sample_id,
            "task_type": task_type,
            "instruction": instruction.strip(),
            "source_s3_uri": source,
            "target_canvas_size": {
                "width": target_size["width"],
                "height": target_size["height"],
            },
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
            "selection": dict(selection),
        }
        inference.append(common)
        ground_truth.append(
            {
                **common,
                "target_s3_uri": target,
                "assistant_target_json": label_text,
                "assistant_target": label,
            }
        )

    if len(task_counts) != expected_task_types:
        raise ValueError(f"task type count changed: {len(task_counts)} != {expected_task_types}")
    if set(task_counts.values()) != {expected_per_task}:
        raise ValueError(f"per-task counts changed: {dict(sorted(task_counts.items()))}")
    return inference, ground_truth, dict(sorted(task_counts.items()))


def partition_rows(rows: Sequence[dict[str, Any]], part_count: int) -> list[list[dict[str, Any]]]:
    quotient, remainder = divmod(len(rows), part_count)
    parts: list[list[dict[str, Any]]] = []
    start = 0
    for part_index in range(part_count):
        size = quotient + int(part_index < remainder)
        parts.append(list(rows[start : start + size]))
        start += size
    if start != len(rows) or any(not part for part in parts):
        raise AssertionError("partition accounting failed")
    return parts


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_sha256 = file_sha256(args.benchmark_manifest)
    if benchmark_sha256 != args.benchmark_manifest_sha256:
        raise ValueError(
            f"benchmark manifest SHA-256 changed: {benchmark_sha256} != "
            f"{args.benchmark_manifest_sha256}"
        )
    rows = load_rows(args.benchmark_manifest)
    inference, ground_truth, task_counts = validate_and_project(
        rows,
        expected_rows=args.expected_rows,
        expected_task_types=args.expected_task_types,
        expected_per_task=args.expected_per_task,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference_info = write_jsonl(args.output_dir / "inference_manifest.jsonl", inference)
    ground_truth_info = write_jsonl(args.output_dir / "ground_truth.jsonl", ground_truth)
    part_infos: list[dict[str, Any]] = []
    for part_index, part in enumerate(partition_rows(inference, args.part_count)):
        filename = f"part-{part_index:03d}-of-{args.part_count:03d}.jsonl"
        info = write_jsonl(args.output_dir / "parts" / filename, part)
        part_infos.append(
            {
                **info,
                "part_index": part_index,
                "part_count": args.part_count,
                "first_input_index": part[0]["input_index"],
                "last_input_index": part[-1]["input_index"],
            }
        )
    index = {
        "complete": True,
        "format_version": FORMAT_VERSION,
        "benchmark_format_version": BENCHMARK_FORMAT_VERSION,
        "benchmark": {
            "path": str(args.benchmark_manifest),
            "sha256": benchmark_sha256,
            "rows": len(rows),
        },
        "input": inference_info,
        "ground_truth": ground_truth_info,
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
        },
        "task_counts": task_counts,
        "part_count": args.part_count,
        "parts": part_infos,
    }
    index_path = args.output_dir / "parts" / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        **index,
        "partition_index": {
            "filename": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": file_sha256(index_path),
        },
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1160)
    parser.add_argument("--expected-task-types", type=int, default=29)
    parser.add_argument("--expected-per-task", type=int, default=40)
    parser.add_argument("--part-count", type=int, default=10)
    args = parser.parse_args(argv)
    if not _SHA256_RE.fullmatch(args.benchmark_manifest_sha256):
        parser.error("benchmark manifest SHA-256 is invalid")
    for name in ("expected_rows", "expected_task_types", "expected_per_task", "part_count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.part_count > args.expected_rows:
        parser.error("part count must not exceed expected rows")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
