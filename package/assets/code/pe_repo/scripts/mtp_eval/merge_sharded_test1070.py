#!/usr/bin/env python3
"""Merge four independently generated TEST1070 condition shards.

Each input is a local condition output directory (or a parent containing the
condition directory) produced by ``run_test1070_inference.py``.  Shard-local
``input_index`` values are deliberately not trusted: rows are mapped back to
the canonical manifest by ``images[0].path == s3_image_path``.

All inputs are validated before any output becomes visible.  The resulting
``<output-root>/<condition>`` directory has the same four merged JSONL files
and condition-summary contract consumed by ``finalize_test1070_speed.py``.
S3 shard roots can be added later without changing the merge contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mtp_eval import run_test1070_inference as runner  # noqa: E402


EXPECTED_SHARDS = 4
MERGE_SCHEMA_VERSION = 1
TASKS = tuple(task.output_name for task in runner.TASKS)
_TASK_SPEC_BY_OUTPUT = {task.output_name: task for task in runner.TASKS}
_MANDATORY_SHARED_FIELDS = (
    "model",
    "max_pixels",
    "max_model_len",
    "max_new_tokens_cli_cap",
    "eos_reserve_tokens",
)
_OPTIONAL_SHARED_FIELDS = (
    "target_backbone_fingerprint",
    "source_checkpoint_manifest_sha256",
    "runtime_image_digest",
    "vllm_version",
    "code_revision",
    "code_overlay_sha256",
    "batch_size",
    "attention_backend",
    "dtype",
    "tensor_parallel_size",
    "runtime",
    "engine",
    "provenance",
)
_PARALLEL_TIMING_FIELDS = (
    "generation_wall_seconds",
    "image_decode_seconds",
    "multimodal_preprocess_seconds",
    "elapsed_seconds",
)


class MergeError(RuntimeError):
    """Raised when shard artifacts cannot form one canonical condition."""


@dataclass(frozen=True)
class ShardCondition:
    root: Path
    summary: dict[str, Any]
    tasks: dict[str, list[dict[str, Any]]]
    image_paths: frozenset[str]
    local_index_to_path: dict[int, str]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MergeError(f"missing shard artifact: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MergeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MergeError(f"missing shard artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise MergeError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _require_int(mapping: Mapping[str, Any], key: str, source: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MergeError(f"{source}: {key} must be a non-negative integer")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, source: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MergeError(f"{source}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise MergeError(f"{source}: {key} must be finite and non-negative")
    return result


def _assert_close(actual: float, expected: float, source: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise MergeError(f"{source}: {actual!r} != expected {expected!r}")


def _image_path(row: Mapping[str, Any], source: str) -> str:
    images = row.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], Mapping):
        raise MergeError(f"{source}: images[0].path is missing")
    value = images[0].get("path")
    if not isinstance(value, str) or not value:
        raise MergeError(f"{source}: images[0].path is missing")
    return value


def _validate_request_timing(row: Mapping[str, Any], source: str) -> None:
    timing = row.get("request_timing")
    if timing is None:
        return
    if not isinstance(timing, Mapping):
        raise MergeError(f"{source}: request_timing must be an object")
    for key, value in timing.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MergeError(f"{source}: request_timing.{key} must be numeric or null")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise MergeError(f"{source}: request_timing.{key} must be finite and non-negative")


def _validate_row(row: Mapping[str, Any], condition: str, task: str, source: str) -> None:
    index = row.get("input_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise MergeError(f"{source}: input_index must be a non-negative integer")
    _image_path(row, source)
    if row.get("condition") != condition:
        raise MergeError(f"{source}: row condition {row.get('condition')!r} != {condition!r}")
    if row.get("task") != task:
        raise MergeError(f"{source}: row task {row.get('task')!r} != {task!r}")
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise MergeError(f"{source}: response must be a non-empty string")
    response_sha256 = row.get("response_sha256")
    expected_hash = runner.sha256_text(response)
    if response_sha256 is not None and response_sha256 != expected_hash:
        raise MergeError(f"{source}: response_sha256 does not match response")
    _require_int(row, "prompt_tokens", source)
    _require_int(row, "output_tokens", source)
    _require_int(row, "max_new_tokens_budget", source)
    if not isinstance(row.get("json_valid"), bool):
        raise MergeError(f"{source}: json_valid must be boolean")
    _validate_request_timing(row, source)


def _response_sequence_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        [int(row["input_index"]), runner.sha256_text(str(row["response"]))]
        for row in sorted(rows, key=lambda item: int(item["input_index"]))
    ]
    return runner.sha256_text(json.dumps(canonical, separators=(",", ":")))


def _validate_acceptance(scope: Mapping[str, Any], condition: str, source: str) -> None:
    drafts = _require_number(scope, "drafts", source)
    draft_tokens = _require_number(scope, "draft_tokens", source)
    accepted = _require_number(scope, "accepted_tokens", source)
    if accepted > draft_tokens:
        raise MergeError(f"{source}: accepted_tokens exceeds draft_tokens")
    expected_rate = accepted / draft_tokens if draft_tokens else None
    rate = scope.get("acceptance_rate")
    if expected_rate is None:
        if rate is not None:
            raise MergeError(f"{source}: acceptance_rate must be null without draft tokens")
    elif isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise MergeError(f"{source}: acceptance_rate is missing")
    else:
        _assert_close(float(rate), expected_rate, f"{source}.acceptance_rate")
    if condition == "no_mtp":
        if drafts or draft_tokens or accepted:
            raise MergeError(f"{source}: no_mtp must have zero speculative counters")
    elif drafts <= 0.0 or draft_tokens <= 0.0:
        raise MergeError(f"{source}: MTP condition has no draft counters")


def _validate_task_summary(
    scope: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    condition: str,
    task: str,
    source: str,
) -> None:
    if _require_int(scope, "rows", source) != len(rows):
        raise MergeError(f"{source}: rows differs from JSONL")
    prompt_tokens = sum(_require_int(row, "prompt_tokens", source) for row in rows)
    output_tokens = sum(_require_int(row, "output_tokens", source) for row in rows)
    json_valid = sum(bool(row["json_valid"]) for row in rows)
    if _require_int(scope, "prompt_tokens", source) != prompt_tokens:
        raise MergeError(f"{source}: prompt_tokens differs from JSONL")
    if _require_int(scope, "output_tokens", source) != output_tokens:
        raise MergeError(f"{source}: output_tokens differs from JSONL")
    if _require_int(scope, "json_valid", source) != json_valid:
        raise MergeError(f"{source}: json_valid differs from JSONL")
    wall = _require_number(scope, "generation_wall_seconds", source)
    for field in _PARALLEL_TIMING_FIELDS[1:]:
        _require_number(scope, field, source)
    output_tps = scope.get("output_tokens_per_second")
    if wall and isinstance(output_tps, (int, float)) and not isinstance(output_tps, bool):
        _assert_close(float(output_tps), output_tokens / wall, f"{source}.output_tokens_per_second")
    _validate_acceptance(scope, condition, source)
    expected_hash = _response_sequence_sha256(rows)
    if scope.get("response_sequence_sha256") not in (None, expected_hash):
        raise MergeError(f"{source}: response_sequence_sha256 differs from JSONL")
    if scope.get("task") not in (None, task):
        raise MergeError(f"{source}: task identity mismatch")


def _validate_total_summary(
    total: Mapping[str, Any],
    task_scopes: Mapping[str, Mapping[str, Any]],
    condition: str,
    source: str,
) -> None:
    integer_fields = ("rows", "prompt_tokens", "output_tokens", "json_valid")
    numeric_fields = ("drafts", "draft_tokens", "accepted_tokens")
    for field in integer_fields:
        expected = sum(
            _require_int(scope, field, f"{source}.tasks.{task}") for task, scope in task_scopes.items()
        )
        if _require_int(total, field, source) != expected:
            raise MergeError(f"{source}: {field} does not equal task sum")
    for field in numeric_fields:
        expected = sum(
            _require_number(scope, field, f"{source}.tasks.{task}") for task, scope in task_scopes.items()
        )
        _assert_close(_require_number(total, field, source), expected, f"{source}.{field}")
    wall = _require_number(total, "generation_wall_seconds", source)
    for field in _PARALLEL_TIMING_FIELDS[1:]:
        _require_number(total, field, source)
    output_tps = total.get("output_tokens_per_second")
    if wall and isinstance(output_tps, (int, float)) and not isinstance(output_tps, bool):
        _assert_close(
            float(output_tps),
            _require_int(total, "output_tokens", source) / wall,
            f"{source}.output_tokens_per_second",
        )
    _validate_acceptance(total, condition, source)


def _resolve_condition_dir(root: str | Path, condition: str) -> Path:
    raw = os.fspath(root)
    if raw.startswith("s3://"):
        raise MergeError("S3 shard roots are not implemented yet; materialise them locally first")
    path = Path(raw)
    direct = path / "summary.json"
    nested = path / condition / "summary.json"
    if direct.is_file() and nested.is_file():
        raise MergeError(f"ambiguous shard root contains two condition summaries: {path}")
    if direct.is_file():
        return path.resolve()
    if nested.is_file():
        return (path / condition).resolve()
    raise MergeError(f"cannot find summary.json for {condition!r} under shard root: {path}")


def _load_shard(root: Path, condition: str) -> ShardCondition:
    summary = _read_json(root / "summary.json")
    source = str(root)
    if summary.get("status") != "complete":
        raise MergeError(f"{source}: shard status is not complete")
    if summary.get("condition") != condition:
        raise MergeError(f"{source}: summary condition {summary.get('condition')!r} != {condition!r}")
    if not isinstance(summary.get("model"), str) or not summary["model"]:
        raise MergeError(f"{source}: model must be a non-empty string")
    for field in ("max_pixels", "max_model_len", "max_new_tokens_cli_cap"):
        if _require_int(summary, field, source) <= 0:
            raise MergeError(f"{source}: {field} must be positive")
    _require_int(summary, "eos_reserve_tokens", source)
    scopes = summary.get("tasks")
    if not isinstance(scopes, Mapping) or set(scopes) != set(TASKS):
        raise MergeError(f"{source}: summary does not contain exactly the four TEST1070 tasks")

    tasks: dict[str, list[dict[str, Any]]] = {}
    task_image_paths: dict[str, frozenset[str]] = {}
    task_index_paths: dict[str, dict[int, str]] = {}
    for task in TASKS:
        rows = _read_jsonl(root / f"{task}.jsonl")
        raw_indices: set[int] = set()
        image_paths: set[str] = set()
        for line_number, row in enumerate(rows, 1):
            row_source = f"{root / f'{task}.jsonl'}:{line_number}"
            _validate_row(row, condition, task, row_source)
            raw_index = int(row["input_index"])
            if raw_index in raw_indices:
                raise MergeError(f"{row_source}: duplicate shard-local input_index {raw_index}")
            raw_indices.add(raw_index)
            image_uri = _image_path(row, row_source)
            if image_uri in image_paths:
                raise MergeError(f"{row_source}: duplicate image path {image_uri!r}")
            image_paths.add(image_uri)
        task_scope = scopes[task]
        if not isinstance(task_scope, Mapping):
            raise MergeError(f"{source}.tasks.{task}: summary is not an object")
        _validate_task_summary(task_scope, rows, condition, task, f"{source}.tasks.{task}")
        task_hashes = summary.get("task_hashes")
        expected_hash = _response_sequence_sha256(rows)
        if isinstance(task_hashes, Mapping) and task_hashes.get(task) not in (None, expected_hash):
            raise MergeError(f"{source}.task_hashes.{task}: differs from JSONL")
        tasks[task] = rows
        task_image_paths[task] = frozenset(image_paths)
        task_index_paths[task] = {
            int(row["input_index"]): _image_path(row, f"{root / f'{task}.jsonl'}") for row in rows
        }

    first_paths = task_image_paths[TASKS[0]]
    for task, paths in task_image_paths.items():
        if paths != first_paths:
            raise MergeError(f"{source}: {task} image set differs from the other shard tasks")
    first_index_paths = task_index_paths[TASKS[0]]
    for task, index_paths in task_index_paths.items():
        if index_paths != first_index_paths:
            raise MergeError(
                f"{source}: {task} shard-local input_index to image mapping differs from other tasks"
            )
    if _require_int(summary, "images", source) != len(first_paths):
        raise MergeError(f"{source}: images differs from shard JSONL row count")
    world_size = _require_int(summary, "world_size", source)
    if world_size <= 0:
        raise MergeError(f"{source}: world_size must be positive")
    total = summary.get("total")
    if not isinstance(total, Mapping):
        raise MergeError(f"{source}: total summary is missing")
    _validate_total_summary(total, scopes, condition, f"{source}.total")
    timing = summary.get("timing")
    if timing is not None and not isinstance(timing, Mapping):
        raise MergeError(f"{source}: timing must be an object")
    return ShardCondition(
        root=root,
        summary=summary,
        tasks=tasks,
        image_paths=first_paths,
        local_index_to_path=first_index_paths,
    )


def _shared_value(shards: Sequence[ShardCondition], key: str, *, required: bool) -> Any:
    present = [key in shard.summary for shard in shards]
    if required and not all(present):
        raise MergeError(f"shard summaries are missing required field {key!r}")
    if not any(present):
        return None
    if not all(present):
        raise MergeError(f"optional identity field {key!r} is only present in some shards")
    values = [shard.summary[key] for shard in shards]
    if len({_canonical_json(value) for value in values}) != 1:
        raise MergeError(f"shard summaries disagree on {key!r}")
    return values[0]


def _latency_samples(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        timing = row.get("request_timing")
        if not isinstance(timing, Mapping):
            continue
        value = timing.get("end_to_end_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _parallel_timing(scopes: Sequence[Mapping[str, Any]], field: str, source: str) -> dict[str, float]:
    values = [_require_number(scope, field, source) for scope in scopes]
    # Runner summaries retain the historical ``elapsed`` name for the
    # generation sum even though the wall field was renamed.
    sum_rank_field = (
        "sum_rank_generation_elapsed_seconds" if field == "generation_wall_seconds" else f"sum_rank_{field}"
    )
    missing_sum_rank = [index for index, scope in enumerate(scopes) if sum_rank_field not in scope]
    if missing_sum_rank:
        raise MergeError(
            f"{source}: {sum_rank_field} is missing from source shard scopes "
            f"{missing_sum_rank}; refusing to treat max-rank wall time as a rank sum"
        )
    sum_rank = sum(_require_number(scope, sum_rank_field, source) for scope in scopes)
    return {
        field: max(values, default=0.0),
        f"sum_shard_{field}": sum(values),
        sum_rank_field: sum_rank,
    }


def _aggregate_scope(
    rows: Sequence[Mapping[str, Any]],
    scopes: Sequence[Mapping[str, Any]],
    condition: str,
    source: str,
) -> dict[str, Any]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    json_valid = sum(bool(row["json_valid"]) for row in rows)
    drafts = sum(_require_number(scope, "drafts", source) for scope in scopes)
    draft_tokens = sum(_require_number(scope, "draft_tokens", source) for scope in scopes)
    accepted = sum(_require_number(scope, "accepted_tokens", source) for scope in scopes)
    if accepted > draft_tokens:
        raise MergeError(f"{source}: aggregate accepted_tokens exceeds draft_tokens")
    if condition == "no_mtp" and (drafts or draft_tokens or accepted):
        raise MergeError(f"{source}: no_mtp aggregate has speculative counters")
    if condition != "no_mtp" and (drafts <= 0.0 or draft_tokens <= 0.0):
        raise MergeError(f"{source}: MTP aggregate has no draft counters")
    timing: dict[str, Any] = {}
    for field in _PARALLEL_TIMING_FIELDS:
        timing.update(_parallel_timing(scopes, field, source))
    wall = float(timing["generation_wall_seconds"])
    latency_samples = _latency_samples(rows)
    budgets = [int(row["max_new_tokens_budget"]) for row in rows]
    return {
        "rows": len(rows),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "json_valid": json_valid,
        "json_valid_rate": json_valid / len(rows) if rows else None,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        **timing,
        "generation_elapsed_seconds": wall,
        "output_tokens_per_second": output_tokens / wall if wall else None,
        "tokens_per_second": output_tokens / wall if wall else None,
        "images_per_second": len(rows) / wall if wall else None,
        "request_latency_seconds": runner.latency_summary(latency_samples),
        "request_latency_samples_seconds": latency_samples,
        "min_max_new_tokens_budget": min(budgets, default=None),
        "max_max_new_tokens_budget": max(budgets, default=None),
    }


def _aggregate_summary_timing(shards: Sequence[ShardCondition]) -> dict[str, float]:
    timings = [shard.summary.get("timing", {}) for shard in shards]
    keys = set().union(*(timing.keys() for timing in timings))
    result: dict[str, float] = {}
    for key in sorted(keys):
        if not all(key in timing for timing in timings):
            raise MergeError(f"condition timing field {key!r} is only present in some shards")
        values = []
        for timing in timings:
            value = timing[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MergeError(f"condition timing field {key!r} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise MergeError(f"condition timing field {key!r} must be finite and non-negative")
            values.append(number)
        result[key] = sum(values) if key.startswith("sum_rank_") else max(values)
    return result


def _safe_rank_summary_reference(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{source}: rank summary reference must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise MergeError(f"{source}: unsafe rank summary reference {value!r}")
    return path.as_posix()


def _rank_provenance_unavailable(
    shards: Sequence[ShardCondition],
    reasons: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {
        "availability": "unavailable",
        "expected_rank_count": sum(int(shard.summary["world_size"]) for shard in shards),
        "materialized_rank_count": 0,
        "reasons": list(reasons),
        "finalizer_effect": (
            "top-level rank_summaries is omitted, so complete_rank_summary_provenance and "
            "resolved speculative configuration remain unavailable"
        ),
    }


def _materialize_rank_metadata(
    shards: Sequence[ShardCondition],
    condition: str,
    *,
    canonical_manifest: str,
    canonical_manifest_sha256: str,
    canonical_index: Mapping[str, int],
    aggregate_world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate source rank summaries and normalise their manifest/rank identity."""

    references_by_shard: list[list[str]] = []
    unavailable_reasons: list[str] = []
    for shard_ordinal, shard in enumerate(shards):
        references = shard.summary.get("rank_summaries")
        expected = int(shard.summary["world_size"])
        if not isinstance(references, list):
            unavailable_reasons.append(f"shard {shard_ordinal}: rank_summaries is absent")
            references_by_shard.append([])
            continue
        safe_references = [
            _safe_rank_summary_reference(
                reference,
                f"{shard.root}.rank_summaries[{index}]",
            )
            for index, reference in enumerate(references)
        ]
        if len(set(safe_references)) != len(safe_references):
            raise MergeError(f"{shard.root}: rank_summaries contains duplicate references")
        if len(safe_references) != expected:
            unavailable_reasons.append(
                f"shard {shard_ordinal}: has {len(safe_references)} rank references, expected {expected}"
            )
        missing = [reference for reference in safe_references if not (shard.root / reference).is_file()]
        if missing:
            unavailable_reasons.append(
                f"shard {shard_ordinal}: {len(missing)} referenced rank summaries are missing"
            )
        if not isinstance(shard.summary.get("manifest_sha256"), str):
            unavailable_reasons.append(f"shard {shard_ordinal}: manifest_sha256 is absent")
        references_by_shard.append(safe_references)
    if unavailable_reasons:
        return _rank_provenance_unavailable(shards, unavailable_reasons)

    normalized: list[dict[str, Any]] = []
    global_rank_offset = 0
    for shard_ordinal, (shard, references) in enumerate(zip(shards, references_by_shard)):
        source_rank_summaries = [_read_json(shard.root / reference) for reference in references]
        local_world_size = int(shard.summary["world_size"])
        by_local_rank: dict[int, tuple[str, dict[str, Any]]] = {}
        for reference, source_summary in zip(references, source_rank_summaries):
            source = f"{shard.root / reference}"
            if source_summary.get("status") != "complete":
                raise MergeError(f"{source}: source rank status is not complete")
            if source_summary.get("condition") != condition:
                raise MergeError(f"{source}: source rank condition mismatch")
            if _require_int(source_summary, "world_size", source) != local_world_size:
                raise MergeError(f"{source}: source rank world_size differs from shard summary")
            local_rank = _require_int(source_summary, "rank", source)
            if local_rank >= local_world_size:
                raise MergeError(f"{source}: source rank is outside local world_size")
            if local_rank in by_local_rank:
                raise MergeError(f"{shard.root}: duplicate source rank {local_rank}")
            for field in _MANDATORY_SHARED_FIELDS:
                if _canonical_json(source_summary.get(field)) != _canonical_json(shard.summary[field]):
                    raise MergeError(f"{source}: {field} differs from shard condition summary")
            if source_summary.get("manifest_sha256") != shard.summary.get("manifest_sha256"):
                raise MergeError(f"{source}: manifest_sha256 differs from shard condition summary")
            by_local_rank[local_rank] = (reference, source_summary)
        if sorted(by_local_rank) != list(range(local_world_size)):
            raise MergeError(f"{shard.root}: source rank summaries do not cover all local ranks")

        owned_local_indices: set[int] = set()
        for local_rank in range(local_world_size):
            reference, source_summary = by_local_rank[local_rank]
            source = f"{shard.root / reference}"
            shard_indices = source_summary.get("shard_indices")
            if not isinstance(shard_indices, list):
                raise MergeError(f"{source}: shard_indices must be a list")
            local_indices: list[int] = []
            for value in shard_indices:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise MergeError(f"{source}: shard_indices must contain non-negative integers")
                if value not in shard.local_index_to_path:
                    raise MergeError(f"{source}: shard index {value} has no source JSONL row")
                if value in owned_local_indices:
                    raise MergeError(f"{shard.root}: shard index {value} is assigned to multiple ranks")
                owned_local_indices.add(value)
                local_indices.append(value)
            source_tasks = source_summary.get("tasks")
            if not isinstance(source_tasks, Mapping) or set(source_tasks) != set(TASKS):
                raise MergeError(f"{source}: source rank task summaries are incomplete")
            for task in TASKS:
                task_scope = source_tasks[task]
                if not isinstance(task_scope, Mapping):
                    raise MergeError(f"{source}.tasks.{task}: summary is not an object")
                if _require_int(task_scope, "rows", f"{source}.tasks.{task}") != len(local_indices):
                    raise MergeError(f"{source}.tasks.{task}: rows differs from shard_indices")

            canonical_indices = sorted(
                canonical_index[shard.local_index_to_path[index]] for index in local_indices
            )
            global_rank = global_rank_offset + local_rank
            normalized.append(
                {
                    "status": "complete",
                    "summary_kind": "merged_rank_metadata",
                    "condition": condition,
                    "model": shard.summary["model"],
                    "rank": global_rank,
                    "world_size": aggregate_world_size,
                    "manifest": canonical_manifest,
                    "manifest_sha256": canonical_manifest_sha256,
                    "shard_indices": canonical_indices,
                    "speculative": source_summary.get("speculative"),
                    "seed": source_summary.get("seed"),
                    "max_pixels": shard.summary["max_pixels"],
                    "max_model_len": shard.summary["max_model_len"],
                    "max_new_tokens_cli_cap": shard.summary["max_new_tokens_cli_cap"],
                    "eos_reserve_tokens": shard.summary["eos_reserve_tokens"],
                    "timing": source_summary.get("timing"),
                    "completed_at": source_summary.get("completed_at"),
                    "source_rank": {
                        "shard_ordinal": shard_ordinal,
                        "root": str(shard.root),
                        "reference": reference,
                        "rank": local_rank,
                        "world_size": local_world_size,
                        "manifest": source_summary.get("manifest"),
                        "manifest_sha256": source_summary.get("manifest_sha256"),
                        "shard_indices": local_indices,
                        "summary_sha256": runner.sha256_text(_canonical_json(source_summary)),
                    },
                }
            )
        if owned_local_indices != set(shard.local_index_to_path):
            missing = sorted(set(shard.local_index_to_path) - owned_local_indices)
            raise MergeError(f"{shard.root}: source rank summaries omit local indices {missing[:5]}")
        global_rank_offset += local_world_size

    if len(normalized) != aggregate_world_size:
        raise MergeError(f"materialized {len(normalized)} rank summaries, expected {aggregate_world_size}")
    return normalized, {
        "availability": "complete",
        "expected_rank_count": aggregate_world_size,
        "materialized_rank_count": len(normalized),
        "normalization": (
            "source rank identity and local shard_indices were validated, then rank/world_size, "
            "manifest identity, and shard_indices were mapped to the canonical merged condition"
        ),
        "source_identity_preserved_under": "source_rank",
        "finalizer_effect": (
            "top-level rank_summaries is complete; finalizer can validate seed and resolved "
            "speculative configuration"
        ),
    }


def build_merged_condition(
    manifest: str | Path,
    condition: str,
    shard_roots: Sequence[str | Path],
    *,
    expected_images: int = runner.EXPECTED_IMAGE_COUNT,
    expected_shards: int = EXPECTED_SHARDS,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Validate and build one merged condition entirely in memory."""

    if condition not in runner.CONDITIONS:
        raise MergeError(f"unsupported condition: {condition!r}")
    if expected_images <= 0:
        raise MergeError("expected_images must be positive")
    if expected_shards <= 0:
        raise MergeError("expected_shards must be positive")
    if len(shard_roots) != expected_shards:
        raise MergeError(f"expected exactly {expected_shards} shard roots, got {len(shard_roots)}")
    try:
        manifest_rows = runner.load_manifest(os.fspath(manifest), expected_count=expected_images)
    except Exception as exc:  # noqa: BLE001 - normalise runner validation into the CLI error type
        raise MergeError(f"invalid canonical manifest: {exc}") from exc
    canonical_paths = [str(row["s3_image_path"]) for row in manifest_rows]
    canonical_index = {path: index for index, path in enumerate(canonical_paths)}
    canonical_manifest = os.fspath(manifest)
    canonical_manifest_sha256 = runner.manifest_sha256(manifest_rows)

    resolved_roots = [_resolve_condition_dir(root, condition) for root in shard_roots]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise MergeError("the same shard condition root was supplied more than once")
    shards = [_load_shard(root, condition) for root in resolved_roots]
    for field in _MANDATORY_SHARED_FIELDS:
        _shared_value(shards, field, required=True)
    optional_values = {
        field: _shared_value(shards, field, required=False) for field in _OPTIONAL_SHARED_FIELDS
    }
    aggregate_world_size = sum(int(shard.summary["world_size"]) for shard in shards)
    rank_metadata, rank_provenance = _materialize_rank_metadata(
        shards,
        condition,
        canonical_manifest=canonical_manifest,
        canonical_manifest_sha256=canonical_manifest_sha256,
        canonical_index=canonical_index,
        aggregate_world_size=aggregate_world_size,
    )
    rank_summary_references = [
        f"ranks/rank-{int(rank_summary['rank']):03d}/summary.json" for rank_summary in rank_metadata
    ]

    merged_tasks: dict[str, list[dict[str, Any]]] = {}
    task_summaries: dict[str, dict[str, Any]] = {}
    task_hashes: dict[str, str] = {}
    for task in TASKS:
        by_index: dict[int, dict[str, Any]] = {}
        owners: dict[int, str] = {}
        for shard_ordinal, shard in enumerate(shards):
            for row in shard.tasks[task]:
                image_uri = _image_path(row, f"{shard.root}/{task}.jsonl")
                if image_uri not in canonical_index:
                    raise MergeError(
                        f"{shard.root}/{task}.jsonl: image path is absent from canonical manifest: "
                        f"{image_uri!r}"
                    )
                index = canonical_index[image_uri]
                if index in by_index:
                    raise MergeError(
                        f"{task}: duplicate canonical image {image_uri!r} in {owners[index]} and {shard.root}"
                    )
                merged = dict(row)
                merged["source_shard_ordinal"] = shard_ordinal
                merged["source_shard_input_index"] = int(row["input_index"])
                merged["input_index"] = index
                by_index[index] = merged
                owners[index] = str(shard.root)
        missing = [canonical_paths[index] for index in range(expected_images) if index not in by_index]
        if missing:
            preview = ", ".join(repr(value) for value in missing[:3])
            raise MergeError(
                f"{task}: missing {len(missing)} canonical images after merging; first: {preview}"
            )
        merged_rows = [by_index[index] for index in range(expected_images)]
        scopes = [shard.summary["tasks"][task] for shard in shards]
        task_summary = _aggregate_scope(
            merged_rows,
            scopes,
            condition,
            f"merged.{condition}.tasks.{task}",
        )
        task_summary.update(
            {
                "task": task,
                "training_task_name": _TASK_SPEC_BY_OUTPUT[task].name,
                "response_sequence_sha256": _response_sequence_sha256(merged_rows),
                "output": f"{task}.jsonl",
                "source_shard_rows": [len(shard.tasks[task]) for shard in shards],
            }
        )
        merged_tasks[task] = merged_rows
        task_summaries[task] = task_summary
        task_hashes[task] = task_summary["response_sequence_sha256"]

    all_rows = [row for task in TASKS for row in merged_tasks[task]]
    total_scopes = [shard.summary["total"] for shard in shards]
    total = _aggregate_scope(
        all_rows,
        total_scopes,
        condition,
        f"merged.{condition}.total",
    )
    expected_total_rows = expected_images * len(TASKS)
    if total["rows"] != expected_total_rows:
        raise MergeError(f"merged total has {total['rows']} rows, expected {expected_total_rows}")
    for field in ("prompt_tokens", "output_tokens", "json_valid"):
        expected = sum(int(task_summary[field]) for task_summary in task_summaries.values())
        if int(total[field]) != expected:
            raise MergeError(f"merged total {field} differs from merged task sum")
    for field in ("drafts", "draft_tokens", "accepted_tokens"):
        expected = sum(float(task_summary[field]) for task_summary in task_summaries.values())
        _assert_close(float(total[field]), expected, f"merged.{condition}.total.{field}")

    summary: dict[str, Any] = {
        "status": "complete",
        "condition": condition,
        "model": _shared_value(shards, "model", required=True),
        "world_size": aggregate_world_size,
        "manifest": canonical_manifest,
        "manifest_sha256": canonical_manifest_sha256,
        "images": expected_images,
        "max_pixels": _shared_value(shards, "max_pixels", required=True),
        "max_model_len": _shared_value(shards, "max_model_len", required=True),
        "max_new_tokens_cli_cap": _shared_value(shards, "max_new_tokens_cli_cap", required=True),
        "eos_reserve_tokens": _shared_value(shards, "eos_reserve_tokens", required=True),
        "timing": _aggregate_summary_timing(shards),
        "tasks": task_summaries,
        "task_hashes": task_hashes,
        "total": total,
        "timing_provenance": {
            "generation_wall_aggregation": "max source shard generation_wall_seconds",
            "sum_rank_aggregation": "sum validated source sum_rank_* timings",
            "source_shards_assumed_parallel": True,
            "actual_source_job_overlap_verified": False,
            "interpretation": (
                "normalized parallel makespan for generation, not queue-inclusive observed "
                "end-to-end wall time across independently scheduled jobs"
            ),
        },
        "merge": {
            "schema_version": MERGE_SCHEMA_VERSION,
            "mapping": "canonical manifest s3_image_path -> input_index",
            "timing_semantics": (
                "counts and speculative counters are summed; parallel shard wall times are max; "
                "sum_rank_* timings are summed across all source ranks"
            ),
            "rank_provenance": rank_provenance,
            "source_shard_count": len(shards),
            "source_shards": [
                {
                    "ordinal": ordinal,
                    "root": str(shard.root),
                    "manifest": shard.summary.get("manifest"),
                    "manifest_sha256": shard.summary.get("manifest_sha256"),
                    "images": shard.summary["images"],
                    "world_size": shard.summary["world_size"],
                    "completed_at": shard.summary.get("completed_at"),
                    "rank_summaries": shard.summary.get("rank_summaries"),
                }
                for ordinal, shard in enumerate(shards)
            ],
        },
        "completed_at": runner.utc_now(),
    }
    if rank_summary_references:
        summary["rank_summaries"] = rank_summary_references
    summary.update({key: value for key, value in optional_values.items() if value is not None})
    return summary, merged_tasks, rank_metadata


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return (text + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MergeError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_merged_condition(
    output_root: Path,
    condition: str,
    summary: Mapping[str, Any],
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    rank_metadata: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    condition_dir = output_root / condition
    payloads = {
        condition_dir / f"{task}.jsonl": b"".join(_json_bytes(row, pretty=False) for row in tasks[task])
        for task in TASKS
    }
    payloads[condition_dir / "summary.json"] = _json_bytes(summary, pretty=True)
    for rank_summary in rank_metadata:
        rank = _require_int(rank_summary, "rank", "materialized rank metadata")
        payloads[condition_dir / "ranks" / f"rank-{rank:03d}" / "summary.json"] = _json_bytes(
            rank_summary,
            pretty=True,
        )
    payloads[condition_dir / "status.json"] = _json_bytes(
        {
            "status": "complete",
            "condition": condition,
            "merged_at": summary.get("completed_at"),
            "source_shard_count": len(summary.get("merge", {}).get("source_shards", [])),
        },
        pretty=True,
    )
    existing = [path for path in payloads if path.exists()]
    if existing and not overwrite:
        raise MergeError(f"output already exists (use --overwrite): {existing[0]}")
    for path, payload in payloads.items():
        _write_atomic(path, payload, overwrite=overwrite)
    return condition_dir


def merge_condition(
    manifest: str | Path,
    condition: str,
    shard_roots: Sequence[str | Path],
    output_root: Path,
    *,
    expected_images: int = runner.EXPECTED_IMAGE_COUNT,
    expected_shards: int = EXPECTED_SHARDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = (output_root / condition).resolve()
    source_directories = {_resolve_condition_dir(root, condition) for root in shard_roots}
    if target in source_directories:
        raise MergeError("output condition directory must not also be a source shard directory")
    summary, tasks, rank_metadata = build_merged_condition(
        manifest,
        condition,
        shard_roots,
        expected_images=expected_images,
        expected_shards=expected_shards,
    )
    write_merged_condition(
        output_root,
        condition,
        summary,
        tasks,
        rank_metadata,
        overwrite=overwrite,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Canonical TEST1070 JSONL manifest")
    parser.add_argument("--condition", required=True, choices=runner.CONDITIONS)
    parser.add_argument(
        "--shard-root",
        action="append",
        required=True,
        help="Local shard condition directory or its parent; repeat exactly four times",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=runner.EXPECTED_IMAGE_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = merge_condition(
            args.manifest,
            args.condition,
            args.shard_root,
            args.output_root,
            expected_images=args.expected_images,
            overwrite=args.overwrite,
        )
    except MergeError as exc:
        print(f"merge failed: {exc}", file=os.sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": summary["status"],
                "condition": summary["condition"],
                "images": summary["images"],
                "output": str(args.output_root / args.condition),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
