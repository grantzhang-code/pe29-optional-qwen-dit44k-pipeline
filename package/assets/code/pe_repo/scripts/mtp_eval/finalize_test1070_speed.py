#!/usr/bin/env python3
"""Build a strict three-condition TEST1070 MTP speed report.

The inference workers intentionally write condition-local artifacts.  This
postprocessor reads the three completed condition summaries and their four
merged task JSONL files from either a local directory or one S3 prefix.  It
does not trust a reported speedup until the recorded inputs and generated
outputs are demonstrably like-for-like.

No output is written until all three conditions have been loaded, validated,
and found to have ``status=complete``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse


CONDITIONS = ("trained_mtp", "original_mtp", "no_mtp")
TASKS = ("global_desc_bg", "full_json_no_region_desc", "full_json", "layout_only")
DEFAULT_EXPECTED_IMAGES = 1070
REPORT_SCHEMA_VERSION = 1
LATENCY_FIELDS = {
    "end_to_end_seconds": "end_to_end_seconds",
    "time_to_first_token_seconds": "time_to_first_token_seconds",
    "decode_seconds": "decode_seconds",
    "engine_inference_seconds": "engine_inference_seconds",
}


class FinalizationError(RuntimeError):
    """Raised when a complete, internally consistent report cannot be built."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise FinalizationError(f"unsafe relative artifact path: {value!r}")
    return path.as_posix()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise FinalizationError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _is_s3_not_found(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    details = response.get("Error", {})
    return isinstance(details, Mapping) and str(details.get("Code", "")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


class EvaluationSource:
    """Read evaluation artifacts without materialising JSONL response text."""

    def __init__(self, root: str | Path, *, s3_client: Any | None = None) -> None:
        raw = os.fspath(root)
        self.display_root = raw.rstrip("/")
        self.s3_client = s3_client
        self.bucket: str | None = None
        self.prefix: str | None = None
        self.local_root: Path | None = None
        if raw.startswith("s3://"):
            self.bucket, self.prefix = _parse_s3_uri(raw)
            if self.s3_client is None:
                import boto3

                self.s3_client = boto3.client("s3")
        else:
            if raw.startswith("file://"):
                raw = urlparse(raw).path
            self.local_root = Path(raw)

    def _s3_key(self, relative: str) -> str:
        relative = _safe_relative_path(relative)
        return "/".join(part for part in (self.prefix, relative) if part)

    def _local_path(self, relative: str) -> Path:
        assert self.local_root is not None
        return self.local_root / _safe_relative_path(relative)

    def read_bytes(self, relative: str, *, optional: bool = False) -> bytes | None:
        if self.local_root is not None:
            path = self._local_path(relative)
            if optional and not path.is_file():
                return None
            try:
                return path.read_bytes()
            except FileNotFoundError as exc:
                raise FinalizationError(f"missing evaluation artifact: {path}") from exc
        assert self.bucket is not None and self.s3_client is not None
        key = self._s3_key(relative)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto is intentionally optional
            if optional and _is_s3_not_found(exc):
                return None
            raise FinalizationError(f"cannot read s3://{self.bucket}/{key}: {exc}") from exc
        return response["Body"].read()

    def read_json(self, relative: str, *, optional: bool = False) -> dict[str, Any] | None:
        body = self.read_bytes(relative, optional=optional)
        if body is None:
            return None
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalizationError(f"invalid JSON artifact {relative}: {exc}") from exc
        if not isinstance(value, dict):
            raise FinalizationError(f"JSON artifact is not an object: {relative}")
        return value

    def iter_jsonl(self, relative: str) -> Iterator[tuple[int, dict[str, Any]]]:
        if self.local_root is not None:
            path = self._local_path(relative)
            try:
                handle = path.open("rb")
            except FileNotFoundError as exc:
                raise FinalizationError(f"missing evaluation artifact: {path}") from exc
            with handle:
                for line_number, raw_line in enumerate(handle, 1):
                    if raw_line.strip():
                        yield line_number, _decode_jsonl_row(raw_line, relative, line_number)
            return

        assert self.bucket is not None and self.s3_client is not None
        key = self._s3_key(relative)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise FinalizationError(f"cannot read s3://{self.bucket}/{key}: {exc}") from exc
        body = response["Body"]
        try:
            lines: Iterable[bytes]
            if hasattr(body, "iter_lines"):
                lines = body.iter_lines()
            else:
                lines = body.read().splitlines()
            for line_number, raw_line in enumerate(lines, 1):
                if raw_line.strip():
                    yield line_number, _decode_jsonl_row(raw_line, relative, line_number)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()


def _decode_jsonl_row(raw_line: bytes, relative: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid JSONL row {relative}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"JSONL row is not an object: {relative}:{line_number}")
    return value


@dataclass(frozen=True)
class RowRecord:
    input_index: int
    image_uri: str
    response_sha256: str
    response_sha256_recorded: bool
    prompt_contract_sha256: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    max_new_tokens_budget: int | None
    finish_reason: str | None
    finish_reason_recorded: bool
    stop_reason: str | None
    stop_reason_recorded: bool
    json_valid: bool | None
    condition_recorded: bool
    task_recorded: bool
    request_timing: Mapping[str, float | None]


@dataclass
class ConditionData:
    condition: str
    summary: dict[str, Any]
    tasks: dict[str, dict[int, RowRecord]]
    rank_summaries: list[dict[str, Any]]
    missing_rank_summaries: list[str]


def _optional_nonnegative_int(row: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizationError(f"{source}: {key} must be a non-negative integer")
    return value


def _optional_nonnegative_float(value: Any, source: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{source} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FinalizationError(f"{source} must be finite and non-negative")
    return result


def _prompt_contract_sha256(row: Mapping[str, Any], source: str) -> str | None:
    if "messages" not in row or row["messages"] is None:
        return None
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise FinalizationError(f"{source}: messages must contain system and user entries")
    contract = messages[:2]
    if not all(isinstance(message, dict) for message in contract):
        raise FinalizationError(f"{source}: prompt messages are not objects")
    if [message.get("role") for message in contract] != ["system", "user"]:
        raise FinalizationError(f"{source}: prompt messages do not have system/user roles")
    return sha256_text(_canonical_json(contract))


def _normalise_row(
    value: Mapping[str, Any],
    *,
    condition: str,
    task: str,
    source: str,
) -> RowRecord:
    index = value.get("input_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise FinalizationError(f"{source}: input_index must be a non-negative integer")
    images = value.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], Mapping):
        raise FinalizationError(f"{source}: images[0].path is missing")
    image_uri = images[0].get("path")
    if not isinstance(image_uri, str) or not image_uri:
        raise FinalizationError(f"{source}: images[0].path is missing")
    response = value.get("response")
    if not isinstance(response, str) or not response.strip():
        raise FinalizationError(f"{source}: response must be a non-empty string")
    derived_response_sha256 = sha256_text(response)
    recorded_response_sha256 = value.get("response_sha256")
    if recorded_response_sha256 is not None and recorded_response_sha256 != derived_response_sha256:
        raise FinalizationError(f"{source}: response_sha256 does not match response text")

    timing_value = value.get("request_timing")
    timing: dict[str, float | None] = {field: None for field in LATENCY_FIELDS}
    if timing_value is not None:
        if not isinstance(timing_value, Mapping):
            raise FinalizationError(f"{source}: request_timing must be an object")
        for field in LATENCY_FIELDS:
            timing[field] = _optional_nonnegative_float(
                timing_value.get(field), f"{source}: request_timing.{field}"
            )

    json_valid = value.get("json_valid")
    if json_valid is not None and not isinstance(json_valid, bool):
        raise FinalizationError(f"{source}: json_valid must be boolean when present")
    if "condition" in value and value["condition"] != condition:
        raise FinalizationError(f"{source}: row condition {value['condition']!r} != {condition!r}")
    if "task" in value and value["task"] != task:
        raise FinalizationError(f"{source}: row task {value['task']!r} != {task!r}")

    return RowRecord(
        input_index=index,
        image_uri=image_uri,
        response_sha256=derived_response_sha256,
        response_sha256_recorded=recorded_response_sha256 is not None,
        prompt_contract_sha256=_prompt_contract_sha256(value, source),
        prompt_tokens=_optional_nonnegative_int(value, "prompt_tokens", source),
        output_tokens=_optional_nonnegative_int(value, "output_tokens", source),
        max_new_tokens_budget=_optional_nonnegative_int(value, "max_new_tokens_budget", source),
        finish_reason=_canonical_json(value.get("finish_reason")) if "finish_reason" in value else None,
        finish_reason_recorded="finish_reason" in value,
        stop_reason=_canonical_json(value.get("stop_reason")) if "stop_reason" in value else None,
        stop_reason_recorded="stop_reason" in value,
        json_valid=json_valid,
        condition_recorded="condition" in value,
        task_recorded="task" in value,
        request_timing=timing,
    )


def _response_sequence_sha256(records: Mapping[int, RowRecord]) -> str:
    canonical = [[index, records[index].response_sha256] for index in sorted(records)]
    return sha256_text(json.dumps(canonical, separators=(",", ":")))


def _load_task_rows(
    source: EvaluationSource,
    condition: str,
    task: str,
    expected_images: int,
) -> dict[int, RowRecord]:
    relative = f"{condition}/{task}.jsonl"
    records: dict[int, RowRecord] = {}
    for line_number, value in source.iter_jsonl(relative):
        record = _normalise_row(
            value,
            condition=condition,
            task=task,
            source=f"{relative}:{line_number}",
        )
        if record.input_index in records:
            raise FinalizationError(f"{relative}: duplicate input_index {record.input_index}")
        records[record.input_index] = record
    if sorted(records) != list(range(expected_images)):
        raise FinalizationError(
            f"{relative}: input indices do not exactly cover 0..{expected_images - 1}"
        )
    image_uris = [record.image_uri for record in records.values()]
    if len(set(image_uris)) != expected_images:
        raise FinalizationError(f"{relative}: image paths are not unique")
    return records


def _load_rank_summaries(
    source: EvaluationSource,
    condition: str,
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    references = summary.get("rank_summaries")
    if not isinstance(references, list):
        return [], ["rank_summaries field is absent"]
    loaded: list[dict[str, Any]] = []
    missing: list[str] = []
    for reference in references:
        if not isinstance(reference, str):
            raise FinalizationError(f"{condition}: rank_summaries contains a non-string path")
        relative = f"{condition}/{_safe_relative_path(reference)}"
        value = source.read_json(relative, optional=True)
        if value is None:
            missing.append(reference)
        else:
            loaded.append(value)
    return loaded, missing


def _require_numeric(mapping: Mapping[str, Any], key: str, source: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{source}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FinalizationError(f"{source}: {key} must be finite and non-negative")
    return result


def _require_int(mapping: Mapping[str, Any], key: str, source: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizationError(f"{source}: {key} must be a non-negative integer")
    return value


def _assert_close(actual: float, expected: float, source: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise FinalizationError(f"{source}: {actual!r} != expected {expected!r}")


def _validate_acceptance(scope: Mapping[str, Any], *, condition: str, source: str) -> None:
    drafts = _require_numeric(scope, "drafts", source)
    draft_tokens = _require_numeric(scope, "draft_tokens", source)
    accepted = _require_numeric(scope, "accepted_tokens", source)
    if accepted > draft_tokens:
        raise FinalizationError(f"{source}: accepted_tokens exceeds draft_tokens")
    reported_rate = scope.get("acceptance_rate")
    expected_rate = accepted / draft_tokens if draft_tokens else None
    if expected_rate is None:
        if reported_rate is not None:
            raise FinalizationError(f"{source}: acceptance_rate must be null with no draft tokens")
    elif not isinstance(reported_rate, (int, float)):
        raise FinalizationError(f"{source}: acceptance_rate is missing")
    else:
        _assert_close(float(reported_rate), expected_rate, f"{source}: acceptance_rate")
    if condition == "no_mtp":
        if drafts or draft_tokens or accepted:
            raise FinalizationError(f"{source}: no_mtp unexpectedly reports speculative counters")
    elif drafts <= 0.0 or draft_tokens <= 0.0:
        raise FinalizationError(f"{source}: MTP condition has no draft counters")


def _validate_condition(data: ConditionData, expected_images: int) -> None:
    condition = data.condition
    summary = data.summary
    if summary.get("status") != "complete":
        raise FinalizationError(f"{condition}: status is {summary.get('status')!r}, not complete")
    if summary.get("condition") != condition:
        raise FinalizationError(f"{condition}: summary condition mismatch")
    if _require_int(summary, "images", condition) != expected_images:
        raise FinalizationError(f"{condition}: images does not equal {expected_images}")
    if set(summary.get("tasks", {})) != set(TASKS):
        raise FinalizationError(f"{condition}: summary task set is not the four training tasks")
    if set(data.tasks) != set(TASKS):
        raise FinalizationError(f"{condition}: merged task file set is incomplete")

    task_prompt_tokens = 0
    task_output_tokens = 0
    task_rows = 0
    task_json_valid = 0
    all_json_valid_available = True
    counter_totals = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    for task in TASKS:
        scope_name = f"{condition}.tasks.{task}"
        task_summary = summary["tasks"][task]
        if not isinstance(task_summary, Mapping):
            raise FinalizationError(f"{scope_name}: summary is not an object")
        records = data.tasks[task]
        if _require_int(task_summary, "rows", scope_name) != len(records):
            raise FinalizationError(f"{scope_name}: row count differs from JSONL")
        task_rows += len(records)

        prompt_values = [record.prompt_tokens for record in records.values()]
        if all(value is not None for value in prompt_values):
            observed = sum(int(value) for value in prompt_values if value is not None)
            if _require_int(task_summary, "prompt_tokens", scope_name) != observed:
                raise FinalizationError(f"{scope_name}: prompt_tokens differs from JSONL")
        task_prompt_tokens += _require_int(task_summary, "prompt_tokens", scope_name)

        output_values = [record.output_tokens for record in records.values()]
        if all(value is not None for value in output_values):
            observed = sum(int(value) for value in output_values if value is not None)
            if _require_int(task_summary, "output_tokens", scope_name) != observed:
                raise FinalizationError(f"{scope_name}: output_tokens differs from JSONL")
        task_output_tokens += _require_int(task_summary, "output_tokens", scope_name)

        json_values = [record.json_valid for record in records.values()]
        if all(value is not None for value in json_values):
            observed_json_valid = sum(bool(value) for value in json_values)
            if _require_int(task_summary, "json_valid", scope_name) != observed_json_valid:
                raise FinalizationError(f"{scope_name}: json_valid differs from JSONL")
            task_json_valid += observed_json_valid
        else:
            all_json_valid_available = False

        sequence_sha256 = _response_sequence_sha256(records)
        if task_summary.get("response_sequence_sha256") not in (None, sequence_sha256):
            raise FinalizationError(f"{scope_name}: response sequence hash differs from JSONL")
        task_hashes = summary.get("task_hashes")
        if isinstance(task_hashes, Mapping) and task_hashes.get(task) not in (None, sequence_sha256):
            raise FinalizationError(f"{scope_name}: condition task hash differs from JSONL")

        generation_wall = _require_numeric(task_summary, "generation_wall_seconds", scope_name)
        output_tps = task_summary.get("output_tokens_per_second")
        if generation_wall and isinstance(output_tps, (int, float)):
            _assert_close(
                float(output_tps),
                _require_int(task_summary, "output_tokens", scope_name) / generation_wall,
                f"{scope_name}: output_tokens_per_second",
            )
        _validate_acceptance(task_summary, condition=condition, source=scope_name)
        for key in counter_totals:
            counter_totals[key] += _require_numeric(task_summary, key, scope_name)

    total = summary.get("total")
    if not isinstance(total, Mapping):
        raise FinalizationError(f"{condition}: total summary is missing")
    if _require_int(total, "rows", f"{condition}.total") != task_rows:
        raise FinalizationError(f"{condition}.total: rows does not equal task sum")
    if _require_int(total, "prompt_tokens", f"{condition}.total") != task_prompt_tokens:
        raise FinalizationError(f"{condition}.total: prompt_tokens does not equal task sum")
    if _require_int(total, "output_tokens", f"{condition}.total") != task_output_tokens:
        raise FinalizationError(f"{condition}.total: output_tokens does not equal task sum")
    if all_json_valid_available and _require_int(total, "json_valid", f"{condition}.total") != task_json_valid:
        raise FinalizationError(f"{condition}.total: json_valid does not equal task sum")
    for key, expected in counter_totals.items():
        _assert_close(_require_numeric(total, key, f"{condition}.total"), expected, f"{condition}.total.{key}")
    generation_wall = _require_numeric(total, "generation_wall_seconds", f"{condition}.total")
    output_tps = total.get("output_tokens_per_second")
    if generation_wall and isinstance(output_tps, (int, float)):
        _assert_close(
            float(output_tps),
            task_output_tokens / generation_wall,
            f"{condition}.total.output_tokens_per_second",
        )
    _validate_acceptance(total, condition=condition, source=f"{condition}.total")

    max_model_len = _require_int(summary, "max_model_len", condition)
    eos_reserve = _require_int(summary, "eos_reserve_tokens", condition)
    _require_int(summary, "max_pixels", condition)
    for task, records in data.tasks.items():
        for record in records.values():
            if record.prompt_tokens is not None and record.output_tokens is not None:
                if record.prompt_tokens + record.output_tokens > max_model_len:
                    raise FinalizationError(
                        f"{condition}.{task}[{record.input_index}]: actual context exceeds max_model_len"
                    )
            if record.prompt_tokens is not None and record.max_new_tokens_budget is not None:
                if record.prompt_tokens + record.max_new_tokens_budget + eos_reserve > max_model_len:
                    raise FinalizationError(
                        f"{condition}.{task}[{record.input_index}]: budget exceeds max_model_len"
                    )


def load_conditions(
    input_root: str | Path,
    *,
    expected_images: int = DEFAULT_EXPECTED_IMAGES,
    s3_client: Any | None = None,
) -> tuple[EvaluationSource, dict[str, ConditionData]]:
    source = EvaluationSource(input_root, s3_client=s3_client)
    summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        value = source.read_json(f"{condition}/summary.json")
        assert value is not None
        summaries[condition] = value
    incomplete = {
        condition: summary.get("status")
        for condition, summary in summaries.items()
        if summary.get("status") != "complete"
    }
    if incomplete:
        raise FinalizationError(f"all 3/3 conditions must be complete; incomplete={incomplete}")

    data: dict[str, ConditionData] = {}
    for condition in CONDITIONS:
        summary = summaries[condition]
        rows = {
            task: _load_task_rows(source, condition, task, expected_images)
            for task in TASKS
        }
        rank_summaries, missing_ranks = _load_rank_summaries(source, condition, summary)
        condition_data = ConditionData(
            condition=condition,
            summary=summary,
            tasks=rows,
            rank_summaries=rank_summaries,
            missing_rank_summaries=missing_ranks,
        )
        _validate_condition(condition_data, expected_images)
        data[condition] = condition_data
    return source, data


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    missing = len(values) - len(clean)
    if not clean:
        availability = "unavailable"
    elif missing:
        availability = "partial"
    else:
        availability = "complete"
    return {
        "availability": availability,
        "count": len(clean),
        "missing_count": missing,
        "mean": sum(clean) / len(clean) if clean else None,
        "min": min(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean) if clean else None,
    }


def _field_availability(records: Sequence[RowRecord]) -> dict[str, Any]:
    total = len(records)

    def status(count: int) -> dict[str, Any]:
        return {
            "availability": "complete" if count == total else ("partial" if count else "unavailable"),
            "available_count": count,
            "missing_count": total - count,
        }

    result = {
        "response_sha256_recorded": status(sum(row.response_sha256_recorded for row in records)),
        "prompt_contract": status(sum(row.prompt_contract_sha256 is not None for row in records)),
        "prompt_tokens": status(sum(row.prompt_tokens is not None for row in records)),
        "output_tokens": status(sum(row.output_tokens is not None for row in records)),
        "max_new_tokens_budget": status(sum(row.max_new_tokens_budget is not None for row in records)),
        "finish_reason": status(sum(row.finish_reason_recorded for row in records)),
        "stop_reason": status(sum(row.stop_reason_recorded for row in records)),
        "json_valid": status(sum(row.json_valid is not None for row in records)),
        "condition_label": status(sum(row.condition_recorded for row in records)),
        "task_label": status(sum(row.task_recorded for row in records)),
    }
    for field in LATENCY_FIELDS:
        result[f"request_timing.{field}"] = status(
            sum(row.request_timing[field] is not None for row in records)
        )
    return result


def _latency_report(records: Sequence[RowRecord]) -> dict[str, Any]:
    return {
        output_name: distribution([row.request_timing[source_name] for row in records])
        for output_name, source_name in LATENCY_FIELDS.items()
    }


def _context_report(
    records: Sequence[RowRecord],
    *,
    max_model_len: int,
    max_pixels: int,
    eos_reserve_tokens: int,
) -> dict[str, Any]:
    actual_context: list[int | None] = []
    budgeted_context: list[int | None] = []
    prompt_fraction: list[float | None] = []
    actual_fraction: list[float | None] = []
    budgeted_fraction: list[float | None] = []
    context_headroom: list[int | None] = []
    output_budget_fraction: list[float | None] = []
    for row in records:
        prompt_fraction.append(
            row.prompt_tokens / max_model_len if row.prompt_tokens is not None else None
        )
        if row.prompt_tokens is not None and row.output_tokens is not None:
            used = row.prompt_tokens + row.output_tokens
            actual_context.append(used)
            actual_fraction.append(used / max_model_len)
            context_headroom.append(max_model_len - used)
        else:
            actual_context.append(None)
            actual_fraction.append(None)
            context_headroom.append(None)
        if row.prompt_tokens is not None and row.max_new_tokens_budget is not None:
            budgeted = row.prompt_tokens + row.max_new_tokens_budget + eos_reserve_tokens
            budgeted_context.append(budgeted)
            budgeted_fraction.append(budgeted / max_model_len)
        else:
            budgeted_context.append(None)
            budgeted_fraction.append(None)
        if row.output_tokens is not None and row.max_new_tokens_budget:
            output_budget_fraction.append(row.output_tokens / row.max_new_tokens_budget)
        else:
            output_budget_fraction.append(None)
    return {
        "configured_max_model_len": max_model_len,
        "configured_max_pixels": max_pixels,
        "configured_max_pixels_human": "1536x1536 area" if max_pixels == 1536 * 1536 else None,
        "eos_reserve_tokens": eos_reserve_tokens,
        "prompt_tokens_combined_text_and_visual": distribution([row.prompt_tokens for row in records]),
        "prompt_context_fraction": distribution(prompt_fraction),
        "output_tokens": distribution([row.output_tokens for row in records]),
        "max_new_tokens_budget": distribution([row.max_new_tokens_budget for row in records]),
        "actual_context_tokens_prompt_plus_output": distribution(actual_context),
        "actual_context_fraction": distribution(actual_fraction),
        "budgeted_context_tokens_including_eos_reserve": distribution(budgeted_context),
        "budgeted_context_fraction": distribution(budgeted_fraction),
        "actual_context_headroom_tokens": distribution(context_headroom),
        "output_budget_fraction": distribution(output_budget_fraction),
        "direct_measurements_unavailable": {
            "visual_token_count": "rows only record combined prompt_tokens",
            "source_image_dimensions": "not recorded in rows",
            "processed_image_dimensions": "not recorded in rows",
            "processed_image_pixels": "max_pixels is configuration, not an observed value",
        },
    }


def _acceptance_report(scope: Mapping[str, Any], condition: str) -> dict[str, Any]:
    drafts = float(scope["drafts"])
    draft_tokens = float(scope["draft_tokens"])
    accepted = float(scope["accepted_tokens"])
    return {
        "applicable": condition != "no_mtp",
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "mean_acceptance_length_including_bonus_token": (
            1.0 + accepted / drafts if drafts else None
        ),
        "definition": "accepted_tokens / draft_tokens (vLLM 0.24 counters)",
    }


def _scope_report(
    condition: str,
    scope: Mapping[str, Any],
    records: Sequence[RowRecord],
    *,
    max_model_len: int,
    max_pixels: int,
    eos_reserve_tokens: int,
) -> dict[str, Any]:
    rows = int(scope["rows"])
    output_tokens = int(scope["output_tokens"])
    generation_wall = float(scope["generation_wall_seconds"])
    return {
        "raw": {
            "rows": rows,
            "prompt_tokens": int(scope["prompt_tokens"]),
            "output_tokens": output_tokens,
            "json_valid": scope.get("json_valid"),
            "json_valid_rate": scope.get("json_valid_rate"),
            "generation_wall_seconds": generation_wall,
            "output_tokens_per_second": output_tokens / generation_wall if generation_wall else None,
            "images_per_second": rows / generation_wall if generation_wall else None,
            "image_decode_seconds": scope.get("image_decode_seconds"),
            "multimodal_preprocess_seconds": scope.get("multimodal_preprocess_seconds"),
            "elapsed_seconds": scope.get("elapsed_seconds"),
        },
        "acceptance": _acceptance_report(scope, condition),
        "latency": _latency_report(records),
        "prompt_and_context_utilization": _context_report(
            records,
            max_model_len=max_model_len,
            max_pixels=max_pixels,
            eos_reserve_tokens=eos_reserve_tokens,
        ),
        "row_field_availability": _field_availability(records),
    }


def _condition_report(data: ConditionData) -> dict[str, Any]:
    summary = data.summary
    max_model_len = int(summary["max_model_len"])
    max_pixels = int(summary["max_pixels"])
    eos_reserve = int(summary["eos_reserve_tokens"])
    all_records = [record for task in TASKS for record in data.tasks[task].values()]
    return {
        "model_path_as_recorded": summary.get("model"),
        "manifest": summary.get("manifest"),
        "manifest_sha256": summary.get("manifest_sha256"),
        "world_size": summary.get("world_size"),
        "images": summary.get("images"),
        "max_model_len": max_model_len,
        "max_pixels": max_pixels,
        "max_new_tokens_cli_cap": summary.get("max_new_tokens_cli_cap"),
        "eos_reserve_tokens": eos_reserve,
        "timing_provenance": summary.get("timing_provenance"),
        "total": _scope_report(
            data.condition,
            summary["total"],
            all_records,
            max_model_len=max_model_len,
            max_pixels=max_pixels,
            eos_reserve_tokens=eos_reserve,
        ),
        "tasks": {
            task: _scope_report(
                data.condition,
                summary["tasks"][task],
                list(data.tasks[task].values()),
                max_model_len=max_model_len,
                max_pixels=max_pixels,
                eos_reserve_tokens=eos_reserve,
            )
            for task in TASKS
        },
    }


def _value_report(values: Mapping[str, Any]) -> dict[str, Any]:
    available = {condition: value for condition, value in values.items() if value is not None}
    serialised = {_canonical_json(value) for value in available.values()}
    return {
        "values": dict(values),
        "availability": (
            "complete" if len(available) == len(CONDITIONS) else ("partial" if available else "unavailable")
        ),
        "consistent": len(available) == len(CONDITIONS) and len(serialised) == 1,
    }


def _rank_metadata(data: ConditionData) -> dict[str, Any]:
    if not data.rank_summaries:
        return {
            "availability": "unavailable",
            "missing": data.missing_rank_summaries,
            "seed": None,
            "resolved_speculative_config": None,
            "internally_consistent": False,
        }
    seeds = {summary.get("seed") for summary in data.rank_summaries}
    speculative_values = {
        _canonical_json(summary.get("speculative")) for summary in data.rank_summaries
    }
    expected_count = data.summary.get("world_size")
    complete = not data.missing_rank_summaries and len(data.rank_summaries) == expected_count
    summary_identity_valid = all(
        summary.get("status") == "complete"
        and summary.get("condition") == data.condition
        and summary.get("world_size") == expected_count
        and summary.get("manifest_sha256") == data.summary.get("manifest_sha256")
        and summary.get("max_pixels") == data.summary.get("max_pixels")
        and summary.get("max_model_len") == data.summary.get("max_model_len")
        and summary.get("max_new_tokens_cli_cap") == data.summary.get("max_new_tokens_cli_cap")
        and summary.get("eos_reserve_tokens") == data.summary.get("eos_reserve_tokens")
        for summary in data.rank_summaries
    )
    return {
        "availability": "complete" if complete else "partial",
        "loaded_count": len(data.rank_summaries),
        "missing": data.missing_rank_summaries,
        "seed": next(iter(seeds)) if len(seeds) == 1 else None,
        "resolved_speculative_config": (
            data.rank_summaries[0].get("speculative") if len(speculative_values) == 1 else None
        ),
        "internally_consistent": (
            complete
            and summary_identity_valid
            and len(seeds) == 1
            and None not in seeds
            and len(speculative_values) == 1
        ),
    }


def _speculative_design_check(condition: str, rank_metadata: Mapping[str, Any]) -> dict[str, Any]:
    config = rank_metadata.get("resolved_speculative_config")
    if config is None:
        return {
            "availability": "unavailable",
            "passed": None,
            "expected": "MTP method=mtp,K=1" if condition != "no_mtp" else "speculation disabled",
            "recorded": None,
        }
    if not isinstance(config, Mapping):
        return {
            "availability": "complete",
            "passed": False,
            "expected": "MTP method=mtp,K=1" if condition != "no_mtp" else "speculation disabled",
            "recorded": config,
        }
    if condition == "no_mtp":
        passed = (
            config.get("enabled") is False
            and config.get("method") is None
            and config.get("num_speculative_tokens") == 0
        )
    else:
        architectures = config.get("draft_architectures")
        architecture_valid = (
            not architectures
            or (isinstance(architectures, list) and "Qwen3_5MTP" in architectures)
        )
        passed = (
            config.get("enabled") is True
            and config.get("method") == "mtp"
            and config.get("num_speculative_tokens") == 1
            and architecture_valid
        )
    return {
        "availability": "complete",
        "passed": passed,
        "expected": "MTP method=mtp,K=1" if condition != "no_mtp" else "speculation disabled",
        "recorded": config,
    }


def _prompt_provenance(data: ConditionData, task: str) -> dict[str, Any]:
    records = list(data.tasks[task].values())
    hashes = {row.prompt_contract_sha256 for row in records if row.prompt_contract_sha256 is not None}
    return {
        "sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "available_count": sum(row.prompt_contract_sha256 is not None for row in records),
        "missing_count": sum(row.prompt_contract_sha256 is None for row in records),
        "internally_consistent": len(hashes) == 1 and all(
            row.prompt_contract_sha256 is not None for row in records
        ),
    }


def _image_sequence_sha256(records: Mapping[int, RowRecord]) -> str:
    value = [[index, records[index].image_uri] for index in sorted(records)]
    return sha256_text(json.dumps(value, separators=(",", ":")))


def _extract_optional_shared_field(summary: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value: Any = summary
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                break
            value = value[component]
        else:
            return value
    return None


def _provenance_report(data: Mapping[str, ConditionData]) -> dict[str, Any]:
    shared_fields = (
        "manifest",
        "manifest_sha256",
        "world_size",
        "images",
        "max_pixels",
        "max_model_len",
        "max_new_tokens_cli_cap",
        "eos_reserve_tokens",
    )
    recorded = {
        field: _value_report({condition: item.summary.get(field) for condition, item in data.items()})
        for field in shared_fields
    }
    rank_metadata = {condition: _rank_metadata(item) for condition, item in data.items()}
    speculative_design = {
        condition: _speculative_design_check(condition, rank_metadata[condition])
        for condition in CONDITIONS
    }
    recorded["seed"] = _value_report(
        {condition: rank_metadata[condition].get("seed") for condition in CONDITIONS}
    )
    prompt_contracts: dict[str, Any] = {}
    image_sequences: dict[str, Any] = {}
    for task in TASKS:
        prompt_values = {
            condition: _prompt_provenance(item, task) for condition, item in data.items()
        }
        prompt_contracts[task] = {
            "by_condition": prompt_values,
            "consistent": all(value["internally_consistent"] for value in prompt_values.values())
            and len({value["sha256"] for value in prompt_values.values()}) == 1,
        }
        image_sequences[task] = _value_report(
            {
                condition: _image_sequence_sha256(item.tasks[task])
                for condition, item in data.items()
            }
        )

    optional_specs = {
        "target_backbone_fingerprint": (
            ("target_backbone_fingerprint",),
            ("provenance", "target_backbone_fingerprint"),
        ),
        "source_checkpoint_manifest_sha256": (
            ("source_checkpoint_manifest_sha256",),
            ("provenance", "source_checkpoint_manifest_sha256"),
        ),
        "runtime_image_digest": (
            ("runtime_image_digest",),
            ("runtime", "image_digest"),
            ("provenance", "runtime_image_digest"),
        ),
        "vllm_version": (("vllm_version",), ("runtime", "vllm_version")),
        "code_revision_or_overlay_sha256": (
            ("code_revision",),
            ("code_overlay_sha256",),
            ("provenance", "code_revision"),
        ),
        "batch_size": (("batch_size",), ("engine", "batch_size")),
        "attention_backend": (("attention_backend",), ("engine", "attention_backend")),
        "dtype": (("dtype",), ("engine", "dtype")),
        "tensor_parallel_size": (
            ("tensor_parallel_size",),
            ("engine", "tensor_parallel_size"),
        ),
        "gpu_hardware_and_clock": (
            ("gpu_hardware_and_clock",),
            ("runtime", "gpu_hardware_and_clock"),
        ),
    }
    optional = {
        name: _value_report(
            {
                condition: _extract_optional_shared_field(item.summary, paths)
                for condition, item in data.items()
            }
        )
        for name, paths in optional_specs.items()
    }
    rank_metadata_complete = all(
        value["availability"] == "complete" and value["internally_consistent"]
        for value in rank_metadata.values()
    )
    rank_metadata_unavailable = all(
        value["availability"] == "unavailable" for value in rank_metadata.values()
    )
    resolved_spec_available = rank_metadata_complete and all(
        value["availability"] == "complete" for value in speculative_design.values()
    )
    resolved_spec_valid = resolved_spec_available and all(
        value["passed"] for value in speculative_design.values()
    )
    recorded_consistent = (
        all(recorded[field]["consistent"] for field in shared_fields)
        and all(value["consistent"] for value in prompt_contracts.values())
        and all(value["consistent"] for value in image_sequences.values())
        and (recorded["seed"]["consistent"] or recorded["seed"]["availability"] == "unavailable")
        and (resolved_spec_valid or not resolved_spec_available)
        and (rank_metadata_complete or rank_metadata_unavailable)
    )
    missing_critical = [name for name, value in optional.items() if value["availability"] != "complete"]
    if recorded["seed"]["availability"] != "complete":
        missing_critical.append("seed")
    if not resolved_spec_available:
        missing_critical.append("resolved_speculative_config")
    if not rank_metadata_complete:
        missing_critical.append("complete_rank_summary_provenance")
    fully_verified = recorded_consistent and not missing_critical and all(
        value["consistent"] for value in optional.values()
    )
    return {
        "recorded_shared_fields": recorded,
        "rank_metadata": rank_metadata,
        "resolved_speculative_design": speculative_design,
        "prompt_contracts": prompt_contracts,
        "image_sequences": image_sequences,
        "optional_runtime_and_model_identity": optional,
        "recorded_provenance_consistent": recorded_consistent,
        "full_provenance_verified": fully_verified,
        "critical_provenance_unavailable": missing_critical,
        "mtp_head_identity": {
            "availability": "unavailable",
            "reason": "condition summaries and rows do not record trained/original MTP tensor hashes",
        },
        "interpretation": (
            "Recorded input/config provenance is consistent, but a fully verified claim also requires "
            "the listed runtime/model identity fields."
            if recorded_consistent and not fully_verified
            else None
        ),
    }


def _pair_check(
    baseline: Mapping[int, RowRecord],
    candidate: Mapping[int, RowRecord],
    *,
    field: str,
    availability: str | None = None,
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(candidate))
    comparable = 0
    matching = 0
    for index in shared:
        left = baseline[index]
        right = candidate[index]
        if availability is not None:
            if not getattr(left, availability) or not getattr(right, availability):
                continue
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if availability is None and (left_value is None or right_value is None):
            continue
        comparable += 1
        matching += left_value == right_value
    expected = len(baseline)
    return {
        "expected_pairs": expected,
        "available_pairs": comparable,
        "matching_pairs": matching,
        "match_rate": matching / comparable if comparable else None,
        "complete_and_equal": comparable == expected and matching == expected,
    }


def _task_comparability(
    baseline: Mapping[int, RowRecord], candidate: Mapping[int, RowRecord]
) -> dict[str, Any]:
    checks = {
        "image_uri": _pair_check(baseline, candidate, field="image_uri"),
        "prompt_contract_sha256": _pair_check(
            baseline, candidate, field="prompt_contract_sha256"
        ),
        "prompt_tokens": _pair_check(baseline, candidate, field="prompt_tokens"),
        "max_new_tokens_budget": _pair_check(
            baseline, candidate, field="max_new_tokens_budget"
        ),
        "response_sha256": _pair_check(baseline, candidate, field="response_sha256"),
        "output_tokens": _pair_check(baseline, candidate, field="output_tokens"),
        "finish_reason": _pair_check(
            baseline,
            candidate,
            field="finish_reason",
            availability="finish_reason_recorded",
        ),
        "stop_reason": _pair_check(
            baseline,
            candidate,
            field="stop_reason",
            availability="stop_reason_recorded",
        ),
    }
    reasons = [name for name, check in checks.items() if not check["complete_and_equal"]]
    index_sets_equal = set(baseline) == set(candidate)
    if not index_sets_equal:
        reasons.insert(0, "input_index_set")
    return {
        "passed": index_sets_equal and not reasons,
        "reasons": reasons,
        "checks": checks,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    return float(numerator) / float(denominator) if denominator else None


def _speedup_values(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generation_wall_speedup": _safe_ratio(
            baseline.get("generation_wall_seconds"), candidate.get("generation_wall_seconds")
        ),
        "output_token_throughput_speedup": _safe_ratio(
            candidate.get("output_tokens_per_second"), baseline.get("output_tokens_per_second")
        ),
        "image_throughput_speedup": _safe_ratio(
            candidate.get("images_per_second"), baseline.get("images_per_second")
        ),
    }


def _timing_provenance_comparison(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = baseline_report.get("timing_provenance")
    candidate = candidate_report.get("timing_provenance")
    if baseline is None and candidate is None:
        return {
            "availability": "unavailable",
            "consistent": True,
            "measurement_class": "direct_condition_wall",
            "actual_source_job_overlap_verified": None,
        }
    consistent = (
        baseline is not None
        and candidate is not None
        and _canonical_json(baseline) == _canonical_json(candidate)
    )
    normalized = (
        consistent
        and isinstance(baseline, Mapping)
        and baseline.get("source_shards_assumed_parallel") is True
        and baseline.get("actual_source_job_overlap_verified") is False
    )
    return {
        "availability": "complete" if baseline is not None and candidate is not None else "partial",
        "consistent": consistent,
        "measurement_class": (
            "normalized_parallel_shard_makespan" if normalized else "recorded_condition_wall"
        ),
        "actual_source_job_overlap_verified": (
            baseline.get("actual_source_job_overlap_verified")
            if consistent and isinstance(baseline, Mapping)
            else None
        ),
        "baseline": baseline,
        "candidate": candidate,
    }


def _scope_comparison(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    recorded_provenance_consistent: bool,
    full_provenance_verified: bool,
    timing_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_raw = baseline_report["raw"]
    candidate_raw = candidate_report["raw"]
    observed = _speedup_values(baseline_raw, candidate_raw)
    like_for_like = bool(
        gate["passed"]
        and recorded_provenance_consistent
        and timing_provenance.get("consistent") is True
    )
    if not like_for_like:
        claim_status = "not_comparable"
    elif timing_provenance.get("measurement_class") == "normalized_parallel_shard_makespan":
        claim_status = "qualified_normalized_parallel_wall"
    elif full_provenance_verified:
        claim_status = "fully_verified"
    else:
        claim_status = "qualified_provenance_incomplete"
    return {
        "comparability_gate": gate,
        "baseline_raw": baseline_raw,
        "candidate_raw": candidate_raw,
        "baseline_acceptance": baseline_report["acceptance"],
        "candidate_acceptance": candidate_report["acceptance"],
        "baseline_latency": baseline_report["latency"],
        "candidate_latency": candidate_report["latency"],
        "observed_speedup_ratios": observed,
        "like_for_like_speedup_ratios": observed if like_for_like else None,
        "claim_status": claim_status,
        "timing_provenance": dict(timing_provenance),
    }


def _aggregate_task_gates(task_gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reasons = {
        task: list(gate["reasons"])
        for task, gate in task_gates.items()
        if not gate["passed"]
    }
    return {
        "passed": not reasons,
        "reasons_by_task": reasons,
        "tasks_passed": sum(gate["passed"] for gate in task_gates.values()),
        "tasks_expected": len(TASKS),
    }


def _comparison_report(
    baseline: ConditionData,
    candidate: ConditionData,
    condition_reports: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    task_gates = {
        task: _task_comparability(baseline.tasks[task], candidate.tasks[task])
        for task in TASKS
    }
    total_gate = _aggregate_task_gates(task_gates)
    recorded_consistent = bool(provenance["recorded_provenance_consistent"])
    fully_verified = bool(provenance["full_provenance_verified"])
    baseline_report = condition_reports[baseline.condition]
    candidate_report = condition_reports[candidate.condition]
    timing_provenance = _timing_provenance_comparison(baseline_report, candidate_report)
    return {
        "baseline_condition": baseline.condition,
        "candidate_condition": candidate.condition,
        "total": _scope_comparison(
            baseline_report["total"],
            candidate_report["total"],
            total_gate,
            recorded_provenance_consistent=recorded_consistent,
            full_provenance_verified=fully_verified,
            timing_provenance=timing_provenance,
        ),
        "tasks": {
            task: _scope_comparison(
                baseline_report["tasks"][task],
                candidate_report["tasks"][task],
                task_gates[task],
                recorded_provenance_consistent=recorded_consistent,
                full_provenance_verified=fully_verified,
                timing_provenance=timing_provenance,
            )
            for task in TASKS
        },
    }


def build_report(
    input_root: str | Path,
    *,
    expected_images: int = DEFAULT_EXPECTED_IMAGES,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    if expected_images <= 0:
        raise FinalizationError("expected_images must be positive")
    source, data = load_conditions(
        input_root,
        expected_images=expected_images,
        s3_client=s3_client,
    )
    condition_reports = {
        condition: _condition_report(item) for condition, item in data.items()
    }
    provenance = _provenance_report(data)
    comparisons = {
        f"{candidate}_vs_no_mtp": _comparison_report(
            data["no_mtp"],
            data[candidate],
            condition_reports,
            provenance,
        )
        for candidate in ("trained_mtp", "original_mtp")
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "generated_at": utc_now(),
        "source_root": source.display_root,
        "completion_gate": {
            "required_conditions": list(CONDITIONS),
            "completed_conditions": list(CONDITIONS),
            "passed": True,
        },
        "expected_images_per_task": expected_images,
        "tasks": list(TASKS),
        "provenance": provenance,
        "conditions": condition_reports,
        "comparisons": comparisons,
        "timing_scope_note": (
            "generation_wall_seconds wraps llm.generate. The separately recorded explicit image decode "
            "and prompt-budget multimodal preprocessing are not included in headline speedup ratios."
        ),
    }


def _serialise_report(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def write_report_atomic(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FinalizationError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def upload_report(payload: bytes, uri: str, *, s3_client: Any | None = None) -> None:
    bucket, key = _parse_s3_uri(uri)
    if not key:
        raise FinalizationError("--output-s3 must name a JSON object, not a bucket root")
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )


def finalize_to_file(
    input_root: str | Path,
    output: Path,
    *,
    expected_images: int = DEFAULT_EXPECTED_IMAGES,
    overwrite: bool = False,
    output_s3: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    # Build and serialise before either local or remote output becomes visible.
    report = build_report(
        input_root,
        expected_images=expected_images,
        s3_client=s3_client,
    )
    payload = _serialise_report(report)
    write_report_atomic(output, payload, overwrite=overwrite)
    if output_s3:
        upload_report(payload, output_s3, s3_client=s3_client)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Local directory or shared S3 evaluation root")
    parser.add_argument("--output", type=Path, required=True, help="Local final JSON report")
    parser.add_argument("--output-s3", help="Optional exact S3 object URI for the same final JSON")
    parser.add_argument("--expected-images", type=int, default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = finalize_to_file(
            args.input_root,
            args.output,
            expected_images=args.expected_images,
            overwrite=args.overwrite,
            output_s3=args.output_s3,
        )
    except FinalizationError as exc:
        print(f"finalization failed: {exc}", file=os.sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "output_s3": args.output_s3,
                "conditions": list(CONDITIONS),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
