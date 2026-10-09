#!/usr/bin/env python3
"""Evaluate complete inline image-edit JSON predictions against ground truth.

Visible layout regions are split by their predicted/ground-truth operation:

* ``general`` is non-edited;
* ``add/crop/modify/outpaint/replace/transform`` are edited;
* Source-only ``remove`` records have no Target bbox or parent and are reported
  separately instead of being mixed into visible-region metrics.

Within each split, regions are associated with a maximum-total-IoU one-to-one
assignment.  An assigned pair is a region true positive only when its Target
bbox IoU reaches ``--iou-threshold`` (0.5 by default).  Parent correctness is
label-invariant: a non-root parent is correct when the predicted parent is the
region paired with the ground-truth parent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_full_json import (  # noqa: E402
    VISIBLE_OPERATIONS,
    InlineTargetValidationError,
    validate_inline_target,
)


EDITED_OPERATIONS = tuple(sorted(VISIBLE_OPERATIONS - {"general"}))
GROUND_TRUTH_SUFFIX = ".target_json.with_inline_edit.json"
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_LABEL_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class DocumentStatus:
    value: dict[str, Any] | None
    strict_json_valid: bool
    json_valid: bool
    schema_valid: bool
    strict_json_error: str | None = None
    json_error: str | None = None
    schema_error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "strict_json_valid": self.strict_json_valid,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "strict_json_error": self.strict_json_error,
            "json_error": self.json_error,
            "schema_error": self.schema_error,
        }


@dataclass(frozen=True)
class Region:
    index: int
    label: str
    parent: str
    region_type: str
    bbox: tuple[int, int, int, int]
    operation: str

    @property
    def edited(self) -> bool:
        return self.operation != "general"


@dataclass(frozen=True)
class RegionMatch:
    prediction: Region
    ground_truth: Region
    iou: float


@dataclass(frozen=True)
class PartitionResult:
    predictions: tuple[Region, ...]
    ground_truth: tuple[Region, ...]
    assignments: tuple[RegionMatch, ...]
    true_positives: tuple[RegionMatch, ...]


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant: {token}")


def _load_json_object(text: str) -> dict[str, Any]:
    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    return value


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _strip_response_wrappers(text: str) -> str:
    text = _THINK_RE.sub("", text, count=1)
    text = _FENCE_START_RE.sub("", text, count=1)
    return _FENCE_END_RE.sub("", text, count=1).strip()


def inspect_document(value: Any, *, allow_response_wrappers: bool) -> DocumentStatus:
    """Parse one object and independently apply the exact training schema."""

    strict_value: dict[str, Any] | None = None
    strict_error: str | None = None
    if isinstance(value, Mapping):
        try:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
            strict_value = _load_json_object(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            strict_error = _error(exc)
    elif isinstance(value, str):
        try:
            strict_value = _load_json_object(value.strip())
        except (ValueError, json.JSONDecodeError) as exc:
            strict_error = _error(exc)
    else:
        strict_error = f"TypeError: expected a JSON object or string, got {type(value).__name__}"

    parsed = strict_value
    json_error = strict_error
    if parsed is None and allow_response_wrappers and isinstance(value, str):
        try:
            parsed = _load_json_object(_strip_response_wrappers(value))
            json_error = None
        except (ValueError, json.JSONDecodeError) as exc:
            json_error = _error(exc)

    schema_error = None
    if parsed is not None:
        try:
            validate_inline_target(parsed)
        except InlineTargetValidationError as exc:
            schema_error = _error(exc)
    return DocumentStatus(
        value=parsed,
        strict_json_valid=strict_value is not None,
        json_valid=parsed is not None,
        schema_valid=parsed is not None and schema_error is None,
        strict_json_error=strict_error,
        json_error=json_error,
        schema_error=schema_error,
    )


def _sample_id(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("sample_id", row.get("id"))
    if value is None and isinstance(row.get("meta"), Mapping):
        value = row["meta"].get("sample_id")
    if value is None:
        value = row.get("input_index")
    if value is None or not str(value).strip():
        raise ValueError(f"{context}: missing sample_id/id/input_index")
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _load_json_object(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {_error(exc)}") from exc
            rows.append(value)
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if isinstance(value, list):
        if not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{path}: JSON array rows must be objects")
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"{path}: expected a JSON object, array, or JSONL file")


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_rows(path)
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        sample_id = _sample_id(row, context=f"{path}:{index}")
        if sample_id in result:
            raise ValueError(f"{path}: duplicate prediction sample_id {sample_id!r}")
        result[sample_id] = row
    return result


def _ground_truth_payload(row: Mapping[str, Any]) -> Any:
    for key in ("target_json", "ground_truth", "groundtruth", "label"):
        if key in row:
            return row[key]
    if set(row) == {"structured_description"}:
        return row
    raise ValueError("ground-truth row has no target_json/ground_truth/groundtruth/label field")


def _insert_ground_truth(
    result: dict[str, Any], sample_id: str, value: Any, *, context: str
) -> None:
    if sample_id in result:
        raise ValueError(f"duplicate ground-truth sample_id {sample_id!r}: {context}")
    result[sample_id] = value


def _load_ground_truth_manifest(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    entries = value.get("groundtruth")
    if not isinstance(entries, Mapping):
        raise ValueError(f"{path}: manifest groundtruth must be an object")
    result: dict[str, Any] = {}
    for sample_id, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("filename"), str):
            raise ValueError(f"{path}: invalid groundtruth entry for {sample_id!r}")
        target_path = (path.parent / raw_entry["filename"]).resolve(strict=True)
        target = json.loads(target_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        _insert_ground_truth(result, str(sample_id), target, context=str(target_path))
    return result


def load_ground_truth(path: Path) -> dict[str, Any]:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
            )
            if isinstance(manifest, Mapping) and "groundtruth" in manifest:
                return _load_ground_truth_manifest(manifest_path, manifest)
        files = sorted(path.rglob(f"*{GROUND_TRUTH_SUFFIX}"))
        if not files:
            raise ValueError(f"{path}: no *{GROUND_TRUTH_SUFFIX} files found")
        result: dict[str, Any] = {}
        for target_path in files:
            sample_id = target_path.name[: -len(GROUND_TRUTH_SUFFIX)]
            value = json.loads(target_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            _insert_ground_truth(result, sample_id, value, context=str(target_path))
        return result

    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if isinstance(value, Mapping) and "groundtruth" in value:
            return _load_ground_truth_manifest(path, value)
    result = {}
    for index, row in enumerate(_read_rows(path), start=1):
        sample_id = _sample_id(row, context=f"{path}:{index}")
        _insert_ground_truth(
            result,
            sample_id,
            _ground_truth_payload(row),
            context=f"{path}:{index}",
        )
    return result


def inspect_prediction_row(row: Mapping[str, Any]) -> DocumentStatus:
    response = row.get("response")
    if isinstance(response, str):
        return inspect_document(response, allow_response_wrappers=True)
    return inspect_document(row.get("prediction"), allow_response_wrappers=True)


def _regions(value: Mapping[str, Any]) -> tuple[Region, ...]:
    layout = value["structured_description"]["elements"]["layout"]
    return tuple(
        Region(
            index=index,
            label=raw["label"],
            parent=raw["parent"],
            region_type=raw["type"],
            bbox=tuple(raw["bbox"]),
            operation=raw["edit"]["operation"],
        )
        for index, raw in enumerate(layout)
    )


def _remove_count(value: Mapping[str, Any]) -> int:
    return len(value["structured_description"]["elements"]["remove"])


def bbox_iou(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    """Compute IoU for normalized ``[y0, x0, y1, x1]`` boxes."""

    ay0, ax0, ay1, ax1 = (float(item) for item in first)
    by0, bx0, by1, bx1 = (float(item) for item in second)
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection = height * width
    first_area = (ay1 - ay0) * (ax1 - ax0)
    second_area = (by1 - by0) * (bx1 - bx0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _label_tokens(label: str) -> frozenset[str]:
    tokens = _LABEL_TOKEN_RE.findall(label.casefold())
    if tokens and tokens[-1].isdigit():
        tokens.pop()
    return frozenset(tokens)


def _tie_break_score(prediction: Region, ground_truth: Region) -> float:
    prediction_tokens = _label_tokens(prediction.label)
    ground_truth_tokens = _label_tokens(ground_truth.label)
    union = prediction_tokens | ground_truth_tokens
    label_similarity = len(prediction_tokens & ground_truth_tokens) / len(union) if union else 0.0
    return label_similarity * 1e-7 + float(prediction.region_type == ground_truth.region_type) * 1e-8


def _minimum_cost_assignment(costs: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Hungarian assignment for a rectangular matrix with rows <= columns."""

    row_count = len(costs)
    column_count = len(costs[0]) if row_count else 0
    if row_count == 0:
        return []
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("Hungarian cost matrix must be rectangular with rows <= columns")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        column = 0
        min_value = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                current = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if current < min_value[candidate]:
                    min_value[candidate] = current
                    predecessor[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    return sorted((row - 1, column - 1) for column, row in enumerate(matched_row[1:], start=1) if row)


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment weight matrix must be rectangular")
    if row_count <= column_count:
        maximum = max(max(row) for row in weights)
        costs = [[maximum - value for value in row] for row in weights]
        return _minimum_cost_assignment(costs)
    transposed = [[weights[row][column] for row in range(row_count)] for column in range(column_count)]
    maximum = max(max(row) for row in transposed)
    costs = [[maximum - value for value in row] for row in transposed]
    return sorted((column, row) for row, column in _minimum_cost_assignment(costs))


def match_regions(
    predictions: Sequence[Region],
    ground_truth: Sequence[Region],
    *,
    iou_threshold: float,
) -> PartitionResult:
    weights = [
        [
            bbox_iou(prediction.bbox, target.bbox) + _tie_break_score(prediction, target)
            for target in ground_truth
        ]
        for prediction in predictions
    ]
    assignments = tuple(
        RegionMatch(predictions[prediction_index], ground_truth[target_index], bbox_iou(
            predictions[prediction_index].bbox, ground_truth[target_index].bbox
        ))
        for prediction_index, target_index in _maximum_weight_assignment(weights)
    )
    true_positives = tuple(item for item in assignments if item.iou >= iou_threshold)
    return PartitionResult(tuple(predictions), tuple(ground_truth), assignments, true_positives)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def partition_report(results: Sequence[PartitionResult], *, iou_threshold: float) -> dict[str, Any]:
    predicted = sum(len(result.predictions) for result in results)
    ground_truth = sum(len(result.ground_truth) for result in results)
    true_positive = sum(len(result.true_positives) for result in results)
    assignments = [item for result in results for item in result.assignments]
    true_positive_matches = [item for result in results for item in result.true_positives]
    precision = true_positive / predicted if predicted else float(ground_truth == 0)
    recall = true_positive / ground_truth if ground_truth else float(predicted == 0)
    signed_errors = [len(result.predictions) - len(result.ground_truth) for result in results]
    iou_at_0_5 = sum(item.iou >= 0.5 for item in assignments)
    return {
        "samples": len(results),
        "counts": {
            "predicted": predicted,
            "ground_truth": ground_truth,
            "signed_error": predicted - ground_truth,
            "mean_signed_error_per_sample": _mean([float(value) for value in signed_errors]),
            "mean_absolute_error_per_sample": _mean([float(abs(value)) for value in signed_errors]),
            "exact_count_samples": sum(value == 0 for value in signed_errors),
            "exact_count_rate": _ratio(sum(value == 0 for value in signed_errors), len(results)),
        },
        "matching": {
            "assigned_pairs": len(assignments),
            "true_positive_pairs": true_positive,
            "false_positive_regions": predicted - true_positive,
            "false_negative_regions": ground_truth - true_positive,
            "iou_threshold": iou_threshold,
        },
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "bbox": {
            "mean_iou_assigned_pairs": _mean([item.iou for item in assignments]),
            "mean_iou_true_positive_pairs": _mean([item.iou for item in true_positive_matches]),
            "iou_at_0_5_count": iou_at_0_5,
            "iou_at_0_5_rate": _ratio(iou_at_0_5, len(assignments)),
        },
    }


def _match_public(item: RegionMatch, *, iou_threshold: float) -> dict[str, Any]:
    return {
        "prediction_index": item.prediction.index,
        "prediction_label": item.prediction.label,
        "prediction_operation": item.prediction.operation,
        "ground_truth_index": item.ground_truth.index,
        "ground_truth_label": item.ground_truth.label,
        "ground_truth_operation": item.ground_truth.operation,
        "iou": item.iou,
        "true_positive": item.iou >= iou_threshold,
    }


def operation_report(results: Sequence[PartitionResult]) -> dict[str, Any]:
    counts = {
        operation: {"true_positive": 0, "false_positive": 0, "false_negative": 0, "support": 0}
        for operation in EDITED_OPERATIONS
    }
    confusion = {
        expected: {predicted: 0 for predicted in EDITED_OPERATIONS}
        for expected in EDITED_OPERATIONS
    }
    correct = 0
    matched = 0
    total_predictions = 0
    total_ground_truth = 0
    for result in results:
        matched_prediction = {item.prediction.index for item in result.true_positives}
        matched_ground_truth = {item.ground_truth.index for item in result.true_positives}
        total_predictions += len(result.predictions)
        total_ground_truth += len(result.ground_truth)
        for item in result.true_positives:
            predicted = item.prediction.operation
            expected = item.ground_truth.operation
            matched += 1
            counts[expected]["support"] += 1
            confusion[expected][predicted] += 1
            if predicted == expected:
                correct += 1
                counts[expected]["true_positive"] += 1
            else:
                counts[predicted]["false_positive"] += 1
                counts[expected]["false_negative"] += 1
        for prediction in result.predictions:
            if prediction.index not in matched_prediction:
                counts[prediction.operation]["false_positive"] += 1
        for target in result.ground_truth:
            if target.index not in matched_ground_truth:
                counts[target.operation]["support"] += 1
                counts[target.operation]["false_negative"] += 1

    per_class: dict[str, Any] = {}
    present_f1: list[float] = []
    all_f1: list[float] = []
    for operation in EDITED_OPERATIONS:
        item = counts[operation]
        precision = _ratio(item["true_positive"], item["true_positive"] + item["false_positive"])
        recall = _ratio(item["true_positive"], item["true_positive"] + item["false_negative"])
        f1 = _f1(precision, recall)
        zero_filled_f1 = f1 if f1 is not None else 0.0
        all_f1.append(zero_filled_f1)
        if item["support"]:
            present_f1.append(zero_filled_f1)
        per_class[operation] = {**item, "precision": precision, "recall": recall, "f1": f1}
    return {
        "scope": "visible edited layout regions; Source-only remove records are excluded",
        "spatially_matched_regions": matched,
        "predicted_edited_regions": total_predictions,
        "ground_truth_edited_regions": total_ground_truth,
        "matched_region_coverage": _ratio(matched, total_ground_truth),
        "accuracy_on_spatially_matched_regions": _ratio(correct, matched),
        "correct_operation_and_region_rate": _ratio(correct, total_ground_truth),
        "macro_f1": _mean(present_f1),
        "macro_f1_all_configured_classes": _mean(all_f1),
        "classes_in_macro_f1": [
            operation for operation in EDITED_OPERATIONS if counts[operation]["support"]
        ],
        "per_class": per_class,
        "confusion_on_spatially_matched_regions": confusion,
    }


def hierarchy_report(
    partition_results: Sequence[PartitionResult],
    all_results: Sequence[tuple[Sequence[Region], Sequence[Region], dict[int, int]]],
) -> dict[str, Any]:
    total = 0
    correct = 0
    root_total = 0
    root_correct = 0
    non_root_total = 0
    non_root_correct = 0
    for result, (predictions, ground_truth, target_to_prediction) in zip(
        partition_results, all_results, strict=True
    ):
        prediction_by_label = {region.label: region.index for region in predictions}
        ground_truth_by_label = {region.label: region.index for region in ground_truth}
        for item in result.true_positives:
            total += 1
            expected_parent = item.ground_truth.parent
            predicted_parent = item.prediction.parent
            if expected_parent == "ROOT":
                root_total += 1
                item_correct = predicted_parent == "ROOT"
                root_correct += int(item_correct)
            else:
                non_root_total += 1
                expected_parent_index = ground_truth_by_label[expected_parent]
                mapped_parent_index = target_to_prediction.get(expected_parent_index)
                actual_parent_index = prediction_by_label.get(predicted_parent)
                item_correct = mapped_parent_index is not None and actual_parent_index == mapped_parent_index
                non_root_correct += int(item_correct)
            correct += int(item_correct)
    return {
        "matched_regions": total,
        "correct": correct,
        "parent_match_rate": _ratio(correct, total),
        "root_parent": {
            "regions": root_total,
            "correct": root_correct,
            "match_rate": _ratio(root_correct, root_total),
        },
        "non_root_parent": {
            "regions": non_root_total,
            "correct": non_root_correct,
            "match_rate": _ratio(non_root_correct, non_root_total),
        },
    }


def _validity_report(statuses: Iterable[DocumentStatus]) -> dict[str, Any]:
    values = list(statuses)
    total = len(values)
    strict_valid = sum(item.strict_json_valid for item in values)
    json_valid = sum(item.json_valid for item in values)
    schema_valid = sum(item.schema_valid for item in values)
    schema_errors = Counter(
        item.schema_error.split(":", 1)[0] for item in values if item.schema_error
    )
    json_errors = Counter(item.json_error.split(":", 1)[0] for item in values if item.json_error)
    return {
        "total": total,
        "strict_json_valid": strict_valid,
        "strict_json_valid_rate": _ratio(strict_valid, total),
        "json_valid": json_valid,
        "json_valid_rate": _ratio(json_valid, total),
        "schema_valid": schema_valid,
        "schema_valid_rate": _ratio(schema_valid, total),
        "schema_valid_given_json_rate": _ratio(schema_valid, json_valid),
        "json_error_types": dict(sorted(json_errors.items())),
        "schema_error_types": dict(sorted(schema_errors.items())),
    }


def _limited_ids(values: set[str], limit: int = 100) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(ordered), "sample_ids": ordered[:limit], "omitted": max(0, len(ordered) - limit)}


def evaluate(
    predictions: Mapping[str, Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    prediction_status = {
        sample_id: inspect_prediction_row(row) for sample_id, row in predictions.items()
    }
    ground_truth_status = {
        sample_id: inspect_document(value, allow_response_wrappers=False)
        for sample_id, value in ground_truth.items()
    }
    common_ids = sorted(set(predictions) & set(ground_truth))
    edited_results: list[PartitionResult] = []
    non_edited_results: list[PartitionResult] = []
    hierarchy_inputs: list[tuple[Sequence[Region], Sequence[Region], dict[int, int]]] = []
    details: list[dict[str, Any]] = []
    remove_predicted = 0
    remove_ground_truth = 0
    remove_errors: list[int] = []

    for sample_id in common_ids:
        prediction_inspection = prediction_status[sample_id]
        target_inspection = ground_truth_status[sample_id]
        detail: dict[str, Any] = {
            "sample_id": sample_id,
            "prediction_validity": prediction_inspection.public(),
            "ground_truth_validity": target_inspection.public(),
            "semantic_evaluated": target_inspection.schema_valid,
        }
        if not target_inspection.schema_valid or target_inspection.value is None:
            details.append(detail)
            continue

        target_regions = _regions(target_inspection.value)
        prediction_regions = (
            _regions(prediction_inspection.value)
            if prediction_inspection.schema_valid and prediction_inspection.value is not None
            else ()
        )
        target_remove_count = _remove_count(target_inspection.value)
        prediction_remove_count = (
            _remove_count(prediction_inspection.value)
            if prediction_inspection.schema_valid and prediction_inspection.value is not None
            else 0
        )
        remove_predicted += prediction_remove_count
        remove_ground_truth += target_remove_count
        remove_errors.append(prediction_remove_count - target_remove_count)

        edited = match_regions(
            [region for region in prediction_regions if region.edited],
            [region for region in target_regions if region.edited],
            iou_threshold=iou_threshold,
        )
        non_edited = match_regions(
            [region for region in prediction_regions if not region.edited],
            [region for region in target_regions if not region.edited],
            iou_threshold=iou_threshold,
        )
        edited_results.append(edited)
        non_edited_results.append(non_edited)
        target_to_prediction = {
            item.ground_truth.index: item.prediction.index
            for result in (edited, non_edited)
            for item in result.true_positives
        }
        hierarchy_inputs.append((prediction_regions, target_regions, target_to_prediction))

        sample_hierarchy_input = [(prediction_regions, target_regions, target_to_prediction)]
        detail.update(
            {
                "regions": {
                    "edited": {
                        **partition_report([edited], iou_threshold=iou_threshold),
                        "assignments": [
                            _match_public(item, iou_threshold=iou_threshold)
                            for item in edited.assignments
                        ],
                    },
                    "non_edited": {
                        **partition_report([non_edited], iou_threshold=iou_threshold),
                        "assignments": [
                            _match_public(item, iou_threshold=iou_threshold)
                            for item in non_edited.assignments
                        ],
                    },
                },
                "operations": operation_report([edited]),
                "hierarchy": {
                    "edited": hierarchy_report([edited], sample_hierarchy_input),
                    "non_edited": hierarchy_report([non_edited], sample_hierarchy_input),
                },
                "remove": {
                    "predicted": prediction_remove_count,
                    "ground_truth": target_remove_count,
                    "signed_count_error": prediction_remove_count - target_remove_count,
                },
            }
        )
        details.append(detail)

    summary = {
        "schema_version": 1,
        "matching_contract": {
            "region_assignment": "maximum-total Target-bbox IoU, one-to-one within edited status",
            "region_true_positive_iou_threshold": iou_threshold,
            "iou_at_0_5_threshold": 0.5,
            "edited_operations": list(EDITED_OPERATIONS),
            "non_edited_operation": "general",
            "remove_policy": (
                "Source-only remove has no Target bbox/parent; report count separately and exclude "
                "from visible-region, operation, and hierarchy metrics"
            ),
            "invalid_prediction_policy": "schema-invalid prediction contributes zero visible regions",
            "parent_policy": "compare parent through qualifying region correspondence; ROOT is explicit",
            "operation_policy": (
                "accuracy is conditional on spatially matched edited regions; macro-F1 also counts "
                "unmatched predicted/ground-truth edited regions as per-class FP/FN"
            ),
        },
        "sample_alignment": {
            "prediction_rows": len(predictions),
            "ground_truth_rows": len(ground_truth),
            "common_rows": len(common_ids),
            "prediction_without_ground_truth": _limited_ids(set(predictions) - set(ground_truth)),
            "ground_truth_without_prediction": _limited_ids(set(ground_truth) - set(predictions)),
            "semantic_evaluated_rows": len(edited_results),
        },
        "validity": {
            "predictions": _validity_report(prediction_status.values()),
            "ground_truth": _validity_report(ground_truth_status.values()),
        },
        "regions": {
            "edited": partition_report(edited_results, iou_threshold=iou_threshold),
            "non_edited": partition_report(non_edited_results, iou_threshold=iou_threshold),
        },
        "operations": operation_report(edited_results),
        "hierarchy": {
            "edited": hierarchy_report(edited_results, hierarchy_inputs),
            "non_edited": hierarchy_report(non_edited_results, hierarchy_inputs),
        },
        "remove": {
            "samples": len(remove_errors),
            "predicted": remove_predicted,
            "ground_truth": remove_ground_truth,
            "signed_count_error": remove_predicted - remove_ground_truth,
            "mean_signed_error_per_sample": _mean([float(value) for value in remove_errors]),
            "mean_absolute_error_per_sample": _mean([float(abs(value)) for value in remove_errors]),
        },
    }
    return summary, details


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Inference JSONL/JSON file")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help=(
            "Prepared ground-truth JSONL, a viewer_groundtruth directory/manifest, or a directory "
            f"containing *{GROUND_TRUTH_SUFFIX}"
        ),
    )
    parser.add_argument("--output", type=Path, help="Summary report JSON path")
    parser.add_argument("--details-output", type=Path, help="Optional per-sample details JSONL path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--strict-sample-ids",
        action="store_true",
        help="Fail unless prediction and ground-truth sample-id sets are identical",
    )
    args = parser.parse_args(argv)
    if not args.predictions.is_file():
        parser.error(f"prediction file does not exist: {args.predictions}")
    if not args.ground_truth.exists():
        parser.error(f"ground-truth path does not exist: {args.ground_truth}")
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    if args.strict_sample_ids and set(predictions) != set(ground_truth):
        missing_ground_truth = sorted(set(predictions) - set(ground_truth))
        missing_predictions = sorted(set(ground_truth) - set(predictions))
        raise ValueError(
            "prediction/ground-truth sample IDs differ: "
            f"prediction_without_ground_truth={missing_ground_truth[:10]}, "
            f"ground_truth_without_prediction={missing_predictions[:10]}"
        )
    summary, details = evaluate(predictions, ground_truth, iou_threshold=args.iou_threshold)
    if args.output:
        _write_json(args.output, summary)
    if args.details_output:
        _write_jsonl(args.details_output, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
