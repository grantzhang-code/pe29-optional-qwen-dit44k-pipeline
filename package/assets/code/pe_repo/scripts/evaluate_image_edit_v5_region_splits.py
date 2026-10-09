#!/usr/bin/env python3
"""Evaluate native image-edit-v5 edited and non-edited regions on hard1000.

V5 intentionally has no operation field.  Its native instruction contract
marks unchanged regions explicitly, so this evaluator uses a frozen,
versioned classifier for the edited/non-edited split and does not manufacture
the seven-operation benchmark ontology.  Target bboxes use the hard1000
cardinality-first matching rule at the requested IoU 0.50 operating point.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v5_full_json import (  # noqa: E402
    V5TargetValidationError,
    validate_v5_target,
)
from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    V6TargetValidationError,
    validate_v6_target,
)
from scripts.evaluate_image_edit_full_json import (  # noqa: E402
    _maximum_weight_assignment,
    bbox_iou as _strict_bbox_iou,
)
from scripts.image_edit_instruction_qwen_judge import (  # noqa: E402
    CachedQwenInstructionJudge,
    QWEN_JUDGE_MODEL_NAME,
    QWEN_JUDGE_MODEL_REVISION,
    QWEN_JUDGE_VERSION,
    QWEN_PROMPT_VERSION,
)


EVALUATOR_VERSION = "image-edit-v5-hard1000-region-splits-v4-qwen-score"
V6_CROSS_SCHEMA_EVALUATOR_VERSION = "image-edit-v6-on-v5-hard1000-region-splits-v1"
V6_GROUND_TRUTH_EVALUATOR_VERSION = "image-edit-v6-on-v7r1-hard1000-region-splits-v1"
V7R1_OBSERVED_YXYX_EVALUATOR_VERSION = (
    "image-edit-v6-on-v7r1-hard1000-observed-yxyx-region-splits-v3"
)
V7_SERIES_OBSERVED_YXYX_EVALUATOR_VERSION = (
    "image-edit-v6-on-v7-series-hard1000-observed-yxyx-region-splits-v1"
)
V5_SCHEMA = "image_edit_v5_instruction"
V6_SCHEMA = "image_edit_v6_direct_root_xyxy"
V7R1_GT_SCHEMA = "image_edit_v7r1_direct_root_observed_yxyx"
V7_SERIES_GT_SCHEMA = "image_edit_v7_series_direct_root_observed_yxyx"
OBSERVED_YXYX_GT_SCHEMAS = (V7R1_GT_SCHEMA, V7_SERIES_GT_SCHEMA)
DOCUMENT_SCHEMAS = (V5_SCHEMA, V6_SCHEMA, *OBSERVED_YXYX_GT_SCHEMAS)
PREDICTION_SCHEMAS = (V5_SCHEMA, V6_SCHEMA)
GROUND_TRUTH_SCHEMAS = DOCUMENT_SCHEMAS
PREDICTION_VIEWS = ("raw", "postprocessed")
CLASSIFIER_VERSION = "operation-free-explicit-unchanged-v2"
IOU_THRESHOLDS = (0.50,)
DEFAULT_BOOTSTRAP_SEED = 20260805
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_UNCHANGED_PATTERNS = (
    re.compile(r"^\s*keep\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:keep|preserve|retain|leave|maintain)\b.*\bunchanged\b", re.IGNORECASE),
    re.compile(r"^\s*.+\b(?:remain|remains|stay|stays)\b.*\bunchanged\b", re.IGNORECASE),
    re.compile(r"^\s*(?:do not|don't)\s+(?:change|modify|alter)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class DocumentStatus:
    value: dict[str, Any] | None
    strict_json_valid: bool
    json_valid: bool
    schema_valid: bool
    evaluation_valid: bool
    strict_json_error: str | None = None
    json_error: str | None = None
    schema_error: str | None = None
    evaluation_error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "strict_json_valid": self.strict_json_valid,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "evaluation_valid": self.evaluation_valid,
            "strict_json_error": self.strict_json_error,
            "json_error": self.json_error,
            "schema_error": self.schema_error,
            "evaluation_error": self.evaluation_error,
        }


@dataclass(frozen=True)
class V5Region:
    index: int
    label: str
    parent: str
    region_type: str
    bbox: tuple[float, float, float, float]
    instruction: str
    references: tuple[tuple[float, float, float, float], ...]
    edited: bool
    classification_reason: str


@dataclass(frozen=True)
class RegionPair:
    prediction: V5Region
    ground_truth: V5Region
    iou: float


@dataclass(frozen=True)
class SamplePartition:
    predictions: tuple[V5Region, ...]
    ground_truth: tuple[V5Region, ...]
    all_assignment: tuple[RegionPair, ...]
    threshold_matches: Mapping[float, tuple[RegionPair, ...]]


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _strip_wrappers(text: str) -> str:
    text = _THINK_RE.sub("", text, count=1)
    text = _FENCE_START_RE.sub("", text, count=1)
    return _FENCE_END_RE.sub("", text, count=1).strip()


def _decode_object(value: Any, *, allow_wrappers: bool) -> tuple[dict[str, Any] | None, bool, str | None]:
    if isinstance(value, Mapping):
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            parsed = json.loads(encoded)
            return parsed, True, None
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, False, _error(exc)
    if not isinstance(value, str):
        return None, False, f"TypeError: expected object or JSON string, got {type(value).__name__}"
    try:
        parsed = json.loads(value.strip(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        if not isinstance(parsed, dict):
            raise ValueError("JSON value must be an object")
        return parsed, True, None
    except (ValueError, json.JSONDecodeError) as strict_exc:
        if not allow_wrappers:
            return None, False, _error(strict_exc)
        try:
            parsed = json.loads(
                _strip_wrappers(value),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            if not isinstance(parsed, dict):
                raise ValueError("JSON value must be an object")
            return parsed, False, None
        except (ValueError, json.JSONDecodeError) as exc:
            return None, False, _error(exc)


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return False
    y0, x0, y1, x1 = (float(item) for item in value)
    return 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000


def bbox_iou(first: Any, second: Any) -> float:
    """Score malformed native boxes as zero while retaining their denominator."""

    if not _valid_bbox(first) or not _valid_bbox(second):
        return 0.0
    return _strict_bbox_iou(first, second)


def _raw_bbox(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _evaluation_structure_errors(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    structured = value.get("structured_description")
    if not isinstance(structured, Mapping):
        return ["structured_description must be an object"]
    elements = structured.get("elements")
    if not isinstance(elements, Mapping):
        return ["structured_description.elements must be an object"]
    layout = elements.get("layout")
    remove = elements.get("remove")
    if not isinstance(layout, list):
        errors.append("elements.layout must be a list")
        layout = []
    if not isinstance(remove, list):
        errors.append("elements.remove must be a list")
        remove = []
    labels: set[str] = set()
    for index, raw in enumerate(layout):
        prefix = f"elements.layout[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("label", "parent", "type"):
            if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
                errors.append(f"{prefix}.{field} must be non-empty text")
        label = raw.get("label")
        if isinstance(label, str):
            if label in labels:
                errors.append(f"{prefix}.label is duplicated")
            labels.add(label)
        if "bbox" not in raw:
            errors.append(f"{prefix}.bbox is missing")
        edit = raw.get("edit")
        if not isinstance(edit, Mapping):
            errors.append(f"{prefix}.edit must be an object")
            continue
        if not isinstance(edit.get("instruction"), str) or not str(edit["instruction"]).strip():
            errors.append(f"{prefix}.edit.instruction must be non-empty text")
        references = edit.get("references")
        if not isinstance(references, list):
            errors.append(f"{prefix}.edit.references must be a list")
            continue
        for ref_index, reference in enumerate(references):
            if not isinstance(reference, Mapping) or "reference_bbox" not in reference:
                errors.append(f"{prefix}.edit.references[{ref_index}] is missing reference_bbox")
    for index, raw in enumerate(remove):
        prefix = f"elements.remove[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        references = raw.get("references")
        if not isinstance(raw.get("instruction"), str) or not str(raw["instruction"]).strip():
            errors.append(f"{prefix}.instruction must be non-empty text")
        if not isinstance(references, list):
            errors.append(f"{prefix}.references must be a list")
            continue
        for ref_index, reference in enumerate(references):
            if not isinstance(reference, Mapping) or "reference_bbox" not in reference:
                errors.append(f"{prefix}.references[{ref_index}] is missing reference_bbox")
    return errors


def inspect_document(
    value: Any,
    *,
    allow_wrappers: bool,
    schema: str = V5_SCHEMA,
) -> DocumentStatus:
    if schema not in DOCUMENT_SCHEMAS:
        raise ValueError(f"unsupported document schema: {schema!r}")
    parsed, strict_valid, json_error = _decode_object(value, allow_wrappers=allow_wrappers)
    strict_error = None if strict_valid else json_error
    schema_error = None
    evaluation_error = None
    if parsed is not None:
        try:
            if schema == V5_SCHEMA:
                validate_v5_target(parsed)
            else:
                validate_v6_target(parsed)
        except (V5TargetValidationError, V6TargetValidationError) as exc:
            schema_error = _error(exc)
        structural = _evaluation_structure_errors(parsed) if schema == V5_SCHEMA else []
        if structural:
            evaluation_error = "; ".join(structural[:20])
    return DocumentStatus(
        value=parsed,
        strict_json_valid=strict_valid,
        json_valid=parsed is not None,
        schema_valid=parsed is not None and schema_error is None,
        evaluation_valid=parsed is not None and evaluation_error is None,
        strict_json_error=strict_error,
        json_error=json_error,
        schema_error=schema_error,
        evaluation_error=evaluation_error,
    )


def classify_edit(instruction: str, references: Sequence[Any]) -> tuple[bool, str]:
    """Return ``(is_edited, reason)`` using only declared V5 native semantics."""

    if not references:
        return True, "target_only_no_references"
    if any(pattern.search(instruction) for pattern in _UNCHANGED_PATTERNS):
        return False, "explicit_unchanged_instruction"
    return True, "referenced_change_instruction"


def _regions(value: Mapping[str, Any]) -> tuple[V5Region, ...]:
    layout = value["structured_description"]["elements"]["layout"]
    result: list[V5Region] = []
    for index, raw in enumerate(layout):
        edit = raw["edit"]
        references = tuple(_raw_bbox(ref["reference_bbox"]) for ref in edit["references"])
        edited, reason = classify_edit(edit["instruction"], references)
        result.append(
            V5Region(
                index=index,
                label=raw["label"],
                parent=raw["parent"],
                region_type=raw["type"],
                bbox=_raw_bbox(raw["bbox"]),
                instruction=edit["instruction"],
                references=references,
                edited=edited,
                classification_reason=reason,
            )
        )
    return tuple(result)


def _xyxy_as_yxyx(value: Any) -> tuple[float, ...]:
    bbox = _raw_bbox(value)
    if len(bbox) != 4:
        return bbox
    x0, y0, x1, y1 = bbox
    return y0, x0, y1, x1


def _regions_v6(value: Mapping[str, Any]) -> tuple[V5Region, ...]:
    """Project native V6 XYXY regions into the evaluator's V5 YXYX frame."""

    layout = value["elements"]["layout"]
    result: list[V5Region] = []
    for index, raw in enumerate(layout):
        edit = raw["edit"]
        references = tuple(
            _xyxy_as_yxyx(reference["reference_bbox"])
            for reference in edit["references"]
        )
        edited, reason = classify_edit(edit["instruction"], references)
        result.append(
            V5Region(
                index=index,
                label=raw["label"],
                parent=raw["parent"],
                region_type=raw["type"],
                bbox=_xyxy_as_yxyx(raw["bbox"]),
                instruction=edit["instruction"],
                references=references,
                edited=edited,
                classification_reason=reason,
            )
        )
    return tuple(result)


def _regions_v7_ground_truth(value: Mapping[str, Any]) -> tuple[V5Region, ...]:
    """Project observed V7-series GT into canonical YXYX.

    Despite the published audits claiming XYXY, the V7/V7R1 ``final_json``
    Target, Source-reference, and remove-reference boxes retain YXYX semantics.
    """

    layout = value["elements"]["layout"]
    result: list[V5Region] = []
    for index, raw in enumerate(layout):
        edit = raw["edit"]
        references = tuple(
            _raw_bbox(reference["reference_bbox"])
            for reference in edit["references"]
        )
        edited, reason = classify_edit(edit["instruction"], references)
        result.append(
            V5Region(
                index=index,
                label=raw["label"],
                parent=raw["parent"],
                region_type=raw["type"],
                bbox=_raw_bbox(raw["bbox"]),
                instruction=edit["instruction"],
                references=references,
                edited=edited,
                classification_reason=reason,
            )
        )
    return tuple(result)


def _regions_for_schema(
    value: Mapping[str, Any], schema: str
) -> tuple[V5Region, ...]:
    if schema == V5_SCHEMA:
        return _regions(value)
    if schema in OBSERVED_YXYX_GT_SCHEMAS:
        return _regions_v7_ground_truth(value)
    return _regions_v6(value)


def _remove_boxes(value: Mapping[str, Any]) -> tuple[tuple[float, float, float, float], ...]:
    remove = value["structured_description"]["elements"]["remove"]
    return tuple(
        _raw_bbox(reference["reference_bbox"])
        for edit in remove
        for reference in edit["references"]
    )


def _remove_boxes_v6(
    value: Mapping[str, Any],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        _xyxy_as_yxyx(reference["reference_bbox"])
        for edit in value["elements"]["remove"]
        for reference in edit["references"]
    )


def _remove_boxes_for_schema(
    value: Mapping[str, Any], schema: str
) -> tuple[tuple[float, float, float, float], ...]:
    if schema == V5_SCHEMA:
        return _remove_boxes(value)
    if schema in OBSERVED_YXYX_GT_SCHEMAS:
        return tuple(
            _raw_bbox(reference["reference_bbox"])
            for edit in value["elements"]["remove"]
            for reference in edit["references"]
        )
    return _remove_boxes_v6(value)


def _pair(prediction: V5Region, ground_truth: V5Region) -> RegionPair:
    return RegionPair(prediction, ground_truth, bbox_iou(prediction.bbox, ground_truth.bbox))


def _all_assignment(predictions: Sequence[V5Region], ground_truth: Sequence[V5Region]) -> tuple[RegionPair, ...]:
    weights = [[bbox_iou(pred.bbox, gt.bbox) for gt in ground_truth] for pred in predictions]
    return tuple(_pair(predictions[pred], ground_truth[gt]) for pred, gt in _maximum_weight_assignment(weights))


def _threshold_assignment(
    predictions: Sequence[V5Region], ground_truth: Sequence[V5Region], threshold: float
) -> tuple[RegionPair, ...]:
    if not predictions or not ground_truth:
        return ()
    bonus = min(len(predictions), len(ground_truth)) + 1.0
    ious = [[bbox_iou(pred.bbox, gt.bbox) for gt in ground_truth] for pred in predictions]
    weights = [[bonus + value if value >= threshold else 0.0 for value in row] for row in ious]
    pairs = _maximum_weight_assignment(weights)
    return tuple(
        RegionPair(predictions[pred], ground_truth[gt], ious[pred][gt])
        for pred, gt in pairs
        if ious[pred][gt] >= threshold
    )


def _box_threshold_count(
    predictions: Sequence[Sequence[float]], ground_truth: Sequence[Sequence[float]], threshold: float
) -> int:
    if not predictions or not ground_truth:
        return 0
    bonus = min(len(predictions), len(ground_truth)) + 1.0
    ious = [[bbox_iou(pred, gt) for gt in ground_truth] for pred in predictions]
    weights = [[bonus + value if value >= threshold else 0.0 for value in row] for row in ious]
    return sum(ious[pred][gt] >= threshold for pred, gt in _maximum_weight_assignment(weights))


def _partition(predictions: Sequence[V5Region], ground_truth: Sequence[V5Region]) -> SamplePartition:
    return SamplePartition(
        predictions=tuple(predictions),
        ground_truth=tuple(ground_truth),
        all_assignment=_all_assignment(predictions, ground_truth),
        threshold_matches={
            threshold: _threshold_assignment(predictions, ground_truth, threshold)
            for threshold in IOU_THRESHOLDS
        },
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _threshold_key(value: float) -> str:
    return f"iou_{value:.2f}".replace(".", "_")


def partition_report(results: Sequence[SamplePartition]) -> dict[str, Any]:
    predicted = sum(len(item.predictions) for item in results)
    ground_truth = sum(len(item.ground_truth) for item in results)
    count_errors = [len(item.predictions) - len(item.ground_truth) for item in results]
    assignments = [pair for item in results for pair in item.all_assignment]
    assignment_slots = sum(max(len(item.predictions), len(item.ground_truth)) for item in results)
    thresholds: dict[str, Any] = {}
    for threshold in IOU_THRESHOLDS:
        matched = [pair for item in results for pair in item.threshold_matches[threshold]]
        true_positive = len(matched)
        precision = true_positive / predicted if predicted else float(ground_truth == 0)
        recall = true_positive / ground_truth if ground_truth else float(predicted == 0)
        thresholds[_threshold_key(threshold)] = {
            "true_positive": true_positive,
            "false_positive": predicted - true_positive,
            "false_negative": ground_truth - true_positive,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "mean_iou_true_positive_pairs": (
                sum(pair.iou for pair in matched) / true_positive if true_positive else None
            ),
        }
    return {
        "samples": len(results),
        "counts": {
            "predicted": predicted,
            "ground_truth": ground_truth,
            "signed_error": predicted - ground_truth,
            "mean_signed_error_per_sample": sum(count_errors) / len(results) if results else None,
            "mean_absolute_error_per_sample": (
                sum(abs(value) for value in count_errors) / len(results) if results else None
            ),
            "exact_count_rate": (
                sum(value == 0 for value in count_errors) / len(results) if results else None
            ),
        },
        "bbox": {
            "mean_iou_assignment_slots_unmatched_zero": (
                sum(pair.iou for pair in assignments) / assignment_slots if assignment_slots else None
            ),
            "mean_iou_assigned_pairs": (
                sum(pair.iou for pair in assignments) / len(assignments) if assignments else None
            ),
            "assignment_pairs": len(assignments),
            "assignment_slots": assignment_slots,
        },
        "detection": thresholds,
    }


def _instruction_score_sufficient(
    pairs: Sequence[RegionPair],
    score_pair: Callable[[str, str], int],
) -> tuple[int, int, Counter[int]]:
    total = 0
    distribution: Counter[int] = Counter()
    for pair in pairs:
        score = score_pair(pair.prediction.instruction, pair.ground_truth.instruction)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Qwen instruction score must be an integer from 1 to 5, got {score!r}")
        total += score
        distribution[score] += 1
    return total, len(pairs), distribution


def _detection_prf(true_positive: int, predicted: int, ground_truth: int) -> dict[str, Any]:
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, ground_truth)
    return {
        "true_positive": true_positive,
        "false_positive": predicted - true_positive,
        "false_negative": ground_truth - true_positive,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _hierarchy_sufficient(
    prediction_regions: Sequence[V5Region],
    ground_truth_regions: Sequence[V5Region],
    split_pairs: Sequence[RegionPair],
    all_pairs: Sequence[RegionPair],
) -> tuple[int, int, int, int, int, int]:
    gt_to_prediction = {pair.ground_truth.index: pair.prediction.index for pair in all_pairs}
    prediction_by_label = {region.label: region.index for region in prediction_regions}
    ground_truth_by_label = {region.label: region.index for region in ground_truth_regions}
    correct = total = root_correct = root_total = non_root_correct = non_root_total = 0
    for pair in split_pairs:
        total += 1
        expected_parent = pair.ground_truth.parent
        predicted_parent = pair.prediction.parent
        if expected_parent == "ROOT":
            root_total += 1
            matched = predicted_parent == "ROOT"
            root_correct += int(matched)
        else:
            non_root_total += 1
            expected_index = ground_truth_by_label.get(expected_parent)
            actual_index = prediction_by_label.get(predicted_parent)
            matched = expected_index is not None and gt_to_prediction.get(expected_index) == actual_index
            non_root_correct += int(matched)
        correct += int(matched)
    return correct, total, root_correct, root_total, non_root_correct, non_root_total


def _reference_sufficient(pairs: Sequence[RegionPair], threshold: float) -> tuple[int, int, int]:
    predicted = sum(len(pair.prediction.references) for pair in pairs)
    ground_truth = sum(len(pair.ground_truth.references) for pair in pairs)
    true_positive = sum(
        _box_threshold_count(pair.prediction.references, pair.ground_truth.references, threshold)
        for pair in pairs
    )
    return true_positive, predicted, ground_truth


def _classifier_report(regions: Iterable[V5Region]) -> dict[str, Any]:
    values = list(regions)
    reasons = Counter(region.classification_reason for region in values)
    ambiguous_unchanged = [
        region.instruction
        for region in values
        if region.edited and "unchanged" in region.instruction.casefold()
    ]
    return {
        "regions": len(values),
        "edited": sum(region.edited for region in values),
        "non_edited": sum(not region.edited for region in values),
        "reasons": dict(sorted(reasons.items())),
        "edited_instructions_containing_unchanged": len(ambiguous_unchanged),
        "edited_instructions_containing_unchanged_examples": ambiguous_unchanged[:20],
    }


def _validity_report(statuses: Iterable[DocumentStatus]) -> dict[str, Any]:
    values = list(statuses)
    total = len(values)
    result: dict[str, Any] = {"total": total}
    for field in ("strict_json_valid", "json_valid", "schema_valid", "evaluation_valid"):
        count = sum(bool(getattr(item, field)) for item in values)
        result[field] = count
        result[f"{field}_rate"] = count / total if total else None
    result["schema_error_types"] = dict(
        sorted(Counter((item.schema_error or "").split(":", 1)[0] for item in values if item.schema_error).items())
    )
    result["evaluation_error_types"] = dict(
        sorted(
            Counter((item.evaluation_error or "").split(":", 1)[0] for item in values if item.evaluation_error).items()
        )
    )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _rows_by_id(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{path}:{line_number}: missing sample_id")
        if sample_id in result:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
        order.append(sample_id)
        result[sample_id] = row
    return order, result


def _prediction_payload(row: Mapping[str, Any], *, view: str = "raw") -> Any:
    if view not in PREDICTION_VIEWS:
        raise ValueError(f"unsupported prediction view: {view!r}")
    if view == "postprocessed" and "prediction" in row:
        return row["prediction"]
    response = row.get("response")
    if isinstance(response, str):
        return response
    for key in ("prediction", "raw_prediction", "output", "generated_json"):
        if key in row:
            return row[key]
    return None


def _ground_truth_payload(row: Mapping[str, Any]) -> Any:
    for key in ("target_json", "ground_truth", "annotation"):
        if key in row:
            return row[key]
    raise ValueError("ground-truth row has no target_json/ground_truth/annotation")


def _instruction_pairs_for_qwen(
    predictions: Mapping[str, Mapping[str, Any]],
    ground_truth: Mapping[str, Mapping[str, Any]],
    *,
    prediction_schema: str = V5_SCHEMA,
    ground_truth_schema: str = V5_SCHEMA,
    prediction_view: str = "raw",
) -> list[tuple[str, str]]:
    """Collect the exact instruction pairs scored after Target IoU matching."""

    pairs: set[tuple[str, str]] = set()
    for sample_id, ground_truth_row in ground_truth.items():
        ground_truth_status = inspect_document(
            _ground_truth_payload(ground_truth_row),
            allow_wrappers=False,
            schema=ground_truth_schema,
        )
        if not ground_truth_status.evaluation_valid or ground_truth_status.value is None:
            raise ValueError(f"ground truth is not evaluation-valid: {sample_id}")
        ground_truth_regions = _regions_for_schema(
            ground_truth_status.value, ground_truth_schema
        )

        prediction_row = predictions.get(sample_id)
        prediction_status = (
            inspect_document(
                _prediction_payload(prediction_row, view=prediction_view),
                allow_wrappers=True,
                schema=prediction_schema,
            )
            if prediction_row is not None
            else DocumentStatus(
                None,
                False,
                False,
                False,
                False,
                json_error="missing prediction",
            )
        )
        prediction_regions = (
            _regions_for_schema(prediction_status.value, prediction_schema)
            if prediction_status.schema_valid and prediction_status.value is not None
            else ()
        )
        for edited in (True, False):
            partition = _partition(
                [region for region in prediction_regions if region.edited is edited],
                [region for region in ground_truth_regions if region.edited is edited],
            )
            pairs.update(
                (pair.prediction.instruction, pair.ground_truth.instruction)
                for pair in partition.threshold_matches[0.50]
            )
    return sorted(pairs)


def _bootstrap_confidence_intervals(
    sufficient: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not sufficient or replicates <= 0:
        return {"replicates": replicates, "seed": seed, "metrics": {}}
    try:
        import numpy as np
    except ModuleNotFoundError:
        return _bootstrap_confidence_intervals_python(
            sufficient,
            replicates=replicates,
            seed=seed,
        )
    fields = (
        "predicted",
        "ground_truth",
        "true_positive",
        "iou_sum",
        "iou_slots",
        "count_exact",
        "count_abs_error",
        "samples",
        "hierarchy_correct",
        "hierarchy_total",
        "instruction_score_sum",
        "instruction_score_pairs",
        "reference_true_positive",
        "reference_predicted",
        "reference_ground_truth",
    )
    metric_values: dict[str, list[float]] = {}
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(sufficient), 1.0 / len(sufficient), dtype=np.float64)
    matrices = {
        split: np.asarray(
            [[float(row[split][field]) for field in fields] for row in sufficient],
            dtype=np.float64,
        )
        for split in ("edited", "non_edited")
    }
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        sample_counts = rng.multinomial(len(sufficient), probabilities, size=size)
        for split, matrix in matrices.items():
            aggregates = sample_counts @ matrix
            positions = {field: index for index, field in enumerate(fields)}

            def column(field: str) -> Any:
                return aggregates[:, positions[field]]

            pred = column("predicted")
            gt = column("ground_truth")
            tp = column("true_positive")
            precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred != 0)
            recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt != 0)
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(tp),
                where=(precision + recall) != 0,
            )
            iou = np.divide(
                column("iou_sum"),
                column("iou_slots"),
                out=np.zeros_like(tp),
                where=column("iou_slots") != 0,
            )
            exact = column("count_exact") / column("samples")
            mae = column("count_abs_error") / column("samples")
            hierarchy = np.divide(
                column("hierarchy_correct"),
                column("hierarchy_total"),
                out=np.zeros_like(tp),
                where=column("hierarchy_total") != 0,
            )
            instruction_score = np.divide(
                column("instruction_score_sum"),
                column("instruction_score_pairs"),
                out=np.zeros_like(tp),
                where=column("instruction_score_pairs") != 0,
            )
            ref_precision = np.divide(
                column("reference_true_positive"),
                column("reference_predicted"),
                out=np.zeros_like(tp),
                where=column("reference_predicted") != 0,
            )
            ref_recall = np.divide(
                column("reference_true_positive"),
                column("reference_ground_truth"),
                out=np.zeros_like(tp),
                where=column("reference_ground_truth") != 0,
            )
            ref_f1 = np.divide(
                2 * ref_precision * ref_recall,
                ref_precision + ref_recall,
                out=np.zeros_like(tp),
                where=(ref_precision + ref_recall) != 0,
            )
            for name, values in (
                ("target_bbox_precision_at_0_50", precision),
                ("target_bbox_recall_at_0_50", recall),
                ("target_bbox_f1_at_0_50", f1),
                ("target_bbox_mean_iou_unmatched_zero", iou),
                ("count_exact_rate", exact),
                ("count_mae", mae),
                ("parent_match_rate_at_0_50", hierarchy),
                ("instruction_qwen_mean_score_at_0_50", instruction_score),
                ("source_reference_bbox_f1_at_0_50", ref_f1),
            ):
                metric_values.setdefault(f"{split}.{name}", []).extend(values.tolist())
    metrics = {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence": 0.95,
        }
        for name, values in sorted(metric_values.items())
    }
    return {
        "method": "sample-level nonparametric bootstrap; corpus metrics recomputed from sufficient counts",
        "replicates": replicates,
        "seed": seed,
        "metrics": metrics,
    }


def _bootstrap_confidence_intervals_python(
    sufficient: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Dependency-free fallback used by minimal local test environments."""

    import random

    fields = (
        "predicted",
        "ground_truth",
        "true_positive",
        "iou_sum",
        "iou_slots",
        "count_exact",
        "count_abs_error",
        "samples",
        "hierarchy_correct",
        "hierarchy_total",
        "instruction_score_sum",
        "instruction_score_pairs",
        "reference_true_positive",
        "reference_predicted",
        "reference_ground_truth",
    )
    rng = random.Random(seed)
    metric_values: dict[str, list[float]] = {}

    def divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    for _ in range(replicates):
        indices = rng.choices(range(len(sufficient)), k=len(sufficient))
        for split in ("edited", "non_edited"):
            aggregate = {
                field: sum(float(sufficient[index][split][field]) for index in indices)
                for field in fields
            }
            precision = divide(aggregate["true_positive"], aggregate["predicted"])
            recall = divide(aggregate["true_positive"], aggregate["ground_truth"])
            instruction_score = divide(
                aggregate["instruction_score_sum"], aggregate["instruction_score_pairs"]
            )
            ref_precision = divide(
                aggregate["reference_true_positive"], aggregate["reference_predicted"]
            )
            ref_recall = divide(
                aggregate["reference_true_positive"], aggregate["reference_ground_truth"]
            )
            values = {
                "target_bbox_precision_at_0_50": precision,
                "target_bbox_recall_at_0_50": recall,
                "target_bbox_f1_at_0_50": _f1(precision, recall) or 0.0,
                "target_bbox_mean_iou_unmatched_zero": divide(
                    aggregate["iou_sum"], aggregate["iou_slots"]
                ),
                "count_exact_rate": divide(aggregate["count_exact"], aggregate["samples"]),
                "count_mae": divide(aggregate["count_abs_error"], aggregate["samples"]),
                "parent_match_rate_at_0_50": divide(
                    aggregate["hierarchy_correct"], aggregate["hierarchy_total"]
                ),
                "instruction_qwen_mean_score_at_0_50": instruction_score,
                "source_reference_bbox_f1_at_0_50": _f1(ref_precision, ref_recall) or 0.0,
            }
            for name, value in values.items():
                metric_values.setdefault(f"{split}.{name}", []).append(value)

    def percentile(values: Sequence[float], quantile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "method": "sample-level nonparametric bootstrap; corpus metrics recomputed from sufficient counts",
        "implementation": "dependency-free-python-fallback",
        "replicates": replicates,
        "seed": seed,
        "metrics": {
            name: {
                "lower": percentile(values, 0.025),
                "upper": percentile(values, 0.975),
                "confidence": 0.95,
            }
            for name, values in sorted(metric_values.items())
        },
    }


def evaluate(
    prediction_order: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    ground_truth_order: Sequence[str],
    ground_truth: Mapping[str, Mapping[str, Any]],
    *,
    prediction_schema: str = V5_SCHEMA,
    ground_truth_schema: str = V5_SCHEMA,
    prediction_view: str = "raw",
    instruction_qwen_score: Callable[[str, str], int],
    instruction_judge_metadata: Mapping[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if prediction_schema not in PREDICTION_SCHEMAS:
        raise ValueError(f"unsupported prediction schema: {prediction_schema!r}")
    if ground_truth_schema not in GROUND_TRUTH_SCHEMAS:
        raise ValueError(f"unsupported ground-truth schema: {ground_truth_schema!r}")
    if prediction_view not in PREDICTION_VIEWS:
        raise ValueError(f"unsupported prediction view: {prediction_view!r}")
    if len(set(ground_truth_order)) != len(ground_truth_order):
        raise ValueError("ground-truth order contains duplicates")
    prediction_without_ground_truth = [
        sample_id for sample_id in prediction_order if sample_id not in ground_truth
    ]
    missing = [sample_id for sample_id in ground_truth_order if sample_id not in predictions]

    prediction_status: dict[str, DocumentStatus] = {
        sample_id: inspect_document(
            _prediction_payload(row, view=prediction_view),
            allow_wrappers=True,
            schema=prediction_schema,
        )
        for sample_id, row in predictions.items()
    }
    ground_truth_status: dict[str, DocumentStatus] = {
        sample_id: inspect_document(
            _ground_truth_payload(row),
            allow_wrappers=False,
            schema=ground_truth_schema,
        )
        for sample_id, row in ground_truth.items()
    }
    invalid_ground_truth = [
        sample_id for sample_id in ground_truth_order if not ground_truth_status[sample_id].evaluation_valid
    ]
    if invalid_ground_truth:
        raise ValueError(f"{len(invalid_ground_truth)} ground-truth rows are not evaluation-valid")

    partitions: dict[str, list[SamplePartition]] = {"edited": [], "non_edited": []}
    hierarchy_totals = {
        split: [0, 0, 0, 0, 0, 0] for split in ("edited", "non_edited")
    }
    instruction_score_totals: dict[str, list[int]] = {
        split: [0, 0, 0, 0, 0, 0, 0] for split in ("edited", "non_edited")
    }
    reference_totals = {
        split: {threshold: [0, 0, 0] for threshold in IOU_THRESHOLDS}
        for split in ("edited", "non_edited")
    }
    remove_totals = {threshold: [0, 0, 0] for threshold in IOU_THRESHOLDS}
    remove_count_errors: list[int] = []
    all_prediction_regions: list[V5Region] = []
    all_ground_truth_regions: list[V5Region] = []
    details: list[dict[str, Any]] = []
    bootstrap_sufficient: list[dict[str, dict[str, float]]] = []

    for sample_id in ground_truth_order:
        gt_status = ground_truth_status[sample_id]
        assert gt_status.value is not None
        gt_regions = _regions_for_schema(gt_status.value, ground_truth_schema)
        prediction_row = predictions.get(sample_id)
        pred_status = prediction_status.get(
            sample_id,
            DocumentStatus(None, False, False, False, False, json_error="missing prediction"),
        )
        # Strict schema failure contributes an empty prediction on the fixed cohort.
        pred_regions = (
            _regions_for_schema(pred_status.value, prediction_schema)
            if pred_status.schema_valid and pred_status.value is not None
            else ()
        )
        all_prediction_regions.extend(pred_regions)
        all_ground_truth_regions.extend(gt_regions)

        sample_partitions: dict[str, SamplePartition] = {}
        for split, edited in (("edited", True), ("non_edited", False)):
            partition = _partition(
                [region for region in pred_regions if region.edited is edited],
                [region for region in gt_regions if region.edited is edited],
            )
            partitions[split].append(partition)
            sample_partitions[split] = partition

        all_matches_050 = (
            *sample_partitions["edited"].threshold_matches[0.50],
            *sample_partitions["non_edited"].threshold_matches[0.50],
        )
        sample_sufficient: dict[str, dict[str, float]] = {}
        for split in ("edited", "non_edited"):
            partition = sample_partitions[split]
            matches_050 = partition.threshold_matches[0.50]
            hierarchy = _hierarchy_sufficient(pred_regions, gt_regions, matches_050, all_matches_050)
            for index, value in enumerate(hierarchy):
                hierarchy_totals[split][index] += value
            instruction_score = _instruction_score_sufficient(
                matches_050, instruction_qwen_score
            )
            instruction_score_totals[split][0] += instruction_score[0]
            instruction_score_totals[split][1] += instruction_score[1]
            for score in range(1, 6):
                instruction_score_totals[split][score + 1] += instruction_score[2][score]
            for threshold in IOU_THRESHOLDS:
                reference = _reference_sufficient(partition.threshold_matches[threshold], threshold)
                for index, value in enumerate(reference):
                    reference_totals[split][threshold][index] += value
            reference_050 = _reference_sufficient(matches_050, 0.50)
            count_error = len(partition.predictions) - len(partition.ground_truth)
            sample_sufficient[split] = {
                "predicted": float(len(partition.predictions)),
                "ground_truth": float(len(partition.ground_truth)),
                "true_positive": float(len(matches_050)),
                "iou_sum": float(sum(pair.iou for pair in partition.all_assignment)),
                "iou_slots": float(max(len(partition.predictions), len(partition.ground_truth))),
                "count_exact": float(count_error == 0),
                "count_abs_error": float(abs(count_error)),
                "samples": 1.0,
                "hierarchy_correct": float(hierarchy[0]),
                "hierarchy_total": float(hierarchy[1]),
                "instruction_score_sum": float(instruction_score[0]),
                "instruction_score_pairs": float(instruction_score[1]),
                "reference_true_positive": float(reference_050[0]),
                "reference_predicted": float(reference_050[1]),
                "reference_ground_truth": float(reference_050[2]),
            }
        bootstrap_sufficient.append(sample_sufficient)

        gt_remove = _remove_boxes_for_schema(gt_status.value, ground_truth_schema)
        pred_remove = (
            _remove_boxes_for_schema(pred_status.value, prediction_schema)
            if pred_status.schema_valid and pred_status.value is not None
            else ()
        )
        remove_count_errors.append(len(pred_remove) - len(gt_remove))
        for threshold in IOU_THRESHOLDS:
            remove_totals[threshold][0] += _box_threshold_count(pred_remove, gt_remove, threshold)
            remove_totals[threshold][1] += len(pred_remove)
            remove_totals[threshold][2] += len(gt_remove)

        details.append(
            {
                "sample_id": sample_id,
                "prediction_validity": pred_status.public(),
                "ground_truth_validity": gt_status.public(),
                "sufficient_statistics": sample_sufficient,
                "regions": {
                    split: {
                        "predicted": len(sample_partitions[split].predictions),
                        "ground_truth": len(sample_partitions[split].ground_truth),
                        "instruction_qwen_score_at_iou_0_50": {
                            "score_sum": sample_sufficient[split]["instruction_score_sum"],
                            "matched_regions": int(
                                sample_sufficient[split]["instruction_score_pairs"]
                            ),
                            "mean_score": _ratio(
                                sample_sufficient[split]["instruction_score_sum"],
                                sample_sufficient[split]["instruction_score_pairs"],
                            ),
                        },
                        "matches": {
                            _threshold_key(threshold): [
                                {
                                    "prediction_index": pair.prediction.index,
                                    "ground_truth_index": pair.ground_truth.index,
                                    "iou": pair.iou,
                                }
                                for pair in sample_partitions[split].threshold_matches[threshold]
                            ]
                            for threshold in IOU_THRESHOLDS
                        },
                    }
                    for split in ("edited", "non_edited")
                },
            }
        )

    hierarchy_report: dict[str, Any] = {}
    instruction_score_report: dict[str, Any] = {}
    reference_report: dict[str, Any] = {}
    for split in ("edited", "non_edited"):
        correct, total, root_correct, root_total, non_root_correct, non_root_total = hierarchy_totals[split]
        hierarchy_report[split] = {
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
        score_sum, matched_regions, *distribution = instruction_score_totals[split]
        instruction_score_report[split] = {
            "matched_regions": int(matched_regions),
            "score_sum": score_sum,
            "mean_score": _ratio(score_sum, matched_regions),
            "score_scale": {"minimum": 1, "maximum": 5, "integer_only": True},
            "score_distribution": {
                str(score): distribution[score - 1] for score in range(1, 6)
            },
            "scope": "instruction pairs conditional on a Target-region match at IoU >= 0.50",
        }
        reference_report[split] = {
            _threshold_key(threshold): {
                **_detection_prf(*reference_totals[split][threshold]),
                "scope": (
                    "prediction reference_bbox values converted to V5 YXYX when needed, "
                    "conditional on a Target-region match at the same threshold"
                ),
            }
            for threshold in IOU_THRESHOLDS
        }

    remove_report = {
        "counts": {
            "predicted": sum(item[1] for item in remove_totals.values()) // len(IOU_THRESHOLDS),
            "ground_truth": sum(item[2] for item in remove_totals.values()) // len(IOU_THRESHOLDS),
            "mean_signed_error_per_sample": sum(remove_count_errors) / len(remove_count_errors),
            "mean_absolute_error_per_sample": sum(abs(value) for value in remove_count_errors) / len(remove_count_errors),
            "exact_count_rate": sum(value == 0 for value in remove_count_errors) / len(remove_count_errors),
        },
        "source_bbox_detection": {
            _threshold_key(threshold): _detection_prf(*remove_totals[threshold])
            for threshold in IOU_THRESHOLDS
        },
    }

    report = {
        "schema_version": 1,
        "evaluator_version": (
            V7_SERIES_OBSERVED_YXYX_EVALUATOR_VERSION
            if ground_truth_schema == V7_SERIES_GT_SCHEMA
            else (
                V7R1_OBSERVED_YXYX_EVALUATOR_VERSION
                if ground_truth_schema == V7R1_GT_SCHEMA
                else (
                    V6_GROUND_TRUTH_EVALUATOR_VERSION
                    if ground_truth_schema == V6_SCHEMA
                    else (
                        EVALUATOR_VERSION
                        if prediction_schema == V5_SCHEMA
                        else V6_CROSS_SCHEMA_EVALUATOR_VERSION
                    )
                )
            )
        ),
        "protocol": {
            "benchmark_cohort": "craft-hard1000-v1",
            "prediction_format": prediction_schema,
            "ground_truth_format": ground_truth_schema,
            "prediction_view": prediction_view,
            "bbox_coordinate_adapter": (
                "V7-series Target bbox, Source reference_bbox, and remove reference_bbox kept as "
                "observed YXYX/1000 despite contradictory audit metadata; all V6 prediction "
                "boxes converted from XYXY/1000 to canonical YXYX/1000"
                if ground_truth_schema in OBSERVED_YXYX_GT_SCHEMAS
                and prediction_schema == V6_SCHEMA
                else (
                    "none; prediction and ground truth are native YXYX/1000"
                    if prediction_schema == V5_SCHEMA and ground_truth_schema == V5_SCHEMA
                    else (
                        "V6 prediction bbox/reference_bbox XYXY/1000 converted to YXYX/1000; "
                        "V5 ground truth unchanged"
                        if prediction_schema == V6_SCHEMA and ground_truth_schema == V5_SCHEMA
                        else (
                            "V5 prediction unchanged; direct-root ground-truth bbox/reference_bbox "
                            "converted according to its declared schema"
                            if prediction_schema == V5_SCHEMA
                            else "V6 prediction and V6 ground-truth bbox/reference_bbox "
                            "XYXY/1000 both converted to canonical YXYX/1000"
                        )
                    )
                )
            ),
            "edited_non_edited_classifier": {
                "version": CLASSIFIER_VERSION,
                "policy": (
                    "no references => edited Target-only addition; an explicit unchanged directive at the "
                    "start of a referenced-region instruction => non-edited; all other referenced regions => edited"
                ),
                "patterns": [pattern.pattern for pattern in _UNCHANGED_PATTERNS],
            },
            "target_bbox_matching": (
                "one-to-one within native edited status; maximize threshold-qualified pair count, then summed IoU, "
                "then stable indices"
            ),
            "thresholds": list(IOU_THRESHOLDS),
            "instruction_qwen_metric": {
                "target_bbox_condition": "IoU >= 0.50",
                "aggregation": "mean integer score across matched instruction pairs",
                "score_scale": {
                    "5": "fully correct and semantically equivalent",
                    "4": "mostly correct with only a minor omission or imprecision",
                    "3": "partially correct with a material detail missing, wrong, or unsupported",
                    "2": "weakly related with major omissions or errors",
                    "1": "unrelated, opposite, or fundamentally incompatible",
                },
                "evaluation_dimensions": (
                    "operation, target, source/current state, desired state/attributes, count, "
                    "literal text, spatial/geometry constraints, and preservation requirements"
                ),
                "unmatched_region_policy": (
                    "penalized by bbox metrics; excluded from the conditional instruction-score mean"
                ),
                "model": QWEN_JUDGE_MODEL_NAME,
                "revision": QWEN_JUDGE_MODEL_REVISION,
                "judge_version": QWEN_JUDGE_VERSION,
                "prompt_version": QWEN_PROMPT_VERSION,
                "input_mode": "text-only",
                "thinking": False,
                "structured_output": "JSON with integer score 1--5 and concise reason",
            },
            "invalid_prediction_policy": (
                f"strict {prediction_schema} schema failure contributes an empty prediction"
            ),
            "operation_metrics": {
                "available": False,
                "reason": "the operation-free schemas contain no operation field; operations are not guessed",
            },
            "official_common_benchmark_comparable": False,
            "comparability_note": (
                "Uses supplied operation-free full-inventory labels on frozen hard1000 IDs/images; "
                "this is not the benchmark's pinned medium-v10 operation profile."
            ),
        },
        "sample_alignment": {
            "ground_truth_rows": len(ground_truth_order),
            "prediction_rows": len(prediction_order),
            "covered_rows": len(ground_truth_order) - len(missing),
            "missing_prediction_rows": len(missing),
            "missing_prediction_ids": missing[:100],
            "prediction_without_native_ground_truth_rows": len(prediction_without_ground_truth),
            "prediction_without_native_ground_truth_ids": prediction_without_ground_truth[:100],
        },
        "validity": {
            "predictions": _validity_report(
                prediction_status.get(
                    sample_id,
                    DocumentStatus(None, False, False, False, False, json_error="missing prediction"),
                )
                for sample_id in ground_truth_order
            ),
            "all_predictions": _validity_report(
                prediction_status[sample_id] for sample_id in prediction_order
            ),
            "ground_truth": _validity_report(ground_truth_status.values()),
        },
        "classifier_audit": {
            "predictions": _classifier_report(all_prediction_regions),
            "ground_truth": _classifier_report(all_ground_truth_regions),
        },
        "regions": {split: partition_report(partitions[split]) for split in ("edited", "non_edited")},
        "hierarchy_at_iou_0_50": hierarchy_report,
        "instruction_qwen_score_at_iou_0_50": instruction_score_report,
        "instruction_judge": dict(instruction_judge_metadata),
        "source_reference_bbox": reference_report,
        "remove": remove_report,
        "confidence_intervals": _bootstrap_confidence_intervals(
            bootstrap_sufficient,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }
    return report, details


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    def format_instruction_score(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f} / 5"

    prediction_schema = report["protocol"]["prediction_format"]
    ground_truth_schema = report["protocol"]["ground_truth_format"]
    lines = [
        "# Operation-free image-edit checkpoint evaluation on CRAFT hard1000",
        "",
        f"- Frozen cohort predictions: {report['sample_alignment']['prediction_rows']}",
        f"- Labelled/evaluated samples: {report['sample_alignment']['ground_truth_rows']}",
        f"- Ground-truth schema: {ground_truth_schema}",
        f"- Predictions without supplied ground truth: "
        f"{report['sample_alignment']['prediction_without_native_ground_truth_rows']}",
        f"- Strict JSON valid: {report['validity']['predictions']['strict_json_valid_rate']:.2%}",
        f"- Strict prediction schema valid ({prediction_schema}): "
        f"{report['validity']['predictions']['schema_valid_rate']:.2%}",
        f"- Ground truth evaluation-valid: {report['validity']['ground_truth']['evaluation_valid_rate']:.2%}",
        f"- Instruction metric: Qwen 1--5 semantic score with "
        f"{QWEN_JUDGE_MODEL_NAME}@{QWEN_JUDGE_MODEL_REVISION[:8]}",
        "- Operation metrics: unsupported by the operation-free schemas",
        "- Official common-benchmark comparable: no (no operation field)",
        "- Aggregate score: not defined by the benchmark",
        "",
        "## Target-region bbox detection",
        "",
        "| Split | IoU | Precision | Recall | F1 | Mean IoU of true-positive pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, label in (("edited", "Edited"), ("non_edited", "Non-edited")):
        region = report["regions"][split]
        for threshold in IOU_THRESHOLDS:
            detection = region["detection"][_threshold_key(threshold)]
            lines.append(
                f"| {label} | {threshold:.2f} | {percentage(detection['precision'])} | "
                f"{percentage(detection['recall'])} | {percentage(detection['f1'])} | "
                f"{percentage(detection['mean_iou_true_positive_pairs'])} |"
            )
    lines.extend(
        [
            "",
            "## Region-plan metrics",
            "",
            "| Split | Mean IoU (unmatched=0) | Count exact | Count MAE | Count bias/sample | "
            "Parent match@0.50 | Instruction Qwen score@0.50 | Source-reference bbox F1@0.50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, label in (("edited", "Edited"), ("non_edited", "Non-edited")):
        region = report["regions"][split]
        hierarchy = report["hierarchy_at_iou_0_50"][split]
        instruction = report["instruction_qwen_score_at_iou_0_50"][split]
        references = report["source_reference_bbox"][split]["iou_0_50"]
        lines.append(
            f"| {label} | {percentage(region['bbox']['mean_iou_assignment_slots_unmatched_zero'])} | "
            f"{percentage(region['counts']['exact_count_rate'])} | "
            f"{region['counts']['mean_absolute_error_per_sample']:.3f} | "
            f"{region['counts']['mean_signed_error_per_sample']:+.3f} | "
            f"{percentage(hierarchy['parent_match_rate'])} | "
            f"{format_instruction_score(instruction['mean_score'])} | "
            f"{percentage(references['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Headline 95% confidence intervals",
            "",
            "| Split | Target bbox F1@0.50 | Mean IoU (unmatched=0) | Instruction Qwen score@0.50 |",
            "|---|---:|---:|---:|",
        ]
    )
    intervals = report["confidence_intervals"]["metrics"]
    for split, label in (("edited", "Edited"), ("non_edited", "Non-edited")):
        region = report["regions"][split]
        f1 = region["detection"]["iou_0_50"]["f1"]
        mean_iou = region["bbox"]["mean_iou_assignment_slots_unmatched_zero"]
        instruction_score = report["instruction_qwen_score_at_iou_0_50"][split][
            "mean_score"
        ]
        f1_ci = intervals.get(f"{split}.target_bbox_f1_at_0_50")
        iou_ci = intervals.get(f"{split}.target_bbox_mean_iou_unmatched_zero")
        instruction_ci = intervals.get(f"{split}.instruction_qwen_mean_score_at_0_50")
        f1_text = percentage(f1)
        iou_text = percentage(mean_iou)
        instruction_text = format_instruction_score(instruction_score)
        if f1_ci:
            f1_text += f" [{percentage(f1_ci['lower'])}, {percentage(f1_ci['upper'])}]"
        if iou_ci:
            iou_text += f" [{percentage(iou_ci['lower'])}, {percentage(iou_ci['upper'])}]"
        if instruction_ci and instruction_score is not None:
            instruction_text += (
                f" [{instruction_ci['lower']:.3f}, {instruction_ci['upper']:.3f}]"
            )
        lines.append(f"| {label} | {f1_text} | {iou_text} | {instruction_text} |")
    lines.extend(
        [
            "",
            "## Instruction score distribution",
            "",
            "| Split | 1 | 2 | 3 | 4 | 5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, label in (("edited", "Edited"), ("non_edited", "Non-edited")):
        distribution = report["instruction_qwen_score_at_iou_0_50"][split][
            "score_distribution"
        ]
        lines.append(
            f"| {label} | {distribution['1']} | {distribution['2']} | "
            f"{distribution['3']} | {distribution['4']} | {distribution['5']} |"
        )
    ci_info = report["confidence_intervals"]
    lines.extend(
        [
            "",
            f"Confidence intervals use {ci_info['replicates']:,} sample-level bootstrap replicates "
            f"(seed {ci_info['seed']}).",
            "",
        ]
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--instruction-judgments", type=Path, required=True)
    parser.add_argument("--prediction-schema", choices=PREDICTION_SCHEMAS, default=V5_SCHEMA)
    parser.add_argument(
        "--ground-truth-schema", choices=GROUND_TRUTH_SCHEMAS, default=V5_SCHEMA
    )
    parser.add_argument("--prediction-view", choices=PREDICTION_VIEWS, default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_order, predictions = _rows_by_id(args.predictions.resolve(strict=True))
    ground_truth_order, ground_truth = _rows_by_id(args.ground_truth.resolve(strict=True))
    instruction_judgments = args.instruction_judgments.resolve(strict=True)
    instruction_pairs = _instruction_pairs_for_qwen(
        predictions,
        ground_truth,
        prediction_schema=args.prediction_schema,
        ground_truth_schema=args.ground_truth_schema,
        prediction_view=args.prediction_view,
    )
    print(f"loading {len(instruction_pairs)} unique Qwen instruction judgments", file=sys.stderr)
    instruction_qwen_score = CachedQwenInstructionJudge(
        instruction_judgments,
        instruction_pairs,
    )
    report, details = evaluate(
        prediction_order,
        predictions,
        ground_truth_order,
        ground_truth,
        prediction_schema=args.prediction_schema,
        ground_truth_schema=args.ground_truth_schema,
        prediction_view=args.prediction_view,
        instruction_qwen_score=instruction_qwen_score,
        instruction_judge_metadata=instruction_qwen_score.metadata,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["provenance"] = {
        "predictions": {"path": str(args.predictions), "sha256": _sha256_file(args.predictions)},
        "ground_truth": {"path": str(args.ground_truth), "sha256": _sha256_file(args.ground_truth)},
        "instruction_judgments": {
            "path": str(instruction_judgments),
            "sha256": _sha256_file(instruction_judgments),
        },
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
    }
    _write_json(args.output, report)
    if args.details_output:
        _write_jsonl(args.details_output, details)
    if args.report_output:
        _write_markdown(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
