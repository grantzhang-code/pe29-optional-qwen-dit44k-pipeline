"""Rule-based image-edit-v6 rewards for ms-swift GRPO.

The module has no runtime dependency on Swift.  When loaded through Swift's
``--external_plugins`` option it registers two outcome rewards:

* ``image_edit_v6_schema``: strict JSON plus the authoritative V6 validator.
* ``image_edit_v6_content``: dense, label-invariant spatial/structure/content
  agreement with the dataset's canonical ``solution`` column.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
import math
import os
import re
from typing import Any, Mapping, Sequence


# The reward plugin only needs the frozen validator.  Avoid registering the SFT
# dataset if Swift imports this external plugin before the GRPO dataset plugin.
_old_library_only = os.environ.get("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY")
os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.image_edit_v6_full_json import canonical_v6_target
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = _old_library_only


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_target(value: Any) -> dict[str, Any]:
    # Python's default decoder silently keeps the last duplicate object key.
    # Reject duplicates explicitly so a completion cannot hide malformed or
    # conflicting fields behind parser-dependent behaviour.
    if isinstance(value, str):
        value = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {token}")
            ),
        )
    canonical = canonical_v6_target(value)
    parsed = json.loads(canonical)
    # canonical_v6_target already guarantees a mapping; keep the type explicit
    # for static analysis and defensive use outside the production dataset.
    if not isinstance(parsed, dict):
        raise ValueError("image-edit-v6 target must be a JSON object")
    return parsed


def score_schema(completion: Any) -> float:
    """Return 1 only for a strict, schema-valid V6 JSON completion."""

    # Rollout completions are text.  Do not accept an already-decoded mapping:
    # the reward also enforces the requested on-wire JSON-only format.
    if not isinstance(completion, str):
        return 0.0
    try:
        _parse_target(completion)
    except (TypeError, ValueError):
        return 0.0
    return 1.0


def _tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [token.casefold() for token in _TOKEN_RE.findall(value)]


def _token_f1(first: Any, second: Any) -> float:
    a = Counter(_tokens(first))
    b = Counter(_tokens(second))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def _optional_text_score(first: Any, second: Any) -> float:
    if first is None and second is None:
        return 1.0
    if first is None or second is None:
        return 0.0
    return _token_f1(first, second)


def _palette_score(first: Any, second: Any) -> float:
    a = set(first or [])
    b = set(second or [])
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def bbox_iou(first: Sequence[int | float], second: Sequence[int | float]) -> float:
    """Compute IoU for V6 normalized ``[x0, y0, x1, y1]`` boxes."""

    ax0, ay0, ax1, ay1 = (float(item) for item in first)
    bx0, by0, bx1, by1 = (float(item) for item in second)
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _minimum_cost_assignment(costs: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Hungarian assignment for a rectangular matrix with rows <= columns."""

    row_count = len(costs)
    column_count = len(costs[0]) if row_count else 0
    if row_count == 0:
        return []
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment cost matrix must be rectangular with rows <= columns")
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
    return sorted(
        (row - 1, column - 1)
        for column, row in enumerate(matched_row[1:], start=1)
        if row
    )


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment weight matrix must be rectangular")
    if row_count <= column_count:
        maximum = max(max(row) for row in weights)
        return _minimum_cost_assignment([[maximum - value for value in row] for row in weights])
    transposed = [
        [weights[row][column] for row in range(row_count)]
        for column in range(column_count)
    ]
    maximum = max(max(row) for row in transposed)
    return sorted(
        (column, row)
        for row, column in _minimum_cost_assignment(
            [[maximum - value for value in row] for row in transposed]
        )
    )


def _reference_score(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    weights = [
        [
            0.85 * bbox_iou(a["reference_bbox"], b["reference_bbox"])
            + 0.15 * float(a.get("reference_id") == b.get("reference_id"))
            for b in second
        ]
        for a in first
    ]
    pairs = _maximum_weight_assignment(weights)
    return sum(weights[i][j] for i, j in pairs) / max(len(first), len(second))


def _label_score(first: str, second: str) -> float:
    # Numbering is an ordering device, not semantic identity.
    return _token_f1(re.sub(r"\s+\d+>$", ">", first), re.sub(r"\s+\d+>$", ">", second))


def _region_score(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    values = (
        (0.15, float(first["type"] == second["type"])),
        (0.05, _label_score(first["label"], second["label"])),
        (0.18, _token_f1(first["desc"], second["desc"])),
        (0.08, _palette_score(first["color_palette"], second["color_palette"])),
        (0.12, _optional_text_score(first.get("text"), second.get("text"))),
        (0.07, _optional_text_score(first.get("entity"), second.get("entity"))),
        (0.20, _token_f1(first["edit"]["instruction"], second["edit"]["instruction"])),
        (0.15, _reference_score(first["edit"]["references"], second["edit"]["references"])),
    )
    return sum(weight * value for weight, value in values)


def _document_score(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_style = first["style_description"]
    second_style = second["style_description"]
    first_elements = first["elements"]
    second_elements = second["elements"]
    values = (
        (0.18, _token_f1(first["high_level_description"], second["high_level_description"])),
        (0.25, _token_f1(first["normalized_edit_instruction"], second["normalized_edit_instruction"])),
        (0.17, _token_f1(first_elements["background"], second_elements["background"])),
        (0.10, _token_f1(first_style["aesthetics"], second_style["aesthetics"])),
        (0.10, _token_f1(first_style["lighting"], second_style["lighting"])),
        (0.10, _token_f1(first_style["medium"], second_style["medium"])),
        (0.10, _palette_score(first_style["color_palette"], second_style["color_palette"])),
    )
    return sum(weight * value for weight, value in values)


def _removal_score(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    weights = [
        [
            0.5 * _token_f1(a["instruction"], b["instruction"])
            + 0.5 * _reference_score(a["references"], b["references"])
            for b in second
        ]
        for a in first
    ]
    pairs = _maximum_weight_assignment(weights)
    return sum(weights[i][j] for i, j in pairs) / max(len(first), len(second))


@dataclass(frozen=True)
class ContentScore:
    bbox: float
    hierarchy: float
    regions: float
    document: float
    removals: float

    @property
    def total(self) -> float:
        return (
            0.35 * self.bbox
            + 0.20 * self.hierarchy
            + 0.20 * self.regions
            + 0.15 * self.document
            + 0.10 * self.removals
        )


def content_score(completion: Any, solution: Any) -> ContentScore | None:
    """Return a dense content breakdown, or ``None`` for invalid V6 output."""

    try:
        prediction = _parse_target(completion)
        target = _parse_target(solution)
    except (TypeError, ValueError):
        return None

    predicted_regions = prediction["elements"]["layout"]
    target_regions = target["elements"]["layout"]
    weights = [
        [
            bbox_iou(first["bbox"], second["bbox"])
            + 1e-8 * float(first["type"] == second["type"])
            for second in target_regions
        ]
        for first in predicted_regions
    ]
    pairs = _maximum_weight_assignment(weights)
    denominator = max(len(predicted_regions), len(target_regions))
    bbox = sum(
        bbox_iou(predicted_regions[i]["bbox"], target_regions[j]["bbox"])
        for i, j in pairs
    ) / denominator
    regions = sum(
        _region_score(predicted_regions[i], target_regions[j]) for i, j in pairs
    ) / denominator

    predicted_label_to_index = {
        item["label"]: index for index, item in enumerate(predicted_regions)
    }
    target_label_to_index = {item["label"]: index for index, item in enumerate(target_regions)}
    assignment = dict(pairs)
    correct_hierarchy = 0
    for prediction_index, target_index in pairs:
        predicted_parent = predicted_regions[prediction_index]["parent"]
        target_parent = target_regions[target_index]["parent"]
        if predicted_parent == "ROOT" and target_parent == "ROOT":
            correct_hierarchy += 1
            continue
        predicted_parent_index = predicted_label_to_index.get(predicted_parent)
        target_parent_index = target_label_to_index.get(target_parent)
        if (
            predicted_parent_index is not None
            and target_parent_index is not None
            and assignment.get(predicted_parent_index) == target_parent_index
        ):
            correct_hierarchy += 1
    hierarchy = correct_hierarchy / denominator

    return ContentScore(
        bbox=bbox,
        hierarchy=hierarchy,
        regions=regions,
        document=_document_score(prediction, target),
        removals=_removal_score(
            prediction["elements"]["remove"], target["elements"]["remove"]
        ),
    )


def score_content(completion: Any, solution: Any) -> float:
    """Return a bounded dense content reward; invalid output receives zero."""

    result = content_score(completion, solution)
    if result is None:
        return 0.0
    return min(1.0, max(0.0, result.total))


class ImageEditV6SchemaReward:
    def __init__(self, args: Any = None, **_: Any) -> None:
        self.args = args

    def __call__(self, completions: Sequence[Any], **_: Any) -> list[float]:
        return [score_schema(completion) for completion in completions]


class ImageEditV6ContentReward:
    def __init__(self, args: Any = None, **_: Any) -> None:
        self.args = args

    def __call__(
        self,
        completions: Sequence[Any],
        solution: Sequence[Any],
        **_: Any,
    ) -> list[float]:
        if len(completions) != len(solution):
            raise ValueError("completions and solution must have the same batch size")
        return [score_content(completion, target) for completion, target in zip(completions, solution)]


def register_with_swift() -> tuple[str, str]:
    """Register rewards in the ms-swift 4.4.2 ORM registry."""

    from swift.rewards import orms

    orms["image_edit_v6_schema"] = ImageEditV6SchemaReward
    orms["image_edit_v6_content"] = ImageEditV6ContentReward
    return "image_edit_v6_schema", "image_edit_v6_content"


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift.rewards") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


if _swift_is_available():
    register_with_swift()
