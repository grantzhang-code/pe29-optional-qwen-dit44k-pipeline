#!/usr/bin/env python3
"""Map PE xyxy boxes to the native MDS intake's grouped yxyx schema.

NativeEditAnnotation owns the subsequent conversion to element.edit and
remove references. Multiple removal boxes become independent occurrences
with the same instruction so every Source endpoint survives that intake.
"""

from __future__ import annotations

import copy
import math
from typing import Any


def _ordered(value: Any, fields: tuple[str, ...], path: str,
             optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """Restore the PE training schema even for previously key-sorted caches.

    Never sort lists: element, action, palette and source endpoint order are
    semantic. Unknown fields fail admission instead of disappearing silently.
    """
    if (not isinstance(value, dict) or set(value) - set(fields)
            or set(fields) - set(optional) - set(value)):
        raise ValueError(f"{path}: fields differ from the PE training schema")
    return {key: copy.deepcopy(value[key]) for key in fields if key in value}


def ordered_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    result = _ordered(prediction, (
        "normalized_edit_instruction", "high_level_description",
        "style_description", "elements",
    ), "prediction")
    result["style_description"] = _ordered(result["style_description"], (
        "aesthetics", "lighting", "medium", "color_palette",
    ), "style_description")
    elements = _ordered(result["elements"], (
        "background", "layout", "reference_edits",
    ), "elements")
    if not isinstance(elements["layout"], list):
        raise ValueError("elements.layout must be an array")
    elements["layout"] = [_ordered(region, (
        "type", "label", "parent", "bbox", "text", "desc", "entity", "color_palette",
    ), f"layout[{index}]", optional=("text", "entity"))
        for index, region in enumerate(elements["layout"])]
    if not isinstance(elements["reference_edits"], list):
        raise ValueError("elements.reference_edits must be an array")
    groups = []
    for index, raw in enumerate(elements["reference_edits"]):
        group = _ordered(raw, ("reference_id", "edit", "remove"), f"reference_edits[{index}]")
        if not isinstance(group["edit"], list) or not isinstance(group["remove"], list):
            raise ValueError("reference edits and removals must be arrays")
        group["edit"] = [_ordered(action, ("label", "instruction", "reference_bbox"),
                                  "edit", optional=("reference_bbox",)) for action in group["edit"]]
        group["remove"] = [_ordered(action, ("instruction", "reference_bbox"), "remove")
                           for action in group["remove"]]
        groups.append(group)
    elements["reference_edits"] = groups
    result["elements"] = elements
    return result


def _box(value: Any, path: str) -> list[int | float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1000 for x in value)
        or not (value[0] < value[2] and value[1] < value[3])
    ):
        raise ValueError(f"{path}: expected a positive-area xyxy_1000 box")
    x1, y1, x2, y2 = value
    return [y1, x1, y2, x2]


def _reference_boxes(value: Any, path: str) -> list[list[int | float]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return [_box(box, f"{path}[{index}]") for index, box in enumerate(value)]
    return [_box(value, path)]


def adapt(prediction: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        raise ValueError("PE prediction is not an object")
    result = ordered_prediction(prediction)
    elements = result.pop("elements", None)
    if not isinstance(elements, dict) or set(elements) != {"background", "layout", "reference_edits"}:
        raise ValueError("PE elements schema changed")
    if not isinstance(result.get("normalized_edit_instruction"), str) or not result["normalized_edit_instruction"].strip():
        raise ValueError("normalized_edit_instruction is empty")
    layout = elements["layout"]
    groups = elements["reference_edits"]
    if not isinstance(layout, list) or not layout or not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("expected nonempty layout and one frame_0 group")
    for index, region in enumerate(layout):
        if not isinstance(region, dict) or "edit" in region:
            raise ValueError(f"layout[{index}] changed")
        region["bbox"] = _box(region.get("bbox"), f"layout[{index}].bbox")
    group = groups[0]
    if not isinstance(group, dict) or group.get("reference_id") != "frame_0":
        raise ValueError("only frame_0 is supported")
    edits = group.get("edit")
    removals = group.get("remove")
    if not isinstance(edits, list) or not isinstance(removals, list):
        raise ValueError("PE action cardinality changed")
    positions = {r["label"]: i for i, r in enumerate(layout)}
    if len(positions) != len(layout):
        raise ValueError("duplicate layout labels")
    previous = -1
    for index, action in enumerate(edits):
        if not isinstance(action, dict) or action.get("label") not in positions or positions[action["label"]] <= previous:
            raise ValueError(f"edit[{index}] does not match layout")
        previous = positions[action["label"]]
        instruction = action.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"edit[{index}] has no instruction")
        boxes = _reference_boxes(action.get("reference_bbox"), f"edit[{index}].reference_bbox")
        if len(boxes) > 1:
            raise ValueError(f"edit[{index}]: native intake cannot represent multiple boxes for one target/frame")
        if boxes:
            action["reference_bbox"] = boxes[0]
    native_removals = []
    for index, action in enumerate(removals):
        if not isinstance(action, dict):
            raise ValueError(f"remove[{index}] changed")
        instruction = action.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or instruction.strip().casefold() == "keep":
            raise ValueError(f"remove[{index}] has no valid instruction")
        boxes = _reference_boxes(action.get("reference_bbox"), f"remove[{index}].reference_bbox")
        if not boxes:
            raise ValueError(f"remove[{index}] has no Source box")
        native_removals.extend({"instruction": instruction, "reference_bbox": box} for box in boxes)
    group["remove"] = native_removals
    result["compositional_deconstruction"] = {
        "background": elements["background"],
        "elements": layout,
        "reference_edits": groups,
    }
    return result
