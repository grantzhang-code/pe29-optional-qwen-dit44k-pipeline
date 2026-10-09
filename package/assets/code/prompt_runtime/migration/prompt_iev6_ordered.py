"""Source-only prompt and ordered Target plus reference edits label."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


PROMPT_VERSION = "iev6_source_target_size_ordered_reference_edits_v5"
PROMPT_SHA256 = "f67343e90f0b1015523d8decce85f742314de295a34295028892c63a48802e82"
PROJECTION_VERSION = "iev6_direct_root_description_with_reference_edits_v2"
SYSTEM_PROMPT = (
    "You are an image layout planner. Given one Source image, an edit instruction, "
    "and the Target canvas size, describe the complete intended Target as one JSON object, "
    "including unchanged regions. Use fields in this order: normalized_edit_instruction, "
    "high_level_description, style_description (aesthetics, lighting, medium, color_palette), "
    "elements (background, layout, reference_edits). List layout regions parent before child. Each region "
    "has type, label, parent, bbox in integer Target xyxy_1000 coordinates, optional text, "
    "desc, optional entity, then color_palette, in that order. After layout, provide one "
    "frame_0 reference_edits group with edit entries aligned one-to-one with layout, and "
    "remove entries for Source content absent from Target. Each edit has label, instruction, "
    "and optional Source reference_bbox; each remove has instruction and Source reference_bbox. "
    "Use 'keep' for unchanged Source-grounded regions. Each reference_bbox is one Source xyxy_1000 "
    "box [x1, y1, x2, y2], or a non-empty list of such boxes when one action covers multiple regions. "
    "Output only the JSON."
)
STYLE_FIELDS = ("aesthetics", "lighting", "medium", "color_palette")
REQUIRED_REGION_FIELDS = {"type", "label", "parent", "bbox", "desc", "color_palette"}
OPTIONAL_REGION_FIELDS = {"text", "entity"}


def _encoded_instruction(instruction: str) -> str:
    encoded = json.dumps(instruction, ensure_ascii=False, allow_nan=False)
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e")


def build_user_prompt(instruction: str, target_size: tuple[int, int]) -> str:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("raw instruction must be non-empty")
    if len(target_size) != 2 or any(type(n) is not int or n <= 0 for n in target_size):
        raise ValueError("target size must be positive width and height")
    width, height = target_size
    return (
        "<image>\nSource: frame_0\n"
        f"Edit instruction (JSON): {_encoded_instruction(instruction)}\n"
        f"Target canvas size (width x height): <image_size>{width}x{height}</image_size>\n"
        "Describe the complete Target as JSON. After elements.layout, output "
        "elements.reference_edits with exactly one frame_0 group: one edit entry per "
        "layout region in the same order, plus remove entries for Source content "
        "absent from Target. Return only JSON."
    )


def prompt_sha256() -> str:
    payload = json.dumps(
        {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT,
         "user": build_user_prompt("__PROMPT_CONTRACT_INSTRUCTION__", (123, 456))},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonempty(value: Any, path: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def ordered_target(final_json: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable annotation to exactly the requested training JSON."""
    if set(final_json) != {
        "high_level_description", "style_description",
        "compositional_deconstruction", "normalized_edit_instruction",
    }:
        raise ValueError("final_json has unexpected top-level fields")
    style = final_json["style_description"]
    composition = final_json["compositional_deconstruction"]
    if not isinstance(style, Mapping) or set(style) != set(STYLE_FIELDS):
        raise ValueError("style_description fields differ from contract")
    if not isinstance(composition, Mapping) or set(composition) != {
        "background", "elements", "reference_edits",
    }:
        raise ValueError("compositional_deconstruction fields differ from contract")
    layout = composition["elements"]
    if not isinstance(layout, list) or not layout:
        raise ValueError("elements layout must be a non-empty array")
    labels: set[str] = set()
    ordered_layout: list[dict[str, Any]] = []
    for index, region in enumerate(layout):
        if not isinstance(region, Mapping):
            raise ValueError(f"layout[{index}] is not an object")
        keys = set(region)
        if not REQUIRED_REGION_FIELDS <= keys or not keys <= REQUIRED_REGION_FIELDS | OPTIONAL_REGION_FIELDS:
            raise ValueError(f"layout[{index}] has unexpected fields")
        label = _nonempty(region["label"], f"layout[{index}].label")
        parent = _nonempty(region["parent"], f"layout[{index}].parent")
        if label in labels or (parent != "ROOT" and parent not in labels):
            raise ValueError(f"layout[{index}] label or parent is invalid")
        bbox = region["bbox"]
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(n) is not int for n in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000)):
            raise ValueError(f"layout[{index}] bbox is not xyxy_1000")
        fields = ["type", "label", "parent", "bbox"]
        if "text" in region:
            fields.append("text")
        fields.append("desc")
        if "entity" in region:
            fields.append("entity")
        fields.append("color_palette")
        ordered_layout.append({key: region[key] for key in fields})
        labels.add(label)
    groups = composition["reference_edits"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("reference_edits must contain exactly one frame_0 group")
    group = groups[0]
    if (not isinstance(group, Mapping) or set(group) != {"reference_id", "edit", "remove"}
            or group["reference_id"] != "frame_0"):
        raise ValueError("reference_edits group differs from frame_0 contract")
    edits, removals = group["edit"], group["remove"]
    if not isinstance(edits, list) or len(edits) != len(ordered_layout) or not isinstance(removals, list):
        raise ValueError("reference_edits must align one-to-one with layout")

    def source_bbox(value: Any, path: str) -> list[int] | list[list[int]]:
        """Preserve the published single- or multi-box Source provenance."""
        if not isinstance(value, list) or not value:
            raise ValueError(f"{path} must be one or more Source xyxy_1000 boxes")
        if len(value) == 4 and all(type(n) is int for n in value):
            boxes = [value]
        else:
            boxes = value
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(n) is not int for n in box)
                    or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
                raise ValueError(f"{path} must be one or more Source xyxy_1000 boxes")
        return value

    ordered_edits = []
    for index, (edit, region) in enumerate(zip(edits, ordered_layout, strict=True)):
        if (not isinstance(edit, Mapping) or set(edit) not in
                ({"label", "instruction"}, {"label", "instruction", "reference_bbox"})
                or edit["label"] != region["label"]):
            raise ValueError(f"reference_edits.edit[{index}] differs from layout")
        ordered_edit = {"label": edit["label"],
                        "instruction": _nonempty(edit["instruction"], f"reference_edits.edit[{index}].instruction")}
        if "reference_bbox" in edit:
            ordered_edit["reference_bbox"] = source_bbox(
                edit["reference_bbox"], f"reference_edits.edit[{index}].reference_bbox")
        ordered_edits.append(ordered_edit)
    ordered_removals = []
    for index, removal in enumerate(removals):
        if not isinstance(removal, Mapping) or set(removal) != {"instruction", "reference_bbox"}:
            raise ValueError(f"reference_edits.remove[{index}] differs from contract")
        ordered_removals.append({
            "instruction": _nonempty(removal["instruction"], f"reference_edits.remove[{index}].instruction"),
            "reference_bbox": source_bbox(removal["reference_bbox"], f"reference_edits.remove[{index}].reference_bbox"),
        })
    ordered_references = [{"reference_id": "frame_0", "edit": ordered_edits, "remove": ordered_removals}]

    return {
        "normalized_edit_instruction": _nonempty(
            final_json["normalized_edit_instruction"], "normalized_edit_instruction"),
        "high_level_description": _nonempty(
            final_json["high_level_description"], "high_level_description"),
        "style_description": {key: style[key] for key in STYLE_FIELDS},
        "elements": {"background": composition["background"], "layout": ordered_layout,
                     "reference_edits": ordered_references},
    }
