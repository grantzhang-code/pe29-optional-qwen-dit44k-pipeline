#!/usr/bin/env python3
"""Project packaged Image Edit V6 predictions into the requested JSON shape.

The input must already use normalized YXYX bboxes.  Valid predictions are
renamed and insertion-ordered without changing any semantic value.  Invalid
predictions are emitted as null so an unconverted XYXY payload can never be
mistaken for YXYX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT_VERSION = "image_edit_v6_reference_ordered_yxyx_v1"
SOURCE_ROOT_FIELDS = {
    "normalized_edit_instruction",
    "high_level_description",
    "style_description",
    "elements",
}
STYLE_FIELDS = {"aesthetics", "color_palette", "lighting", "medium"}
SOURCE_COMPOSITION_FIELDS = {"background", "layout", "remove"}
ELEMENT_REQUIRED_FIELDS = {
    "type",
    "label",
    "parent",
    "bbox",
    "desc",
    "edit",
    "color_palette",
}
ELEMENT_OPTIONAL_FIELDS = {"text", "entity"}
EDIT_FIELDS = {"instruction", "references"}
REFERENCE_FIELDS = {"reference_bbox", "reference_id"}
REMOVE_FIELDS = {"instruction", "references"}


def _canonical_bytes(value: Any, *, sort_keys: bool) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} fields differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _bbox_yxyx(value: Any, *, path: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path} must contain four coordinates")
    if any(type(coordinate) is not int for coordinate in value):
        raise ValueError(f"{path} coordinates must be integers")
    top, left, bottom, right = value
    if not (0 <= top < bottom <= 1000 and 0 <= left < right <= 1000):
        raise ValueError(f"{path} is outside normalized YXYX bounds: {value!r}")
    return list(value)


def _ordered_reference(value: Any, *, path: str) -> dict[str, Any]:
    source = _object(value, path=path)
    _exact_fields(source, REFERENCE_FIELDS, path=path)
    reference_id = source["reference_id"]
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError(f"{path}.reference_id must be a non-empty string")
    return {
        "reference_bbox": _bbox_yxyx(
            source["reference_bbox"], path=f"{path}.reference_bbox"
        ),
        "reference_id": reference_id,
    }


def _ordered_edit(value: Any, *, path: str) -> dict[str, Any]:
    source = _object(value, path=path)
    _exact_fields(source, EDIT_FIELDS, path=path)
    instruction = source["instruction"]
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{path}.instruction must be a non-empty string")
    references = [
        _ordered_reference(reference, path=f"{path}.references[{index}]")
        for index, reference in enumerate(
            _array(source["references"], path=f"{path}.references")
        )
    ]
    return {"instruction": instruction, "references": references}


def _ordered_element(value: Any, *, path: str) -> dict[str, Any]:
    source = _object(value, path=path)
    actual = set(source)
    missing = ELEMENT_REQUIRED_FIELDS - actual
    extra = actual - ELEMENT_REQUIRED_FIELDS - ELEMENT_OPTIONAL_FIELDS
    if missing or extra:
        raise ValueError(
            f"{path} fields differ: missing={sorted(missing)} extra={sorted(extra)}"
        )
    result: dict[str, Any] = {
        "type": source["type"],
        "label": source["label"],
        "parent": source["parent"],
        "bbox": _bbox_yxyx(source["bbox"], path=f"{path}.bbox"),
        "desc": source["desc"],
    }
    # These fields are valid V6 extensions but are absent from the user's
    # compact example.  Keep them deterministically between desc and edit.
    for optional_field in ("text", "entity"):
        if optional_field in source:
            result[optional_field] = source[optional_field]
    result["edit"] = _ordered_edit(source["edit"], path=f"{path}.edit")
    result["color_palette"] = source["color_palette"]
    return result


def _ordered_remove(value: Any, *, path: str) -> dict[str, Any]:
    source = _object(value, path=path)
    _exact_fields(source, REMOVE_FIELDS, path=path)
    instruction = source["instruction"]
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{path}.instruction must be a non-empty string")
    references = [
        _ordered_reference(reference, path=f"{path}.references[{index}]")
        for index, reference in enumerate(
            _array(source["references"], path=f"{path}.references")
        )
    ]
    return {"instruction": instruction, "references": references}


def ordered_prediction(value: Any, *, path: str = "prediction") -> dict[str, Any]:
    """Return one prediction with the exact requested key insertion order."""

    source = _object(value, path=path)
    _exact_fields(source, SOURCE_ROOT_FIELDS, path=path)
    style = _object(source["style_description"], path=f"{path}.style_description")
    _exact_fields(style, STYLE_FIELDS, path=f"{path}.style_description")
    composition = _object(source["elements"], path=f"{path}.elements")
    _exact_fields(
        composition, SOURCE_COMPOSITION_FIELDS, path=f"{path}.elements"
    )
    layout = _array(composition["layout"], path=f"{path}.elements.layout")
    removals = _array(composition["remove"], path=f"{path}.elements.remove")

    return {
        "normalized_edit_instruction": source["normalized_edit_instruction"],
        "high_level_description": source["high_level_description"],
        "style_description": {
            "aesthetics": style["aesthetics"],
            "color_palette": style["color_palette"],
            "lighting": style["lighting"],
            "medium": style["medium"],
        },
        "compositional_deconstruction": {
            "background": composition["background"],
            "elements": [
                _ordered_element(element, path=f"{path}.elements.layout[{index}]")
                for index, element in enumerate(layout)
            ],
            "remove": [
                _ordered_remove(removal, path=f"{path}.elements.remove[{index}]")
                for index, removal in enumerate(removals)
            ],
        },
    }


def restore_source_prediction(value: Any) -> dict[str, Any]:
    """Reverse the projection so callers can prove semantic equivalence."""

    ordered = _object(value, path="ordered_prediction")
    composition = _object(
        ordered["compositional_deconstruction"],
        path="ordered_prediction.compositional_deconstruction",
    )
    return {
        "normalized_edit_instruction": ordered["normalized_edit_instruction"],
        "high_level_description": ordered["high_level_description"],
        "style_description": dict(ordered["style_description"]),
        "elements": {
            "background": composition["background"],
            "layout": list(composition["elements"]),
            "remove": list(composition["remove"]),
        },
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(_canonical_bytes(row, sort_keys=False) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_distinct(paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input and output paths must be distinct")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_jsonl.resolve(strict=True)
    output_path = args.output_jsonl.resolve()
    summary_path = args.summary_json.resolve()
    success_path = args.success_json.resolve()
    _ensure_distinct([input_path, output_path, summary_path, success_path])

    source_sha256 = _file_sha256(input_path)
    if args.expected_input_sha256 and source_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"input sha256 mismatch: {source_sha256} != {args.expected_input_sha256}"
        )

    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_sample_ids: set[str] = set()
    projected_rows = 0
    skipped_invalid_rows = 0
    layout_elements = 0
    remove_records = 0
    layout_bboxes = 0
    reference_bboxes = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_row = _object(json.loads(line), path=f"line[{line_number}]")
            input_index = source_row.get("input_index")
            if type(input_index) is not int or input_index < 0:
                raise ValueError(f"line[{line_number}].input_index is invalid")
            if input_index in seen_indices:
                raise ValueError(f"duplicate input_index: {input_index}")
            seen_indices.add(input_index)
            sample_id = source_row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"line[{line_number}].sample_id is invalid")
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)

            valid = source_row.get("json_valid") is True and source_row.get(
                "schema_valid"
            ) is True
            prediction: dict[str, Any] | None
            projection_status: str
            if valid:
                if source_row.get("bbox_coordinate_order") != "yxyx_1000":
                    raise ValueError(
                        f"line[{line_number}] is not packaged as yxyx_1000"
                    )
                packaging = _object(
                    source_row.get("bbox_packaging"),
                    path=f"line[{line_number}].bbox_packaging",
                )
                if packaging.get("status") != "packaged":
                    raise ValueError(
                        f"line[{line_number}] has unexpected bbox packaging status"
                    )
                cleanup = _object(
                    source_row.get("hld_label_cleanup"),
                    path=f"line[{line_number}].hld_label_cleanup",
                )
                if cleanup.get("version") != args.expected_hld_postprocess_version:
                    raise ValueError(
                        f"line[{line_number}] has unexpected HLD cleanup version"
                    )
                if cleanup.get("remaining_invalid_numbered_label_occurrences") != 0:
                    raise ValueError(
                        f"line[{line_number}] has remaining invalid HLD labels"
                    )
                source_prediction = _object(
                    source_row.get("prediction"), path=f"line[{line_number}].prediction"
                )
                prediction = ordered_prediction(
                    source_prediction, path=f"line[{line_number}].prediction"
                )
                if restore_source_prediction(prediction) != source_prediction:
                    raise ValueError(
                        f"line[{line_number}] projection is not semantically reversible"
                    )
                composition = prediction["compositional_deconstruction"]
                layout_elements += len(composition["elements"])
                remove_records += len(composition["remove"])
                layout_bboxes += len(composition["elements"])
                reference_bboxes += sum(
                    len(element["edit"]["references"])
                    for element in composition["elements"]
                ) + sum(
                    len(removal["references"])
                    for removal in composition["remove"]
                )
                projected_rows += 1
                projection_status = "projected"
            else:
                prediction = None
                skipped_invalid_rows += 1
                projection_status = "skipped_invalid_prediction"

            row = {
                "input_index": input_index,
                "sample_id": sample_id,
                "instruction": source_row.get("instruction"),
                "reference_image": source_row.get("reference_image"),
                "target_image": source_row.get("target_image"),
                "prediction": prediction,
                "json_valid": source_row.get("json_valid"),
                "schema_valid": source_row.get("schema_valid"),
                "json_error": source_row.get("json_error"),
                "schema_error": source_row.get("schema_error"),
                "bbox_coordinate_order": "yxyx_1000",
                "target_schema": FORMAT_VERSION,
                "prompt_version": source_row.get("prompt_version"),
                "prompt_contract_sha256": source_row.get("prompt_contract_sha256"),
                "prompt_tokens": source_row.get("prompt_tokens"),
                "output_tokens": source_row.get("output_tokens"),
                "finish_reason": source_row.get("finish_reason"),
                "stop_reason": source_row.get("stop_reason"),
                "hld_label_cleanup": source_row.get("hld_label_cleanup"),
                "dataset_source": source_row.get("dataset_source"),
                "caption_projection": {
                    "version": FORMAT_VERSION,
                    "status": projection_status,
                    "source_target_schema": source_row.get("target_schema"),
                    "source_bbox_coordinate_order": source_row.get(
                        "bbox_coordinate_order"
                    ),
                    "semantic_roundtrip_verified": valid,
                },
            }
            rows.append(row)

    if len(rows) != args.expected_rows:
        raise ValueError(f"row count mismatch: {len(rows)} != {args.expected_rows}")
    if sorted(seen_indices) != list(range(args.expected_rows)):
        raise ValueError("input_index values are not the complete contiguous range")

    _atomic_write_jsonl(output_path, rows)
    output_sha256 = _file_sha256(output_path)
    summary = {
        "version": FORMAT_VERSION,
        "complete": True,
        "source": str(input_path),
        "source_sha256": source_sha256,
        "output": str(output_path),
        "output_sha256": output_sha256,
        "bbox_coordinate_order": "yxyx_1000",
        "rows": len(rows),
        "projected_rows": projected_rows,
        "skipped_invalid_rows": skipped_invalid_rows,
        "layout_elements": layout_elements,
        "remove_records": remove_records,
        "layout_bboxes_validated": layout_bboxes,
        "reference_bboxes_validated": reference_bboxes,
        "total_bboxes_validated": layout_bboxes + reference_bboxes,
        "prediction_field_order": [
            "normalized_edit_instruction",
            "high_level_description",
            "style_description",
            "compositional_deconstruction",
        ],
        "style_description_field_order": [
            "aesthetics",
            "color_palette",
            "lighting",
            "medium",
        ],
        "compositional_deconstruction_field_order": [
            "background",
            "elements",
            "remove",
        ],
        "element_field_order": [
            "type",
            "label",
            "parent",
            "bbox",
            "desc",
            "text (optional)",
            "entity (optional)",
            "edit",
            "color_palette",
        ],
        "edit_field_order": ["instruction", "references"],
        "reference_field_order": ["reference_bbox", "reference_id"],
        "semantic_roundtrip_verified_rows": projected_rows,
        "hld_postprocess_version": args.expected_hld_postprocess_version,
        "remaining_invalid_numbered_hld_labels": 0,
    }
    _atomic_write_json(summary_path, summary)
    success = {
        "version": FORMAT_VERSION,
        "complete": True,
        "rows": len(rows),
        "projected_rows": projected_rows,
        "skipped_invalid_rows": skipped_invalid_rows,
        "bbox_coordinate_order": "yxyx_1000",
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "summary_sha256": _file_sha256(summary_path),
        "semantic_roundtrip_verified": True,
        "hld_postprocess_verified": True,
    }
    _atomic_write_json(success_path, success)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--success-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument(
        "--expected-hld-postprocess-version",
        default="image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
