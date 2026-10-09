"""Package postprocessed image-edit V6 predictions from XYXY to YXYX."""

from __future__ import annotations

import copy
from typing import Any, Mapping


PACKAGING_VERSION = "image_edit_v6_prediction_xyxy_to_yxyx_v1"
SOURCE_BBOX_COORDINATE_ORDER = "xyxy_1000"
PACKAGED_BBOX_COORDINATE_ORDER = "yxyx_1000"
_BBOX_SCALE = 1000


class V6BBoxPackagingError(ValueError):
    """Raised when a V6 prediction cannot be packaged without ambiguity."""


def _xyxy_as_yxyx(value: Any, *, path: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise V6BBoxPackagingError(f"{path} must contain four integers")
    x0, y0, x1, y1 = value
    if not (
        0 <= x0 < x1 <= _BBOX_SCALE
        and 0 <= y0 < y1 <= _BBOX_SCALE
    ):
        raise V6BBoxPackagingError(
            f"{path} must be a positive XYXY/1000 bbox, got {value!r}"
        )
    return [y0, x0, y1, x1]


def _convert_references(value: Any, *, path: str) -> int:
    if not isinstance(value, list):
        raise V6BBoxPackagingError(f"{path} must be an array")
    converted = 0
    for index, reference in enumerate(value):
        reference_path = f"{path}[{index}]"
        if not isinstance(reference, dict):
            raise V6BBoxPackagingError(f"{reference_path} must be an object")
        reference["reference_bbox"] = _xyxy_as_yxyx(
            reference.get("reference_bbox"),
            path=f"{reference_path}.reference_bbox",
        )
        converted += 1
    return converted


def package_v6_prediction_yxyx(
    prediction: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deep-copy one direct-root V6 prediction and convert known bbox fields."""

    if not isinstance(prediction, Mapping):
        raise V6BBoxPackagingError("prediction must be an object")
    packaged = copy.deepcopy(dict(prediction))
    elements = packaged.get("elements")
    if not isinstance(elements, dict):
        raise V6BBoxPackagingError("prediction.elements must be an object")
    layout = elements.get("layout")
    remove = elements.get("remove")
    if not isinstance(layout, list) or not isinstance(remove, list):
        raise V6BBoxPackagingError(
            "prediction.elements.layout and prediction.elements.remove must be arrays"
        )

    layout_bbox_count = 0
    reference_bbox_count = 0
    for index, element in enumerate(layout):
        path = f"prediction.elements.layout[{index}]"
        if not isinstance(element, dict):
            raise V6BBoxPackagingError(f"{path} must be an object")
        element["bbox"] = _xyxy_as_yxyx(
            element.get("bbox"), path=f"{path}.bbox"
        )
        layout_bbox_count += 1
        edit = element.get("edit")
        if not isinstance(edit, dict):
            raise V6BBoxPackagingError(f"{path}.edit must be an object")
        reference_bbox_count += _convert_references(
            edit.get("references"), path=f"{path}.edit.references"
        )

    for index, removal in enumerate(remove):
        path = f"prediction.elements.remove[{index}]"
        if not isinstance(removal, dict):
            raise V6BBoxPackagingError(f"{path} must be an object")
        reference_bbox_count += _convert_references(
            removal.get("references"), path=f"{path}.references"
        )

    audit = {
        "version": PACKAGING_VERSION,
        "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
        "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
        "layout_bboxes_converted": layout_bbox_count,
        "reference_bboxes_converted": reference_bbox_count,
        "total_bboxes_converted": layout_bbox_count + reference_bbox_count,
    }
    return packaged, audit


def package_v6_inference_row_yxyx(
    row: Mapping[str, Any],
    *,
    expected_hld_postprocess_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Package one inference row after verifying the HLD cleanup boundary."""

    if not isinstance(row, Mapping):
        raise V6BBoxPackagingError("inference row must be an object")
    packaged_row = copy.deepcopy(dict(row))
    if row.get("schema_valid") is not True:
        audit = {
            "version": PACKAGING_VERSION,
            "status": "skipped_invalid_prediction",
            "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
            "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
            "total_bboxes_converted": 0,
        }
        packaged_row["bbox_packaging"] = audit
        packaged_row["bbox_coordinate_order"] = PACKAGED_BBOX_COORDINATE_ORDER
        return packaged_row, audit

    cleanup = row.get("hld_label_cleanup")
    if not isinstance(cleanup, Mapping):
        raise V6BBoxPackagingError(
            "schema-valid V6 row is missing hld_label_cleanup audit"
        )
    if cleanup.get("version") != expected_hld_postprocess_version:
        raise V6BBoxPackagingError(
            "HLD postprocess version mismatch: "
            f"{cleanup.get('version')!r} != {expected_hld_postprocess_version!r}"
        )
    if cleanup.get("remaining_invalid_numbered_label_occurrences") != 0:
        raise V6BBoxPackagingError("HLD postprocess left invalid numbered labels")

    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping):
        raise V6BBoxPackagingError("schema-valid row has no prediction object")
    packaged_prediction, audit = package_v6_prediction_yxyx(prediction)
    audit = {**audit, "status": "packaged"}
    packaged_row["prediction"] = packaged_prediction

    original_prediction = row.get("original_prediction")
    if original_prediction is not None:
        if not isinstance(original_prediction, Mapping):
            raise V6BBoxPackagingError("original_prediction must be an object")
        packaged_original, original_audit = package_v6_prediction_yxyx(
            original_prediction
        )
        packaged_row["original_prediction"] = packaged_original
        audit["original_prediction_bboxes_converted"] = original_audit[
            "total_bboxes_converted"
        ]

    packaged_row["bbox_packaging"] = audit
    packaged_row["bbox_coordinate_order"] = PACKAGED_BBOX_COORDINATE_ORDER
    return packaged_row, audit
