"""Source-only PE SFT dataset backed by node-local cached MDS shards.

The current prompt sees exactly one Source image, the raw editing instruction,
and the intended Target canvas size.  The size comes from an offline,
row-hash-bound sidecar; Target pixels and URI never enter the prompt or hot
DataLoader path.  The current reference-edits mode emits the complete Target
layout and keeps Source-to-Target provenance in the row's frozen
``elements.reference_edits`` structure.  The earlier sparse-region modes
remain available for their frozen training configurations.  Exact Source bytes
are embedded offline into 256 MiB MDS objects, so training reads large objects
sequentially and reuses a bounded node-local NVMe cache.

The prior source-only-v3 prompt remains supported for its frozen H200 config.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlsplit


_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        _as_list,
        _env_bool,
        data_stream_shard_context,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


_old_v6_library_only = os.environ.get("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY")
os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.image_edit_v6_full_json import (
        ANNOTATION_CONTRACT_SHA256,
        ANNOTATION_CONTRACT_VERSION,
        TARGET_SCHEMA,
        V6TargetValidationError,
        _MISSING,
        _bbox_order_is_xyxy,
        _nested_value,
        _warning_codes,
        canonical_v6_target as _canonical_v6_target_compact,
    )
finally:
    if _old_v6_library_only is None:
        os.environ.pop("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = _old_v6_library_only


LEGACY_PROMPT_VERSION = "image_edit_pe_sparse_edited_regions_source_only_v3"
LEGACY_PROMPT_CONTRACT_SHA256 = (
    "8c017fe7908046bf14827ca5e19532f9b2537b639a74e7acb0f66fa8dc8b1474"
)
PROMPT_VERSION = "image_edit_pe_sparse_edited_regions_target_size_v4"
# Checked-in prompt pin. Update this literal and PROMPT_VERSION together when
# SYSTEM_PROMPT or build_user_prompt changes.
PROMPT_CONTRACT_SHA256 = "be81ac83b4779d068fda638abe693f54ba7948d341b68bc633e670e2d981c610"
LABEL_PROJECTION_VERSION = "non_keep_layout_only_deleted_parent_to_root_v2"
REFERENCE_EDITS_PROMPT_VERSION = "image_edit_pe_full_layout_reference_edits_target_size_v7"
# Checked-in prompt pin. Update this literal and REFERENCE_EDITS_PROMPT_VERSION
# together when REFERENCE_EDITS_SYSTEM_PROMPT or
# build_reference_edits_user_prompt changes.
REFERENCE_EDITS_PROMPT_CONTRACT_SHA256 = (
    "24724be83895c5590219f09c38280a9edc55874cb4aab432bfeb1f95d04a27e0"
)
REFERENCE_EDITS_LABEL_PROJECTION_VERSION = "full_layout_reference_edits_exact_v1"
REFERENCE_EDITS_SOURCE_FORMAT_VERSION = "image-edit-v6-reference-edits-jsonl-v1"
REFERENCE_EDITS_SOURCE_SCHEMA_VERSION = "image-edit-v6-reference-edits-v1"
REFERENCE_EDITS_SUBSET_FORMAT_VERSION = (
    "image-edit-v6-pe-keep-ancestor-complement-jsonl-v1"
)
REFERENCE_EDITS_EXCLUSION_AUDIT_SHA256 = (
    "c482ea6de5a6fd90ec176176391c299b2adaf4b449dc7e60698d68922ac7aef9"
)
MDS_FORMAT_VERSION = "image-edit-pe-mds-original-source-v1"
MDS_ALGORITHM_VERSION = "image-edit-pe-jsonl-exact-source-to-mds-v1"
MDS_COLUMNS = {
    "row_json": "str",
    "source_image": "bytes",
    "source_sha256": "str",
}
EXPECTED_ROWS = 11_771_901
ABLATION_5M_ROWS = 5_000_000
REFERENCE_EDITS_ROWS = 4_372_300
SAMPLING_ALGORITHM_VERSION = "python-mt19937-global-srs-bitset-v1"
DEFAULT_DATASET_NAME = "image_edit_v6_pe_mds_qwen35"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CHANNEL = "image_edit_v6_pe_source_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_SIZE_SIDECAR_FORMAT_VERSION = "image-edit-pe-target-size-sidecar-v1"
TARGET_SIZE_SIDECAR_ALGORITHM_VERSION = (
    "image-edit-pe-target-size-from-frozen-row-parquet-metadata-or-target-header-v2"
)
TARGET_SIZE_SIDECAR_MAGIC = b"IEPTS001"
TARGET_SIZE_SIDECAR_HEADER = struct.Struct("<8sI")
TARGET_SIZE_SIDECAR_RECORD = struct.Struct("<II32s")
STREAM_ORDER_VERSION = "image-edit-pe-mds-rank-stream-v1"


SYSTEM_PROMPT = (
    "You infer the intended Target image from a Source image, an editing instruction, and "
    "the Target canvas size. "
    "Return only valid JSON describing the Target. In elements.layout, include only regions "
    'that undergo an actual edit; omit every region whose edit.instruction is "keep".'
)


REFERENCE_EDITS_SYSTEM_PROMPT = (
    "You infer the intended complete Target image from one Source image, a raw editing "
    "instruction, and the Target canvas size. Return exactly one valid JSON object describing "
    "the Target. Represent the complete Target region hierarchy in elements.layout, followed "
    "by elements.reference_edits for Source-to-Target provenance. elements.reference_edits "
    "contains exactly one frame_0 group whose edit entries align one-to-one and in order with "
    "elements.layout and whose remove entries describe Source content absent from the Target."
)


LEGACY_SYSTEM_PROMPT = (
    "You infer the intended Target image from a Source image and an editing instruction. "
    "Return only valid JSON describing the Target. In elements.layout, include only regions "
    'that undergo an actual edit; omit every region whose edit.instruction is "keep".'
)


def _encoded_instruction(instruction: str) -> str:
    """Encode untrusted instruction text without creating image placeholders."""

    encoded = json.dumps(instruction, ensure_ascii=False, allow_nan=False)
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e")


def build_user_prompt(
    instruction: str,
    target_size: tuple[int, int],
) -> str:
    """Build an Image2JSON-style prompt with Source pixels and Target size."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("editing instruction must be non-empty")
    if (
        not isinstance(target_size, tuple)
        or len(target_size) != 2
        or any(type(value) is not int or value <= 0 for value in target_size)
    ):
        raise ValueError("target_size must be a (positive width, positive height) tuple")
    target_width, target_height = target_size
    return (
        "<image>\n"
        "The image is Source `frame_0`. Apply the editing instruction to predict the "
        "complete Target image at the specified canvas size.\n"
        "Raw edit instruction (JSON):\n"
        f"{_encoded_instruction(instruction)}\n"
        "Target canvas size (width x height):\n"
        "<image_size>\n"
        f"{target_width}x{target_height}\n"
        "</image_size>\n"
        "Generate one JSON object for the intended final Target containing "
        "high_level_description, "
        "style_description, elements.background, elements.layout, elements.remove, and "
        "normalized_edit_instruction. The global descriptions, style, and background must "
        "describe the complete Target image. Include style_description.color_palette.\n\n"
        "elements.layout is a sparse list of edited Target regions. Include a region only "
        'when its edit.instruction is not exactly "keep"; never include unchanged regions. '
        "If no Target region is edited, use an empty layout list. Every included layout item "
        "must contain type, label, parent, bbox in xyxy_1000 coordinates normalized relative "
        "to the Target canvas, desc, color_palette, "
        "and edit. Its edit.instruction must state the actual change and must not be "
        '"keep". Preserve parent only when that parent region is also included; otherwise '
        'replace parent with "ROOT". When an included region depicts '
        "an identifiable named entity, include its name in the optional entity field. When "
        'exact visible text is expected for an included item of type "text", include it in '
        "the text field.\n\n"
        "For layout-item edit.references, use Source provenance with reference_id "
        '"frame_0"; Target-only additions may use empty references. Put explicitly removed '
        "Source identities in elements.remove; removals do not appear in elements.layout. "
        "Describe the intended final Target, not the unchanged Source. Return only valid JSON "
        "and no prose or Markdown."
    )


def build_reference_edits_user_prompt(
    instruction: str,
    target_size: tuple[int, int],
) -> str:
    """Build the complete-layout reference-edits TextImage2Json prompt."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("editing instruction must be non-empty")
    if (
        not isinstance(target_size, tuple)
        or len(target_size) != 2
        or any(type(value) is not int or value <= 0 for value in target_size)
    ):
        raise ValueError("target_size must be a (positive width, positive height) tuple")
    target_width, target_height = target_size
    return (
        "<image>\n"
        "The image is Source `frame_0`. Apply the editing instruction to predict the "
        "complete Target image at the specified canvas size.\n"
        "Raw edit instruction (JSON):\n"
        f"{_encoded_instruction(instruction)}\n"
        "Target canvas size (width x height):\n"
        "<image_size>\n"
        f"{target_width}x{target_height}\n"
        "</image_size>\n"
        "Generate one JSON object for the intended final Target containing "
        "high_level_description, style_description, elements.background, elements.layout, "
        "elements.reference_edits, and normalized_edit_instruction. The global descriptions, "
        "style, background, hierarchy, and inventory must describe the complete Target image. "
        "Include style_description.color_palette.\n\n"
        "elements.layout must contain every Target region, whether changed or unchanged, in "
        "parent-before-child order. Every layout item must contain type, label, parent, bbox "
        "in xyxy_1000 coordinates normalized relative to the Target canvas, desc, and "
        "color_palette; identifiable entities may include entity, and text regions may include "
        "their exact visible text.\n\n"
        "Place elements.reference_edits immediately after elements.layout. It must be a list "
        "containing exactly one object with reference_id `frame_0`, an edit list, and a remove "
        "list. The edit list must have exactly one entry for every layout item, in the same "
        "order. Each edit entry contains the corresponding layout label, an instruction, and, "
        "when Source provenance exists, reference_bbox immediately after instruction. Format "
        "reference_bbox as the four-integer Source xyxy_1000 list [x1, y1, x2, y2]. Use the "
        "exact string `keep` for an unchanged Source-grounded region. For a Target-only "
        "addition, the edit entry consists of label and instruction.\n\n"
        "The frame_0 remove list records Source content absent from the Target. Each remove "
        "entry contains an instruction followed by its required reference_bbox in the same "
        "[x1, y1, x2, y2] format. All descriptive fields, layout regions, and Target bboxes "
        "describe the intended final Target. Return only valid JSON and no prose or Markdown."
    )


def _build_legacy_user_prompt(instruction: str) -> str:
    """Reproduce the frozen source-only-v3 prompt exactly."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("editing instruction must be non-empty")
    return (
        "<image>\n"
        "The image is Source `frame_0`. Apply the editing instruction to predict the "
        "complete Target image.\n"
        "Raw edit instruction (JSON):\n"
        f"{_encoded_instruction(instruction)}\n"
        "Generate one JSON object for the intended final Target containing "
        "high_level_description, "
        "style_description, elements.background, elements.layout, elements.remove, and "
        "normalized_edit_instruction. The global descriptions, style, and background must "
        "describe the complete Target image. Include style_description.color_palette.\n\n"
        "elements.layout is a sparse list of edited Target regions. Include a region only "
        'when its edit.instruction is not exactly "keep"; never include unchanged regions. '
        "If no Target region is edited, use an empty layout list. Every included layout item "
        "must contain type, label, parent, bbox in xyxy_1000 coordinates, desc, color_palette, "
        "and edit. Its edit.instruction must state the actual change and must not be "
        '"keep". Preserve parent only when that parent region is also included; otherwise '
        'replace parent with "ROOT". When an included region depicts '
        "an identifiable named entity, include its name in the optional entity field. When "
        'exact visible text is expected for an included item of type "text", include it in '
        "the text field.\n\n"
        "For layout-item edit.references, use Source provenance with reference_id "
        '"frame_0"; Target-only additions may use empty references. Put explicitly removed '
        "Source identities in elements.remove; removals do not appear in elements.layout. "
        "Describe the intended final Target, not the unchanged Source. Return only valid JSON "
        "and no prose or Markdown."
    )


def _prompt_contract_sha256(
    *,
    prompt_version: str,
    system_prompt: str,
    user_prompt: str,
    label_projection_version: str = LABEL_PROJECTION_VERSION,
    assistant_json_serialization: str = (
        "validate complete final_json; retain layout items where "
        "edit.instruction!='keep'; replace deleted direct parent with ROOT; "
        "json.dumps(ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)"
    ),
) -> str:
    payload = json.dumps(
        {
            "prompt_version": prompt_version,
            "system_prompt": system_prompt,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "__ASSISTANT_TARGET__"},
            ],
            "images": ["__REFERENCE_IMAGE__"],
            "label_projection_version": label_projection_version,
            "assistant_json_serialization": assistant_json_serialization,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_contract_sha256() -> str:
    return _prompt_contract_sha256(
        prompt_version=PROMPT_VERSION,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(
            "__PROMPT_CONTRACT_INSTRUCTION__",
            (1234, 567),
        ),
    )


def _legacy_prompt_contract_sha256() -> str:
    return _prompt_contract_sha256(
        prompt_version=LEGACY_PROMPT_VERSION,
        system_prompt=LEGACY_SYSTEM_PROMPT,
        user_prompt=_build_legacy_user_prompt("__PROMPT_CONTRACT_INSTRUCTION__"),
    )


def reference_edits_prompt_contract_sha256() -> str:
    return _prompt_contract_sha256(
        prompt_version=REFERENCE_EDITS_PROMPT_VERSION,
        system_prompt=REFERENCE_EDITS_SYSTEM_PROMPT,
        user_prompt=build_reference_edits_user_prompt(
            "__PROMPT_CONTRACT_INSTRUCTION__",
            (1234, 567),
        ),
        label_projection_version=REFERENCE_EDITS_LABEL_PROJECTION_VERSION,
        assistant_json_serialization=(
            "validate exact full-layout reference_edits final_json; preserve all layout, edit, "
            "remove, reference_bbox records, and publication field order; "
            "json.dumps(ensure_ascii=False,sort_keys=False,separators=(',',':'),allow_nan=False)"
        ),
    )


def _prompt_uses_target_size(prompt_version: str) -> bool:
    return prompt_version in {PROMPT_VERSION, REFERENCE_EDITS_PROMPT_VERSION}


def _label_projection_for_prompt(prompt_version: str) -> str:
    if prompt_version == REFERENCE_EDITS_PROMPT_VERSION:
        return REFERENCE_EDITS_LABEL_PROJECTION_VERSION
    return LABEL_PROJECTION_VERSION


def canonical_v6_target(value: Any) -> str:
    """Apply the inherited strict V7 schema and compact serialization."""

    return _canonical_v6_target_compact(value)


def canonical_sparse_edit_target(value: Any) -> str:
    """Validate the complete label, then retain only non-``keep`` layout regions.

    When an edited region's original parent is omitted because it is unchanged,
    the projected parent is always ``ROOT``. We intentionally do not search for
    a retained grandparent.
    """

    target = json.loads(canonical_v6_target(value))
    layout = target["elements"]["layout"]
    edited_labels = {region["label"] for region in layout if region["edit"]["instruction"] != "keep"}
    sparse_layout: list[dict[str, Any]] = []
    for region in layout:
        if region["label"] not in edited_labels:
            continue
        if region["parent"] != "ROOT" and region["parent"] not in edited_labels:
            region["parent"] = "ROOT"
        sparse_layout.append(region)
    target["elements"]["layout"] = sparse_layout
    return json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reference_bbox_list(value: Any, path: str) -> list[list[int]]:
    """Normalize one-or-many reference boxes for inherited V6 validation."""

    boxes: list[Any]
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(type(coordinate) is int for coordinate in value)
    ):
        boxes = [value]
    elif isinstance(value, list) and value:
        boxes = value
    else:
        raise V6TargetValidationError(
            f"{path} must be one xyxy_1000 box or a non-empty list of boxes"
        )
    normalized: list[list[int]] = []
    for index, box in enumerate(boxes):
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(coordinate) is not int for coordinate in box)
        ):
            raise V6TargetValidationError(f"{path}[{index}] must contain four integers")
        normalized.append(list(box))
    return normalized


def canonical_reference_edits_target(value: Any) -> str:
    """Validate and serialize the complete-layout ``reference_edits`` label.

    The publication is a lossless structural rewrite of the inherited V6
    target. Reconstructing that target lets the frozen validator continue to
    enforce hierarchy, palettes, normalized bboxes, edit semantics, and remove
    semantics while this function additionally enforces the new field layout.
    """

    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant: {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise V6TargetValidationError("final_json is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise V6TargetValidationError("final_json must be an object")
    try:
        target = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise V6TargetValidationError("final_json is not JSON serializable") from exc

    elements = target.get("elements")
    if not isinstance(elements, Mapping):
        raise V6TargetValidationError("elements must be an object")
    if set(elements) != {"background", "layout", "reference_edits"}:
        raise V6TargetValidationError(
            "elements keys must be background, layout, and reference_edits"
        )
    layout = elements.get("layout")
    groups = elements.get("reference_edits")
    if not isinstance(layout, list) or not layout:
        raise V6TargetValidationError("elements.layout must be a non-empty array")
    if not isinstance(groups, list) or len(groups) != 1:
        raise V6TargetValidationError(
            "elements.reference_edits must contain exactly one frame_0 group"
        )
    group = groups[0]
    if not isinstance(group, Mapping) or set(group) != {"reference_id", "edit", "remove"}:
        raise V6TargetValidationError("the reference_edits group has invalid keys")
    if group.get("reference_id") != "frame_0":
        raise V6TargetValidationError("reference_edits[0].reference_id must be frame_0")
    edits = group.get("edit")
    removals = group.get("remove")
    if not isinstance(edits, list) or len(edits) != len(layout):
        raise V6TargetValidationError(
            "reference_edits[0].edit must align one-to-one with elements.layout"
        )
    if not isinstance(removals, list):
        raise V6TargetValidationError("reference_edits[0].remove must be an array")

    reconstructed_layout: list[dict[str, Any]] = []
    for index, (raw_region, raw_edit) in enumerate(zip(layout, edits, strict=True)):
        region_path = f"elements.layout[{index}]"
        edit_path = f"elements.reference_edits[0].edit[{index}]"
        if not isinstance(raw_region, Mapping) or "edit" in raw_region:
            raise V6TargetValidationError(f"{region_path} must not contain inline edit")
        if not isinstance(raw_edit, Mapping) or not {"label", "instruction"} <= set(raw_edit):
            raise V6TargetValidationError(f"{edit_path} must contain label and instruction")
        if not set(raw_edit) <= {"label", "instruction", "reference_bbox"}:
            raise V6TargetValidationError(f"{edit_path} has invalid keys")
        if raw_edit.get("label") != raw_region.get("label"):
            raise V6TargetValidationError(f"{edit_path}.label does not match layout order")
        references: list[dict[str, Any]] = []
        if "reference_bbox" in raw_edit:
            references = [
                {"reference_bbox": box, "reference_id": "frame_0"}
                for box in _reference_bbox_list(
                    raw_edit["reference_bbox"],
                    f"{edit_path}.reference_bbox",
                )
            ]
        region = dict(raw_region)
        region["edit"] = {
            "instruction": raw_edit.get("instruction"),
            "references": references,
        }
        reconstructed_layout.append(region)

    reconstructed_removals: list[dict[str, Any]] = []
    for index, raw_removal in enumerate(removals):
        path = f"elements.reference_edits[0].remove[{index}]"
        if not isinstance(raw_removal, Mapping) or set(raw_removal) != {
            "instruction",
            "reference_bbox",
        }:
            raise V6TargetValidationError(
                f"{path} must contain exactly instruction and reference_bbox"
            )
        references = [
            {"reference_bbox": box, "reference_id": "frame_0"}
            for box in _reference_bbox_list(
                raw_removal["reference_bbox"],
                f"{path}.reference_bbox",
            )
        ]
        reconstructed_removals.append(
            {"instruction": raw_removal.get("instruction"), "references": references}
        )

    reconstructed = dict(target)
    reconstructed["elements"] = {
        "background": elements.get("background"),
        "layout": reconstructed_layout,
        "remove": reconstructed_removals,
    }
    _canonical_v6_target_compact(reconstructed)
    return json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ImageEditV6PEMdsConfig:
    dataset_name: str
    val_dataset_name: str
    mds_remote: str
    mds_local_cache_dir: str
    mds_manifest_sha256: str
    mds_index_sha256: str
    source_manifest_sha256: str
    prompt_version: str = PROMPT_VERSION
    prompt_contract_sha256: str = PROMPT_CONTRACT_SHA256
    label_projection_version: str = LABEL_PROJECTION_VERSION
    source_format_version: str = ""
    source_schema_version: str = ""
    target_size_remote: str = ""
    target_size_manifest_sha256: str = ""
    target_size_local_cache_dir: str = ""
    expected_rows: int = EXPECTED_ROWS
    source_expected_rows: int = EXPECTED_ROWS
    sampling_algorithm: str = ""
    sampling_seed: int = 0
    sampling_bitset_sha256: str = ""
    cache_limit_bytes: int = 450 * 1024**3
    download_retry: int = 8
    download_timeout_seconds: int = 300
    instruction_field: str = "instruction"
    reference_image_field: str = "source_s3_uri"
    target_image_field: str = "target_s3_uri"
    target_json_field: str = "final_json"
    sample_id_field: str = "sample_id"
    status_field: str = "status"
    audit_valid_field: str = "call2_audit.valid"
    contract_version_field: str = "contract_version"
    contract_sha256_field: str = "contract_sha256"
    bbox_coordinate_order_field: str = "bbox_coordinate_order"
    allowed_statuses: tuple[str, ...] = ("validated", "manual_review_required")
    source_name: str = "image_edit_v6_pe_mds"
    rank_shard: bool = True
    repeat: bool = True
    shuffle_shards: bool = True
    sample_shuffle_buffer: int = 32
    seed: int = 43
    max_samples: int = 0
    reject_invalid_rows: bool = True
    minimum_physical_shards: int = 64


def streaming_resume_dataset_contract(config: ImageEditV6PEMdsConfig) -> dict[str, Any]:
    """Return the content/order contract used to authenticate loader replay.

    Cache locations and retry policy are intentionally excluded: they affect
    performance, not sample identity or order.  Everything that can change a
    row, its rank assignment, its shuffle order, prompt, or label is pinned.
    """

    return {
        "version": 1,
        "kind": "image-edit-v6-pe-mds-stream-contract",
        "stream_order_version": STREAM_ORDER_VERSION,
        "mds_format_version": MDS_FORMAT_VERSION,
        "mds_algorithm_version": MDS_ALGORITHM_VERSION,
        "dataset_name": config.dataset_name,
        "mds_remote": config.mds_remote,
        "mds_manifest_sha256": config.mds_manifest_sha256,
        "mds_index_sha256": config.mds_index_sha256,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_format_version": config.source_format_version,
        "source_schema_version": config.source_schema_version,
        "expected_rows": config.expected_rows,
        "source_expected_rows": config.source_expected_rows,
        "sampling_algorithm": config.sampling_algorithm,
        "sampling_seed": config.sampling_seed,
        "sampling_bitset_sha256": config.sampling_bitset_sha256,
        "target_size_remote": config.target_size_remote,
        "target_size_manifest_sha256": config.target_size_manifest_sha256,
        "target_size_sidecar_format_version": TARGET_SIZE_SIDECAR_FORMAT_VERSION,
        "target_size_sidecar_algorithm_version": TARGET_SIZE_SIDECAR_ALGORITHM_VERSION,
        "prompt_version": config.prompt_version,
        "prompt_contract_sha256": config.prompt_contract_sha256,
        "label_projection_version": config.label_projection_version,
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "instruction_field": config.instruction_field,
        "reference_image_field": config.reference_image_field,
        "target_image_field": config.target_image_field,
        "target_json_field": config.target_json_field,
        "sample_id_field": config.sample_id_field,
        "status_field": config.status_field,
        "audit_valid_field": config.audit_valid_field,
        "contract_version_field": config.contract_version_field,
        "contract_sha256_field": config.contract_sha256_field,
        "bbox_coordinate_order_field": config.bbox_coordinate_order_field,
        "allowed_statuses": list(config.allowed_statuses),
        "source_name": config.source_name,
        "rank_shard": config.rank_shard,
        "repeat": config.repeat,
        "shuffle_shards": config.shuffle_shards,
        "sample_shuffle_buffer": config.sample_shuffle_buffer,
        "seed": config.seed,
        "max_samples": config.max_samples,
        "reject_invalid_rows": config.reject_invalid_rows,
        "minimum_physical_shards": config.minimum_physical_shards,
    }


def make_row(
    source_row: Mapping[str, Any],
    *,
    source_image: bytes,
    target_size: tuple[int, int] | None = None,
    source_path: str,
    config: ImageEditV6PEMdsConfig,
) -> dict[str, Any] | None:
    def reject(reason: str, cause: Exception | None = None) -> None:
        if config.reject_invalid_rows:
            error = ValueError(f"{source_path}: invalid image-edit-v6-PE row: {reason}")
            if cause is not None:
                raise error from cause
            raise error
        return None

    status = _nested_value(source_row, config.status_field)
    if status is _MISSING or str(status) not in config.allowed_statuses:
        return reject("status is missing or is not allowed")
    audit = source_row.get("call2_audit")
    if not isinstance(audit, Mapping):
        return reject("call2_audit must be a mapping")
    if _nested_value(source_row, config.audit_valid_field) is not True:
        return reject("call2_audit.valid must be true")
    if audit.get("errors") != []:
        return reject("call2_audit.errors must be empty")
    if config.prompt_version == REFERENCE_EDITS_PROMPT_VERSION:
        membership = source_row.get("subset_membership")
        if (
            not isinstance(membership, Mapping)
            or membership.get("format_version")
            != REFERENCE_EDITS_SUBSET_FORMAT_VERSION
            or membership.get("exclusion_audit_manifest_sha256")
            != REFERENCE_EDITS_EXCLUSION_AUDIT_SHA256
            or type(membership.get("base_sampled_row_ordinal")) is not int
            or not 0 <= int(membership["base_sampled_row_ordinal"]) < ABLATION_5M_ROWS
            or type(membership.get("global_source_row_index")) is not int
            or int(membership["global_source_row_index"]) < 0
            or type(membership.get("source_part_index")) is not int
            or not 0 <= int(membership["source_part_index"]) < 136
            or type(membership.get("source_part_row_index")) is not int
            or int(membership["source_part_row_index"]) < 0
        ):
            return reject("reference-edits subset membership is invalid")
    else:
        if _nested_value(source_row, config.contract_version_field) != ANNOTATION_CONTRACT_VERSION:
            return reject("annotation contract version mismatch")
        if _nested_value(source_row, config.contract_sha256_field) != ANNOTATION_CONTRACT_SHA256:
            return reject("annotation contract SHA-256 mismatch")
        if not _bbox_order_is_xyxy(
            _nested_value(source_row, config.bbox_coordinate_order_field)
        ):
            return reject("bbox coordinate order is not XYXY")
    # These are redundant row-level publication hints.  Older/compacted rows in
    # the frozen 5M publication may omit them, while the publication manifest
    # pins the schema and the canonical target validator below enforces it.  If
    # either hint is present, still fail closed on an explicit contradiction.
    schema_valid = source_row.get("schema_valid", _MISSING)
    if schema_valid is not _MISSING and schema_valid is not True:
        return reject("schema_valid must be true when present")
    target_schema = source_row.get("target_schema", _MISSING)
    if target_schema is not _MISSING and target_schema != TARGET_SCHEMA:
        return reject("target_schema is not the direct-root XYXY contract")

    instruction = _nested_value(source_row, config.instruction_field)
    reference_uri = _nested_value(source_row, config.reference_image_field)
    target_uri = _nested_value(source_row, config.target_image_field)
    target_json = _nested_value(source_row, config.target_json_field)
    if (
        not isinstance(instruction, str)
        or not instruction.strip()
        or not isinstance(reference_uri, str)
        or not reference_uri.strip()
        or not isinstance(target_uri, str)
        or not target_uri.strip()
        or reference_uri.strip() == target_uri.strip()
        or target_json is _MISSING
    ):
        return reject("instruction/media/final_json binding is incomplete or invalid")
    if not isinstance(source_image, bytes) or not source_image:
        return reject("embedded Source image must contain non-empty bytes")
    try:
        if config.prompt_version == PROMPT_VERSION:
            if target_size is None:
                return reject("target-size-v4 prompt requires a row-bound Target canvas size")
            user_prompt = build_user_prompt(instruction, target_size)
            system_prompt = SYSTEM_PROMPT
            assistant_target = canonical_sparse_edit_target(target_json)
        elif config.prompt_version == REFERENCE_EDITS_PROMPT_VERSION:
            if target_size is None:
                return reject("reference-edits-v5 prompt requires a row-bound Target canvas size")
            user_prompt = build_reference_edits_user_prompt(instruction, target_size)
            system_prompt = REFERENCE_EDITS_SYSTEM_PROMPT
            assistant_target = canonical_reference_edits_target(target_json)
        elif config.prompt_version == LEGACY_PROMPT_VERSION:
            if target_size is not None:
                return reject("source-only-v3 prompt must not receive a Target canvas size")
            user_prompt = _build_legacy_user_prompt(instruction)
            system_prompt = LEGACY_SYSTEM_PROMPT
            assistant_target = canonical_sparse_edit_target(target_json)
        else:  # Defensive: load_runtime_config rejects this earlier.
            return reject(f"unsupported prompt version {config.prompt_version!r}")
    except (V6TargetValidationError, ValueError) as exc:
        return reject(
            f"final_json or instruction failed validation: {exc}",
            exc,
        )

    sample_id = _nested_value(source_row, config.sample_id_field)
    if sample_id is _MISSING:
        sample_id = ""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_target},
        ],
        "images": [{"bytes": source_image, "path": None}],
        "channel": CHANNEL,
        "meta": {
            "jsonl_path": source_path,
            "source_name": config.source_name,
            "sample_id": str(sample_id),
            "status": str(status),
            "reference_id": "frame_0",
            "reference_image": reference_uri.strip(),
            "target_image": target_uri.strip(),
            "prompt_version": config.prompt_version,
            "prompt_contract_sha256": config.prompt_contract_sha256,
            "target_width": target_size[0] if target_size is not None else 0,
            "target_height": target_size[1] if target_size is not None else 0,
            "label_projection_version": config.label_projection_version,
            "source_format_version": config.source_format_version,
            "source_schema_version": config.source_schema_version,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
            "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
            "bbox_coordinate_order": "xyxy",
            "review_warning_codes": _warning_codes(audit.get("warnings")),
        },
    }


def _config_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    config = document.get("image_edit_v6_pe_mds_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("image_edit_v6_pe_mds_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("IMAGE_EDIT_V6_PE_MDS_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _config_or_env(config: Mapping[str, Any], key: str, env_name: str) -> str:
    value = os.environ.get(env_name) or config.get(key) or ""
    return str(value).strip()


def _lowercase_sha256(value: str, field: str, *, allow_pending: bool = False) -> str:
    if allow_pending and value.startswith("PENDING_"):
        return value
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def _normalize_s3_prefix(uri: str) -> str:
    bucket, key = _parse_s3_uri(uri.rstrip("/"))
    return f"s3://{bucket}/{key}/"


def _s3_client(config: ImageEditV6PEMdsConfig):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
        config=Config(
            connect_timeout=5,
            read_timeout=config.download_timeout_seconds,
            tcp_keepalive=True,
            retries={"max_attempts": config.download_retry, "mode": "adaptive"},
            max_pool_connections=8,
        ),
    )


def _get_s3_bytes(
    client: Any,
    remote: str,
    relative: str,
    *,
    expected_sha256: str | None = None,
) -> bytes:
    bucket, prefix = _parse_s3_uri(remote.rstrip("/"))
    key = f"{prefix.rstrip('/')}/{relative.lstrip('/')}"
    payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if expected_sha256 is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"{remote}{relative} SHA-256 mismatch: {actual} != {expected_sha256}")
    return payload


def _validate_prompt_and_config(config: Mapping[str, Any], *, resolve_mds: bool) -> None:
    actual = prompt_contract_sha256()
    if actual != PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit-v6 PE prompt changed without updating its version/hash: "
            f"actual={actual} declared={PROMPT_CONTRACT_SHA256}"
        )
    legacy_actual = _legacy_prompt_contract_sha256()
    if legacy_actual != LEGACY_PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit-v6 PE legacy prompt implementation drifted: "
            f"actual={legacy_actual} declared={LEGACY_PROMPT_CONTRACT_SHA256}"
        )
    reference_edits_actual = reference_edits_prompt_contract_sha256()
    if reference_edits_actual != REFERENCE_EDITS_PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit-v6 PE reference-edits prompt changed without updating its version/hash: "
            f"actual={reference_edits_actual} "
            f"declared={REFERENCE_EDITS_PROMPT_CONTRACT_SHA256}"
        )
    prompt_version = str(config.get("prompt_version") or PROMPT_VERSION)
    prompt_hash = str(config.get("prompt_contract_sha256") or PROMPT_CONTRACT_SHA256)
    prompt_contracts = {
        PROMPT_VERSION: PROMPT_CONTRACT_SHA256,
        LEGACY_PROMPT_VERSION: LEGACY_PROMPT_CONTRACT_SHA256,
        REFERENCE_EDITS_PROMPT_VERSION: REFERENCE_EDITS_PROMPT_CONTRACT_SHA256,
    }
    if prompt_version not in prompt_contracts:
        raise ValueError("image_edit_v6_pe_mds_config prompt_version mismatch")
    if prompt_hash != prompt_contracts[prompt_version]:
        raise ValueError("image_edit_v6_pe_mds_config prompt hash mismatch")
    target_size_values = (
        str(config.get("target_size_remote") or "").strip(),
        str(config.get("target_size_manifest_sha256") or "").strip(),
        str(config.get("target_size_local_cache_dir") or "").strip(),
    )
    if _prompt_uses_target_size(prompt_version):
        if not all(target_size_values):
            raise ValueError("target-size prompt requires the complete target-size sidecar config")
        _normalize_s3_prefix(target_size_values[0])
        _lowercase_sha256(
            target_size_values[1],
            "target-size sidecar manifest SHA-256",
            allow_pending=not resolve_mds,
        )
        if not Path(target_size_values[2]).is_absolute():
            raise ValueError("target_size_local_cache_dir must be absolute")
    elif any(target_size_values):
        raise ValueError("source-only-v3 prompt must not declare a target-size sidecar")
    expected_label_projection = _label_projection_for_prompt(prompt_version)
    if (
        str(config.get("label_projection_version") or expected_label_projection)
        != expected_label_projection
    ):
        raise ValueError("image_edit_v6_pe_mds_config label projection mismatch")

    source_format_version = str(config.get("source_format_version") or "")
    source_schema_version = str(config.get("source_schema_version") or "")
    if prompt_version == REFERENCE_EDITS_PROMPT_VERSION:
        if (
            source_format_version != REFERENCE_EDITS_SOURCE_FORMAT_VERSION
            or source_schema_version != REFERENCE_EDITS_SOURCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "reference-edits-v5 requires the frozen source format and schema versions"
            )
    elif source_format_version or source_schema_version:
        raise ValueError("legacy PE prompts must not declare a reference-edits source schema")

    authoritative_fields = {
        "instruction_field": "instruction",
        "reference_image_field": "source_s3_uri",
        "target_image_field": "target_s3_uri",
        "target_json_field": "final_json",
        "sample_id_field": "sample_id",
        "status_field": "status",
        "audit_valid_field": "call2_audit.valid",
        "contract_version_field": "contract_version",
        "contract_sha256_field": "contract_sha256",
        "bbox_coordinate_order_field": "bbox_coordinate_order",
    }
    for key, expected in authoritative_fields.items():
        if str(config.get(key) or expected) != expected:
            raise ValueError(f"image_edit_v6_pe_mds_config.{key} must be {expected!r}")

    hashes = (
        (
            _config_or_env(
                config,
                "mds_manifest_sha256",
                "IMAGE_EDIT_PE_MDS_MANIFEST_SHA256",
            ),
            "MDS manifest SHA-256",
        ),
        (
            _config_or_env(
                config,
                "mds_index_sha256",
                "IMAGE_EDIT_PE_MDS_INDEX_SHA256",
            ),
            "MDS index SHA-256",
        ),
        (
            _config_or_env(
                config,
                "source_manifest_sha256",
                "IMAGE_EDIT_PE_SOURCE_MANIFEST_SHA256",
            ),
            "source JSONL manifest SHA-256",
        ),
    )
    for value, field in hashes:
        _lowercase_sha256(value, field, allow_pending=not resolve_mds)


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_mds: bool = True,
) -> ImageEditV6PEMdsConfig:
    config = _config_root(document if document is not None else _load_config_document())
    _validate_prompt_and_config(config, resolve_mds=resolve_mds)

    remote = _normalize_s3_prefix(
        str(
            config.get("mds_remote")
            or "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
            "iev6-full72-pe64-dedup-merged-20260827-v1-mds-source-v1/"
        )
    )
    allowed_statuses = tuple(str(value) for value in _as_list(config.get("allowed_statuses"))) or (
        "validated",
        "manual_review_required",
    )
    if allowed_statuses != ("validated", "manual_review_required"):
        raise ValueError("PE training requires both reviewed V6 status classes")

    prompt_version = str(config.get("prompt_version") or PROMPT_VERSION)
    expected_rows = int(config.get("expected_rows") or EXPECTED_ROWS)
    if expected_rows not in {EXPECTED_ROWS, ABLATION_5M_ROWS, REFERENCE_EDITS_ROWS}:
        raise ValueError(
            f"expected_rows must be the full publication ({EXPECTED_ROWS}) or "
            f"one of the exact subsets ({ABLATION_5M_ROWS}, {REFERENCE_EDITS_ROWS})"
        )
    default_source_rows = (
        REFERENCE_EDITS_ROWS
        if prompt_version == REFERENCE_EDITS_PROMPT_VERSION
        else EXPECTED_ROWS
    )
    source_expected_rows = int(config.get("source_expected_rows") or default_source_rows)
    if expected_rows == REFERENCE_EDITS_ROWS:
        if (
            prompt_version != REFERENCE_EDITS_PROMPT_VERSION
            or source_expected_rows != REFERENCE_EDITS_ROWS
        ):
            raise ValueError(
                "the reference-edits publication requires its v5 prompt and exact row count"
            )
    elif prompt_version == REFERENCE_EDITS_PROMPT_VERSION:
        raise ValueError("reference-edits-v5 may only read its frozen publication")
    elif source_expected_rows != EXPECTED_ROWS:
        raise ValueError(f"source_expected_rows must pin the merged publication at {EXPECTED_ROWS}")
    sampling_algorithm = str(config.get("sampling_algorithm") or "")
    sampling_seed = int(config.get("sampling_seed") or 0)
    sampling_bitset_sha256 = str(config.get("sampling_bitset_sha256") or "")
    if expected_rows == ABLATION_5M_ROWS:
        if (
            sampling_algorithm != SAMPLING_ALGORITHM_VERSION
            or sampling_seed < 0
            or SHA256_RE.fullmatch(sampling_bitset_sha256) is None
        ):
            raise ValueError("the 5M ablation subset requires its exact sampling algorithm and seed")
    elif sampling_algorithm or sampling_seed or sampling_bitset_sha256:
        raise ValueError("an unsampled PE publication must not declare a sampling contract")
    cache_limit_gib = int(config.get("cache_limit_gib") or 450)
    if cache_limit_gib < 32:
        raise ValueError("cache_limit_gib must be at least 32")
    minimum_physical_shards = int(config.get("minimum_physical_shards") or 64)
    if minimum_physical_shards < 64:
        raise ValueError("minimum_physical_shards must cover all 64 DP ranks")

    result = ImageEditV6PEMdsConfig(
        dataset_name=str(config.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        mds_remote=remote,
        mds_local_cache_dir=str(
            config.get("mds_local_cache_dir") or "/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/image-edit-v6-pe-mds"
        ),
        mds_manifest_sha256=_config_or_env(
            config,
            "mds_manifest_sha256",
            "IMAGE_EDIT_PE_MDS_MANIFEST_SHA256",
        ),
        mds_index_sha256=_config_or_env(
            config,
            "mds_index_sha256",
            "IMAGE_EDIT_PE_MDS_INDEX_SHA256",
        ),
        source_manifest_sha256=_config_or_env(
            config,
            "source_manifest_sha256",
            "IMAGE_EDIT_PE_SOURCE_MANIFEST_SHA256",
        ),
        prompt_version=prompt_version,
        prompt_contract_sha256=str(
            config.get("prompt_contract_sha256") or PROMPT_CONTRACT_SHA256
        ),
        label_projection_version=str(
            config.get("label_projection_version")
            or _label_projection_for_prompt(prompt_version)
        ),
        source_format_version=str(config.get("source_format_version") or ""),
        source_schema_version=str(config.get("source_schema_version") or ""),
        target_size_remote=(
            _normalize_s3_prefix(str(config["target_size_remote"]))
            if config.get("target_size_remote")
            else ""
        ),
        target_size_manifest_sha256=str(
            config.get("target_size_manifest_sha256") or ""
        ),
        target_size_local_cache_dir=str(
            config.get("target_size_local_cache_dir") or ""
        ),
        expected_rows=expected_rows,
        source_expected_rows=source_expected_rows,
        sampling_algorithm=sampling_algorithm,
        sampling_seed=sampling_seed,
        sampling_bitset_sha256=sampling_bitset_sha256,
        cache_limit_bytes=cache_limit_gib * 1024**3,
        download_retry=int(config.get("download_retry") or 8),
        download_timeout_seconds=int(config.get("download_timeout_seconds") or 300),
        allowed_statuses=allowed_statuses,
        source_name=str(config.get("source_name") or "image_edit_v6_pe_mds"),
        rank_shard=_env_bool(
            "IMAGE_EDIT_V6_PE_MDS_RANK_SHARD",
            bool(config.get("rank_shard", True)),
        ),
        repeat=_env_bool(
            "IMAGE_EDIT_V6_PE_MDS_REPEAT",
            bool(config.get("repeat", True)),
        ),
        shuffle_shards=_env_bool(
            "IMAGE_EDIT_V6_PE_MDS_SHUFFLE_SHARDS",
            bool(config.get("shuffle_shards", True)),
        ),
        sample_shuffle_buffer=int(
            os.environ.get("IMAGE_EDIT_V6_PE_MDS_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 32
        ),
        seed=int(os.environ.get("IMAGE_EDIT_V6_PE_MDS_SEED") or config.get("seed") or 43),
        max_samples=int(os.environ.get("IMAGE_EDIT_V6_PE_MDS_MAX_SAMPLES") or config.get("max_samples") or 0),
        reject_invalid_rows=_env_bool(
            "IMAGE_EDIT_V6_PE_MDS_REJECT_INVALID_ROWS",
            bool(config.get("reject_invalid_rows", True)),
        ),
        minimum_physical_shards=minimum_physical_shards,
    )
    if (
        result.download_retry < 1
        or result.download_timeout_seconds < 1
        or result.sample_shuffle_buffer < 1
        or result.max_samples < 0
    ):
        raise ValueError("MDS retry/timeout/buffer/sample limits are invalid")
    return result


def _validate_mds_metadata(
    config: ImageEditV6PEMdsConfig,
) -> list[dict[str, Any]]:
    client = _s3_client(config)
    manifest_payload = _get_s3_bytes(
        client,
        config.mds_remote,
        "manifest.json",
        expected_sha256=config.mds_manifest_sha256,
    )
    index_payload = _get_s3_bytes(
        client,
        config.mds_remote,
        "index.json",
        expected_sha256=config.mds_index_sha256,
    )
    success_payload = _get_s3_bytes(client, config.mds_remote, "_SUCCESS")
    manifest = json.loads(manifest_payload)
    index = json.loads(index_payload)
    success = json.loads(success_payload)

    source = manifest.get("source_dataset", {})
    source_descriptor = source.get("manifest", {}) if isinstance(source, Mapping) else {}
    format_spec = manifest.get("format", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("format_version") != MDS_FORMAT_VERSION
        or manifest.get("algorithm_version") != MDS_ALGORITHM_VERSION
        or _normalize_s3_prefix(str(manifest.get("output_prefix") or "")) != config.mds_remote
        or format_spec.get("columns") != MDS_COLUMNS
        or format_spec.get("compression") is not None
        or format_spec.get("image_transform") != "none"
        or format_spec.get("mds_version") != 2
        or index.get("version") != 2
        or not isinstance(index.get("shards"), list)
        or success.get("complete") is not True
        or success.get("format_version") != MDS_FORMAT_VERSION
        or success.get("manifest", {}).get("sha256") != config.mds_manifest_sha256
        or success.get("index", {}).get("sha256") != config.mds_index_sha256
        or source_descriptor.get("sha256") != config.source_manifest_sha256
        or success.get("source_manifest_sha256") != config.source_manifest_sha256
    ):
        raise ValueError("immutable PE MDS metadata contract mismatch")
    if config.source_format_version and (
        source.get("format_version") != config.source_format_version
        or source.get("schema_version") != config.source_schema_version
    ):
        raise ValueError("immutable PE MDS source schema contract mismatch")

    sampling = manifest.get("sampling")
    if config.expected_rows == ABLATION_5M_ROWS:
        selection_bitset = sampling.get("selection_bitset") if isinstance(sampling, Mapping) else None
        if (
            not isinstance(sampling, Mapping)
            or sampling.get("algorithm") != config.sampling_algorithm
            or int(sampling.get("population_rows") or 0) != config.source_expected_rows
            or int(sampling.get("sample_count") or 0) != config.expected_rows
            or int(sampling.get("seed", -1)) != config.sampling_seed
            or not isinstance(selection_bitset, Mapping)
            or selection_bitset.get("sha256") != config.sampling_bitset_sha256
            or int(success.get("sample_count") or 0) != config.expected_rows
            or int(success.get("sample_seed", -1)) != config.sampling_seed
            or success.get("sampling_bitset_sha256") != config.sampling_bitset_sha256
            or success.get("sampling_plan_sha256") != sampling.get("manifest_sha256")
        ):
            raise ValueError("immutable PE MDS 5M sampling contract mismatch")
    elif sampling is not None or int(success.get("sample_count") or 0) != 0:
        raise ValueError("full PE MDS publication unexpectedly declares sampling")

    shards: list[dict[str, Any]] = []
    seen_basenames: set[str] = set()
    total_rows = 0
    for shard_id, raw_shard in enumerate(index["shards"]):
        if not isinstance(raw_shard, Mapping):
            raise ValueError(f"MDS shard {shard_id} is not an object")
        raw_data = raw_shard.get("raw_data")
        names = raw_shard.get("column_names")
        encodings = raw_shard.get("column_encodings")
        sizes = raw_shard.get("column_sizes")
        samples = raw_shard.get("samples")
        if (
            raw_shard.get("version") != 2
            or raw_shard.get("format") != "mds"
            or raw_shard.get("compression") is not None
            or raw_shard.get("zip_data") is not None
            or not isinstance(raw_data, Mapping)
            or not isinstance(names, list)
            or not isinstance(encodings, list)
            or not isinstance(sizes, list)
            or len(names) != len(encodings)
            or len(names) != len(sizes)
            or dict(zip(names, encodings)) != MDS_COLUMNS
            or type(samples) is not int
            or samples < 1
        ):
            raise ValueError(f"MDS shard {shard_id} violates the frozen schema")
        basename = raw_data.get("basename")
        byte_count = raw_data.get("bytes")
        sha1 = raw_data.get("hashes", {}).get("sha1") if isinstance(raw_data.get("hashes"), Mapping) else None
        path = Path(str(basename))
        if (
            not isinstance(basename, str)
            or not basename
            or path.is_absolute()
            or ".." in path.parts
            or basename in seen_basenames
            or type(byte_count) is not int
            or byte_count < 1
            or not isinstance(sha1, str)
            or re.fullmatch(r"[0-9a-f]{40}", sha1) is None
        ):
            raise ValueError(f"MDS shard {shard_id} has invalid object identity")
        seen_basenames.add(basename)
        descriptor = dict(raw_shard)
        descriptor["_shard_id"] = shard_id
        shards.append(descriptor)
        total_rows += samples

    if (
        len(shards) < config.minimum_physical_shards
        or total_rows != config.expected_rows
        or int(source.get("rows") or 0) != config.expected_rows
        or int(source.get("rows_available") or config.expected_rows) != config.source_expected_rows
        or int(success.get("rows") or 0) != config.expected_rows
        or int(manifest.get("mds_shards") or 0) != len(shards)
    ):
        raise ValueError("PE MDS row or physical-shard coverage mismatch")
    if config.target_size_remote:
        _attach_target_size_sidecars(client, config, shards)
    return shards


def _attach_target_size_sidecars(
    client: Any,
    config: ImageEditV6PEMdsConfig,
    shards: Sequence[dict[str, Any]],
) -> None:
    """Validate and attach one immutable row-aligned size file per MDS shard."""

    manifest_payload = _get_s3_bytes(
        client,
        config.target_size_remote,
        "manifest.json",
        expected_sha256=config.target_size_manifest_sha256,
    )
    success_payload = _get_s3_bytes(client, config.target_size_remote, "_SUCCESS")
    manifest = json.loads(manifest_payload)
    success = json.loads(success_payload)
    outputs = manifest.get("outputs")
    source_mds = manifest.get("source_mds")
    record_format = manifest.get("record_format")
    if (
        manifest.get("complete") is not True
        or manifest.get("format_version") != TARGET_SIZE_SIDECAR_FORMAT_VERSION
        or manifest.get("algorithm_version") != TARGET_SIZE_SIDECAR_ALGORITHM_VERSION
        or not isinstance(outputs, list)
        or not isinstance(source_mds, Mapping)
        or source_mds.get("prefix") != config.mds_remote
        or source_mds.get("manifest_sha256") != config.mds_manifest_sha256
        or source_mds.get("index_sha256") != config.mds_index_sha256
        or int(source_mds.get("rows") or 0) != config.expected_rows
        or int(source_mds.get("shards") or 0) != len(shards)
        or not isinstance(record_format, Mapping)
        or record_format.get("magic_ascii") != TARGET_SIZE_SIDECAR_MAGIC.decode("ascii")
        or int(record_format.get("header_bytes") or 0) != TARGET_SIZE_SIDECAR_HEADER.size
        or int(record_format.get("record_bytes") or 0) != TARGET_SIZE_SIDECAR_RECORD.size
        or record_format.get("row_binding") != "sha256(exact MDS row_json UTF-8 bytes)"
        or success.get("complete") is not True
        or success.get("format_version") != TARGET_SIZE_SIDECAR_FORMAT_VERSION
        or success.get("manifest_sha256") != config.target_size_manifest_sha256
        or int(success.get("rows") or 0) != config.expected_rows
        or int(success.get("mds_shards") or 0) != len(shards)
    ):
        raise ValueError("immutable target-size sidecar metadata contract mismatch")

    by_mds_basename: dict[str, Mapping[str, Any]] = {}
    total_rows = 0
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("target-size sidecar output is not an object")
        mds_basename = output.get("mds_basename")
        relative_key = output.get("relative_key")
        samples = output.get("samples")
        byte_count = output.get("bytes")
        digest = output.get("sha256")
        if (
            not isinstance(mds_basename, str)
            or not mds_basename
            or mds_basename in by_mds_basename
            or not isinstance(relative_key, str)
            or not relative_key
            or Path(relative_key).is_absolute()
            or ".." in Path(relative_key).parts
            or type(samples) is not int
            or samples < 1
            or type(byte_count) is not int
            or byte_count != TARGET_SIZE_SIDECAR_HEADER.size + samples * TARGET_SIZE_SIDECAR_RECORD.size
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("target-size sidecar output identity is invalid")
        by_mds_basename[mds_basename] = output
        total_rows += samples

    if len(by_mds_basename) != len(shards) or total_rows != config.expected_rows:
        raise ValueError("target-size sidecar coverage does not match MDS")
    for shard in shards:
        raw_data = shard["raw_data"]
        basename = str(raw_data["basename"])
        output = by_mds_basename.get(basename)
        if (
            output is None
            or int(output["samples"]) != int(shard["samples"])
            or output.get("mds_sha1") != raw_data["hashes"]["sha1"]
        ):
            raise ValueError(f"target-size sidecar does not bind MDS shard {basename}")
        shard["_target_size_sidecar"] = dict(output)


def _safe_cache_path(root: Path, basename: str) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / basename).resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError(f"MDS shard path escapes local cache: {basename!r}")
    return target


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - pinned MDS integrity field.
    with path.open("rb") as source:
        while payload := source.read(8 * 1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def _locked_file(path: Path, *, shared: bool = False, blocking: bool = True):
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def lock():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return lock()


def _shard_lock_path(root: Path, basename: str) -> Path:
    lock_name = hashlib.sha256(basename.encode("utf-8")).hexdigest()
    return root / ".locks" / f"{lock_name}.lock"


def _evict_for_download(
    root: Path,
    *,
    required_bytes: int,
    limit_bytes: int,
    keep: Path,
) -> None:
    if required_bytes > limit_bytes:
        raise ValueError("one MDS shard exceeds the configured cache limit")
    with _locked_file(root / ".eviction.lock"):
        candidates: list[tuple[float, Path, int]] = []
        resident = 0
        for path in root.rglob("*.mds"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            resident += stat.st_size
            if path != keep:
                candidates.append((stat.st_mtime, path, stat.st_size))
        for _mtime, path, size in sorted(candidates):
            if resident + required_bytes <= limit_bytes:
                break
            basename = path.relative_to(root).as_posix()
            try:
                # A shared lock is held while a worker mmaps the shard. Skip
                # active files; unlinking an unopened path can otherwise race
                # the narrow handoff from downloader to reader.
                with _locked_file(_shard_lock_path(root, basename), blocking=False):
                    path.unlink()
            except (FileNotFoundError, BlockingIOError):
                continue
            resident -= size
        if resident + required_bytes > limit_bytes:
            raise RuntimeError("MDS cache has insufficient evictable space")


def _download_cached_shard(
    descriptor: Mapping[str, Any],
    config: ImageEditV6PEMdsConfig,
) -> Path:
    raw_data = descriptor["raw_data"]
    basename = str(raw_data["basename"])
    expected_bytes = int(raw_data["bytes"])
    expected_sha1 = str(raw_data["hashes"]["sha1"])
    root = Path(config.mds_local_cache_dir)
    target = _safe_cache_path(root, basename)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _shard_lock_path(root, basename)
    # A cache hit must coexist with an mmap reader (including a training
    # worker while the background prefetcher checks this shard). Only a
    # missing/incomplete shard needs the exclusive download lock.
    try:
        with _locked_file(lock_path, shared=True):
            if target.is_file() and target.stat().st_size == expected_bytes:
                os.utime(target, None)
                return target
    except FileNotFoundError:
        pass
    with _locked_file(lock_path):
        if target.is_file() and target.stat().st_size == expected_bytes:
            os.utime(target, None)
            return target
        if target.exists():
            target.unlink()
        _evict_for_download(
            root,
            required_bytes=expected_bytes,
            limit_bytes=config.cache_limit_bytes,
            keep=target,
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".partial",
            dir=target.parent,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        bucket, prefix = _parse_s3_uri(config.mds_remote.rstrip("/"))
        key = f"{prefix.rstrip('/')}/{basename}"
        started = time.monotonic()
        try:
            client = _s3_client(config)
            with temporary.open("wb") as output:
                client.download_fileobj(bucket, key, output)
            if temporary.stat().st_size != expected_bytes:
                raise IOError(
                    f"MDS shard size mismatch for {basename}: {temporary.stat().st_size} != {expected_bytes}"
                )
            actual_sha1 = _sha1_file(temporary)
            if actual_sha1 != expected_sha1:
                raise IOError(f"MDS shard SHA-1 mismatch for {basename}: {actual_sha1} != {expected_sha1}")
            os.replace(temporary, target)
            print(
                f"[dlms] cached MDS shard={basename} bytes={expected_bytes} "
                f"seconds={time.monotonic() - started:.2f}",
                flush=True,
            )
            return target
        finally:
            temporary.unlink(missing_ok=True)


def _target_size_lock_path(root: Path, relative_key: str) -> Path:
    lock_name = hashlib.sha256(relative_key.encode("utf-8")).hexdigest()
    return root / ".locks" / f"{lock_name}.lock"


def _download_target_size_sidecar(
    descriptor: Mapping[str, Any],
    config: ImageEditV6PEMdsConfig,
) -> Path:
    relative_key = str(descriptor["relative_key"])
    expected_bytes = int(descriptor["bytes"])
    expected_sha256 = str(descriptor["sha256"])
    root = Path(config.target_size_local_cache_dir)
    target = _safe_cache_path(root, relative_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _locked_file(_target_size_lock_path(root, relative_key)):
        if target.is_file() and target.stat().st_size == expected_bytes:
            if hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
                return target
            target.unlink()
        elif target.exists():
            target.unlink()
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".partial",
            dir=target.parent,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        bucket, prefix = _parse_s3_uri(config.target_size_remote.rstrip("/"))
        key = f"{prefix.rstrip('/')}/{relative_key.lstrip('/')}"
        try:
            payload = _s3_client(config).get_object(Bucket=bucket, Key=key)["Body"].read()
            if len(payload) != expected_bytes:
                raise IOError(
                    f"target-size sidecar byte count mismatch: {len(payload)} != {expected_bytes}"
                )
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise IOError(
                    "target-size sidecar SHA-256 mismatch: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            temporary.write_bytes(payload)
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)


def _decode_target_size_sidecar(
    payload: bytes,
    *,
    expected_samples: int,
) -> tuple[tuple[int, int, bytes], ...]:
    expected_bytes = (
        TARGET_SIZE_SIDECAR_HEADER.size
        + expected_samples * TARGET_SIZE_SIDECAR_RECORD.size
    )
    if len(payload) != expected_bytes:
        raise ValueError("target-size sidecar length does not match its MDS shard")
    magic, samples = TARGET_SIZE_SIDECAR_HEADER.unpack_from(payload, 0)
    if magic != TARGET_SIZE_SIDECAR_MAGIC or samples != expected_samples:
        raise ValueError("target-size sidecar header does not match its MDS shard")
    records: list[tuple[int, int, bytes]] = []
    offset = TARGET_SIZE_SIDECAR_HEADER.size
    for _ in range(samples):
        width, height, row_sha256 = TARGET_SIZE_SIDECAR_RECORD.unpack_from(payload, offset)
        if width <= 0 or height <= 0:
            raise ValueError("target-size sidecar contains a non-positive dimension")
        records.append((width, height, row_sha256))
        offset += TARGET_SIZE_SIDECAR_RECORD.size
    return tuple(records)


def _decode_mds_value(encoding: str, payload: bytes) -> Any:
    if encoding == "bytes":
        return payload
    if encoding == "str":
        return payload.decode("utf-8")
    if encoding == "int64":
        return struct.unpack("<q", payload)[0]
    if encoding == "float64":
        return struct.unpack("<d", payload)[0]
    raise ValueError(f"unsupported MDS encoding: {encoding}")


def _iter_local_mds(
    path: Path,
    descriptor: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    import mmap

    sample_count = int(descriptor["samples"])
    with path.open("rb") as source:
        payload = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            actual_count = struct.unpack_from("<I", payload, 0)[0]
            if actual_count != sample_count:
                raise ValueError("MDS sample count differs from root index")
            offsets = struct.unpack_from(f"<{sample_count + 1}I", payload, 4)
            if (
                offsets[0] <= 4 + 4 * (sample_count + 1)
                or offsets[-1] != len(payload)
                or any(left >= right for left, right in zip(offsets, offsets[1:]))
            ):
                raise ValueError("MDS shard offsets are corrupt")
            names = list(descriptor["column_names"])
            encodings = list(descriptor["column_encodings"])
            sizes = list(descriptor["column_sizes"])
            variable_count = sum(value is None for value in sizes)
            for sample_index in range(sample_count):
                encoded = payload[offsets[sample_index] : offsets[sample_index + 1]]
                variable_sizes = struct.unpack_from(f"<{variable_count}I", encoded, 0)
                variable_index = 0
                cursor = 4 * variable_count
                result: dict[str, Any] = {}
                for name, encoding, fixed_size in zip(names, encodings, sizes, strict=True):
                    if fixed_size is None:
                        size = variable_sizes[variable_index]
                        variable_index += 1
                    else:
                        size = int(fixed_size)
                    end = cursor + size
                    if end > len(encoded):
                        raise ValueError("MDS sample column exceeds its boundary")
                    result[name] = _decode_mds_value(encoding, encoded[cursor:end])
                    cursor = end
                if cursor != len(encoded):
                    raise ValueError("MDS sample has trailing bytes")
                yield result
        finally:
            payload.close()


def _iter_mds_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    config: ImageEditV6PEMdsConfig,
    repeat: bool,
    max_samples: int,
    stream_seed: int,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    pass_index = 0
    while True:
        pass_shards = list(shards)
        rng = random.Random(config.seed + stream_seed + pass_index)
        if config.shuffle_shards:
            rng.shuffle(pass_shards)
        pass_emitted = 0
        buffer: list[dict[str, Any]] = []
        for shard_position, descriptor in enumerate(pass_shards, start=1):
            basename = str(descriptor["raw_data"]["basename"])
            root = Path(config.mds_local_cache_dir)
            context = f"{config.mds_remote}{basename}"
            print(
                "[dlms] opening MDS shard "
                f"pass={pass_index} position={shard_position}/{len(pass_shards)} "
                f"shard_id={descriptor['_shard_id']} samples={descriptor['samples']} "
                f"path={basename}",
                flush=True,
            )
            target_size_records: tuple[tuple[int, int, bytes], ...] | None = None
            if _prompt_uses_target_size(config.prompt_version):
                sidecar_descriptor = descriptor.get("_target_size_sidecar")
                if not isinstance(sidecar_descriptor, Mapping):
                    raise ValueError(f"{context}: target-size-v4 sidecar is missing")
                sidecar_path = _download_target_size_sidecar(sidecar_descriptor, config)
                target_size_records = _decode_target_size_sidecar(
                    sidecar_path.read_bytes(),
                    expected_samples=int(descriptor["samples"]),
                )
            while True:
                local = _download_cached_shard(descriptor, config)
                try:
                    with _locked_file(_shard_lock_path(root, basename), shared=True):
                        if not local.is_file():
                            continue
                        for sample_index, sample in enumerate(_iter_local_mds(local, descriptor)):
                            if set(sample) != set(MDS_COLUMNS):
                                raise ValueError(f"{context}: MDS sample schema drifted")
                            row_json = sample["row_json"]
                            source_image = sample["source_image"]
                            source_sha256 = sample["source_sha256"]
                            if (
                                not isinstance(row_json, str)
                                or not isinstance(source_image, bytes)
                                or not isinstance(source_sha256, str)
                                or SHA256_RE.fullmatch(source_sha256) is None
                            ):
                                raise ValueError(f"{context}: MDS sample types are invalid")
                            target_size: tuple[int, int] | None = None
                            if target_size_records is not None:
                                target_width, target_height, expected_row_sha256 = (
                                    target_size_records[sample_index]
                                )
                                actual_row_sha256 = hashlib.sha256(
                                    row_json.encode("utf-8")
                                ).digest()
                                if actual_row_sha256 != expected_row_sha256:
                                    raise ValueError(
                                        f"{context}: target-size sidecar row binding mismatch "
                                        f"at sample {sample_index}"
                                    )
                                target_size = (target_width, target_height)
                            source_row = json.loads(row_json)
                            if not isinstance(source_row, Mapping):
                                raise ValueError(f"{context}: row_json is not an object")
                            row = make_row(
                                source_row,
                                source_image=source_image,
                                target_size=target_size,
                                source_path=context,
                                config=config,
                            )
                            if row is None:
                                continue
                            emitted += 1
                            pass_emitted += 1
                            buffer.append(row)
                            if len(buffer) >= config.sample_shuffle_buffer:
                                yield buffer.pop(rng.randrange(len(buffer)))
                            if max_samples > 0 and emitted >= max_samples:
                                while buffer:
                                    yield buffer.pop(rng.randrange(len(buffer)))
                                return
                        os.utime(local, None)
                    break
                except FileNotFoundError:
                    # Eviction won the downloader-to-reader handoff; refetch.
                    continue
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
        if not repeat:
            return
        if pass_emitted == 0:
            raise RuntimeError("no valid image-edit-v6 PE rows found in MDS shards")
        pass_index += 1


def _iter_mds_shard_payload(
    shard_payload: Mapping[str, Any],
    *,
    config: ImageEditV6PEMdsConfig,
    repeat: bool,
    max_samples: int,
    stream_seed: int,
) -> Iterator[dict[str, Any]]:
    """Keep one DP rank's physical shards in a single repeat domain.

    ``IterableDataset.from_generator`` treats every top-level list in
    ``gen_kwargs`` as a sharding axis and invokes the generator once per list
    element.  Passing ``shards`` as that top-level list while also repeating
    inside ``_iter_mds_shards`` makes the first physical shard repeat forever;
    later shards are never reached.  The mapping wrapper deliberately makes
    the rank-local shard list one logical HF shard, so an epoch covers every
    physical shard before repetition starts.
    """

    shards = shard_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("MDS shard payload must contain a non-empty shards list")
    if any(not isinstance(descriptor, Mapping) for descriptor in shards):
        raise ValueError("MDS shard payload contains a non-mapping descriptor")
    yield from _iter_mds_shards(
        shards,
        config=config,
        repeat=repeat,
        max_samples=max_samples,
        stream_seed=stream_seed,
    )


def _build_hf_stream(
    shards: Sequence[Mapping[str, Any]],
    *,
    config: ImageEditV6PEMdsConfig,
    repeat: bool,
    max_samples: int,
):
    from datasets import Features, IterableDataset, List, Value

    features = Features(
        {
            "messages": List(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ),
            "images": List(
                {
                    "bytes": Value("binary"),
                    "path": Value("string"),
                }
            ),
            "channel": Value("string"),
            "meta": {
                "jsonl_path": Value("string"),
                "source_name": Value("string"),
                "sample_id": Value("string"),
                "status": Value("string"),
                "reference_id": Value("string"),
                "reference_image": Value("string"),
                "target_image": Value("string"),
                "prompt_version": Value("string"),
                "prompt_contract_sha256": Value("string"),
                "label_projection_version": Value("string"),
                "source_format_version": Value("string"),
                "source_schema_version": Value("string"),
                "target_width": Value("int64"),
                "target_height": Value("int64"),
                "annotation_contract_version": Value("string"),
                "annotation_contract_sha256": Value("string"),
                "bbox_coordinate_order": Value("string"),
                "review_warning_codes": Value("string"),
            },
        }
    )
    # Keep the list nested. A top-level list is auto-split by Hugging Face and
    # is incompatible with the generator's whole-rank repeat loop: the first
    # physical shard would repeat forever and starve every later shard.
    return IterableDataset.from_generator(
        _iter_mds_shard_payload,
        gen_kwargs={
            "shard_payload": {"shards": list(shards)},
            "config": config,
            "repeat": repeat,
            "max_samples": max_samples,
            "stream_seed": sum(int(item["_shard_id"]) for item in shards),
        },
        features=features,
    )


class ImageEditV6PEMdsLoader:
    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        strict: bool = False,
        remove_unused_columns: bool = True,
        **_: Any,
    ) -> None:
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = streaming
        self.strict = strict
        self.remove_unused_columns = remove_unused_columns

    def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
        del dataset_syntax, use_hf
        config = load_runtime_config()
        shards = _validate_mds_metadata(config)
        data_rank, data_world_size = data_stream_shard_context()
        if not config.rank_shard:
            raise ValueError("PE MDS requires rank_shard=true")
        if data_world_size > len(shards):
            raise ValueError(
                "PE MDS requires at least one physical shard per DP rank: "
                f"shards={len(shards)} data_world_size={data_world_size}"
            )
        rank_shards = shards[data_rank::data_world_size]
        is_validation = bool(dataset_meta and dataset_meta.dataset_name == config.val_dataset_name)
        dataset = _build_hf_stream(
            rank_shards,
            config=config,
            repeat=False if is_validation else config.repeat,
            max_samples=1000 if is_validation else config.max_samples,
        )
        dataset = dataset_meta.preprocess_func(
            dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
        )
        if self.remove_unused_columns:
            from swift.dataset.preprocessor import RowPreprocessor

            dataset = RowPreprocessor.remove_useless_columns(dataset)
        # The rank-local DataLoader patch must apply a restored training cursor
        # only to the train stream.  Keep the marker on the final HF iterable;
        # IterablePackingDataset retains it through its ``dataset`` attribute.
        dataset._dlms_streaming_resume_role = (
            "validation" if is_validation else "train"
        )
        return dataset


def register_with_swift() -> tuple[str, str]:
    from dlms.checkpoint.streaming_resume import install_dataset_contract
    from swift.dataset import MessagesPreprocessor
    from swift.dataset.dataset_meta import BaseDatasetLoader, DatasetMeta
    from swift.dataset.register import register_dataset

    class SwiftImageEditV6PEMdsLoader(ImageEditV6PEMdsLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_mds=False)
    stream_contract_sha256 = install_dataset_contract(
        streaming_resume_dataset_contract(config)
    )
    for name, tags in (
        (
            config.dataset_name,
            [
                "vision",
                "single-image",
                "sft",
                "structured-json",
                "image-edit-v6",
                "pe",
                "mds",
            ],
        ),
        (
            config.val_dataset_name,
            [
                "vision",
                "single-image",
                "sft",
                "structured-json",
                "image-edit-v6",
                "pe",
                "mds",
                "validation",
            ],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImageEditV6PEMdsLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    print(
        f"[dlms] image-edit-v6 PE MDS enabled: dataset={config.dataset_name} "
        f"prompt={config.prompt_version} source_only=true "
        f"target_size={_prompt_uses_target_size(config.prompt_version)} remote={config.mds_remote} "
        f"stream_contract={stream_contract_sha256}",
        flush=True,
    )
    return config.dataset_name, config.val_dataset_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


if _swift_is_available() and not _env_bool("DLMS_IMAGE_EDIT_V6_PE_MDS_LIBRARY_ONLY", False):
    register_with_swift()
