"""Two-image SFT contract for the direct-root image-edit-v6 Target JSON.

The authoritative annotation row is the immutable-projected ``final_json``
published after image-edit-v6 Call 2.  This loader supplies Source ``frame_0``,
Target, and the raw user instruction to Qwen3.5, then supervises the complete
direct-root V6 object.  The normalized instruction remains an assistant label;
putting it in the user prompt would leak the target.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        _as_list,
        _env_bool,
        _expand_paths,
        _read_uri_bytes,
        _s3_client,
        _split_s3_uri,
        data_stream_shard_context,
        iter_jsonl,
        shard_paths,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


PROMPT_VERSION = "image_edit_v6_direct_root_xyxy_v7_qwen35_v7"
# Checked-in prompt pin. Update this literal and PROMPT_VERSION together when
# SYSTEM_PROMPT or build_user_prompt changes.
PROMPT_CONTRACT_SHA256 = "7984173abd05fe72ea188ff4065db1305c49df1c41c01f18fc3a284046299f6f"
ANNOTATION_CONTRACT_VERSION = "image_edit_v6_edit_evidence_remove_repair_xy_v7"
ANNOTATION_CONTRACT_SHA256 = (
    "121b58cb71123e2c2ce78850455d208b626d313792768f5bb8d260beb936156f"
)
TARGET_SCHEMA = "image_edit_v6_direct_root_xyxy"
MANIFEST_FORMAT = "image_edit_v6_training_manifest_v1"
ROW_FORMAT = "image_edit_v6_training_row_v1"
PENDING_PUBLICATION = "PENDING_AFTER_CALL2_PUBLICATION"
DEFAULT_DATASET_NAME = "image_edit_v6_full_json_qwen35"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CHANNEL = "image_edit_v6_full_json"


SYSTEM_PROMPT = (
    "You produce a complete structured image-edit description from ordered Source and Target "
    "images plus a raw edit instruction. Return only valid JSON in the requested full "
    "image-edit-v6 schema."
)


def _encoded_instruction(instruction: str) -> str:
    """Encode untrusted text without creating extra Qwen image placeholders."""

    encoded = json.dumps(instruction, ensure_ascii=False, allow_nan=False)
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e")


def build_user_prompt(instruction: str) -> str:
    """Build the ordered, injection-safe two-image user prompt."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must be non-empty")
    return (
        "<image>\n"
        "The first image is Source `frame_0`.\n"
        "<image>\n"
        "The second image is Target.\n"
        "Raw edit instruction (JSON):\n"
        f"{_encoded_instruction(instruction)}\n"
        "Generate the complete direct-root JSON containing high_level_description, "
        "style_description, elements.background, elements.layout, elements.remove, and "
        "normalized_edit_instruction. Include style_description.color_palette. Every layout "
        "item must contain type, label, parent, bbox in xyxy_1000 coordinates, desc, "
        "color_palette, and edit. When a layout item depicts an identifiable named entity, "
        "include its name in the optional entity field. When exact visible text is available "
        'for an item of type "text", include it in the text field.\n\n'
        "Use Target pixels for the complete Target inventory and all descriptions, style, "
        "hierarchy, palettes, and bboxes. Compare both images for observed regional changes; "
        "the raw instruction guides attention but does not prove success. Regional edits "
        "describe the observed Source-to-Target pair. Assign each visible change to the "
        "smallest affected Target region; keep unaffected ancestors, children, siblings, "
        "carriers, nearby objects, and background unchanged.\n\n"
        "Type is one of coarse, medium, fine, text, face, or hand. Labels are unique numbered "
        "angle-bracket noun phrases; parent is ROOT or an earlier label. Every palette is "
        "nonempty uppercase #RRGGBB. Every Target bbox and Source reference_bbox contains "
        "integer [x0,y0,x1,y1] coordinates in 0–1000.\n\n"
        "Every edit contains exactly instruction and references. Nonempty references use "
        "same-granularity Source provenance containing exactly reference_bbox and reference_id "
        '"frame_0"; the instruction describes the observed change or no change. Empty '
        "references identify a Target-only addition; describe that addition with a natural, "
        "executable instruction.\n\n"
        "Every remove item contains exactly instruction and references with nonempty Source "
        "references. Its instruction directs removal and states "
        "that the Source identity is absent in Target. Use remove only for an explicitly "
        "requested, locally bounded Source identity proven absent in a comparable Target view. "
        "Crop, occlusion, movement, replacement, viewpoint or framing change, cutout or "
        "masking, uncertainty, and wholesale background or context replacement are not "
        "removals. A visible Target surface revealed or repaired after removal remains an "
        "ordinary layout item; its edit references the removed Source and describes the "
        "visible reconstruction, never unchanged or a provenance-free addition.\n\n"
        "For normalized_edit_instruction, the raw instruction defines the intended edit and "
        "the image pair grounds it. Write one executable instruction, normally 35–75 words, "
        "anchored to "
        "<frame>0</frame> or a concrete Source identity. Resolve pronouns, repeated objects, "
        "spatial anchors, exact text and numbers, identities, and relations. Preserve every "
        "explicit raw constraint even when Target violates it; treat the visible violation as a "
        "failure artifact, not an intended edit. Exclude unrelated Target artifacts and "
        "unintended identity, text, or value drift."
    )


def prompt_contract_sha256() -> str:
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_user_prompt("__PROMPT_CONTRACT_INSTRUCTION__"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


STYLE_KEYS = frozenset({"aesthetics", "lighting", "medium", "color_palette"})
ROOT_KEYS = frozenset(
    {
        "high_level_description",
        "style_description",
        "elements",
        "normalized_edit_instruction",
    }
)
ELEMENT_KEYS = frozenset({"background", "layout", "remove"})
REGION_REQUIRED_KEYS = frozenset(
    {"type", "label", "parent", "bbox", "desc", "color_palette", "edit"}
)
REGION_OPTIONAL_KEYS = frozenset({"text", "entity"})
REGION_TYPES = frozenset({"coarse", "medium", "fine", "text", "face", "hand"})
EDIT_KEYS = frozenset({"instruction", "references"})
REFERENCE_KEYS = frozenset({"reference_bbox", "reference_id"})
FORBIDDEN_KEYS = frozenset(
    {"operation", "structured_description", "normalized_edit_instruction_prompt"}
)
LABEL_RE = re.compile(r"^<[^<>\r\n]+ [1-9][0-9]*>$")
COLOR_RE = re.compile(r"^#[0-9A-F]{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()


class V6TargetValidationError(ValueError):
    """The assistant label violates the frozen image-edit-v6 target shape."""


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V6TargetValidationError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise V6TargetValidationError(f"{path} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise V6TargetValidationError(
            f"{path} keys must be {sorted(expected)}, got {sorted(actual)}"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V6TargetValidationError(f"{path} must be a non-empty string")
    return value


def _bbox(value: Any, path: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise V6TargetValidationError(f"{path} must contain four integers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise V6TargetValidationError(f"{path} is outside normalized XYXY bounds")


def _palette(value: Any, path: str) -> None:
    colors = _array(value, path)
    if not colors:
        raise V6TargetValidationError(f"{path} must be non-empty")
    for index, color in enumerate(colors):
        if not isinstance(color, str) or COLOR_RE.fullmatch(color) is None:
            raise V6TargetValidationError(
                f"{path}[{index}] must be uppercase #RRGGBB"
            )


def _references(value: Any, path: str) -> list[Any]:
    references = _array(value, path)
    seen: set[str] = set()
    for index, raw_reference in enumerate(references):
        reference_path = f"{path}[{index}]"
        reference = _object(raw_reference, reference_path)
        _exact_keys(reference, set(REFERENCE_KEYS), reference_path)
        if reference.get("reference_id") != "frame_0":
            raise V6TargetValidationError(
                f"{reference_path}.reference_id must be frame_0"
            )
        _bbox(reference.get("reference_bbox"), f"{reference_path}.reference_bbox")
        encoded = json.dumps(reference, sort_keys=True, separators=(",", ":"))
        if encoded in seen:
            raise V6TargetValidationError(f"{reference_path} duplicates Source provenance")
        seen.add(encoded)
    return references


def _layout_edit(value: Any, path: str) -> None:
    edit = _object(value, path)
    _exact_keys(edit, set(EDIT_KEYS), path)
    instruction = _text(edit.get("instruction"), f"{path}.instruction")
    references = _references(edit.get("references"), f"{path}.references")
    lower = instruction.lower()
    if "unchanged" in lower and not references:
        raise V6TargetValidationError(
            f"{path}.instruction cannot classify a Target-only addition as unchanged"
        )


def _remove_edit(value: Any, path: str) -> None:
    edit = _object(value, path)
    _exact_keys(edit, set(EDIT_KEYS), path)
    instruction = _text(edit.get("instruction"), f"{path}.instruction")
    references = _references(edit.get("references"), f"{path}.references")
    lower = instruction.lower()
    if not references:
        raise V6TargetValidationError(f"{path}.references must be non-empty")
    if re.search(r"\b(remove|delete|erase|eliminate)\b", lower) is None:
        raise V6TargetValidationError(f"{path}.instruction must direct removal")
    if "target" not in lower or "unchanged" in lower:
        raise V6TargetValidationError(
            f"{path}.instruction must state Target absence and cannot be unchanged"
        )


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(FORBIDDEN_KEYS & set(value)) or any(
            _contains_forbidden_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def validate_v6_target(value: Any) -> Mapping[str, Any]:
    target = _object(value, "final_json")
    _exact_keys(target, set(ROOT_KEYS), "final_json")
    if _contains_forbidden_key(target):
        raise V6TargetValidationError("final_json contains a forbidden schema key")
    _text(target.get("high_level_description"), "high_level_description")
    _text(target.get("normalized_edit_instruction"), "normalized_edit_instruction")

    style = _object(target.get("style_description"), "style_description")
    _exact_keys(style, set(STYLE_KEYS), "style_description")
    for field in ("aesthetics", "lighting", "medium"):
        _text(style.get(field), f"style_description.{field}")
    _palette(style.get("color_palette"), "style_description.color_palette")

    elements = _object(target.get("elements"), "elements")
    _exact_keys(elements, set(ELEMENT_KEYS), "elements")
    _text(elements.get("background"), "elements.background")
    layout = _array(elements.get("layout"), "elements.layout")
    if not layout:
        raise V6TargetValidationError("elements.layout must be non-empty")
    labels: list[str] = []
    for index, raw_region in enumerate(layout):
        path = f"elements.layout[{index}]"
        region = _object(raw_region, path)
        actual_keys = set(region)
        if not REGION_REQUIRED_KEYS <= actual_keys or not actual_keys <= (
            REGION_REQUIRED_KEYS | REGION_OPTIONAL_KEYS
        ):
            raise V6TargetValidationError(f"{path} has invalid keys: {sorted(actual_keys)}")
        if region.get("type") not in REGION_TYPES:
            raise V6TargetValidationError(f"{path}.type is invalid: {region.get('type')!r}")
        label = _text(region.get("label"), f"{path}.label")
        if LABEL_RE.fullmatch(label) is None or label in labels:
            raise V6TargetValidationError(f"{path}.label is invalid or duplicated")
        parent = _text(region.get("parent"), f"{path}.parent")
        if parent != "ROOT" and parent not in labels:
            raise V6TargetValidationError(f"{path}.parent must name an earlier region")
        _bbox(region.get("bbox"), f"{path}.bbox")
        _text(region.get("desc"), f"{path}.desc")
        _palette(region.get("color_palette"), f"{path}.color_palette")
        for field in REGION_OPTIONAL_KEYS & actual_keys:
            _text(region.get(field), f"{path}.{field}")
        _layout_edit(region.get("edit"), f"{path}.edit")
        labels.append(label)

    seen_removals: set[str] = set()
    for index, raw_removal in enumerate(_array(elements.get("remove"), "elements.remove")):
        path = f"elements.remove[{index}]"
        _remove_edit(raw_removal, path)
        encoded = json.dumps(raw_removal, sort_keys=True, separators=(",", ":"))
        if encoded in seen_removals:
            raise V6TargetValidationError(f"{path} duplicates a removal")
        seen_removals.add(encoded)
    return target


def canonical_v6_target(value: Any) -> str:
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
    validated = validate_v6_target(value)
    try:
        return json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V6TargetValidationError("final_json is not JSON serializable") from exc


@dataclass(frozen=True)
class ImageEditV6FullJsonConfig:
    dataset_name: str
    val_dataset_name: str
    paths: tuple[str, ...]
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
    allowed_statuses: tuple[str, ...] = ("validated",)
    source_name: str = "image_edit_v6_full_json"
    val_path_count: int = 0
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 2048
    seed: int = 42
    max_samples: int = 0
    reject_invalid_rows: bool = False
    minimum_physical_shards: int = 128
    shard_identities: tuple[tuple[str, str, str, int, int], ...] = ()
    required_source_stage: str = ""
    required_source_partitions: int = 0
    minimum_shards_per_source_partition: int = 1
    exclusion_policy: str = ""
    exclusion_sample_ids_sha256: str = ""
    sequence_length_audit_max_length: int = 0
    sequence_length_audit_model_manifest_sha256: str = ""
    prompt_version: str = PROMPT_VERSION
    prompt_contract_sha256: str = PROMPT_CONTRACT_SHA256


def _nested_value(row: Mapping[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _warning_codes(value: Any) -> str:
    if not isinstance(value, list):
        return "[]"
    codes = [
        str(item.get("code") or "")
        for item in value
        if isinstance(item, Mapping) and item.get("code")
    ]
    return json.dumps(codes, separators=(",", ":"))


def _bbox_order_is_xyxy(value: Any) -> bool:
    if value == "xyxy":
        return True
    if not isinstance(value, Mapping):
        return False
    return all(
        value.get(field) == "xyxy"
        for field in ("target_layout", "reference_bbox", "remove_reference_bbox")
    )


def make_row(
    source_row: Mapping[str, Any],
    *,
    source_path: str,
    config: ImageEditV6FullJsonConfig,
) -> dict[str, Any] | None:
    def reject(reason: str, cause: Exception | None = None) -> None:
        if config.reject_invalid_rows:
            error = ValueError(f"{source_path}: invalid image-edit-v6 row: {reason}")
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
    errors = audit.get("errors")
    warnings = audit.get("warnings")
    manual_review = str(status) == "manual_review_required"
    if errors != []:
        return reject("call2_audit.errors must be empty")
    if audit.get("manual_review_required") is not manual_review:
        return reject("status and call2_audit.manual_review_required disagree")
    if not isinstance(warnings, list) or bool(warnings) is not manual_review:
        return reject("status and call2_audit.warnings disagree")
    if _nested_value(source_row, config.contract_version_field) != ANNOTATION_CONTRACT_VERSION:
        return reject("annotation contract version mismatch")
    if _nested_value(source_row, config.contract_sha256_field) != ANNOTATION_CONTRACT_SHA256:
        return reject("annotation contract SHA-256 mismatch")
    if not _bbox_order_is_xyxy(
        _nested_value(source_row, config.bbox_coordinate_order_field)
    ):
        return reject("bbox coordinate order is not XYXY")

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
    try:
        assistant_target = canonical_v6_target(target_json)
        user_prompt = build_user_prompt(instruction)
    except (V6TargetValidationError, ValueError) as exc:
        return reject("final_json or instruction failed validation", exc)

    sample_id = _nested_value(source_row, config.sample_id_field)
    if sample_id is _MISSING:
        sample_id = ""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_target},
        ],
        "images": [reference_uri.strip(), target_uri.strip()],
        "channel": CHANNEL,
        "meta": {
            "jsonl_path": source_path,
            "source_name": config.source_name,
            "sample_id": str(sample_id),
            "status": str(status),
            "reference_id": "frame_0",
            "reference_image": reference_uri.strip(),
            "target_image": target_uri.strip(),
            "prompt_version": PROMPT_VERSION,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
            "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
            "bbox_coordinate_order": "xyxy",
            "review_warning_codes": _warning_codes(warnings),
        },
    }


def _config_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    config = document.get("image_edit_v6_full_json_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("image_edit_v6_full_json_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("IMAGE_EDIT_V6_FULL_JSON_CONFIG") or os.environ.get(
        "JOB_CONFIG_YAML"
    )
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _manifest_documents(
    config: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any], str]]:
    manifests = _as_list(
        config.get("paths_manifests")
        or config.get("paths_manifest")
        or config.get("manifests")
        or config.get("manifest")
    )
    result: list[tuple[str, Mapping[str, Any], str]] = []
    for raw_uri in manifests:
        uri = str(raw_uri)
        manifest_version_id = str(config.get("manifest_version_id") or "")
        if uri.startswith("s3://") and "manifest_version_id" in config:
            if not manifest_version_id or manifest_version_id == PENDING_PUBLICATION:
                raise ValueError(
                    "manifest_version_id is pending; pin it after Call-2 publication"
                )
            bucket, key = _split_s3_uri(uri)
            raw = _s3_client().get_object(
                Bucket=bucket, Key=key, VersionId=manifest_version_id
            )["Body"].read()
        else:
            raw = _read_uri_bytes(uri)
        try:
            document = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{uri}: V6 manifest must be a JSON object") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"{uri}: V6 manifest must be a mapping")
        result.append((uri, document, hashlib.sha256(raw).hexdigest()))
    return result


def _validate_contracts(
    config: Mapping[str, Any],
    manifests: Sequence[tuple[str, Mapping[str, Any], str]],
    allowed_statuses: Sequence[str],
    paths: Sequence[str],
    minimum_physical_shards: int,
    required_source_stage: str,
    required_source_partitions: int,
    minimum_shards_per_source_partition: int,
    exclusion_policy: str,
    exclusion_sample_ids_sha256: str,
    sequence_length_audit_max_length: int,
    sequence_length_audit_model_manifest_sha256: str,
) -> None:
    actual = prompt_contract_sha256()
    if actual != PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit-v6 prompt changed without updating PROMPT_VERSION and "
            f"PROMPT_CONTRACT_SHA256: actual={actual} declared={PROMPT_CONTRACT_SHA256}"
        )
    if "prompt_version" in config or "prompt_contract_sha256" in config:
        if (
            str(config.get("prompt_version") or "") != PROMPT_VERSION
            or str(config.get("prompt_contract_sha256") or "")
            != PROMPT_CONTRACT_SHA256
        ):
            raise ValueError("image_edit_v6_full_json_config prompt contract mismatch")
    if "annotation_contract_version" in config or "annotation_contract_sha256" in config:
        if (
            str(config.get("annotation_contract_version") or "")
            != ANNOTATION_CONTRACT_VERSION
            or str(config.get("annotation_contract_sha256") or "")
            != ANNOTATION_CONTRACT_SHA256
        ):
            raise ValueError("image_edit_v6_full_json_config annotation contract mismatch")
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
    for key, expected_value in authoritative_fields.items():
        if str(config.get(key) or expected_value) != expected_value:
            raise ValueError(
                f"image_edit_v6_full_json_config.{key} must be {expected_value!r}"
            )

    if manifests and "manifest_sha256" in config:
        if len(manifests) != 1:
            raise ValueError("immutable V6 publication supports exactly one manifest")
        declared_manifest_sha256 = str(config.get("manifest_sha256") or "")
        if SHA256_RE.fullmatch(declared_manifest_sha256) is None:
            raise ValueError(
                "manifest_sha256 is pending; publish Call-2 data and pin its canonical SHA-256"
            )
        if manifests[0][2] != declared_manifest_sha256:
            raise ValueError(
                f"{manifests[0][0]}: manifest SHA-256 mismatch: "
                f"actual={manifests[0][2]} expected={declared_manifest_sha256}"
            )

    for uri, document, _manifest_digest in manifests:
        if document.get("format_version") != MANIFEST_FORMAT or document.get("complete") is not True:
            raise ValueError(f"{uri}: V6 training manifest is incomplete or incompatible")
        if "required_total_rows" in config:
            required_total_rows = int(config.get("required_total_rows") or 0)
            if required_total_rows < 1:
                raise ValueError(
                    "required_total_rows is pending; pin it after Call-2 publication"
                )
            if int(document.get("total_rows") or 0) != required_total_rows:
                raise ValueError(f"{uri}: total_rows does not match the pinned config")
        if "required_status_counts" in config:
            required_status_counts = config.get("required_status_counts")
            actual_status_counts = document.get("status_counts")
            if (
                not isinstance(required_status_counts, Mapping)
                or not required_status_counts
                or dict(actual_status_counts or {}) != dict(required_status_counts)
                or sum(int(value) for value in required_status_counts.values())
                != int(document.get("total_rows") or 0)
            ):
                raise ValueError(
                    f"{uri}: status_counts do not match the pinned publication"
                )
        contract = document.get("dataset_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"{uri}: dataset_contract must be a mapping")
        expected = {
            "output_format": "prepared",
            "row_format_version": ROW_FORMAT,
            "runtime_loader": "dlms/datasets/image_edit_v6_full_json.py",
            "target_schema": TARGET_SCHEMA,
            "target_json_field": "final_json",
            "instruction_field": "instruction",
            "reference_image_field": "source_s3_uri",
            "target_image_field": "target_s3_uri",
            "audit_valid_field": "call2_audit.valid",
            "image_order": ["reference frame_0", "target"],
            "selection": "authoritative_call2_final_json",
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
            "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        }
        for key, expected_value in expected.items():
            if contract.get(key) != expected_value:
                raise ValueError(
                    f"{uri}: dataset_contract.{key}={contract.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        if tuple(str(value) for value in _as_list(contract.get("allowed_statuses"))) != tuple(
            allowed_statuses
        ):
            raise ValueError(f"{uri}: allowed_statuses do not match the training config")
        if not _bbox_order_is_xyxy(contract.get("bbox_coordinate_order")):
            raise ValueError(f"{uri}: bbox coordinate contract must be XYXY")
        if required_source_stage and contract.get("source_stage") != required_source_stage:
            raise ValueError(
                f"{uri}: dataset_contract.source_stage must be {required_source_stage!r}"
            )
        if exclusion_policy:
            exclusion = contract.get("exclusion")
            if not isinstance(exclusion, Mapping):
                raise ValueError(f"{uri}: dataset_contract.exclusion must be a mapping")
            if (
                exclusion.get("policy") != exclusion_policy
                or exclusion.get("sample_ids_sha256")
                != exclusion_sample_ids_sha256
                or exclusion.get("overlap_rows") != 0
            ):
                raise ValueError(f"{uri}: evaluation exclusion contract mismatch")
        declared_shards = int(document.get("physical_shard_count") or 0)
        if declared_shards < minimum_physical_shards:
            raise ValueError(
                f"{uri}: physical_shard_count={declared_shards} is below "
                f"minimum {minimum_physical_shards} for DP128 rank-local streaming"
            )
        if required_source_partitions:
            source_partitions = document.get("source_partitions")
            if not isinstance(source_partitions, list):
                raise ValueError(f"{uri}: source_partitions must be an array")
            expected_indices = list(range(required_source_partitions))
            actual_indices: list[int] = []
            source_partition_rows: dict[int, int] = {}
            for item in source_partitions:
                if not isinstance(item, Mapping):
                    raise ValueError(f"{uri}: source partition record must be a mapping")
                index = int(item.get("partition_index", -1))
                actual_indices.append(index)
                input_rows = int(item.get("input_rows") or 0)
                excluded_rows = int(item.get("excluded_rows") or 0)
                rejected_rows = int(item.get("rejected_rows") or 0)
                output_rows = int(item.get("output_rows") or 0)
                if (
                    item.get("stage") != required_source_stage
                    or item.get("complete") is not True
                    or input_rows < 1
                    or excluded_rows < 0
                    or rejected_rows < 0
                    or output_rows < 1
                    or input_rows != excluded_rows + rejected_rows + output_rows
                    or SHA256_RE.fullmatch(str(item.get("sha256") or "")) is None
                    or int(item.get("bytes") or 0) < 1
                    or str(item.get("version_id") or "") in {"", "null"}
                    or not str(item.get("uri") or "").startswith("s3://")
                ):
                    raise ValueError(f"{uri}: source partition {index} is incomplete")
                source_partition_rows[index] = output_rows
            if sorted(actual_indices) != expected_indices:
                raise ValueError(
                    f"{uri}: source partitions must be exactly 0.."
                    f"{required_source_partitions - 1}"
                )

            document_paths = [str(value) for value in _as_list(document.get("paths"))]
            required_path_prefix = str(config.get("required_path_prefix") or "")
            if (
                len(document_paths) != len(set(document_paths))
                or any(not path.startswith("s3://") for path in document_paths)
                or (
                    required_path_prefix
                    and any(not path.startswith(required_path_prefix) for path in document_paths)
                )
            ):
                raise ValueError(f"{uri}: output shard paths are not unique/pinned S3 paths")
            shard_records = document.get("shards")
            if not isinstance(shard_records, list) or len(shard_records) != declared_shards:
                raise ValueError(f"{uri}: shards do not match physical_shard_count")
            if any(not isinstance(item, Mapping) for item in shard_records):
                raise ValueError(f"{uri}: output shard record must be a mapping")
            if [str(item.get("path") or "") for item in shard_records] != document_paths:
                raise ValueError(f"{uri}: paths must exactly match ordered shard records")
            part_indexes: dict[int, list[int]] = {
                index: [] for index in expected_indices
            }
            part_counts: dict[int, set[int]] = {
                index: set() for index in expected_indices
            }
            output_rows = 0
            output_rows_by_partition = {
                index: 0 for index in expected_indices
            }
            for item in shard_records:
                partition = int(item.get("source_partition", -1))
                rows = int(item.get("rows") or 0)
                if (
                    partition not in part_indexes
                    or rows < 1
                    or SHA256_RE.fullmatch(str(item.get("sha256") or "")) is None
                    or int(item.get("bytes") or 0) < 1
                    or str(item.get("version_id") or "") in {"", "null"}
                ):
                    raise ValueError(f"{uri}: invalid output shard record")
                part_indexes[partition].append(int(item.get("part_index", -1)))
                part_counts[partition].add(int(item.get("part_count") or 0))
                output_rows += rows
                output_rows_by_partition[partition] += rows
            for partition in expected_indices:
                indexes = sorted(part_indexes[partition])
                counts = part_counts[partition]
                if (
                    len(indexes) < minimum_shards_per_source_partition
                    or len(counts) != 1
                    or counts != {len(indexes)}
                    or indexes != list(range(len(indexes)))
                ):
                    raise ValueError(
                        f"{uri}: source partition {partition} output shards are incomplete"
                    )
                if output_rows_by_partition[partition] != source_partition_rows[partition]:
                    raise ValueError(
                        f"{uri}: source partition {partition} row conservation failed"
                    )
            if output_rows != int(document.get("total_rows") or 0):
                raise ValueError(f"{uri}: output shard rows do not match total_rows")
            sample_id_audit = document.get("sample_id_audit")
            if (
                not isinstance(sample_id_audit, Mapping)
                or sample_id_audit.get("complete") is not True
                or sample_id_audit.get("checked_rows") != output_rows
                or sample_id_audit.get("unique_sample_ids") != output_rows
                or sample_id_audit.get("duplicate_rows") != 0
                or SHA256_RE.fullmatch(
                    str(sample_id_audit.get("ordered_sample_ids_sha256") or "")
                )
                is None
            ):
                raise ValueError(f"{uri}: sample_id uniqueness audit is incomplete")
        if sequence_length_audit_max_length:
            audit = document.get("sequence_length_audit")
            if not isinstance(audit, Mapping) or audit.get("complete") is not True:
                raise ValueError(f"{uri}: complete sequence_length_audit is required")
            total_rows = int(document.get("total_rows") or 0)
            expected_audit = {
                "max_length": sequence_length_audit_max_length,
                "max_pixels": int(
                    config.get("sequence_length_audit_max_pixels") or 0
                ),
                "image_max_token_num": int(
                    config.get("sequence_length_audit_image_max_token_num") or 0
                ),
                "model_type": str(
                    config.get("sequence_length_audit_model_type") or ""
                ),
                "template_mode": str(
                    config.get("sequence_length_audit_template_mode") or ""
                ),
                "truncation_strategy": str(
                    config.get("sequence_length_audit_truncation_strategy") or ""
                ),
                "add_non_thinking_prefix": bool(
                    config.get("sequence_length_audit_add_non_thinking_prefix", False)
                ),
                "runtime_image_digest": str(
                    config.get("sequence_length_audit_runtime_image_digest") or ""
                ),
                "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
                "model_artifact_manifest_sha256": (
                    sequence_length_audit_model_manifest_sha256
                ),
                "checked_rows": total_rows,
                "overlength_rows": 0,
                "assistant_truncation_rows": 0,
                "encode_error_rows": 0,
                "zero_supervised_assistant_rows": 0,
                "json_eos_incomplete_rows": 0,
            }
            if total_rows < 1:
                raise ValueError(f"{uri}: total_rows must be positive")
            for key, expected_value in expected_audit.items():
                if audit.get(key) != expected_value:
                    raise ValueError(
                        f"{uri}: sequence_length_audit.{key}={audit.get(key)!r}, "
                        f"expected {expected_value!r}"
                    )
            max_observed_length = int(audit.get("max_observed_length") or 0)
            if not 0 < max_observed_length <= sequence_length_audit_max_length:
                raise ValueError(f"{uri}: sequence length maximum is invalid")
    if manifests and len(paths) < minimum_physical_shards:
        raise ValueError(
            f"V6 manifest resolved {len(paths)} paths; at least "
            f"{minimum_physical_shards} are required"
        )


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> ImageEditV6FullJsonConfig:
    config = _config_root(document if document is not None else _load_config_document())
    allowed_statuses = tuple(str(value) for value in _as_list(config.get("allowed_statuses")))
    if "allowed_statuses" not in config:
        allowed_statuses = ("validated",)
    if not allowed_statuses:
        raise ValueError("allowed_statuses must be non-empty")
    minimum_physical_shards = int(config.get("minimum_physical_shards") or 128)
    if minimum_physical_shards < 1:
        raise ValueError("minimum_physical_shards must be positive")
    required_source_stage = str(config.get("required_source_stage") or "")
    required_source_partitions = int(config.get("required_source_partitions") or 0)
    minimum_shards_per_source_partition = int(
        config.get("minimum_shards_per_source_partition") or 1
    )
    if required_source_partitions < 0 or minimum_shards_per_source_partition < 1:
        raise ValueError("invalid source partition requirements")
    if required_source_partitions and not required_source_stage:
        raise ValueError(
            "required_source_stage is required with required_source_partitions"
        )
    exclusion_policy = str(config.get("exclusion_policy") or "")
    exclusion_sample_ids_sha256 = str(
        config.get("exclusion_sample_ids_sha256") or ""
    )
    if exclusion_policy and SHA256_RE.fullmatch(exclusion_sample_ids_sha256) is None:
        raise ValueError("exclusion_sample_ids_sha256 must be a lowercase SHA-256")
    sequence_length_audit_max_length = int(
        config.get("sequence_length_audit_max_length") or 0
    )
    sequence_length_audit_model_manifest_sha256 = str(
        config.get("sequence_length_audit_model_manifest_sha256") or ""
    )
    if sequence_length_audit_max_length < 0:
        raise ValueError("sequence_length_audit_max_length cannot be negative")
    if (
        sequence_length_audit_max_length
        and not sequence_length_audit_model_manifest_sha256
    ):
        raise ValueError(
            "sequence_length_audit_model_manifest_sha256 is required when "
            "sequence_length_audit_max_length is set"
        )
    if sequence_length_audit_max_length:
        required_sequence_audit_fields = (
            "sequence_length_audit_max_pixels",
            "sequence_length_audit_image_max_token_num",
            "sequence_length_audit_model_type",
            "sequence_length_audit_template_mode",
            "sequence_length_audit_truncation_strategy",
            "sequence_length_audit_runtime_image_digest",
        )
        missing = [key for key in required_sequence_audit_fields if not config.get(key)]
        if missing:
            raise ValueError(
                f"missing sequence-length audit configuration: {', '.join(missing)}"
            )
    manifests = _manifest_documents(config) if resolve_paths else []
    if manifests:
        paths = tuple(
            str(path)
            for _uri, document, _digest in manifests
            for path in _as_list(document.get("paths"))
        )
    else:
        paths = tuple(_expand_paths(config)) if resolve_paths else ()
    _validate_contracts(
        config,
        manifests,
        allowed_statuses,
        paths,
        minimum_physical_shards,
        required_source_stage,
        required_source_partitions,
        minimum_shards_per_source_partition,
        exclusion_policy,
        exclusion_sample_ids_sha256,
        sequence_length_audit_max_length,
        sequence_length_audit_model_manifest_sha256,
    )
    shard_identities = tuple(
        (
            str(item["path"]),
            str(item["version_id"]),
            str(item["sha256"]),
            int(item["bytes"]),
            int(item["rows"]),
        )
        for _uri, document, _digest in manifests
        for item in document.get("shards", [])
    )
    return ImageEditV6FullJsonConfig(
        dataset_name=str(config.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        paths=paths,
        instruction_field=str(config.get("instruction_field") or "instruction"),
        reference_image_field=str(config.get("reference_image_field") or "source_s3_uri"),
        target_image_field=str(config.get("target_image_field") or "target_s3_uri"),
        target_json_field=str(config.get("target_json_field") or "final_json"),
        sample_id_field=str(config.get("sample_id_field") or "sample_id"),
        status_field=str(config.get("status_field") or "status"),
        audit_valid_field=str(config.get("audit_valid_field") or "call2_audit.valid"),
        contract_version_field=str(
            config.get("contract_version_field") or "contract_version"
        ),
        contract_sha256_field=str(
            config.get("contract_sha256_field") or "contract_sha256"
        ),
        bbox_coordinate_order_field=str(
            config.get("bbox_coordinate_order_field") or "bbox_coordinate_order"
        ),
        allowed_statuses=allowed_statuses,
        source_name=str(config.get("source_name") or "image_edit_v6_full_json"),
        val_path_count=int(config.get("val_path_count") or 0),
        val_max_samples=int(config.get("val_max_samples") or 1000),
        rank_shard=_env_bool(
            "IMAGE_EDIT_V6_FULL_JSON_RANK_SHARD", bool(config.get("rank_shard", True))
        ),
        repeat=_env_bool(
            "IMAGE_EDIT_V6_FULL_JSON_REPEAT", bool(config.get("repeat", True))
        ),
        shuffle_paths=_env_bool(
            "IMAGE_EDIT_V6_FULL_JSON_SHUFFLE_PATHS",
            bool(config.get("shuffle_paths", True)),
        ),
        sample_shuffle_buffer=int(
            os.environ.get("IMAGE_EDIT_V6_FULL_JSON_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 2048
        ),
        seed=int(os.environ.get("IMAGE_EDIT_V6_FULL_JSON_SEED") or config.get("seed") or 42),
        max_samples=int(
            os.environ.get("IMAGE_EDIT_V6_FULL_JSON_MAX_SAMPLES")
            or config.get("max_samples")
            or 0
        ),
        reject_invalid_rows=_env_bool(
            "IMAGE_EDIT_V6_FULL_JSON_REJECT_INVALID_ROWS",
            bool(config.get("reject_invalid_rows", False)),
        ),
        minimum_physical_shards=minimum_physical_shards,
        shard_identities=shard_identities,
        required_source_stage=required_source_stage,
        required_source_partitions=required_source_partitions,
        minimum_shards_per_source_partition=minimum_shards_per_source_partition,
        exclusion_policy=exclusion_policy,
        exclusion_sample_ids_sha256=exclusion_sample_ids_sha256,
        sequence_length_audit_max_length=sequence_length_audit_max_length,
        sequence_length_audit_model_manifest_sha256=(
            sequence_length_audit_model_manifest_sha256
        ),
    )


def _iter_verified_jsonl(
    path: str,
    identity: tuple[str, str, int, int] | None,
) -> Iterator[dict[str, Any]]:
    """Read the exact immutable shard version pinned by the manifest."""

    if identity is None:
        yield from iter_jsonl(path)
        return
    version_id, expected_sha256, expected_bytes, expected_rows = identity
    if path.startswith("s3://"):
        bucket, key = _split_s3_uri(path)
        raw = _s3_client().get_object(
            Bucket=bucket, Key=key, VersionId=version_id
        )["Body"].read()
    else:
        raw = Path(path).read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path}: immutable shard byte count changed: "
            f"actual={len(raw)} expected={expected_bytes}"
        )
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: immutable shard SHA-256 mismatch: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    actual_rows = sum(1 for line in io.BytesIO(raw) if line.strip())
    if actual_rows != expected_rows:
        raise ValueError(
            f"{path}: immutable shard row count changed: "
            f"actual={actual_rows} expected={expected_rows}"
        )
    for line_number, line in enumerate(io.BytesIO(raw), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        yield value


def _iter_source(
    path_payload: Mapping[str, Any],
    *,
    config: ImageEditV6FullJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    paths = list(path_payload["paths"])
    if data_rank is None or data_world_size is None:
        data_rank, data_world_size = data_stream_shard_context()
    if config.rank_shard and data_world_size > len(paths):
        raise ValueError(
            "rank-local image-edit-v6 streaming requires at least one physical JSONL "
            f"shard per DP rank: paths={len(paths)}, data_world_size={data_world_size}"
        )
    emitted = 0
    shard_identities = {
        item[0]: item[1:] for item in config.shard_identities
    }
    pass_index = 0
    while True:
        pass_paths = list(paths)
        rng = random.Random(config.seed + pass_index)
        if shuffle_paths:
            rng.shuffle(pass_paths)
        if config.rank_shard:
            pass_paths = shard_paths(pass_paths, data_rank, data_world_size)
        if not pass_paths:
            return
        buffer: list[dict[str, Any]] = []
        pass_emitted = 0
        for path in pass_paths:
            for source_row in _iter_verified_jsonl(path, shard_identities.get(path)):
                row = make_row(source_row, source_path=path, config=config)
                if row is None:
                    continue
                emitted += 1
                pass_emitted += 1
                if config.sample_shuffle_buffer <= 1:
                    yield row
                else:
                    buffer.append(row)
                    if len(buffer) >= config.sample_shuffle_buffer:
                        yield buffer.pop(rng.randrange(len(buffer)))
                if max_samples > 0 and emitted >= max_samples:
                    while buffer:
                        yield buffer.pop(rng.randrange(len(buffer)))
                    return
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
        if not repeat:
            return
        if pass_emitted == 0:
            raise RuntimeError(f"No valid image-edit-v6 rows found in {len(pass_paths)} paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: ImageEditV6FullJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
):
    from datasets import Features, IterableDataset, Value

    data_rank, data_world_size = data_stream_shard_context()
    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "images": [Value("string")],
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
                "annotation_contract_version": Value("string"),
                "annotation_contract_sha256": Value("string"),
                "bbox_coordinate_order": Value("string"),
                "review_warning_codes": Value("string"),
            },
        }
    )
    return IterableDataset.from_generator(
        _iter_source,
        gen_kwargs={
            "path_payload": {"paths": list(paths)},
            "config": config,
            "repeat": repeat,
            "shuffle_paths": shuffle_paths,
            "max_samples": max_samples,
            "data_rank": data_rank,
            "data_world_size": data_world_size,
        },
        features=features,
    )


class ImageEditV6FullJsonLoader:
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
        if not self.streaming:
            raise ValueError("image-edit-v6 full JSON requires --streaming true")
        config = load_runtime_config()
        requested_name = getattr(dataset_meta, "dataset_name", None) or config.dataset_name
        is_validation = requested_name == config.val_dataset_name
        if config.val_path_count > 0:
            val_paths = config.paths[: config.val_path_count]
            train_paths = config.paths[config.val_path_count :]
        else:
            val_paths = config.paths
            train_paths = config.paths
        if is_validation:
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
            )
        else:
            dataset = _build_hf_stream(
                train_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=config.max_samples,
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
        return dataset


def register_with_swift() -> tuple[str, str]:
    from swift.dataset import MessagesPreprocessor
    from swift.dataset.dataset_meta import BaseDatasetLoader, DatasetMeta
    from swift.dataset.register import register_dataset

    from dlms.patches.s3_image_load import apply as apply_s3_image_load

    class SwiftImageEditV6FullJsonLoader(ImageEditV6FullJsonLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "multi-image", "sft", "structured-json", "image-edit-v6"]),
        (
            config.val_dataset_name,
            ["vision", "multi-image", "sft", "structured-json", "image-edit-v6", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImageEditV6FullJsonLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    apply_s3_image_load()
    print(
        f"[dlms] image-edit-v6 full JSON stream enabled: dataset={config.dataset_name} "
        f"prompt={PROMPT_VERSION} annotation={ANNOTATION_CONTRACT_VERSION}",
        flush=True,
    )
    return config.dataset_name, config.val_dataset_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


if _swift_is_available() and not _env_bool("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY", False):
    register_with_swift()
