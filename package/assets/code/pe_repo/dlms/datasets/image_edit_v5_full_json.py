"""Two-image SFT contract for complete image-edit-v5 Target JSON.

Prepared JSONL rows deliberately contain no chat messages.  This loader turns
one ordered Source image (``frame_0``), one Target image, and the dataset edit
instruction into a Qwen conversation at runtime, then supervises the complete
V5 object.  V5 differs from the older inline V4 task: every region edit is
``{instruction, references}``, and an ``operation`` key is forbidden.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
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
        data_stream_shard_context,
        iter_jsonl,
        shard_paths,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


PROMPT_VERSION = "image_edit_v5_full_json_instruction_v3"
# Checked-in contract pin.  Update this literal together with PROMPT_VERSION
# whenever SYSTEM_PROMPT or build_user_prompt changes.
PROMPT_CONTRACT_SHA256 = "7f7b4e791ac1632475bf04bf7a45b260d759484bc6fe0cb86bda8a06cedfe442"
DEFAULT_DATASET_NAME = "image_edit_v5_full_json_qwen35"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CHANNEL = "image_edit_v5_full_json"
STRICT_TARGET_STATUSES = frozenset({"validated"})
PRESERVED_TARGET_STATUSES = frozenset({"manual_review_required"})


SYSTEM_PROMPT = """Given two ordered images—the first is Reference `frame_0` and the second is Target—plus an edit instruction, produce the complete image-edit-v5 annotation. Return only valid JSON in the schema below.

Target pixels determine visible inventory, labels, hierarchy, Target bboxes, descriptions, background, and style. Compare Reference with Target to determine local correspondence and whether each region changed or stayed unchanged. Use the dataset instruction only to route attention; never claim an instructed result that is absent from Target. Inventory the whole useful Target, not only edited regions.

The only top-level key is `structured_description`, containing exactly `high_level_description`, `style_description{aesthetics,lighting,medium,color_palette}`, and `elements{background,layout,remove}`. Each layout item contains exactly `label,parent,type,bbox,desc,edit`; type is `coarse|medium|fine`, parent is `ROOT` or an earlier label, and bbox is integer `[y0,x0,y1,x1]` on 0–1000.

Every layout edit contains exactly `instruction,references`. A nonempty references array uses same-granularity Source provenance shaped `{reference_id:"frame_0",reference_bbox}`. Its instruction must explicitly say `referenced Source` and locally describe either the concrete change or that the region remains unchanged in the Target. An empty references array means a Target-only addition, and its instruction explicitly says add, create, or introduce.

`remove` contains only Source identities intentionally absent from Target. Each remove item contains exactly `instruction,references`, uses nonempty Source provenance, and explicitly directs removal. Do not treat crop, occlusion, viewpoint change, movement, uncertainty, or replacement as removal. No Markdown, metadata, comments, analysis, wrappers, or extra text."""


def build_user_prompt(instruction: str) -> str:
    """Create the injection-safe per-sample prompt with two image slots."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must be non-empty")
    encoded = json.dumps(instruction, ensure_ascii=False, allow_nan=False)
    return (
        "First image — Reference (`frame_0`):\n"
        "<image>\n"
        "Second image — Target:\n"
        "<image>\n"
        f"Instruction:\n{encoded}\n"
        "Generate the complete image-edit-v5 JSON containing `structured_description` "
        "with `high_level_description`, `style_description`, `elements.background`, "
        "`elements.layout`, and `elements.remove`. Every layout item must contain "
        "`label`, `parent`, `type`, `bbox`, `desc`, and `edit`; every edit contains "
        "only `instruction` and `references`."
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


LAYOUT_TYPES = frozenset({"coarse", "medium", "fine"})
STYLE_KEYS = frozenset({"aesthetics", "lighting", "medium", "color_palette"})
COLOR_CHARS = frozenset("0123456789abcdefABCDEF")
_MISSING = object()


class V5TargetValidationError(ValueError):
    """The assistant target is not a strict image-edit-v5 object."""


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V5TargetValidationError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise V5TargetValidationError(f"{path} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise V5TargetValidationError(
            f"{path} keys must be {sorted(expected)}, got {sorted(actual)}"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V5TargetValidationError(f"{path} must be a non-empty string")
    return value


def _bbox(value: Any, path: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise V5TargetValidationError(f"{path} must contain four integers")
    y0, x0, y1, x1 = value
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        raise V5TargetValidationError(f"{path} is outside normalized yxyx bounds")


def _references(value: Any, path: str) -> list[Any]:
    references = _array(value, path)
    for index, raw_reference in enumerate(references):
        reference_path = f"{path}[{index}]"
        reference = _object(raw_reference, reference_path)
        _exact_keys(reference, {"reference_id", "reference_bbox"}, reference_path)
        if reference.get("reference_id") != "frame_0":
            raise V5TargetValidationError(
                f"{reference_path}.reference_id must be frame_0"
            )
        _bbox(reference.get("reference_bbox"), f"{reference_path}.reference_bbox")
    return references


def _layout_edit(value: Any, path: str) -> None:
    edit = _object(value, path)
    _exact_keys(edit, {"instruction", "references"}, path)
    instruction = _text(edit.get("instruction"), f"{path}.instruction")
    references = _references(edit.get("references"), f"{path}.references")
    lower = instruction.lower()
    if references and "referenced source" not in lower:
        raise V5TargetValidationError(
            f'{path}.instruction must use the phrase "referenced Source"'
        )
    if not references and not any(word in lower for word in ("add", "create", "introduce")):
        raise V5TargetValidationError(
            f"{path}.instruction must explicitly add/create/introduce a Target-only region"
        )


def _remove_edit(value: Any, path: str) -> None:
    edit = _object(value, path)
    _exact_keys(edit, {"instruction", "references"}, path)
    instruction = _text(edit.get("instruction"), f"{path}.instruction")
    references = _references(edit.get("references"), f"{path}.references")
    lower = instruction.lower()
    if not references:
        raise V5TargetValidationError(f"{path}.references must be non-empty")
    if "referenced source" not in lower:
        raise V5TargetValidationError(
            f'{path}.instruction must use the phrase "referenced Source"'
        )
    if not any(word in lower for word in ("remove", "delete", "erase", "eliminate")):
        raise V5TargetValidationError(f"{path}.instruction must explicitly direct removal")


def validate_v5_target(value: Any) -> Mapping[str, Any]:
    target = _object(value, "target_json")
    _exact_keys(target, {"structured_description"}, "target_json")
    structured = _object(target.get("structured_description"), "structured_description")
    _exact_keys(
        structured,
        {"high_level_description", "style_description", "elements"},
        "structured_description",
    )
    _text(structured.get("high_level_description"), "high_level_description")

    style = _object(structured.get("style_description"), "style_description")
    _exact_keys(style, set(STYLE_KEYS), "style_description")
    for field in ("aesthetics", "lighting", "medium"):
        _text(style.get(field), f"style_description.{field}")
    palette = _array(style.get("color_palette"), "style_description.color_palette")
    if not palette:
        raise V5TargetValidationError("style_description.color_palette must be non-empty")
    for index, color in enumerate(palette):
        if (
            not isinstance(color, str)
            or len(color) not in (7, 9)
            or not color.startswith("#")
            or any(character not in COLOR_CHARS for character in color[1:])
        ):
            raise V5TargetValidationError(
                f"style_description.color_palette[{index}] must be #RRGGBB or #RRGGBBAA"
            )

    elements = _object(structured.get("elements"), "elements")
    _exact_keys(elements, {"background", "layout", "remove"}, "elements")
    _text(elements.get("background"), "elements.background")
    layout = _array(elements.get("layout"), "elements.layout")
    if not layout:
        raise V5TargetValidationError("elements.layout must be non-empty")
    for index, raw_region in enumerate(layout):
        path = f"elements.layout[{index}]"
        region = _object(raw_region, path)
        _exact_keys(region, {"label", "parent", "type", "bbox", "desc", "edit"}, path)
        for field in ("label", "parent", "desc"):
            _text(region.get(field), f"{path}.{field}")
        if region.get("type") not in LAYOUT_TYPES:
            raise V5TargetValidationError(f"{path}.type is invalid: {region.get('type')!r}")
        _bbox(region.get("bbox"), f"{path}.bbox")
        _layout_edit(region.get("edit"), f"{path}.edit")

    for index, removal in enumerate(_array(elements.get("remove"), "elements.remove")):
        _remove_edit(removal, f"elements.remove[{index}]")
    return target


def canonical_v5_target(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant: {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise V5TargetValidationError("target_json is not strict JSON") from exc
    validated = validate_v5_target(value)
    try:
        return json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V5TargetValidationError("target_json is not JSON serializable") from exc


def contains_operation_key(value: Any) -> bool:
    """Return whether an exact ``operation`` key occurs at any JSON depth."""

    if isinstance(value, Mapping):
        return "operation" in value or any(
            contains_operation_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(contains_operation_key(child) for child in value)
    return False


def canonical_training_target(value: Any, *, status: str) -> str:
    """Serialize a label according to its authoritative V5 review status.

    Validated labels retain the strict V5 schema gate.  Manual-review labels
    are intentionally kept without semantic repair, so they need only be
    complete JSON objects.  Failed labels are never accepted for training.
    """

    if contains_operation_key(value):
        raise V5TargetValidationError("target_json contains forbidden operation key")
    if status in STRICT_TARGET_STATUSES:
        return canonical_v5_target(value)
    if status not in PRESERVED_TARGET_STATUSES:
        raise V5TargetValidationError(f"unsupported training status: {status!r}")
    if not isinstance(value, Mapping):
        raise V5TargetValidationError("manual-review target_json must be an object")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V5TargetValidationError(
            "manual-review target_json is not JSON serializable"
        ) from exc


@dataclass(frozen=True)
class ImageEditV5FullJsonConfig:
    dataset_name: str
    val_dataset_name: str
    paths: tuple[str, ...]
    instruction_field: str = "instruction"
    reference_images_field: str = "reference_images"
    target_image_field: str = "target_image"
    target_json_field: str = "target_json"
    sample_id_field: str = "sample_id"
    status_field: str = "status"
    allowed_statuses: tuple[str, ...] = ("validated",)
    media_uri_fields: tuple[str, ...] = ("s3_uri", "uri", "gcs_uri", "path")
    source_name: str = "image_edit_v5_full_json"
    val_path_count: int = 0
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 2048
    seed: int = 42
    max_samples: int = 0
    prompt_version: str = PROMPT_VERSION
    prompt_contract_sha256: str = PROMPT_CONTRACT_SHA256


def _nested_value(row: Mapping[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _media_uri(value: Any, fields: Sequence[str]) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for field in fields:
            candidate = value.get(field)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def make_row(
    source_row: Mapping[str, Any],
    *,
    source_path: str,
    config: ImageEditV5FullJsonConfig,
) -> dict[str, Any] | None:
    status = _nested_value(source_row, config.status_field) if config.status_field else ""
    if config.status_field and (status is _MISSING or str(status) not in config.allowed_statuses):
        return None
    instruction = _nested_value(source_row, config.instruction_field)
    references = _nested_value(source_row, config.reference_images_field)
    target_image = _nested_value(source_row, config.target_image_field)
    target_json = _nested_value(source_row, config.target_json_field)
    if (
        not isinstance(instruction, str)
        or not instruction.strip()
        or not isinstance(references, list)
        or len(references) != 1
        or target_image is _MISSING
        or target_json is _MISSING
    ):
        return None
    reference = references[0]
    if not isinstance(reference, Mapping) or reference.get("reference_id") != "frame_0":
        return None
    reference_uri = _media_uri(reference, config.media_uri_fields)
    target_uri = _media_uri(target_image, config.media_uri_fields)
    if not reference_uri or not target_uri or reference_uri == target_uri:
        return None
    try:
        assistant_target = canonical_training_target(target_json, status=str(status))
        user_prompt = build_user_prompt(instruction)
    except (V5TargetValidationError, ValueError):
        return None
    sample_id = _nested_value(source_row, config.sample_id_field)
    if sample_id is _MISSING:
        sample_id = ""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_target},
        ],
        "images": [reference_uri, target_uri],
        "channel": CHANNEL,
        "meta": {
            "jsonl_path": source_path,
            "source_name": config.source_name,
            "sample_id": str(sample_id),
            "status": str(status),
            "reference_id": "frame_0",
            "reference_image": reference_uri,
            "target_image": target_uri,
            "prompt_version": PROMPT_VERSION,
        },
    }


def _config_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    config = document.get("image_edit_v5_full_json_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("image_edit_v5_full_json_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("IMAGE_EDIT_V5_FULL_JSON_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _manifest_contracts(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    manifests = _as_list(
        config.get("paths_manifests")
        or config.get("paths_manifest")
        or config.get("manifests")
        or config.get("manifest")
    )
    result: list[tuple[str, Mapping[str, Any]]] = []
    for raw_uri in manifests:
        uri = str(raw_uri)
        try:
            document = json.loads(_read_uri_bytes(uri).decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(document, Mapping):
            continue
        contract = document.get("dataset_contract")
        if contract is None:
            continue
        if not isinstance(contract, Mapping):
            raise ValueError(f"{uri}: dataset_contract must be a mapping")
        result.append((uri, contract))
    return result


def _validate_contracts(
    config: Mapping[str, Any],
    contracts: Sequence[tuple[str, Mapping[str, Any]]],
    allowed_statuses: Sequence[str],
) -> None:
    actual = prompt_contract_sha256()
    if actual != PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit-v5 prompt changed without updating PROMPT_VERSION and "
            f"PROMPT_CONTRACT_SHA256: actual={actual} declared={PROMPT_CONTRACT_SHA256}"
        )
    declarations: list[tuple[str, str, str]] = []
    if "prompt_version" in config or "prompt_contract_sha256" in config:
        declarations.append(
            (
                "image_edit_v5_full_json_config",
                str(config.get("prompt_version") or ""),
                str(config.get("prompt_contract_sha256") or ""),
            )
        )
    for uri, contract in contracts:
        declarations.append(
            (
                uri,
                str(contract.get("prompt_version") or ""),
                str(contract.get("prompt_contract_sha256") or ""),
            )
        )
        if contract.get("output_format") not in (None, "prepared"):
            raise ValueError(f"{uri}: image-edit-v5 loader requires output_format=prepared")
        if contract.get("target_schema") not in (None, "image_edit_v5_instruction"):
            raise ValueError(f"{uri}: incompatible target_schema={contract.get('target_schema')!r}")
        if contract.get("allowed_statuses") is not None:
            manifest_statuses = tuple(
                str(value) for value in _as_list(contract.get("allowed_statuses"))
            )
            if manifest_statuses != tuple(allowed_statuses):
                raise ValueError(
                    f"{uri}: configured allowed_statuses={list(allowed_statuses)!r} do not "
                    f"match manifest allowed_statuses={list(manifest_statuses)!r}"
                )
    for source, version, digest in declarations:
        if version != PROMPT_VERSION or digest != PROMPT_CONTRACT_SHA256:
            raise ValueError(
                f"{source}: prompt contract mismatch; got {version!r}/{digest!r}, "
                f"expected {PROMPT_VERSION!r}/{PROMPT_CONTRACT_SHA256!r}"
            )


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> ImageEditV5FullJsonConfig:
    config = _config_root(document if document is not None else _load_config_document())
    contracts = _manifest_contracts(config) if resolve_paths else []
    allowed_statuses = tuple(str(value) for value in _as_list(config.get("allowed_statuses")))
    if "allowed_statuses" not in config:
        allowed_statuses = ("validated",)
    _validate_contracts(config, contracts, allowed_statuses)
    status_field = str(config.get("status_field", "status") or "")
    if bool(status_field) != bool(allowed_statuses):
        raise ValueError("status_field and allowed_statuses must be configured together")
    media_uri_fields = tuple(str(value) for value in _as_list(config.get("media_uri_fields")))
    if not media_uri_fields:
        media_uri_fields = ("s3_uri", "uri", "gcs_uri", "path")
    return ImageEditV5FullJsonConfig(
        dataset_name=str(config.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        paths=tuple(_expand_paths(config)) if resolve_paths else (),
        instruction_field=str(config.get("instruction_field") or "instruction"),
        reference_images_field=str(config.get("reference_images_field") or "reference_images"),
        target_image_field=str(config.get("target_image_field") or "target_image"),
        target_json_field=str(config.get("target_json_field") or "target_json"),
        sample_id_field=str(config.get("sample_id_field") or "sample_id"),
        status_field=status_field,
        allowed_statuses=allowed_statuses,
        media_uri_fields=media_uri_fields,
        source_name=str(config.get("source_name") or "image_edit_v5_full_json"),
        val_path_count=int(config.get("val_path_count") or 0),
        val_max_samples=int(config.get("val_max_samples") or 1000),
        rank_shard=_env_bool("IMAGE_EDIT_V5_FULL_JSON_RANK_SHARD", bool(config.get("rank_shard", True))),
        repeat=_env_bool("IMAGE_EDIT_V5_FULL_JSON_REPEAT", bool(config.get("repeat", True))),
        shuffle_paths=_env_bool(
            "IMAGE_EDIT_V5_FULL_JSON_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))
        ),
        sample_shuffle_buffer=int(
            os.environ.get("IMAGE_EDIT_V5_FULL_JSON_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 2048
        ),
        seed=int(os.environ.get("IMAGE_EDIT_V5_FULL_JSON_SEED") or config.get("seed") or 42),
        max_samples=int(
            os.environ.get("IMAGE_EDIT_V5_FULL_JSON_MAX_SAMPLES")
            or config.get("max_samples")
            or 0
        ),
    )


def _iter_source(
    path_payload: Mapping[str, Any],
    *,
    config: ImageEditV5FullJsonConfig,
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
            "rank-local image-edit-v5 streaming requires at least one physical JSONL "
            f"shard per DP rank: paths={len(paths)}, data_world_size={data_world_size}"
        )
    emitted = 0
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
            for source_row in iter_jsonl(path):
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
            raise RuntimeError(f"No valid image-edit-v5 rows found in {len(pass_paths)} paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: ImageEditV5FullJsonConfig,
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


class ImageEditV5FullJsonLoader:
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
            raise ValueError("image-edit-v5 full JSON requires --streaming true")
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

    class SwiftImageEditV5FullJsonLoader(ImageEditV5FullJsonLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "multi-image", "sft", "structured-json", "image-edit-v5"]),
        (
            config.val_dataset_name,
            ["vision", "multi-image", "sft", "structured-json", "image-edit-v5", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImageEditV5FullJsonLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    apply_s3_image_load()
    print(
        f"[dlms] image-edit-v5 full JSON stream enabled: dataset={config.dataset_name} "
        f"prompt={PROMPT_VERSION}",
        flush=True,
    )
    return config.dataset_name, config.val_dataset_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


if _swift_is_available():
    register_with_swift()
