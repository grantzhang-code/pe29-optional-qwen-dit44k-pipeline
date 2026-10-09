"""Single-task, two-image SFT contract for inline image-edit Target JSON.

Each prepared JSONL row contains one ordered reference image (``frame_0``),
one Target image, an edit instruction, and the complete
``target_json.with_inline_edit.json`` object.  The emitted Swift row always
keeps the media order ``[reference, target]`` and supervises one canonical,
compact JSON object.  This module deliberately does not mix in any of the
legacy CRAFT partial-output tasks.

The expected prepared row shape is::

    {
      "sample_id": "sample-1",
      "status": "validated",
      "instruction": "Change the leaves to green.",
      "reference_images": [
        {"reference_id": "frame_0", "uri": "s3://bucket/reference.jpg"}
      ],
      "target_image": "s3://bucket/target.jpg",
      "target_json": {"structured_description": {"...": "..."}}
    }

Field paths and media URI keys are configurable under
``image_edit_full_json_config`` in ``JOB_CONFIG_YAML``.  The default status
gate accepts only auto-validated annotations; manual-review rows require an
explicit opt-in.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Reuse the established manifest, S3, and DP-rank helpers without registering
# the baseline4 dataset as an import side effect.
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


PROMPT_VERSION = "image_edit_full_json_inline_v4_baseline_style"
# This is deliberately a checked-in literal rather than a value computed at
# import time.  A prompt edit must update both PROMPT_VERSION and this digest,
# making otherwise-silent training-contract drift fail loudly in tests/loaders.
PROMPT_CONTRACT_SHA256 = "d732ff763d314efee3fe0b57116b4b95678e529d4d445229899c806e686e95ce"
DEFAULT_DATASET_NAME = "image_edit_full_json_inline_qwen35"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CHANNEL = "image_edit_full_json"

SYSTEM_PROMPT = """Given ordered Reference `frame_0`, Target, and an instruction, produce a complete structured image-edit annotation. Return only valid JSON in the schema described below.

Target pixels alone determine visible inventory, labels, hierarchy, Target bboxes, descriptions, background, and style. Reference and instruction determine only correspondence, operation, Source bbox, and genuine removal. Inventory the whole useful Target, not only edited regions; never invent an instructed result absent from Target or mention edit history.

The only top-level key is `structured_description`, with exactly `high_level_description`, `style_description{aesthetics,lighting,medium,color_palette}`, and `elements{background,layout,remove}`. Each layout item has exactly `label,parent,type,bbox,desc,edit`; labels are unique `<name N>`, parents are `ROOT` or earlier labels, types are `coarse|medium|fine`, and bboxes are integer `[y0,x0,y1,x1]` on 0–1000.

Each edit is `{operation,references}`: `general` same unchanged; `modify` same identity changed; `transform` moved/resized; `replace` different identity; `crop`/`outpaint` contracted/expanded framing; `add` no Source counterpart. Add has no references; all others use same-granularity references shaped `{reference_id:"frame_0",reference_bbox}`. `remove` contains only `{operation:"remove",references}` for referenced Source identities intentionally absent from Target, never crop, occlusion, viewpoint, movement, uncertainty, or replaced identities. Mention each visible label once in `high_level_description`. No Markdown, metadata, comments, or extra text."""


def build_user_prompt(instruction: str) -> str:
    """Build the short per-sample prompt with an injection-safe JSON string."""

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must be non-empty")
    encoded = json.dumps(instruction, ensure_ascii=False, allow_nan=False)
    return (
        "Reference image (`frame_0`):\n"
        "<image>\n"
        "Target image:\n"
        "<image>\n"
        f"Instruction:\n{encoded}\n"
        "Generate the complete JSON containing `structured_description` with "
        "`high_level_description`, `style_description`, `elements.background`, "
        "`elements.layout`, and `elements.remove`. Every layout item must contain "
        "`label`, `parent`, `type`, `bbox`, `desc`, and `edit`."
    )


def prompt_contract_sha256() -> str:
    """Hash the exact static prompt contract used to materialize Swift rows."""

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


VISIBLE_OPERATIONS = frozenset({"add", "crop", "general", "modify", "outpaint", "replace", "transform"})
LAYOUT_TYPES = frozenset({"coarse", "medium", "fine"})
LABEL_RE = re.compile(r"^<[^<>\r\n]+ [1-9][0-9]*>$")
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")
_MISSING = object()


class InlineTargetValidationError(ValueError):
    """A target cannot be supervised without changing its declared schema."""


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InlineTargetValidationError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise InlineTargetValidationError(f"{path} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise InlineTargetValidationError(f"{path} keys must be {sorted(expected)}, got {sorted(actual)}")


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InlineTargetValidationError(f"{path} must be a non-empty string")
    return value


def _validate_bbox(value: Any, path: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise InlineTargetValidationError(f"{path} must contain four integers")
    y0, x0, y1, x1 = value
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        raise InlineTargetValidationError(f"{path} is outside normalized yxyx bounds")


def _validate_references(value: Any, path: str) -> list[Any]:
    references = _require_list(value, path)
    for index, raw_reference in enumerate(references):
        reference_path = f"{path}[{index}]"
        reference = _require_object(raw_reference, reference_path)
        _require_exact_keys(reference, {"reference_id", "reference_bbox"}, reference_path)
        if reference.get("reference_id") != "frame_0":
            raise InlineTargetValidationError(f"{reference_path}.reference_id must be frame_0")
        _validate_bbox(reference.get("reference_bbox"), f"{reference_path}.reference_bbox")
    return references


def _validate_visible_edit(value: Any, path: str) -> None:
    edit = _require_object(value, path)
    _require_exact_keys(edit, {"operation", "references"}, path)
    operation = edit.get("operation")
    if operation not in VISIBLE_OPERATIONS:
        raise InlineTargetValidationError(f"{path}.operation is invalid: {operation!r}")
    references = _validate_references(edit.get("references"), f"{path}.references")
    if operation == "add" and references:
        raise InlineTargetValidationError(f"{path}.references must be empty for add")
    if operation != "add" and not references:
        raise InlineTargetValidationError(f"{path}.references must be non-empty for {operation}")


def _validate_remove_edit(value: Any, path: str) -> None:
    edit = _require_object(value, path)
    _require_exact_keys(edit, {"operation", "references"}, path)
    if edit.get("operation") != "remove":
        raise InlineTargetValidationError(f"{path}.operation must be remove")
    references = _validate_references(edit.get("references"), f"{path}.references")
    if not references:
        raise InlineTargetValidationError(f"{path}.references must be non-empty")


def validate_inline_target(value: Any) -> Mapping[str, Any]:
    """Validate the exact observed ``target_json.with_inline_edit`` contract."""

    target = _require_object(value, "target_json")
    _require_exact_keys(target, {"structured_description"}, "target_json")
    structured = _require_object(target.get("structured_description"), "structured_description")
    _require_exact_keys(
        structured,
        {"high_level_description", "style_description", "elements"},
        "structured_description",
    )
    _require_nonempty_string(
        structured.get("high_level_description"), "structured_description.high_level_description"
    )

    style = _require_object(structured.get("style_description"), "style_description")
    _require_exact_keys(
        style,
        {"aesthetics", "lighting", "medium", "color_palette"},
        "style_description",
    )
    for field in ("aesthetics", "lighting", "medium"):
        _require_nonempty_string(style.get(field), f"style_description.{field}")
    palette = _require_list(style.get("color_palette"), "style_description.color_palette")
    if not palette:
        raise InlineTargetValidationError("style_description.color_palette must be non-empty")
    for index, color in enumerate(palette):
        if not isinstance(color, str) or COLOR_RE.fullmatch(color) is None:
            raise InlineTargetValidationError(
                f"style_description.color_palette[{index}] must be #RRGGBB or #RRGGBBAA"
            )

    elements = _require_object(structured.get("elements"), "elements")
    _require_exact_keys(elements, {"background", "layout", "remove"}, "elements")
    _require_nonempty_string(elements.get("background"), "elements.background")

    layout = _require_list(elements.get("layout"), "elements.layout")
    if not layout:
        raise InlineTargetValidationError("elements.layout must be non-empty")
    seen_labels: set[str] = set()
    for index, raw_region in enumerate(layout):
        path = f"elements.layout[{index}]"
        region = _require_object(raw_region, path)
        _require_exact_keys(region, {"label", "parent", "type", "bbox", "desc", "edit"}, path)
        label = _require_nonempty_string(region.get("label"), f"{path}.label")
        if LABEL_RE.fullmatch(label) is None:
            raise InlineTargetValidationError(f"{path}.label has invalid syntax: {label!r}")
        if label in seen_labels:
            raise InlineTargetValidationError(f"{path}.label is duplicated: {label!r}")
        parent = _require_nonempty_string(region.get("parent"), f"{path}.parent")
        if parent != "ROOT" and parent not in seen_labels:
            raise InlineTargetValidationError(
                f"{path}.parent must be ROOT or an earlier exact label: {parent!r}"
            )
        if region.get("type") not in LAYOUT_TYPES:
            raise InlineTargetValidationError(f"{path}.type is invalid: {region.get('type')!r}")
        _validate_bbox(region.get("bbox"), f"{path}.bbox")
        _require_nonempty_string(region.get("desc"), f"{path}.desc")
        _validate_visible_edit(region.get("edit"), f"{path}.edit")
        seen_labels.add(label)

    remove = _require_list(elements.get("remove"), "elements.remove")
    for index, edit in enumerate(remove):
        _validate_remove_edit(edit, f"elements.remove[{index}]")
    return target


def canonical_inline_target(value: Any) -> str:
    """Return a compact, stable assistant target or raise on schema drift."""

    if isinstance(value, str):

        def reject_nonstandard_constant(token: str) -> None:
            raise ValueError(f"non-standard JSON constant: {token}")

        try:
            value = json.loads(value, parse_constant=reject_nonstandard_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InlineTargetValidationError("target_json is not strict JSON") from exc
    validated = validate_inline_target(value)
    try:
        return json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InlineTargetValidationError("target_json is not JSON serializable") from exc


@dataclass(frozen=True)
class ImageEditFullJsonConfig:
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
    source_name: str = "image_edit_full_json_inline"
    val_path_count: int = 0
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 8192
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
    config: ImageEditFullJsonConfig,
) -> dict[str, Any] | None:
    """Convert one prepared pair to an exact two-image Swift messages row."""

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
    if not reference_uri or not target_uri:
        return None
    try:
        assistant_target = canonical_inline_target(target_json)
        user_prompt = build_user_prompt(instruction)
    except (InlineTargetValidationError, ValueError):
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
    config = document.get("image_edit_full_json_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("image_edit_full_json_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("IMAGE_EDIT_FULL_JSON_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _declared_manifest_contracts(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Read optional dataset contracts from JSON path manifests.

    Legacy/plain-text manifests remain supported and simply contribute no
    contract.  Prepared manifests emitted by the shard builder carry the
    contract, so loading them automatically pins prompt materialization.
    """

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


def _validate_prompt_contract(
    config: Mapping[str, Any],
    *,
    manifest_contracts: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[str, str]:
    actual_digest = prompt_contract_sha256()
    if actual_digest != PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "image-edit prompt source changed without updating PROMPT_VERSION and "
            f"PROMPT_CONTRACT_SHA256: actual={actual_digest} "
            f"declared={PROMPT_CONTRACT_SHA256}"
        )

    declarations: list[tuple[str, str, str]] = []
    configured_version = config.get("prompt_version")
    configured_digest = config.get("prompt_contract_sha256")
    if configured_version is not None or configured_digest is not None:
        declarations.append(
            (
                "image_edit_full_json_config",
                str(configured_version or ""),
                str(configured_digest or ""),
            )
        )
    for uri, contract in manifest_contracts:
        declarations.append(
            (
                uri,
                str(contract.get("prompt_version") or ""),
                str(contract.get("prompt_contract_sha256") or ""),
            )
        )

    for source, version, digest in declarations:
        if version != PROMPT_VERSION or digest != PROMPT_CONTRACT_SHA256:
            raise ValueError(
                f"{source}: prepared dataset prompt contract does not match this loader: "
                f"version={version!r}, sha256={digest!r}; expected "
                f"version={PROMPT_VERSION!r}, sha256={PROMPT_CONTRACT_SHA256!r}"
            )
    return PROMPT_VERSION, PROMPT_CONTRACT_SHA256


def _validate_manifest_dataset_contracts(
    contracts: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    allowed_statuses: Sequence[str],
) -> None:
    for uri, contract in contracts:
        if contract.get("output_format") not in (None, "prepared"):
            raise ValueError(f"{uri}: image-edit loader requires output_format=prepared")
        raw_statuses = contract.get("allowed_statuses")
        if raw_statuses is None:
            continue
        manifest_statuses = tuple(str(value) for value in _as_list(raw_statuses))
        if manifest_statuses != tuple(allowed_statuses):
            raise ValueError(
                f"{uri}: configured allowed_statuses={list(allowed_statuses)!r} do not "
                f"match prepared manifest allowed_statuses={list(manifest_statuses)!r}"
            )


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> ImageEditFullJsonConfig:
    config = _config_root(document if document is not None else _load_config_document())
    manifest_contracts = _declared_manifest_contracts(config) if resolve_paths else []
    prompt_version, prompt_digest = _validate_prompt_contract(config, manifest_contracts=manifest_contracts)
    allowed_statuses = tuple(str(value) for value in _as_list(config.get("allowed_statuses")))
    if "allowed_statuses" not in config:
        allowed_statuses = ("validated",)
    _validate_manifest_dataset_contracts(manifest_contracts, allowed_statuses=allowed_statuses)
    status_field = str(config.get("status_field", "status") or "")
    if bool(status_field) != bool(allowed_statuses):
        raise ValueError("status_field and allowed_statuses must be configured together")
    media_uri_fields = tuple(str(value) for value in _as_list(config.get("media_uri_fields")))
    if not media_uri_fields:
        media_uri_fields = ("s3_uri", "uri", "gcs_uri", "path")
    return ImageEditFullJsonConfig(
        dataset_name=str(config.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        paths=_expand_paths(config) if resolve_paths else (),
        instruction_field=str(config.get("instruction_field") or "instruction"),
        reference_images_field=str(config.get("reference_images_field") or "reference_images"),
        target_image_field=str(config.get("target_image_field") or "target_image"),
        target_json_field=str(config.get("target_json_field") or "target_json"),
        sample_id_field=str(config.get("sample_id_field") or "sample_id"),
        status_field=status_field,
        allowed_statuses=allowed_statuses,
        media_uri_fields=media_uri_fields,
        source_name=str(config.get("source_name") or "image_edit_full_json_inline"),
        val_path_count=int(config.get("val_path_count") or 0),
        val_max_samples=int(config.get("val_max_samples") or 1000),
        rank_shard=_env_bool("IMAGE_EDIT_FULL_JSON_RANK_SHARD", bool(config.get("rank_shard", True))),
        repeat=_env_bool("IMAGE_EDIT_FULL_JSON_REPEAT", bool(config.get("repeat", True))),
        shuffle_paths=_env_bool(
            "IMAGE_EDIT_FULL_JSON_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))
        ),
        sample_shuffle_buffer=int(
            os.environ.get("IMAGE_EDIT_FULL_JSON_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 8192
        ),
        seed=int(os.environ.get("IMAGE_EDIT_FULL_JSON_SEED") or config.get("seed") or 42),
        max_samples=int(os.environ.get("IMAGE_EDIT_FULL_JSON_MAX_SAMPLES") or config.get("max_samples") or 0),
        prompt_version=prompt_version,
        prompt_contract_sha256=prompt_digest,
    )


def _iter_source(
    path_payload: Mapping[str, Any],
    *,
    config: ImageEditFullJsonConfig,
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
            "rank-local image-edit streaming requires at least one physical JSONL "
            f"shard per data-parallel rank: paths={len(paths)}, "
            f"data_world_size={data_world_size}. Rebuild with --output-parts >= "
            "the DP world size or disable rank-local streaming."
        )
    emitted = 0
    pass_index = 0
    while True:
        pass_paths = list(paths)
        path_rng = random.Random(config.seed + pass_index)
        if shuffle_paths:
            path_rng.shuffle(pass_paths)
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
                        yield buffer.pop(path_rng.randrange(len(buffer)))
                if max_samples > 0 and emitted >= max_samples:
                    while buffer:
                        yield buffer.pop(path_rng.randrange(len(buffer)))
                    return

        while buffer:
            yield buffer.pop(path_rng.randrange(len(buffer)))
        if not repeat:
            return
        if pass_emitted == 0:
            raise RuntimeError(f"No valid image-edit full-JSON rows found in {len(pass_paths)} paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: ImageEditFullJsonConfig,
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


class ImageEditFullJsonLoader:
    """Loader mixed into ms-swift's ``BaseDatasetLoader`` at registration."""

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
            raise ValueError("image-edit full-JSON requires --streaming true")
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
            if not val_paths:
                raise ValueError("No image-edit full-JSON validation paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
            )
        else:
            if not train_paths:
                raise ValueError("No image-edit full-JSON training paths resolved")
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

    class SwiftImageEditFullJsonLoader(ImageEditFullJsonLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "multi-image", "sft", "structured-json", "image-edit"]),
        (
            config.val_dataset_name,
            ["vision", "multi-image", "sft", "structured-json", "image-edit", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImageEditFullJsonLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    apply_s3_image_load()
    print(
        f"[dlms] image-edit full-JSON stream enabled: dataset={config.dataset_name} prompt={PROMPT_VERSION}",
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
