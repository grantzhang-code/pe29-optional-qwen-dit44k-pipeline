"""Streaming Image2JSON dataset for the cleaned Jimmy 1M + 5M exports.

The source rows are accepted from their actual content, not from historical
``status``, ``cleaned`` or ``validation.strict_success`` metadata.  Every
accepted Jimmy row has the canonical ``final_caption.annotation`` structure
and a top-level ``s3_image_path``.  Converted ChartGalaxy rows use the same
annotation envelope and their audited nested ``image.image_export_uri``.  All
rows are projected into exactly two training tasks:

* 30% ``full_json``: descriptions, background and complete hierarchy;
* 70% ``layout_only``: hierarchy and geometry only.

An optional controlled treatment replaces 10 percentage points of the
``layout_only`` stream with the converted ChartGalaxy rev8 training split.  In
that mode the physical source mix is 90% Jimmy and 10% ChartGalaxy, while the
overall task mix remains exactly 30% ``full_json`` and 70% ``layout_only``.
ChartGalaxy rows are provenance-gated and can only produce ``layout_only``.

Both element-level and style-level ``color_palette`` values are intentionally
removed.  Exact non-empty ``text`` is retained only on elements whose type is
``text``.  Source bboxes are ``[y1, x1, y2, x2]`` and targets are
``[x1, y1, x2, y2]`` in the 0--1000 coordinate system.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Import the established S3/config/DP helpers without registering mixed4.
_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        TaskSpec,
        _as_list,
        _config_root,
        _env_bool,
        _expand_paths,
        _load_config_document,
        _paths_from_prefix,
        _read_uri_bytes,
        compact_json,
        data_stream_shard_context,
        shard_paths,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


DEFAULT_DATASET_NAME = "image2json_jimmy_1m5m_two_task_qwen35_4b_mtp"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
DEFAULT_SOURCE_NAME = "jimmy_vote_gt50_1m5m"
FULL_JSON_WEIGHT = 0.30
LAYOUT_ONLY_WEIGHT = 0.70
PRIMARY_SOURCE_WEIGHT_WITH_CHART = 0.90
CHARTGALAXY_SOURCE_WEIGHT = 0.10
PRIMARY_FULL_JSON_WEIGHT_WITH_CHART = FULL_JSON_WEIGHT / PRIMARY_SOURCE_WEIGHT_WITH_CHART
PRIMARY_LAYOUT_ONLY_WEIGHT_WITH_CHART = (
    LAYOUT_ONLY_WEIGHT - CHARTGALAXY_SOURCE_WEIGHT
) / PRIMARY_SOURCE_WEIGHT_WITH_CHART
CHARTGALAXY_DATASET_REVISION_ID = "chartgalaxy_craft_v2_rev8"
CHARTGALAXY_SOURCE_KIND = "chartgalaxy_craft_rev8"
OVERALL_TASK_WEIGHTS = {
    "full_json": FULL_JSON_WEIGHT,
    "layout_only": LAYOUT_ONLY_WEIGHT,
}

FULL_JSON_SYSTEM_PROMPT = (
    "You produce a complete structured design description from an image. "
    "Return only valid JSON in the requested full schema."
)
FULL_JSON_USER_PROMPT = (
    "<image>\nGenerate the complete JSON containing high_level_description, "
    "style_description, elements.background, and elements.layout. Every layout item must "
    "contain type, label, parent, bbox in xyxy_1000 coordinates, and desc. When exact visible "
    'text is available for an item of type "text", include it in the text field.'
)
LAYOUT_ONLY_SYSTEM_PROMPT = (
    "You extract only layout hierarchy and geometry from an image. Return only valid JSON and no prose."
)
LAYOUT_ONLY_USER_PROMPT = (
    "<image>\nGenerate only elements.layout. Every layout item must contain only type, label, "
    "parent, and bbox in xyxy_1000 coordinates. Do not output high_level_description, "
    "style_description, elements.background, or desc."
)

TASKS = (
    TaskSpec(
        name="full_json",
        probability=FULL_JSON_WEIGHT,
        system_prompt=FULL_JSON_SYSTEM_PROMPT,
        user_prompt=FULL_JSON_USER_PROMPT,
        layout_fields=("type", "label", "parent", "bbox", "text", "desc"),
        include_high_level_description=True,
        include_style_description=True,
        include_background=True,
    ),
    TaskSpec(
        name="layout_only",
        probability=LAYOUT_ONLY_WEIGHT,
        system_prompt=LAYOUT_ONLY_SYSTEM_PROMPT,
        user_prompt=LAYOUT_ONLY_USER_PROMPT,
        layout_fields=("type", "label", "parent", "bbox"),
        include_high_level_description=False,
        include_style_description=False,
        include_background=False,
    ),
)
TASK_BY_NAME = {task.name: task for task in TASKS}


@dataclass(frozen=True)
class Image2JSONConfig:
    dataset_name: str
    val_dataset_name: str
    paths: tuple[str, ...]
    source_name: str = DEFAULT_SOURCE_NAME
    val_path_count: int = 32
    val_max_samples: int = 1000
    rank_shard: bool = True
    val_rank_shard: bool = False
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 8192
    seed: int = 42
    max_samples: int = 0
    layout_only_paths: tuple[str, ...] = ()
    layout_only_source_name: str = ""
    primary_source_weight: float = 1.0
    layout_only_source_weight: float = 0.0

    @property
    def has_layout_only_source_mix(self) -> bool:
        return self.layout_only_source_weight > 0.0


@dataclass(frozen=True)
class _PreparedSource:
    image_uri: str
    annotation: Mapping[str, Any]
    custom_id: str
    global_row: str
    model: str
    forced_task_name: str = ""


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_s3_uri(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    uri = value.strip()
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not uri.startswith("s3://") or not bucket or not separator or not key:
        return None
    return uri


def _metadata_string(value: Any) -> str:
    return "" if value is None else str(value)


def _integer_setting(
    config: Mapping[str, Any],
    *,
    env_name: str,
    config_name: str,
    default: int,
) -> int:
    value = os.environ.get(env_name)
    if value is None or value == "":
        value = config.get(config_name)
    if value is None or value == "":
        value = default
    return int(value)


def _bbox_to_xyxy_1000(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) for item in value):
        return None
    try:
        y1, x1, y2, x2 = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    coordinates = (y1, x1, y2, x2)
    if not all(math.isfinite(item) and 0.0 <= item <= 1000.0 for item in coordinates):
        return None
    if y1 > y2 or x1 > x2:
        return None
    # xyxy_1000 is an integer-coordinate target schema. Parsing as float is
    # useful for rejecting NaN/Inf and accepting integral numeric strings, but
    # retaining those temporary floats would serialize every coordinate as
    # ``412.0`` and waste target tokens. Reject genuinely fractional source
    # coordinates instead of silently rounding them.
    if not all(item.is_integer() for item in coordinates):
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def _without_color_palette(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_color_palette(item) for key, item in value.items() if key != "color_palette"}
    if isinstance(value, list):
        return [_without_color_palette(item) for item in value]
    if isinstance(value, tuple):
        return [_without_color_palette(item) for item in value]
    return value


def _without_null_mapping_values(value: Any) -> Any:
    """Drop null-valued mapping entries without reordering the remaining keys."""
    if isinstance(value, Mapping):
        return {key: _without_null_mapping_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_null_mapping_values(item) for item in value]
    if isinstance(value, tuple):
        return [_without_null_mapping_values(item) for item in value]
    return value


def _normalize_annotation(annotation: Any) -> dict[str, Any] | None:
    if not isinstance(annotation, Mapping):
        return None
    high_level = annotation.get("high_level_description")
    style = annotation.get("style_description")
    elements = annotation.get("elements")
    if not _nonempty_string(high_level) or not isinstance(style, Mapping):
        return None
    if not isinstance(elements, Mapping):
        return None
    background = elements.get("background")
    source_layout = elements.get("layout")
    if not _nonempty_string(background) or not isinstance(source_layout, list) or not source_layout:
        return None

    layout: list[dict[str, Any]] = []
    labels: set[str] = set()
    parent_by_label: dict[str, str | None] = {}
    for source_element in source_layout:
        if not isinstance(source_element, Mapping):
            return None
        element_type = source_element.get("type")
        label = source_element.get("label")
        if not _nonempty_string(element_type) or not _nonempty_string(label):
            return None
        if label in labels or "parent" not in source_element:
            return None
        parent = source_element.get("parent")
        if parent is not None and not _nonempty_string(parent):
            return None
        bbox = _bbox_to_xyxy_1000(source_element.get("bbox"))
        desc = source_element.get("desc")
        if bbox is None or not _nonempty_string(desc):
            return None

        item: dict[str, Any] = {
            "type": element_type,
            "label": label,
            "parent": parent,
            "bbox": bbox,
        }
        text = source_element.get("text")
        if element_type == "text" and _nonempty_string(text):
            item["text"] = text
        item["desc"] = desc
        layout.append(item)
        labels.add(label)
        parent_by_label[label] = parent

    # Validate the hierarchy from content instead of trusting validation flags.
    for label, parent in parent_by_label.items():
        if parent is None or parent.upper() == "ROOT":
            continue
        if parent == label or parent not in labels:
            return None
        seen = {label}
        cursor: str | None = parent
        while cursor is not None and cursor.upper() != "ROOT":
            if cursor in seen:
                return None
            seen.add(cursor)
            cursor = parent_by_label.get(cursor)

    return {
        "high_level_description": high_level,
        # These optional style fields are absent semantically when their value
        # is null. Keep source order for every remaining key. This cleanup is
        # deliberately scoped to style_description: layout parent=null is part
        # of the required hierarchy schema and must remain in the target.
        "style_description": _without_null_mapping_values(_without_color_palette(style)),
        "elements": {
            "background": background,
            "layout": layout,
        },
    }


def _has_chartgalaxy_marker(source_row: Mapping[str, Any]) -> bool:
    training = source_row.get("training")
    return (
        source_row.get("dataset_revision_id") == CHARTGALAXY_DATASET_REVISION_ID
        or (
            isinstance(training, Mapping)
            and training.get("source_kind") == CHARTGALAXY_SOURCE_KIND
        )
    )


def _normalize_chartgalaxy_layout(annotation: Any) -> dict[str, Any] | None:
    """Validate only the geometry contract used by the forced layout task.

    This intentionally does not relax or refactor the Jimmy full-annotation
    normalizer above.  ChartGalaxy never enters ``full_json``, so unrelated
    global/style/background/description fields cannot reject or leak into its
    target.
    """
    if not isinstance(annotation, Mapping):
        return None
    elements = annotation.get("elements")
    if not isinstance(elements, Mapping):
        return None
    source_layout = elements.get("layout")
    if not isinstance(source_layout, list) or not source_layout:
        return None

    layout: list[dict[str, Any]] = []
    labels: set[str] = set()
    parent_by_label: dict[str, str | None] = {}
    for source_element in source_layout:
        if not isinstance(source_element, Mapping):
            return None
        element_type = source_element.get("type")
        label = source_element.get("label")
        if not _nonempty_string(element_type) or not _nonempty_string(label):
            return None
        if label in labels or "parent" not in source_element:
            return None
        parent = source_element.get("parent")
        if parent is not None and not _nonempty_string(parent):
            return None
        bbox = _bbox_to_xyxy_1000(source_element.get("bbox"))
        if bbox is None:
            return None
        layout.append(
            {
                "type": element_type,
                "label": label,
                "parent": parent,
                "bbox": bbox,
            }
        )
        labels.add(label)
        parent_by_label[label] = parent

    for label, parent in parent_by_label.items():
        if parent is None or parent.upper() == "ROOT":
            continue
        if parent == label or parent not in labels:
            return None
        seen = {label}
        cursor: str | None = parent
        while cursor is not None and cursor.upper() != "ROOT":
            if cursor in seen:
                return None
            seen.add(cursor)
            cursor = parent_by_label.get(cursor)

    return {"elements": {"layout": layout}}


def _prepare_chartgalaxy_source_row(source_row: Mapping[str, Any]) -> _PreparedSource | None:
    training = source_row.get("training")
    if (
        source_row.get("dataset_revision_id") != CHARTGALAXY_DATASET_REVISION_ID
        or not isinstance(training, Mapping)
        or training.get("source_kind") != CHARTGALAXY_SOURCE_KIND
        or training.get("channel") != "chartgalaxy"
        or "layout_only" not in _as_list(training.get("available_tasks"))
    ):
        return None
    final_caption = source_row.get("final_caption")
    image = source_row.get("image")
    if not isinstance(final_caption, Mapping) or not isinstance(image, Mapping):
        return None
    annotation = _normalize_chartgalaxy_layout(final_caption.get("annotation"))
    image_uri = _normalized_s3_uri(
        image.get("image_export_uri") or image.get("image_export_read_uri")
    )
    if annotation is None or image_uri is None:
        return None
    return _PreparedSource(
        image_uri=image_uri,
        annotation=annotation,
        custom_id=_metadata_string(source_row.get("custom_id")),
        global_row=_metadata_string(source_row.get("global_row")),
        model=_metadata_string(source_row.get("model")),
        forced_task_name="layout_only",
    )


def _prepare_source_row(source_row: Any) -> _PreparedSource | None:
    if not isinstance(source_row, Mapping):
        return None
    if _has_chartgalaxy_marker(source_row):
        return _prepare_chartgalaxy_source_row(source_row)
    final_caption = source_row.get("final_caption")
    if not isinstance(final_caption, Mapping):
        return None
    annotation = _normalize_annotation(final_caption.get("annotation"))
    image_uri = _normalized_s3_uri(source_row.get("s3_image_path"))
    if annotation is None or image_uri is None:
        return None
    return _PreparedSource(
        image_uri=image_uri,
        annotation=annotation,
        custom_id=_metadata_string(source_row.get("custom_id")),
        global_row=_metadata_string(source_row.get("global_row")),
        model=_metadata_string(source_row.get("model")),
    )


def annotation_view(annotation: Mapping[str, Any], task: TaskSpec) -> dict[str, Any]:
    source_elements = annotation["elements"]
    layout = []
    for source_element in source_elements["layout"]:
        item = {field: source_element[field] for field in task.layout_fields if field in source_element}
        layout.append(item)

    target: dict[str, Any] = {}
    if task.include_high_level_description:
        target["high_level_description"] = annotation["high_level_description"]
    if task.include_style_description:
        target["style_description"] = annotation["style_description"]
    target_elements: dict[str, Any] = {}
    if task.include_background:
        target_elements["background"] = source_elements["background"]
    target_elements["layout"] = layout
    target["elements"] = target_elements
    return _without_color_palette(target)


def _make_prepared_row(
    prepared: _PreparedSource,
    *,
    source_path: str,
    task: TaskSpec,
    source_name: str,
) -> dict[str, Any]:
    if prepared.forced_task_name:
        task = TASK_BY_NAME[prepared.forced_task_name]
    return {
        "messages": [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt},
            {"role": "assistant", "content": compact_json(annotation_view(prepared.annotation, task))},
        ],
        "images": [prepared.image_uri],
        "channel": task.name,
        "meta": {
            "jsonl_path": source_path,
            "source_name": source_name,
            "custom_id": prepared.custom_id,
            "global_row": prepared.global_row,
            "model": prepared.model,
            "image_path": prepared.image_uri,
        },
    }


def make_image2json_row(
    source_row: Any,
    *,
    source_path: str,
    task: TaskSpec,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> dict[str, Any] | None:
    """Project one valid Jimmy row without consulting metadata status flags."""
    prepared = _prepare_source_row(source_row)
    if prepared is None:
        return None
    return _make_prepared_row(
        prepared,
        source_path=source_path,
        task=task,
        source_name=source_name,
    )


def choose_task(
    random_value: float,
    *,
    full_json_weight: float = FULL_JSON_WEIGHT,
) -> TaskSpec:
    if not 0.0 <= random_value <= 1.0:
        raise ValueError(f"random_value must be in [0, 1], got {random_value}")
    if not 0.0 <= full_json_weight <= 1.0:
        raise ValueError(f"full_json_weight must be in [0, 1], got {full_json_weight}")
    return TASKS[0] if random_value < full_json_weight else TASKS[1]


def _iter_jsonl_objects(path: str) -> Iterator[Mapping[str, Any]]:
    """Read a shard safely; malformed records do not kill a multi-node run."""
    if path.startswith("s3://"):
        lines = io.BytesIO(_read_uri_bytes(path))
    else:
        lines = Path(path).open("rb")
    skipped = 0
    first_bad_line = 0
    try:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped += 1
                first_bad_line = first_bad_line or line_number
                continue
            if not isinstance(value, Mapping):
                skipped += 1
                first_bad_line = first_bad_line or line_number
                continue
            yield value
    finally:
        lines.close()
    if skipped:
        print(
            f"[dlms] Image2JSON skipped {skipped} malformed/non-object records in {path}; "
            f"first bad line={first_bad_line}",
            flush=True,
        )


def _assert_exact_task_mix(config: Mapping[str, Any]) -> None:
    raw_tasks = config.get("task_mix")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASKS):
        raise ValueError("Image2JSON task_mix must contain exactly full_json and layout_only")
    for actual, expected in zip(raw_tasks, TASKS):
        if not isinstance(actual, Mapping) or actual.get("name") != expected.name:
            raise ValueError(f"Image2JSON task order/name drifted; expected {expected.name!r}")
        if not math.isclose(float(actual.get("probability", -1)), expected.probability, abs_tol=1e-12):
            raise ValueError(f"Image2JSON probability drifted for {expected.name}")
        checks = {
            "system_prompt": expected.system_prompt,
            "user_prompt": expected.user_prompt,
            "layout_fields": list(expected.layout_fields),
            "include_high_level_description": expected.include_high_level_description,
            "include_style_description": expected.include_style_description,
            "include_background": expected.include_background,
        }
        for field, expected_value in checks.items():
            if actual.get(field) != expected_value:
                raise ValueError(f"Image2JSON {expected.name}.{field} drifted from the golden contract")


def _source_mix_sections(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source_mix = config.get("source_mix")
    if source_mix is None:
        return {}, {}
    if not isinstance(source_mix, Mapping):
        raise ValueError("Image2JSON source_mix must be a mapping")
    primary = source_mix.get("primary")
    additional = source_mix.get("additional")
    if not isinstance(primary, Mapping) or not isinstance(additional, list) or len(additional) != 1:
        raise ValueError("Image2JSON source_mix requires one primary and one layout-only source")
    layout_only_source = additional[0]
    if not isinstance(layout_only_source, Mapping):
        raise ValueError("Image2JSON layout-only source must be a mapping")
    if not math.isclose(
        float(primary.get("probability", -1)),
        PRIMARY_SOURCE_WEIGHT_WITH_CHART,
        abs_tol=1e-12,
    ):
        raise ValueError("Image2JSON primary source probability must be 0.90")
    if not math.isclose(
        float(layout_only_source.get("probability", -1)),
        CHARTGALAXY_SOURCE_WEIGHT,
        abs_tol=1e-12,
    ):
        raise ValueError("Image2JSON ChartGalaxy source probability must be 0.10")
    if (
        layout_only_source.get("task") != "layout_only"
        or layout_only_source.get("channel") != "layout_only"
    ):
        raise ValueError("Image2JSON ChartGalaxy source must use task/channel=layout_only")
    if not _nonempty_string(primary.get("name")) or not _nonempty_string(
        layout_only_source.get("name")
    ):
        raise ValueError("Image2JSON source_mix names must be non-empty strings")
    return primary, layout_only_source


def _expand_paths_with_prefix_logging(
    config: Mapping[str, Any], *, env_paths: str | None = None
) -> tuple[str, ...]:
    """Resolve every configured prefix independently and report its shard count."""
    prefixes = [str(prefix) for prefix in _as_list(config.get("prefixes") or config.get("prefix"))]
    if not prefixes:
        return _expand_paths(config, env_paths=env_paths)

    # Let the shared resolver handle explicit paths/manifests/roots, but remove
    # prefixes so each one is listed exactly once below and can be audited.
    non_prefix_config = dict(config)
    non_prefix_config.pop("prefixes", None)
    non_prefix_config.pop("prefix", None)
    paths = list(_expand_paths(non_prefix_config, env_paths=env_paths))
    for prefix in prefixes:
        try:
            prefix_paths = _paths_from_prefix(prefix)
        except Exception as exc:
            print(
                f"[dlms] Image2JSON data prefix ERROR: {prefix} ({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        state = "OK" if prefix_paths else "MISSING"
        print(
            f"[dlms] Image2JSON data prefix {state}: {prefix} non_empty_jsonl_shards={len(prefix_paths)}",
            flush=True,
        )
        if prefix_paths:
            paths.extend(prefix_paths)
    return tuple(dict.fromkeys(paths))


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> Image2JSONConfig:
    config = _config_root(document if document is not None else _load_config_document())
    _assert_exact_task_mix(config)
    primary_source, layout_only_source = _source_mix_sections(config)
    paths: tuple[str, ...] = ()
    layout_only_paths: tuple[str, ...] = ()
    if resolve_paths:
        path_override = os.environ.get("IMAGE2JSON_JSONL_PATHS") or os.environ.get(
            "CRAFT_HIERARCHY_JSONL_PATHS"
        )
        paths = _expand_paths_with_prefix_logging(config, env_paths=path_override)
        if layout_only_source:
            layout_path_override = os.environ.get("IMAGE2JSON_LAYOUT_ONLY_JSONL_PATHS")
            layout_only_paths = _expand_paths(
                layout_only_source,
                env_paths=layout_path_override,
            )
            state = "OK" if layout_only_paths else "MISSING"
            print(
                f"[dlms] Image2JSON fixed layout-only source {state}: "
                f"{layout_only_source['name']} jsonl_shards={len(layout_only_paths)}",
                flush=True,
            )

    return Image2JSONConfig(
        dataset_name=(
            os.environ.get("CRAFT_HIERARCHY_DATASET_NAME")
            or str(config.get("dataset_name") or DEFAULT_DATASET_NAME)
        ),
        val_dataset_name=(
            os.environ.get("CRAFT_HIERARCHY_VAL_DATASET_NAME")
            or str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME)
        ),
        paths=paths,
        source_name=str(
            primary_source.get("name")
            if primary_source
            else config.get("source_name") or DEFAULT_SOURCE_NAME
        ),
        val_path_count=_integer_setting(
            config,
            env_name="CRAFT_HIERARCHY_VAL_PATH_COUNT",
            config_name="val_path_count",
            default=32,
        ),
        val_max_samples=_integer_setting(
            config,
            env_name="CRAFT_HIERARCHY_VAL_MAX_SAMPLES",
            config_name="val_max_samples",
            default=1000,
        ),
        rank_shard=_env_bool("CRAFT_HIERARCHY_RANK_SHARD", bool(config.get("rank_shard", True))),
        val_rank_shard=_env_bool("CRAFT_HIERARCHY_VAL_RANK_SHARD", bool(config.get("val_rank_shard", False))),
        repeat=_env_bool("CRAFT_HIERARCHY_REPEAT", bool(config.get("repeat", True))),
        shuffle_paths=_env_bool("CRAFT_HIERARCHY_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))),
        sample_shuffle_buffer=_integer_setting(
            config,
            env_name="CRAFT_HIERARCHY_SAMPLE_SHUFFLE_BUFFER",
            config_name="sample_shuffle_buffer",
            default=8192,
        ),
        seed=_integer_setting(
            config,
            env_name="CRAFT_HIERARCHY_DATA_SEED",
            config_name="seed",
            default=42,
        ),
        max_samples=_integer_setting(
            config,
            env_name="CRAFT_HIERARCHY_MAX_SAMPLES",
            config_name="max_samples",
            default=0,
        ),
        layout_only_paths=layout_only_paths,
        layout_only_source_name=str(layout_only_source.get("name") or ""),
        primary_source_weight=(
            float(primary_source.get("probability")) if primary_source else 1.0
        ),
        layout_only_source_weight=(
            float(layout_only_source.get("probability")) if layout_only_source else 0.0
        ),
    )


def _iter_image2json_source(
    path_payload: Mapping[str, Any],
    *,
    seed: int,
    shuffle_paths: bool,
    sample_shuffle_buffer: int,
    rank_shard: bool,
    repeat: bool,
    max_samples: int,
    fixed_task_name: str = "",
    source_name: str = DEFAULT_SOURCE_NAME,
    full_json_weight: float = FULL_JSON_WEIGHT,
    require_chartgalaxy: bool = False,
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    paths = list(path_payload["paths"])
    fixed_task = TASK_BY_NAME.get(fixed_task_name) if fixed_task_name else None
    if fixed_task_name and fixed_task is None:
        raise ValueError(f"Unknown fixed task: {fixed_task_name!r}")
    if data_rank is None or data_world_size is None:
        data_rank, data_world_size = data_stream_shard_context()

    emitted = 0
    pass_index = 0
    while True:
        pass_paths = list(paths)
        path_rng = random.Random(seed + pass_index)
        if shuffle_paths:
            path_rng.shuffle(pass_paths)
        if rank_shard:
            pass_paths = shard_paths(pass_paths, data_rank, data_world_size)
        if not pass_paths:
            return

        task_rng = random.Random(seed + pass_index * 1_000_003 + data_rank * 10_007)
        buffer: list[dict[str, Any]] = []
        pass_emitted = 0
        for path in pass_paths:
            rejected_in_path = 0
            for source_row in _iter_jsonl_objects(path):
                prepared = _prepare_source_row(source_row)
                if prepared is None or (
                    require_chartgalaxy and prepared.forced_task_name != "layout_only"
                ):
                    rejected_in_path += 1
                    continue
                # Invalid content must not consume RNG and perturb later tasks.
                if prepared.forced_task_name:
                    task = TASK_BY_NAME[prepared.forced_task_name]
                else:
                    task = fixed_task or choose_task(
                        task_rng.random(),
                        full_json_weight=full_json_weight,
                    )
                row = _make_prepared_row(
                    prepared,
                    source_path=path,
                    task=task,
                    source_name=source_name,
                )
                emitted += 1
                pass_emitted += 1
                if sample_shuffle_buffer <= 1:
                    yield row
                else:
                    buffer.append(row)
                    if len(buffer) >= sample_shuffle_buffer:
                        yield buffer.pop(path_rng.randrange(len(buffer)))

                if max_samples > 0 and emitted >= max_samples:
                    while buffer:
                        yield buffer.pop(path_rng.randrange(len(buffer)))
                    return

            if rejected_in_path:
                print(
                    f"[dlms] Image2JSON rejected {rejected_in_path} structurally invalid rows in {path}",
                    flush=True,
                )

        while buffer:
            yield buffer.pop(path_rng.randrange(len(buffer)))
        if not repeat:
            return
        if pass_emitted == 0:
            raise RuntimeError(f"No valid Image2JSON rows found in {len(pass_paths)} JSONL paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: Image2JSONConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    fixed_task_name: str = "",
    rank_shard: bool | None = None,
    seed_offset: int = 0,
    full_json_weight: float = FULL_JSON_WEIGHT,
    require_chartgalaxy: bool = False,
    source_name: str | None = None,
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
                "custom_id": Value("string"),
                "global_row": Value("string"),
                "model": Value("string"),
                "image_path": Value("string"),
            },
        }
    )
    return IterableDataset.from_generator(
        _iter_image2json_source,
        gen_kwargs={
            "path_payload": {"paths": list(paths)},
            "seed": config.seed + seed_offset,
            "shuffle_paths": shuffle_paths,
            "sample_shuffle_buffer": config.sample_shuffle_buffer,
            "rank_shard": config.rank_shard if rank_shard is None else rank_shard,
            "repeat": repeat,
            "max_samples": max_samples,
            "fixed_task_name": fixed_task_name,
            "source_name": source_name or config.source_name,
            "full_json_weight": full_json_weight,
            "require_chartgalaxy": require_chartgalaxy,
            "data_rank": data_rank,
            "data_world_size": data_world_size,
        },
        features=features,
    )


class Image2JSONTwoTaskLoader:
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
            raise ValueError("Image2JSON requires --streaming true")
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
                raise ValueError("No Image2JSON validation JSONL paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
                fixed_task_name="full_json",
                rank_shard=config.val_rank_shard,
            )
        else:
            if not train_paths:
                raise ValueError("No Image2JSON training JSONL paths resolved after val split")
            if config.has_layout_only_source_mix:
                if not config.layout_only_paths:
                    raise ValueError("No ChartGalaxy layout-only JSONL paths resolved")
                from datasets import interleave_datasets

                primary = _build_hf_stream(
                    train_paths,
                    config=config,
                    repeat=config.repeat,
                    shuffle_paths=config.shuffle_paths,
                    max_samples=0,
                    full_json_weight=PRIMARY_FULL_JSON_WEIGHT_WITH_CHART,
                )
                layout_only = _build_hf_stream(
                    config.layout_only_paths,
                    config=config,
                    repeat=config.repeat,
                    shuffle_paths=config.shuffle_paths,
                    max_samples=0,
                    fixed_task_name="layout_only",
                    seed_offset=1_000_003,
                    require_chartgalaxy=True,
                    source_name=config.layout_only_source_name,
                )
                data_rank, _ = data_stream_shard_context()
                dataset = interleave_datasets(
                    [primary, layout_only],
                    probabilities=[
                        config.primary_source_weight,
                        config.layout_only_source_weight,
                    ],
                    # Independent DP replicas should not all choose the same
                    # source for a given global step. TP peers share one DP
                    # rank, so this remains model-parallel safe.
                    seed=config.seed + data_rank * 10_007,
                    stopping_strategy="all_exhausted" if config.repeat else "first_exhausted",
                )
                if config.max_samples > 0:
                    dataset = dataset.take(config.max_samples)
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

    class SwiftImage2JSONTwoTaskLoader(Image2JSONTwoTaskLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "sft", "structured-json", "image2json", "jimmy-1m5m"]),
        (
            config.val_dataset_name,
            ["vision", "sft", "structured-json", "image2json", "jimmy-1m5m", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImage2JSONTwoTaskLoader,
                tags=tags,
            ),
            exist_ok=True,
        )

    apply_s3_image_load()
    requested_channels = tuple(_as_list(os.environ.get("CRAFT_PACKING_REQUIRED_CHANNELS")))
    if requested_channels:
        raise ValueError("Image2JSON uses ordinary packing; CRAFT_PACKING_REQUIRED_CHANNELS must be empty")
    source_message = "Jimmy vote-gt50 1M+5M"
    if config.has_layout_only_source_mix:
        source_message += "+ChartGalaxy rev8 source mix=0.90/0.10"
    print(
        f"[dlms] Image2JSON stream enabled: {source_message}; "
        "tasks full_json=0.30/layout_only=0.70; color_palette disabled",
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
