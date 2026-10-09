"""CRAFT baseline4 streaming dataset plugin for ms-swift 4.4.

This is a small, callable migration of the reference FSDP data contract.  It
keeps the four exact CRAFT tasks and adds the region source as a fifth packing
channel:

* 70% ``region_layout`` (region source, layout-only target)
* 5.333% ``global_desc_bg``
* 10.667% ``full_json_no_region_desc``
* 8% ``full_json``
* 6% ``layout_only``

The module is intended for ``--custom_register_path``.  It reads the existing
``craft_hierarchy_jsonl_config`` mapping from ``JOB_CONFIG_YAML`` (or from
``CRAFT_MIXED4_CONFIG``), registers train/validation datasets, and optionally
applies strict channel-balanced packing when
``CRAFT_PACKING_REQUIRED_CHANNELS`` is non-empty. S3 access is lazy; importing
the plugin never reads a shard or manifest.

Megatron note: native ms-swift streaming uses a DP-group dispatcher.  In that
mode every model-parallel group must build the same full stream and the
dispatcher partitions samples, so this loader deliberately does not path-shard.
If a runtime supplies rank-local iterable loading, set
``SWIFT_RANK_LOCAL_ITERABLE_DATASET=true``; paths are then sharded by Megatron
data-parallel rank (never by global TP/PP rank).
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

DEFAULT_DATASET_NAME = "craft_hierarchy_baseline4_region7030"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
PRIMARY_SOURCE_WEIGHT = 0.30
REGION_SOURCE_WEIGHT = 0.70


@dataclass(frozen=True)
class TaskSpec:
    name: str
    probability: float
    system_prompt: str
    user_prompt: str
    layout_fields: tuple[str, ...]
    include_high_level_description: bool
    include_style_description: bool
    include_background: bool


TASKS = (
    TaskSpec(
        name="global_desc_bg",
        probability=0.17777777777777778,
        system_prompt=(
            "You extract global design context, background, and layout geometry from an image. "
            "Return only valid JSON in the requested compact schema."
        ),
        user_prompt=(
            "<image>\nGenerate JSON containing high_level_description, elements.background, and "
            "elements.layout. Every layout item must contain only type, label, parent, and bbox in "
            "xyxy_1000 coordinates. Do not output style_description, desc, or color_palette."
        ),
        layout_fields=("type", "label", "parent", "bbox"),
        include_high_level_description=True,
        include_style_description=False,
        include_background=True,
    ),
    TaskSpec(
        name="full_json_no_region_desc",
        probability=0.35555555555555557,
        system_prompt=(
            "You extract global design context, style, background, region color palettes, and "
            "layout geometry from an image. Return only valid JSON in the requested schema, "
            "without region-level captions."
        ),
        user_prompt=(
            "<image>\nGenerate JSON containing high_level_description, style_description, "
            "elements.background, and elements.layout. Every layout item must contain only type, "
            "label, parent, bbox in xyxy_1000 coordinates, and color_palette. Do not output a desc "
            "key or any region-level caption. Do not output prose or Markdown."
        ),
        layout_fields=("type", "label", "parent", "bbox", "color_palette"),
        include_high_level_description=True,
        include_style_description=True,
        include_background=True,
    ),
    TaskSpec(
        name="full_json",
        probability=0.26666666666666666,
        system_prompt=(
            "You produce a complete structured design description from an image. "
            "Return only valid JSON in the requested full schema."
        ),
        user_prompt=(
            "<image>\nGenerate the complete JSON containing high_level_description, "
            "style_description, elements.background, and elements.layout. Every layout item must "
            "contain type, label, parent, bbox in xyxy_1000 coordinates, desc, and color_palette."
        ),
        layout_fields=("type", "label", "parent", "bbox", "desc", "color_palette"),
        include_high_level_description=True,
        include_style_description=True,
        include_background=True,
    ),
    TaskSpec(
        name="layout_only",
        probability=0.20,
        system_prompt=(
            "You extract only layout hierarchy and geometry from an image. "
            "Return only valid JSON and no prose."
        ),
        user_prompt=(
            "<image>\nGenerate only elements.layout. Every layout item must contain only type, "
            "label, parent, and bbox in xyxy_1000 coordinates. Do not output "
            "high_level_description, style_description, elements.background, desc, or color_palette."
        ),
        layout_fields=("type", "label", "parent", "bbox"),
        include_high_level_description=False,
        include_style_description=False,
        include_background=False,
    ),
)
TASK_BY_NAME = {task.name: task for task in TASKS}

# This is the effective source x task distribution, not just the conditional
# distribution within the 30% primary CRAFT stream.
OVERALL_CHANNEL_WEIGHTS = {
    "global_desc_bg": PRIMARY_SOURCE_WEIGHT * TASK_BY_NAME["global_desc_bg"].probability,
    "full_json_no_region_desc": (
        PRIMARY_SOURCE_WEIGHT * TASK_BY_NAME["full_json_no_region_desc"].probability
    ),
    "full_json": PRIMARY_SOURCE_WEIGHT * TASK_BY_NAME["full_json"].probability,
    "layout_only": PRIMARY_SOURCE_WEIGHT * TASK_BY_NAME["layout_only"].probability,
    "region_layout": REGION_SOURCE_WEIGHT,
}

IMAGE_PATH_FIELDS = ("image_export_uri", "image_export_read_uri")
COMPLETED_STATUSES = frozenset({"completed"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class Mixed4Config:
    dataset_name: str
    val_dataset_name: str
    primary_paths: tuple[str, ...]
    region_paths: tuple[str, ...]
    val_path_count: int = 32
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 64
    seed: int = 42
    max_samples: int = 0


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def to_xyxy_1000(bbox: Any) -> list[float] | None:
    """Convert source ``[y1, x1, y2, x2]`` coordinates to xyxy_1000."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        y1, x1, y2, x2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2]


def annotation_view(annotation: Mapping[str, Any], task: TaskSpec) -> dict[str, Any]:
    elements = annotation.get("elements") or {}
    layout: list[dict[str, Any]] = []
    for element in elements.get("layout") or []:
        if not isinstance(element, Mapping):
            continue
        item: dict[str, Any] = {}
        for field in task.layout_fields:
            if field == "bbox":
                converted = to_xyxy_1000(element.get("bbox"))
                if converted is not None:
                    item["bbox"] = converted
            elif field in element:
                item[field] = element[field]
        if item:
            layout.append(item)

    view: dict[str, Any] = {}
    if task.include_high_level_description:
        view["high_level_description"] = annotation.get("high_level_description", "")
    if task.include_style_description:
        view["style_description"] = annotation.get("style_description", "")
    target_elements: dict[str, Any] = {}
    if task.include_background:
        target_elements["background"] = elements.get("background", "")
    target_elements["layout"] = layout
    view["elements"] = target_elements
    return view


def _annotation_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    final_caption = row.get("final_caption") or {}
    if isinstance(final_caption, Mapping):
        annotation = final_caption.get("annotation")
        if isinstance(annotation, Mapping):
            return annotation
    annotation = row.get("annotation")
    return annotation if isinstance(annotation, Mapping) else None


def _image_uri_from_row(row: Mapping[str, Any]) -> str:
    image = row.get("image") or {}
    if isinstance(image, Mapping):
        for field in IMAGE_PATH_FIELDS:
            value = image.get(field)
            if isinstance(value, str) and value:
                return value
    for field in IMAGE_PATH_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def make_row(
    source_row: Mapping[str, Any],
    *,
    source_path: str,
    task: TaskSpec,
    channel: str | None = None,
    source_name: str = "craft",
) -> dict[str, Any] | None:
    """Map one exported annotation to the exact Swift messages/images row."""
    status = str(source_row.get("status") or "")
    if status not in COMPLETED_STATUSES:
        return None
    annotation = _annotation_from_row(source_row)
    image_uri = _image_uri_from_row(source_row)
    if not annotation or not image_uri:
        return None
    target = annotation_view(annotation, task)
    return {
        "messages": [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt},
            {"role": "assistant", "content": compact_json(target)},
        ],
        "images": [image_uri],
        "channel": channel or task.name,
        "meta": {
            "jsonl_path": source_path,
            "source_name": source_name,
            "custom_id": str(source_row.get("custom_id") or ""),
            "sample_id": str(source_row.get("sample_id") or ""),
            "global_row": str(source_row.get("global_row") or ""),
            "dataset_revision_id": str(source_row.get("dataset_revision_id") or ""),
            "status": status,
            "image_path": image_uri,
        },
    }


def choose_primary_task(random_value: float) -> TaskSpec:
    if not 0.0 <= random_value <= 1.0:
        raise ValueError(f"random_value must be in [0, 1], got {random_value}")
    cumulative = 0.0
    for task in TASKS:
        cumulative += task.probability
        if random_value < cumulative:
            return task
    return TASKS[-1]


def shard_paths(paths: Sequence[str], rank: int, world_size: int) -> list[str]:
    """Deterministically assign shuffled physical shards to data ranks."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return [path for index, path in enumerate(paths) if index % world_size == rank]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in _TRUE_VALUES


def _megatron_data_parallel_context() -> tuple[int, int] | None:
    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    except (ImportError, RuntimeError, AssertionError):
        pass
    return None


def data_stream_shard_context() -> tuple[int, int]:
    """Return the rank that owns stream I/O, respecting Megatron DP topology.

    Explicit ``CRAFT_DATA_PARALLEL_{RANK,WORLD_SIZE}`` values are useful for
    offline workers.  Under Megatron rank-local iteration we query mpu.  Under
    stock ms-swift's dispatcher, only DP-group rank zero reads and scatters the
    common stream, so physical path sharding must be disabled (0/1).
    """
    explicit_rank = os.environ.get("CRAFT_DATA_PARALLEL_RANK")
    explicit_size = os.environ.get("CRAFT_DATA_PARALLEL_WORLD_SIZE")
    if explicit_rank is not None or explicit_size is not None:
        if explicit_rank is None or explicit_size is None:
            raise ValueError(
                "CRAFT_DATA_PARALLEL_RANK and CRAFT_DATA_PARALLEL_WORLD_SIZE must be set together"
            )
        return int(explicit_rank), int(explicit_size)

    megatron_context = _megatron_data_parallel_context()
    if megatron_context is not None:
        if _env_bool("SWIFT_RANK_LOCAL_ITERABLE_DATASET", False):
            return megatron_context
        return 0, 1

    return int(os.environ.get("RANK", "0")), max(int(os.environ.get("WORLD_SIZE", "1")), 1)


_S3_CLIENT = None


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        from botocore.config import Config

        _S3_CLIENT = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
    return _S3_CLIENT


def _split_s3_uri(uri: str) -> tuple[str, str]:
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def _read_uri_bytes(path: str) -> bytes:
    if path.startswith("s3://"):
        bucket, key = _split_s3_uri(path)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(path).read_bytes()


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    if path.startswith("s3://"):
        lines = io.BytesIO(_read_uri_bytes(path))
    else:
        lines = Path(path).open("rb")
    try:
        for line in lines:
            if line.strip():
                yield json.loads(line)
    finally:
        lines.close()


def _is_dataset_jsonl(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.endswith(".jsonl") and not name.startswith("example.")


def _paths_from_prefix(prefix: str) -> list[str]:
    if prefix.startswith("s3://"):
        bucket, key_prefix = _split_s3_uri(prefix.rstrip("/") + "/")
        paginator = _s3_client().get_paginator("list_objects_v2")
        paths = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            paths.extend(
                f"s3://{bucket}/{item['Key']}"
                for item in page.get("Contents", [])
                if item.get("Size", 0) > 0 and _is_dataset_jsonl(item["Key"])
            )
        return sorted(paths)

    root = Path(prefix)
    if root.is_file():
        return [str(root)] if root.stat().st_size > 0 and _is_dataset_jsonl(str(root)) else []
    return sorted(
        str(path)
        for path in root.rglob("*.jsonl")
        if path.is_file() and path.stat().st_size > 0 and _is_dataset_jsonl(str(path))
    )


def _paths_from_manifest(path: str) -> list[str]:
    text = _read_uri_bytes(path).decode("utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return [
            line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(document, list):
        entries = document
    elif isinstance(document, Mapping):
        entries = document.get("paths") or document.get("shards") or document.get("jsonl_paths") or []
    else:
        raise ValueError(f"{path} manifest must be a JSON list/object or newline-delimited paths")

    result = []
    for entry in entries:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            result.append(entry["path"])
        elif isinstance(entry, Mapping) and isinstance(entry.get("s3_path"), str):
            result.append(entry["s3_path"])
        else:
            raise ValueError(f"{path} manifest contains unsupported entry: {entry!r}")
    return result


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_as_list(item) if isinstance(item, (list, tuple)) else [item])
        return result
    return [value]


def _expand_paths(section: Mapping[str, Any], *, env_paths: str | None = None) -> tuple[str, ...]:
    paths = [str(path) for path in _as_list(env_paths or section.get("paths") or section.get("path"))]
    for manifest in _as_list(
        section.get("paths_manifests")
        or section.get("paths_manifest")
        or section.get("manifests")
        or section.get("manifest")
    ):
        paths.extend(_paths_from_manifest(str(manifest)))
    for prefix in _as_list(section.get("prefixes") or section.get("prefix")):
        paths.extend(_paths_from_prefix(str(prefix)))
    for root in _as_list(section.get("roots") or section.get("root")):
        paths.extend(_paths_from_prefix(f"{str(root).rstrip('/')}/jsonl"))
    return tuple(dict.fromkeys(paths))


def _config_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    config = document.get("craft_hierarchy_jsonl_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("craft_hierarchy_jsonl_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("CRAFT_MIXED4_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _assert_exact_task_mix(config: Mapping[str, Any]) -> None:
    raw_tasks = config.get("task_mix")
    if raw_tasks is None:
        return
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASKS):
        raise ValueError(
            "baseline4 task_mix must contain exactly global_desc_bg, "
            "full_json_no_region_desc, full_json, layout_only"
        )
    for actual, expected in zip(raw_tasks, TASKS):
        if not isinstance(actual, Mapping) or actual.get("name") != expected.name:
            raise ValueError(f"mixed4 task order/name drifted; expected {expected.name!r}")
        if not math.isclose(float(actual.get("probability")), expected.probability, abs_tol=1e-12):
            raise ValueError(f"mixed4 probability drifted for {expected.name}")
        checks = {
            "system_prompt": expected.system_prompt,
            "user_prompt": expected.user_prompt,
            "layout_fields": list(expected.layout_fields),
            "include_high_level_description": expected.include_high_level_description,
            "include_style_description": expected.include_style_description,
            "include_background": expected.include_background,
        }
        for field, expected_value in checks.items():
            if field in actual and actual[field] != expected_value:
                raise ValueError(f"mixed4 {expected.name}.{field} drifted from the golden contract")


def _region_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source_mix = config.get("source_mix") or {}
    if not source_mix:
        return {}
    if not isinstance(source_mix, Mapping):
        raise ValueError("source_mix must be a mapping")
    primary = source_mix.get("primary") or {}
    additional = source_mix.get("additional") or []
    if not isinstance(primary, Mapping) or not isinstance(additional, list) or len(additional) != 1:
        raise ValueError("mixed4 source_mix requires one primary and exactly one region source")
    region = additional[0]
    if not isinstance(region, Mapping):
        raise ValueError("mixed4 region source must be a mapping")
    if not math.isclose(float(primary.get("probability", 0)), PRIMARY_SOURCE_WEIGHT, abs_tol=1e-12):
        raise ValueError("mixed4 primary source probability must be 0.3")
    if not math.isclose(float(region.get("probability", 0)), REGION_SOURCE_WEIGHT, abs_tol=1e-12):
        raise ValueError("mixed4 region source probability must be 0.7")
    if region.get("task") != "layout_only" or region.get("channel") != "region_layout":
        raise ValueError("mixed4 region source must use task=layout_only and channel=region_layout")
    return region


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> Mixed4Config:
    config = _config_root(document if document is not None else _load_config_document())
    _assert_exact_task_mix(config)
    region = _region_section(config)

    primary_paths: tuple[str, ...] = ()
    region_paths: tuple[str, ...] = ()
    if resolve_paths:
        primary_override = os.environ.get("CRAFT_MIXED4_PRIMARY_PATHS") or os.environ.get(
            "CRAFT_HIERARCHY_JSONL_PATHS"
        )
        region_override = os.environ.get("CRAFT_MIXED4_REGION_PATHS")
        primary_paths = _expand_paths(config, env_paths=primary_override)
        region_paths = _expand_paths(region, env_paths=region_override)

    return Mixed4Config(
        dataset_name=(
            os.environ.get("CRAFT_HIERARCHY_DATASET_NAME")
            or str(config.get("dataset_name") or DEFAULT_DATASET_NAME)
        ),
        val_dataset_name=(
            os.environ.get("CRAFT_HIERARCHY_VAL_DATASET_NAME")
            or str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME)
        ),
        primary_paths=primary_paths,
        region_paths=region_paths,
        val_path_count=int(
            os.environ.get("CRAFT_HIERARCHY_VAL_PATH_COUNT") or config.get("val_path_count") or 32
        ),
        val_max_samples=int(
            os.environ.get("CRAFT_HIERARCHY_VAL_MAX_SAMPLES") or config.get("val_max_samples") or 1000
        ),
        rank_shard=_env_bool("CRAFT_HIERARCHY_RANK_SHARD", bool(config.get("rank_shard", True))),
        repeat=_env_bool("CRAFT_HIERARCHY_REPEAT", bool(config.get("repeat", True))),
        shuffle_paths=_env_bool("CRAFT_HIERARCHY_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))),
        sample_shuffle_buffer=int(
            os.environ.get("CRAFT_HIERARCHY_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 64
        ),
        seed=int(os.environ.get("CRAFT_HIERARCHY_DATA_SEED") or config.get("seed") or 42),
        max_samples=int(os.environ.get("CRAFT_HIERARCHY_MAX_SAMPLES") or config.get("max_samples") or 0),
    )


def _iter_craft_source(
    path_payload: Mapping[str, Any],
    *,
    seed: int,
    shuffle_paths: bool,
    sample_shuffle_buffer: int,
    rank_shard: bool,
    repeat: bool,
    max_samples: int,
    fixed_task_name: str = "",
    channel_override: str = "",
    source_name: str = "craft",
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    paths = list(path_payload["paths"])
    fixed_task = TASK_BY_NAME.get(fixed_task_name) if fixed_task_name else None
    if fixed_task_name and fixed_task is None:
        raise ValueError(f"Unknown fixed task: {fixed_task_name!r}")

    emitted = 0
    pass_index = 0
    if data_rank is None or data_world_size is None:
        data_rank, data_world_size = data_stream_shard_context()
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
            for source_row in iter_jsonl(path):
                task = fixed_task or choose_primary_task(task_rng.random())
                row = make_row(
                    source_row,
                    source_path=path,
                    task=task,
                    channel=channel_override,
                    source_name=source_name,
                )
                if row is None:
                    continue
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

        while buffer:
            yield buffer.pop(path_rng.randrange(len(buffer)))
        if not repeat:
            return
        if pass_emitted == 0:
            raise RuntimeError(f"No valid completed CRAFT rows found in {len(pass_paths)} JSONL paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: Mixed4Config,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    seed_offset: int = 0,
    fixed_task_name: str = "",
    channel_override: str = "",
    source_name: str = "craft",
):
    from datasets import Features, IterableDataset, Value

    # Resolve this in the distributed parent process.  DataLoader workers may
    # not retain Megatron's process-group objects, but they must use the same
    # DP/native-dispatch decision as their parent rank.
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
                "sample_id": Value("string"),
                "global_row": Value("string"),
                "dataset_revision_id": Value("string"),
                "status": Value("string"),
                "image_path": Value("string"),
            },
        }
    )
    return IterableDataset.from_generator(
        _iter_craft_source,
        gen_kwargs={
            # A nested list prevents HF from treating every physical path as
            # an independent generator shard and dropping paths on workers.
            "path_payload": {"paths": list(paths)},
            "seed": config.seed + seed_offset,
            "shuffle_paths": shuffle_paths,
            "sample_shuffle_buffer": config.sample_shuffle_buffer,
            "rank_shard": config.rank_shard,
            "repeat": repeat,
            "max_samples": max_samples,
            "fixed_task_name": fixed_task_name,
            "channel_override": channel_override,
            "source_name": source_name,
            "data_rank": data_rank,
            "data_world_size": data_world_size,
        },
        features=features,
    )


class CraftMixed4Loader:
    """Loader implementation mixed into ms-swift's ``BaseDatasetLoader``."""

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
            raise ValueError("CRAFT mixed4 requires --streaming true")
        config = load_runtime_config()
        requested_name = getattr(dataset_meta, "dataset_name", None) or config.dataset_name
        is_validation = requested_name == config.val_dataset_name

        if config.val_path_count > 0:
            val_paths = config.primary_paths[: config.val_path_count]
            train_paths = config.primary_paths[config.val_path_count :]
        else:
            val_paths = config.primary_paths
            train_paths = config.primary_paths

        if is_validation:
            if not val_paths:
                raise ValueError("No primary CRAFT validation JSONL paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
                source_name="craft",
            )
        else:
            if not train_paths:
                raise ValueError("No primary CRAFT training JSONL paths resolved after val split")
            if not config.region_paths:
                raise ValueError("No region-layout JSONL paths resolved")
            from datasets import interleave_datasets

            primary = _build_hf_stream(
                train_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=0,
                source_name="craft",
            )
            region = _build_hf_stream(
                config.region_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=0,
                seed_offset=1_000_003,
                fixed_task_name="layout_only",
                channel_override="region_layout",
                source_name="reve2_region",
            )
            dataset = interleave_datasets(
                [primary, region],
                probabilities=[PRIMARY_SOURCE_WEIGHT, REGION_SOURCE_WEIGHT],
                seed=config.seed,
                stopping_strategy="all_exhausted" if config.repeat else "first_exhausted",
            )
            if config.max_samples > 0:
                dataset = dataset.take(config.max_samples)

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
    """Register the plugin against the pinned ms-swift 4.4 public API."""
    from swift.dataset import MessagesPreprocessor
    from swift.dataset.dataset_meta import BaseDatasetLoader, DatasetMeta
    from swift.dataset.register import register_dataset

    from dlms.patches.channel_balanced_packing import apply as apply_channel_packing
    from dlms.patches.s3_image_load import apply as apply_s3_image_load

    class SwiftCraftMixed4Loader(CraftMixed4Loader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "sft", "structured-json", "craft", "mixed4"]),
        (
            config.val_dataset_name,
            ["vision", "sft", "structured-json", "craft", "mixed4", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftCraftMixed4Loader,
                tags=tags,
            ),
            exist_ok=True,
        )

    apply_s3_image_load()
    requested_channels = tuple(_as_list(os.environ.get("CRAFT_PACKING_REQUIRED_CHANNELS")))
    if requested_channels:
        apply_channel_packing(requested_channels)
    else:
        print(
            "[dlms] standard streaming packing enabled; baseline4 proportions are owned by the 70/30 interleave",
            flush=True,
        )
    return config.dataset_name, config.val_dataset_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


# custom_register_path executes this file for its side effects.  Keeping all
# third-party imports behind this guard makes row/packing golden tests offline.
if _swift_is_available() and not _env_bool("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", False):
    register_with_swift()
