"""Canonical CRAFT/Reve streaming dataset for two-task MTP training.

The source and task mixtures are deliberately separate:

* 44% token-compressed CRAFT (short or medium bucket), split 15/22
  ``full_json`` and 7/22 ``layout_only``;
* 56% Reve2 region data, always ``layout_only``.

The resulting externally visible channels are exactly 30% ``full_json`` and
70% ``layout_only``.  Full JSON preserves the optional exact ``text`` value on
layout elements; layout-only targets retain the historical geometry schema.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

# Reuse the proven S3, canonical-row and DP rank-local helpers without
# registering the legacy mixed4 dataset as a side effect.
_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        TASK_BY_NAME as _MIXED4_TASK_BY_NAME,
        TaskSpec,
        _as_list,
        _config_root,
        _env_bool,
        _expand_paths,
        _load_config_document,
        data_stream_shard_context,
        iter_jsonl,
        make_row,
        shard_paths,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


DEFAULT_DATASET_NAME = "craft_short_v4_two_task_qwen35_4b_mtp"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"

SHORT_SOURCE_WEIGHT = 0.44
REGION_SOURCE_WEIGHT = 0.56
PRIMARY_FULL_JSON_WEIGHT = 15 / 22
PRIMARY_LAYOUT_ONLY_WEIGHT = 7 / 22

OVERALL_TASK_WEIGHTS = {
    "full_json": SHORT_SOURCE_WEIGHT * PRIMARY_FULL_JSON_WEIGHT,
    "layout_only": (SHORT_SOURCE_WEIGHT * PRIMARY_LAYOUT_ONLY_WEIGHT + REGION_SOURCE_WEIGHT),
}


def _full_json_task() -> TaskSpec:
    source = _MIXED4_TASK_BY_NAME["full_json"]
    return TaskSpec(
        name=source.name,
        probability=PRIMARY_FULL_JSON_WEIGHT,
        system_prompt=source.system_prompt,
        user_prompt=(
            "<image>\nGenerate the complete JSON containing high_level_description, "
            "style_description, elements.background, and elements.layout. Every layout item must "
            "contain type, label, parent, bbox in xyxy_1000 coordinates, desc, and color_palette. "
            'When exact visible text is available for an item of type "text", include it in the '
            "text field."
        ),
        layout_fields=("type", "label", "parent", "bbox", "text", "desc", "color_palette"),
        include_high_level_description=source.include_high_level_description,
        include_style_description=source.include_style_description,
        include_background=source.include_background,
    )


def _layout_only_task() -> TaskSpec:
    source = _MIXED4_TASK_BY_NAME["layout_only"]
    return TaskSpec(
        name=source.name,
        probability=PRIMARY_LAYOUT_ONLY_WEIGHT,
        system_prompt=source.system_prompt,
        user_prompt=source.user_prompt,
        layout_fields=source.layout_fields,
        include_high_level_description=source.include_high_level_description,
        include_style_description=source.include_style_description,
        include_background=source.include_background,
    )


TASKS = (_full_json_task(), _layout_only_task())
TASK_BY_NAME = {task.name: task for task in TASKS}


@dataclass(frozen=True)
class ShortTwoTaskConfig:
    dataset_name: str
    val_dataset_name: str
    primary_paths: tuple[str, ...]
    region_paths: tuple[str, ...]
    primary_source_name: str = "tokencompress_short"
    val_path_count: int = 32
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 8192
    seed: int = 42
    max_samples: int = 0


def _assert_exact_task_mix(config: Mapping[str, Any]) -> None:
    raw_tasks = config.get("task_mix")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASKS):
        raise ValueError("CRAFT two-task task_mix must contain exactly full_json and layout_only")
    for actual, expected in zip(raw_tasks, TASKS):
        if not isinstance(actual, Mapping) or actual.get("name") != expected.name:
            raise ValueError(f"CRAFT two-task task order/name drifted; expected {expected.name!r}")
        if not math.isclose(float(actual.get("probability", -1)), expected.probability, abs_tol=1e-12):
            raise ValueError(f"CRAFT two-task primary probability drifted for {expected.name}")
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
                raise ValueError(f"CRAFT two-task {expected.name}.{field} drifted from the golden contract")


def _source_sections(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source_mix = config.get("source_mix")
    if not isinstance(source_mix, Mapping):
        raise ValueError("CRAFT two-task source_mix must be a mapping")
    primary = source_mix.get("primary")
    additional = source_mix.get("additional")
    if not isinstance(primary, Mapping) or not isinstance(additional, list) or len(additional) != 1:
        raise ValueError("CRAFT two-task source_mix requires one primary and one Reve2 source")
    region = additional[0]
    if not isinstance(region, Mapping):
        raise ValueError("CRAFT two-task Reve2 source must be a mapping")
    primary_name = primary.get("name")
    if primary_name not in {"tokencompress_short", "tokencompress_medium"} or not math.isclose(
        float(primary.get("probability", -1)), SHORT_SOURCE_WEIGHT, abs_tol=1e-12
    ):
        raise ValueError(
            "CRAFT two-task primary source must be tokencompress_short or tokencompress_medium "
            "with probability 0.44"
        )
    if region.get("name") != "reve2_region" or not math.isclose(
        float(region.get("probability", -1)), REGION_SOURCE_WEIGHT, abs_tol=1e-12
    ):
        raise ValueError("CRAFT two-task additional source must be reve2_region with probability 0.56")
    if region.get("task") != "layout_only" or region.get("channel") != "layout_only":
        raise ValueError("CRAFT two-task Reve2 source must use task=layout_only and channel=layout_only")
    return primary, region


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> ShortTwoTaskConfig:
    config = _config_root(document if document is not None else _load_config_document())
    _assert_exact_task_mix(config)
    primary, region = _source_sections(config)

    primary_paths: tuple[str, ...] = ()
    region_paths: tuple[str, ...] = ()
    if resolve_paths:
        primary_override = os.environ.get("CRAFT_SHORT_TWO_TASK_PRIMARY_PATHS") or os.environ.get(
            "CRAFT_HIERARCHY_JSONL_PATHS"
        )
        region_override = os.environ.get("CRAFT_SHORT_TWO_TASK_REGION_PATHS")
        # Keep the established CRAFT convention: the primary root/manifest is
        # a top-level dataset property. A source-local path remains supported
        # for offline fixtures and explicit source-scoped configurations.
        primary_paths = _expand_paths(config, env_paths=primary_override)
        if not primary_paths:
            primary_paths = _expand_paths(primary, env_paths=primary_override)
        region_paths = _expand_paths(region, env_paths=region_override)

    return ShortTwoTaskConfig(
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
        primary_source_name=str(primary["name"]),
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
            or 8192
        ),
        seed=int(os.environ.get("CRAFT_HIERARCHY_DATA_SEED") or config.get("seed") or 42),
        max_samples=int(os.environ.get("CRAFT_HIERARCHY_MAX_SAMPLES") or config.get("max_samples") or 0),
    )


def choose_primary_task(random_value: float) -> TaskSpec:
    if not 0.0 <= random_value <= 1.0:
        raise ValueError(f"random_value must be in [0, 1], got {random_value}")
    return TASKS[0] if random_value < PRIMARY_FULL_JSON_WEIGHT else TASKS[1]


def _iter_short_two_task_source(
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
    source_name: str = "tokencompress_short",
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield canonical Swift rows while assigning only the two local tasks."""
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
            for source_row in iter_jsonl(path):
                task = fixed_task or choose_primary_task(task_rng.random())
                row = make_row(
                    source_row,
                    source_path=path,
                    task=task,
                    channel=channel_override or task.name,
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
            raise RuntimeError(f"No valid completed rows found in {len(pass_paths)} JSONL paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: ShortTwoTaskConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    seed_offset: int = 0,
    fixed_task_name: str = "",
    channel_override: str = "",
    source_name: str = "tokencompress_short",
):
    from datasets import Features, IterableDataset, Value

    # Resolve the DP topology in the parent process. DataLoader workers may not
    # retain Megatron process groups, but must use the same rank-local decision.
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
        _iter_short_two_task_source,
        gen_kwargs={
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


class CraftShortTwoTaskLoader:
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
            raise ValueError("CRAFT two-task requires --streaming true")
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
                raise ValueError("No CRAFT two-task validation JSONL paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
                fixed_task_name="full_json",
                channel_override="full_json",
                source_name=config.primary_source_name,
            )
        else:
            if not train_paths:
                raise ValueError("No CRAFT two-task training JSONL paths resolved after val split")
            if not config.region_paths:
                raise ValueError("No Reve2 layout JSONL paths resolved")
            from datasets import interleave_datasets

            primary = _build_hf_stream(
                train_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=0,
                source_name=config.primary_source_name,
            )
            region = _build_hf_stream(
                config.region_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=0,
                seed_offset=1_000_003,
                fixed_task_name="layout_only",
                channel_override="layout_only",
                source_name="reve2_region",
            )
            dataset = interleave_datasets(
                [primary, region],
                probabilities=[SHORT_SOURCE_WEIGHT, REGION_SOURCE_WEIGHT],
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
    from swift.dataset import MessagesPreprocessor
    from swift.dataset.dataset_meta import BaseDatasetLoader, DatasetMeta
    from swift.dataset.register import register_dataset

    from dlms.patches.s3_image_load import apply as apply_s3_image_load

    class SwiftCraftShortTwoTaskLoader(CraftShortTwoTaskLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    source_tag = config.primary_source_name.removeprefix("tokencompress_")
    for name, tags in (
        (config.dataset_name, ["vision", "sft", "structured-json", "craft", source_tag]),
        (
            config.val_dataset_name,
            ["vision", "sft", "structured-json", "craft", source_tag, "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftCraftShortTwoTaskLoader,
                tags=tags,
            ),
            exist_ok=True,
        )

    apply_s3_image_load()
    requested_channels = tuple(_as_list(os.environ.get("CRAFT_PACKING_REQUIRED_CHANNELS")))
    if requested_channels:
        raise ValueError(
            "CRAFT two-task uses ordinary packing; CRAFT_PACKING_REQUIRED_CHANNELS must be empty"
        )
    print(
        f"[dlms] CRAFT two-task stream enabled: source {config.primary_source_name}=0.44/"
        "reve2=0.56; tasks full_json=0.30/layout_only=0.70",
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
