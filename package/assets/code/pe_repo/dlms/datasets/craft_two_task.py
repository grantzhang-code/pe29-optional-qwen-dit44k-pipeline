"""Two-task CRAFT/Reve streaming dataset for Qwen3.5 MTP training.

The stream has exactly two externally visible task channels:

* 30% primary CRAFT rows, always rendered with the ``full_json`` contract;
* 70% Reve2 region rows, always rendered with the ``layout_only`` contract.

The implementation deliberately reuses the proven S3, deterministic shuffle,
and data-parallel rank-local helpers from :mod:`craft_mixed4`, while registering
only this loader when used as an ms-swift ``custom_register_path``.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

# craft_mixed4 remains the shared implementation of the low-level row/S3/rank
# logic. Suppress only its import-time Swift registration; direct imports of the
# old plugin retain their historical behavior.
_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        TASK_BY_NAME as _MIXED4_TASK_BY_NAME,
        TaskSpec,
        _as_list,
        _build_hf_stream,
        _config_root,
        _env_bool,
        _expand_paths,
        _load_config_document,
    )
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


DEFAULT_DATASET_NAME = "craft_hierarchy_two_task_region70_fulljson30_qwen35_4b_mtp"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CRAFT_WEIGHT = 0.30
LAYOUT_WEIGHT = 0.70


def _task_with_probability(name: str, probability: float) -> TaskSpec:
    source = _MIXED4_TASK_BY_NAME[name]
    return TaskSpec(
        name=source.name,
        probability=probability,
        system_prompt=source.system_prompt,
        user_prompt=source.user_prompt,
        layout_fields=source.layout_fields,
        include_high_level_description=source.include_high_level_description,
        include_style_description=source.include_style_description,
        include_background=source.include_background,
    )


TASKS = (
    _task_with_probability("full_json", CRAFT_WEIGHT),
    _task_with_probability("layout_only", LAYOUT_WEIGHT),
)
TASK_BY_NAME = {task.name: task for task in TASKS}
OVERALL_TASK_WEIGHTS = {task.name: task.probability for task in TASKS}


@dataclass(frozen=True)
class TwoTaskConfig:
    dataset_name: str
    val_dataset_name: str
    primary_paths: tuple[str, ...]
    region_paths: tuple[str, ...]
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
        raise ValueError("two-task task_mix must contain exactly full_json and layout_only")
    for actual, expected in zip(raw_tasks, TASKS):
        if not isinstance(actual, Mapping) or actual.get("name") != expected.name:
            raise ValueError(f"two-task task order/name drifted; expected {expected.name!r}")
        if not math.isclose(float(actual.get("probability", -1)), expected.probability, abs_tol=1e-12):
            raise ValueError(f"two-task probability drifted for {expected.name}")
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
                raise ValueError(f"two-task {expected.name}.{field} drifted from the golden contract")


def _region_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source_mix = config.get("source_mix")
    if not isinstance(source_mix, Mapping):
        raise ValueError("two-task source_mix must be a mapping")
    primary = source_mix.get("primary")
    additional = source_mix.get("additional")
    if not isinstance(primary, Mapping) or not isinstance(additional, list) or len(additional) != 1:
        raise ValueError("two-task source_mix requires one CRAFT primary and one Reve2 source")
    region = additional[0]
    if not isinstance(region, Mapping):
        raise ValueError("two-task Reve2 source must be a mapping")
    if primary.get("name") != "craft" or not math.isclose(
        float(primary.get("probability", -1)), CRAFT_WEIGHT, abs_tol=1e-12
    ):
        raise ValueError("two-task primary source must be craft with probability 0.30")
    if region.get("name") != "reve2_region" or not math.isclose(
        float(region.get("probability", -1)), LAYOUT_WEIGHT, abs_tol=1e-12
    ):
        raise ValueError("two-task additional source must be reve2_region with probability 0.70")
    if region.get("task") != "layout_only" or region.get("channel") != "layout_only":
        raise ValueError("two-task Reve2 source must use task=layout_only and channel=layout_only")
    return region


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> TwoTaskConfig:
    config = _config_root(document if document is not None else _load_config_document())
    _assert_exact_task_mix(config)
    region = _region_section(config)

    primary_paths: tuple[str, ...] = ()
    region_paths: tuple[str, ...] = ()
    if resolve_paths:
        primary_override = os.environ.get("CRAFT_TWO_TASK_PRIMARY_PATHS") or os.environ.get(
            "CRAFT_HIERARCHY_JSONL_PATHS"
        )
        region_override = os.environ.get("CRAFT_TWO_TASK_REGION_PATHS")
        primary_paths = _expand_paths(config, env_paths=primary_override)
        region_paths = _expand_paths(region, env_paths=region_override)

    return TwoTaskConfig(
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
        shuffle_paths=_env_bool(
            "CRAFT_HIERARCHY_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))
        ),
        sample_shuffle_buffer=int(
            os.environ.get("CRAFT_HIERARCHY_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 8192
        ),
        seed=int(os.environ.get("CRAFT_HIERARCHY_DATA_SEED") or config.get("seed") or 42),
        max_samples=int(os.environ.get("CRAFT_HIERARCHY_MAX_SAMPLES") or config.get("max_samples") or 0),
    )


class CraftTwoTaskLoader:
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
                raise ValueError("No primary CRAFT validation JSONL paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
                fixed_task_name="full_json",
                channel_override="full_json",
                source_name="craft",
            )
        else:
            if not train_paths:
                raise ValueError("No primary CRAFT training JSONL paths resolved after val split")
            if not config.region_paths:
                raise ValueError("No Reve2 layout JSONL paths resolved")
            from datasets import interleave_datasets

            primary = _build_hf_stream(
                train_paths,
                config=config,
                repeat=config.repeat,
                shuffle_paths=config.shuffle_paths,
                max_samples=0,
                fixed_task_name="full_json",
                channel_override="full_json",
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
                channel_override="layout_only",
                source_name="reve2_region",
            )
            dataset = interleave_datasets(
                [primary, region],
                probabilities=[CRAFT_WEIGHT, LAYOUT_WEIGHT],
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

    class SwiftCraftTwoTaskLoader(CraftTwoTaskLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["vision", "sft", "structured-json", "craft", "two-task"]),
        (
            config.val_dataset_name,
            ["vision", "sft", "structured-json", "craft", "two-task", "validation"],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftCraftTwoTaskLoader,
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
        "[dlms] two-task stream enabled: full_json=0.30, layout_only=0.70; "
        "source identity remains in meta.source_name",
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
