"""GRPO dataset adapter for the two-image image-edit-v6 contract.

The published image-edit-v6 rows are shared with SFT, but GRPO must not put the
gold answer in ``messages``.  This adapter therefore emits a prompt-only
conversation and keeps the canonical target in the top-level ``solution``
column.  ms-swift 4.4.2 forwards ``solution`` and ``meta`` to reward functions
without including them in the model prompt.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Iterator, Mapping, Sequence


# Import the authoritative V6 contract without registering its SFT dataset as a
# side effect.  This file owns the GRPO registrations below.
_old_library_only = os.environ.get("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY")
os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets import image_edit_v6_full_json as _v6
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = _old_library_only


DEFAULT_DATASET_NAME = "image_edit_v6_grpo"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
CHANNEL = "image_edit_v6_grpo"


def dataset_names() -> tuple[str, str]:
    """Return stable Swift registration names, with optional env overrides."""

    train_name = os.environ.get("IMAGE_EDIT_V6_GRPO_DATASET_NAME", DEFAULT_DATASET_NAME).strip()
    val_name = os.environ.get(
        "IMAGE_EDIT_V6_GRPO_VAL_DATASET_NAME", f"{train_name}_val"
    ).strip()
    if not train_name or not val_name or train_name == val_name:
        raise ValueError("image-edit-v6 GRPO train/validation dataset names must be distinct")
    return train_name, val_name


def _sft_row_to_grpo(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = list(row["messages"])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("image-edit-v6 source row has no canonical assistant target")
    solution = messages[-1].get("content")
    if not isinstance(solution, str):
        raise ValueError("image-edit-v6 canonical assistant target must be text")

    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "training_stage": "grpo",
            "target_schema": _v6.TARGET_SCHEMA,
        }
    )
    return {
        "messages": messages[:-1],
        "images": list(row["images"]),
        "solution": solution,
        "channel": CHANNEL,
        "meta": meta,
    }


def make_grpo_row(
    source_row: Mapping[str, Any],
    *,
    source_path: str,
    config: _v6.ImageEditV6FullJsonConfig,
) -> dict[str, Any] | None:
    """Validate a published V6 row and convert it to the Swift GRPO contract."""

    row = _v6.make_row(source_row, source_path=source_path, config=config)
    return None if row is None else _sft_row_to_grpo(row)


def _iter_source(
    path_payload: Mapping[str, Any],
    *,
    config: _v6.ImageEditV6FullJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    # Reuse immutable-shard verification, rank sharding, deterministic shuffling,
    # and row validation from the production SFT loader.
    for row in _v6._iter_source(
        path_payload,
        config=config,
        repeat=repeat,
        shuffle_paths=shuffle_paths,
        max_samples=max_samples,
        data_rank=data_rank,
        data_world_size=data_world_size,
    ):
        yield _sft_row_to_grpo(row)


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: _v6.ImageEditV6FullJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
):
    # Kept lazy so contract and reward unit tests do not require datasets/swift.
    from datasets import Features, IterableDataset, Value

    data_rank, data_world_size = _v6.data_stream_shard_context()
    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "images": [Value("string")],
            "solution": Value("string"),
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
                "training_stage": Value("string"),
                "target_schema": Value("string"),
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


class ImageEditV6GRPOLoader:
    """Swift loader that preserves reward-only columns by construction."""

    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        strict: bool = False,
        remove_unused_columns: bool = False,
        **_: Any,
    ) -> None:
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = streaming
        self.strict = strict
        # Megatron GRPO sets this to False.  Keep it false even if a caller uses
        # the generic loader default: removing extras would silently drop the
        # solution/meta columns needed by the reward plugin.
        self.remove_unused_columns = False
        del remove_unused_columns

    def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
        del dataset_syntax, use_hf
        if not self.streaming:
            raise ValueError("image-edit-v6 GRPO requires --streaming true")
        config = _v6.load_runtime_config()
        train_name, val_name = dataset_names()
        requested_name = getattr(dataset_meta, "dataset_name", None) or train_name
        is_validation = requested_name == val_name
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
        # MessagesPreprocessor retains arbitrary columns when
        # remove_unused_columns=False; ms-swift later places them in
        # OnPolicySample.extra and flattens them into reward kwargs.
        return dataset_meta.preprocess_func(
            dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
        )


def register_with_swift() -> tuple[str, str]:
    from swift.dataset import MessagesPreprocessor
    from swift.dataset.dataset_meta import BaseDatasetLoader, DatasetMeta
    from swift.dataset.register import register_dataset

    from dlms.patches.s3_image_load import apply as apply_s3_image_load

    class SwiftImageEditV6GRPOLoader(ImageEditV6GRPOLoader, BaseDatasetLoader):
        pass

    # Validate the pinned V6 prompt/annotation contract during registration.
    _v6.load_runtime_config(resolve_paths=False)
    train_name, val_name = dataset_names()
    for name, tags in (
        (train_name, ["vision", "multi-image", "grpo", "structured-json", "image-edit-v6"]),
        (
            val_name,
            [
                "vision",
                "multi-image",
                "grpo",
                "structured-json",
                "image-edit-v6",
                "validation",
            ],
        ),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftImageEditV6GRPOLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    apply_s3_image_load()
    print(
        f"[dlms] image-edit-v6 GRPO stream enabled: dataset={train_name} "
        f"prompt={_v6.PROMPT_VERSION} annotation={_v6.ANNOTATION_CONTRACT_VERSION}",
        flush=True,
    )
    return train_name, val_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


if _swift_is_available() and not _v6._env_bool("DLMS_IMAGE_EDIT_V6_GRPO_LIBRARY_ONLY", False):
    register_with_swift()
