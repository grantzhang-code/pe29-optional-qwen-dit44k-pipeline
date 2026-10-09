"""Rank-local 70/30 Text-to-JSON and Image2JSON rehearsal stream.

This loader combines the published original-user-prompt Text-to-JSON dataset
with the cleaned Jimmy 1M+5M Image2JSON two-task stream.  Source selection is
performed independently per Megatron data-parallel rank while TP/PP peers for
the same DP rank see the same decisions.

The production contract is deliberately modality-weighted rather than based
on physical dataset cardinality:

* 70% Text-to-JSON full-schema rows;
* 30% Image2JSON rows, internally split 30% full JSON / 70% layout only.

The resulting expected channels are 70% text full JSON, 9% image full JSON,
and 21% image layout only.  Text rows carry an empty ``images`` list in this
combined schema; their prompts and targets remain byte-identical to the
standalone Text-to-JSON loader.
"""

from __future__ import annotations

import importlib.util
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from dlms.datasets import image2json_two_task, text_to_json
from dlms.datasets.craft_mixed4 import data_stream_shard_context


DEFAULT_DATASET_NAME = "text70_image30_json_qwen35_4b_mtp_from_i2j_c8000"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
DEFAULT_TEXT_WEIGHT = 0.70
DEFAULT_IMAGE_WEIGHT = 0.30


@dataclass(frozen=True)
class TextImageToJsonConfig:
    dataset_name: str
    val_dataset_name: str
    text_weight: float = DEFAULT_TEXT_WEIGHT
    image_weight: float = DEFAULT_IMAGE_WEIGHT
    seed: int = 42
    max_samples: int = 0


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("TEXT_IMAGE_TO_JSON_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def load_runtime_config(document: Mapping[str, Any] | None = None) -> TextImageToJsonConfig:
    document = document if document is not None else _load_config_document()
    raw = document.get("text_image_to_json_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("text_image_to_json_config must be a mapping")
    text_weight = float(raw.get("text_weight", DEFAULT_TEXT_WEIGHT))
    image_weight = float(raw.get("image_weight", DEFAULT_IMAGE_WEIGHT))
    if text_weight <= 0 or image_weight <= 0:
        raise ValueError("text_image_to_json weights must both be positive")
    if abs(text_weight + image_weight - 1.0) > 1e-12:
        raise ValueError("text_image_to_json weights must sum to 1")
    max_samples = int(raw.get("max_samples") or 0)
    if max_samples < 0:
        raise ValueError("text_image_to_json max_samples must be non-negative")
    return TextImageToJsonConfig(
        dataset_name=str(raw.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(raw.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        text_weight=text_weight,
        image_weight=image_weight,
        seed=int(raw.get("seed") or 42),
        max_samples=max_samples,
    )


def _normalise_mixed_row(row: Mapping[str, Any], *, modality: str) -> dict[str, Any]:
    meta = row.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    if modality == "text":
        sample_id = str(meta.get("row_id") or "")
        images: list[str] = []
    else:
        sample_id = str(meta.get("custom_id") or meta.get("global_row") or "")
        raw_images = row.get("images")
        if not isinstance(raw_images, list) or not all(isinstance(value, str) for value in raw_images):
            raise ValueError("Image2JSON mixed rows must contain a string images list")
        images = list(raw_images)
    return {
        "messages": row["messages"],
        "images": images,
        "channel": str(row.get("channel") or ""),
        "meta": {
            "modality": modality,
            "jsonl_path": str(meta.get("jsonl_path") or ""),
            "source_name": str(meta.get("source_name") or ""),
            "sample_id": sample_id,
            "image_path": str(meta.get("image_path") or ""),
        },
    }


def _iter_mixed_source(
    *,
    text_path_payload: Mapping[str, Any],
    image_path_payload: Mapping[str, Any],
    text_config: text_to_json.TextToJsonConfig,
    image_config: image2json_two_task.Image2JSONConfig,
    mix_config: TextImageToJsonConfig,
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    if data_rank is None or data_world_size is None:
        data_rank, data_world_size = data_stream_shard_context()
    text_rows = text_to_json._iter_text_to_json_source(
        text_path_payload,
        config=text_config,
        repeat=text_config.repeat,
        shuffle_paths=text_config.shuffle_paths,
        max_samples=0,
        data_rank=data_rank,
        data_world_size=data_world_size,
    )
    image_rows = image2json_two_task._iter_image2json_source(
        image_path_payload,
        seed=image_config.seed,
        shuffle_paths=image_config.shuffle_paths,
        sample_shuffle_buffer=image_config.sample_shuffle_buffer,
        rank_shard=image_config.rank_shard,
        repeat=image_config.repeat,
        max_samples=0,
        source_name=image_config.source_name,
        data_rank=data_rank,
        data_world_size=data_world_size,
    )
    source_rng = random.Random(mix_config.seed + data_rank * 10_007)
    emitted = 0
    while mix_config.max_samples <= 0 or emitted < mix_config.max_samples:
        modality = "text" if source_rng.random() < mix_config.text_weight else "image"
        try:
            source_row = next(text_rows if modality == "text" else image_rows)
        except StopIteration:
            return
        emitted += 1
        yield _normalise_mixed_row(source_row, modality=modality)


def _build_hf_stream(
    *,
    text_paths: tuple[str, ...],
    image_paths: tuple[str, ...],
    text_config: text_to_json.TextToJsonConfig,
    image_config: image2json_two_task.Image2JSONConfig,
    mix_config: TextImageToJsonConfig,
):
    from datasets import Features, IterableDataset, Value

    data_rank, data_world_size = data_stream_shard_context()
    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "images": [Value("string")],
            "channel": Value("string"),
            "meta": {
                "modality": Value("string"),
                "jsonl_path": Value("string"),
                "source_name": Value("string"),
                "sample_id": Value("string"),
                "image_path": Value("string"),
            },
        }
    )
    return IterableDataset.from_generator(
        _iter_mixed_source,
        gen_kwargs={
            "text_path_payload": {"paths": list(text_paths)},
            "image_path_payload": {"paths": list(image_paths)},
            "text_config": text_config,
            "image_config": image_config,
            "mix_config": mix_config,
            "data_rank": data_rank,
            "data_world_size": data_world_size,
        },
        features=features,
    )


class TextImageToJsonLoader:
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
            raise ValueError("Text/Image-to-JSON mix requires --streaming true")
        document = _load_config_document()
        mix_config = load_runtime_config(document)
        requested_name = getattr(dataset_meta, "dataset_name", None) or mix_config.dataset_name
        if requested_name == mix_config.val_dataset_name:
            raise ValueError("The production Text/Image-to-JSON mix has no validation stream")

        text_config = text_to_json.load_runtime_config(document)
        image_config = image2json_two_task.load_runtime_config(document)
        if image_config.has_layout_only_source_mix:
            raise ValueError("The Text/Image baseline excludes an Image2JSON secondary source")
        text_paths = (
            text_config.paths[text_config.val_path_count :]
            if text_config.val_path_count
            else text_config.paths
        )
        image_paths = (
            image_config.paths[image_config.val_path_count :]
            if image_config.val_path_count
            else image_config.paths
        )
        if not text_paths:
            raise ValueError("No Text-to-JSON training paths resolved")
        if not image_paths:
            raise ValueError("No Image2JSON training paths resolved")

        dataset = _build_hf_stream(
            text_paths=text_paths,
            image_paths=image_paths,
            text_config=text_config,
            image_config=image_config,
            mix_config=mix_config,
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

    class SwiftTextImageToJsonLoader(TextImageToJsonLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config()
    register_dataset(
        DatasetMeta(
            dataset_name=config.dataset_name,
            preprocess_func=MessagesPreprocessor(),
            loader=SwiftTextImageToJsonLoader,
            tags=["multimodal", "sft", "structured-json", "text70", "image30"],
        ),
        exist_ok=True,
    )
    apply_s3_image_load()
    if os.environ.get("CRAFT_PACKING_REQUIRED_CHANNELS", "").strip():
        raise ValueError("Text/Image-to-JSON uses ordinary packing; required channels must be empty")
    print(
        f"[dlms] Text/Image-to-JSON stream enabled: text={config.text_weight:.2f} "
        f"image={config.image_weight:.2f}",
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
