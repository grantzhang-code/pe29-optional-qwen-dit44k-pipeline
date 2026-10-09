"""Rank-local streaming dataset for text-to-JSON SFT.

The source contract is deliberately independent from the CRAFT vision schema::

    {"input_text": "...", "target_json": {"...": "..."}}

Field names, prompts, optional status filtering, and S3/local shard discovery
come from ``text_to_json_config`` in ``JOB_CONFIG_YAML``.  Rows emitted to
ms-swift contain messages only; in particular, they never contain an ``images``
column or an ``<image>`` placeholder.
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

# Reuse the established S3 manifest and DP rank-local helpers without
# registering the CRAFT vision dataset as an import side effect.
_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import (
        _as_list,
        _env_bool,
        _expand_paths,
        _paths_from_manifest,
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


DEFAULT_DATASET_NAME = "text_to_json_qwen35_4b_mtp"
DEFAULT_VAL_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_val"
DEFAULT_SYSTEM_PROMPT = "Convert the user's text into the requested structure. Return only valid JSON."
DEFAULT_USER_PROMPT_TEMPLATE = "{input}"


@dataclass(frozen=True)
class TextToJsonConfig:
    dataset_name: str
    val_dataset_name: str
    paths: tuple[str, ...]
    input_field: str = "input_text"
    target_field: str = "target_json"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    status_field: str = ""
    allowed_statuses: tuple[str, ...] = ()
    id_field: str = "id"
    source_name: str = "text_to_json"
    target_must_be_object: bool = True
    val_path_count: int = 0
    val_max_samples: int = 1000
    rank_shard: bool = True
    repeat: bool = True
    shuffle_paths: bool = True
    sample_shuffle_buffer: int = 8192
    seed: int = 42
    max_samples: int = 0


def _config_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    config = document.get("text_to_json_config", document)
    if not isinstance(config, Mapping):
        raise ValueError("text_to_json_config must be a mapping")
    return config


def _load_config_document() -> Mapping[str, Any]:
    path = os.environ.get("TEXT_TO_JSON_CONFIG") or os.environ.get("JOB_CONFIG_YAML")
    if not path:
        return {}
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML/JSON mapping")
    return document


def _resolve_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    direct_override = os.environ.get("TEXT_TO_JSON_PATHS")
    manifest_override = os.environ.get("TEXT_TO_JSON_MANIFEST")
    if direct_override and manifest_override:
        raise ValueError("Set only one of TEXT_TO_JSON_PATHS and TEXT_TO_JSON_MANIFEST")
    if direct_override:
        return tuple(str(path) for path in _as_list(direct_override))
    if manifest_override:
        _validate_manifest_contract(manifest_override, config)
        return tuple(_paths_from_manifest(manifest_override))
    paths = _expand_paths(config)
    contract = config.get("manifest_contract")
    if contract:
        manifests = _as_list(
            config.get("paths_manifests")
            or config.get("paths_manifest")
            or config.get("manifests")
            or config.get("manifest")
        )
        if len(manifests) != 1:
            raise ValueError("text_to_json_config.manifest_contract requires exactly one manifest")
        _validate_manifest_contract(str(manifests[0]), config)
    return paths


def _manifest_success_path(manifest_path: str) -> str:
    if manifest_path.startswith("s3://"):
        return f"{manifest_path.rsplit('/', 1)[0]}/_SUCCESS.json"
    return str(Path(manifest_path).resolve().parent / "_SUCCESS.json")


def _validate_manifest_contract(manifest_path: str, config: Mapping[str, Any]) -> None:
    contract = config.get("manifest_contract")
    if not contract:
        return
    if not isinstance(contract, Mapping):
        raise ValueError("text_to_json_config.manifest_contract must be a mapping")

    raw = _read_uri_bytes(manifest_path)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON manifest") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{manifest_path}: baseline manifest must be an object")

    source = document.get("source") or {}
    target = document.get("target") or {}
    stats = document.get("stats") or {}
    paths = document.get("paths") or []
    checks = {
        "schema_version": (document.get("schema_version"), contract.get("schema_version")),
        "dataset_revision": (document.get("dataset_revision"), contract.get("dataset_revision")),
        "source.prompt_field": (source.get("prompt_field"), contract.get("prompt_field")),
        "target.task": (target.get("task"), contract.get("target_task")),
        "target.bbox_format": (target.get("bbox_format"), contract.get("bbox_format")),
        "stats.paired_rows": (stats.get("paired_rows"), contract.get("paired_rows")),
        "stats.train_rows": (stats.get("train_rows"), contract.get("train_rows")),
        "stats.validation_rows": (stats.get("validation_rows"), contract.get("validation_rows")),
        "path_count": (len(paths) if isinstance(paths, list) else None, contract.get("path_count")),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if expected is not None and actual != expected
    ]
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(path, str) and path.endswith(".jsonl") and "/train/" in path for path in paths)
    ):
        mismatches.append("paths must be a non-empty list of train/*.jsonl shards")

    if bool(contract.get("require_success", True)):
        success_path = _manifest_success_path(manifest_path)
        try:
            success = json.loads(_read_uri_bytes(success_path))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{success_path}: missing or invalid success marker") from exc
        if not isinstance(success, Mapping) or success.get("status") != "passed":
            mismatches.append(f"success.status={getattr(success, 'get', lambda *_: None)('status')!r}")
        expected_digest = hashlib.sha256(raw).hexdigest()
        if isinstance(success, Mapping) and success.get("manifest_sha256") != expected_digest:
            mismatches.append("success.manifest_sha256 does not match manifest.json")

    if mismatches:
        raise ValueError(f"{manifest_path}: text-to-JSON manifest contract failed: " + "; ".join(mismatches))


def load_runtime_config(
    document: Mapping[str, Any] | None = None,
    *,
    resolve_paths: bool = True,
) -> TextToJsonConfig:
    config = _config_root(document if document is not None else _load_config_document())
    user_prompt_template = str(config.get("user_prompt_template") or DEFAULT_USER_PROMPT_TEMPLATE)
    if user_prompt_template.count("{input}") != 1:
        raise ValueError("text_to_json_config.user_prompt_template must contain {input} exactly once")

    status_field = str(config.get("status_field") or "")
    allowed_statuses = tuple(str(value) for value in _as_list(config.get("allowed_statuses")))
    if bool(status_field) != bool(allowed_statuses):
        raise ValueError("status_field and allowed_statuses must be configured together")

    return TextToJsonConfig(
        dataset_name=str(config.get("dataset_name") or DEFAULT_DATASET_NAME),
        val_dataset_name=str(config.get("val_dataset_name") or DEFAULT_VAL_DATASET_NAME),
        paths=_resolve_paths(config) if resolve_paths else (),
        input_field=str(config.get("input_field") or "input_text"),
        target_field=str(config.get("target_field") or "target_json"),
        system_prompt=str(config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
        user_prompt_template=user_prompt_template,
        status_field=status_field,
        allowed_statuses=allowed_statuses,
        id_field=str(config.get("id_field") or "id"),
        source_name=str(config.get("source_name") or "text_to_json"),
        target_must_be_object=bool(config.get("target_must_be_object", True)),
        val_path_count=int(config.get("val_path_count") or 0),
        val_max_samples=int(config.get("val_max_samples") or 1000),
        rank_shard=_env_bool("TEXT_TO_JSON_RANK_SHARD", bool(config.get("rank_shard", True))),
        repeat=_env_bool("TEXT_TO_JSON_REPEAT", bool(config.get("repeat", True))),
        shuffle_paths=_env_bool("TEXT_TO_JSON_SHUFFLE_PATHS", bool(config.get("shuffle_paths", True))),
        sample_shuffle_buffer=int(
            os.environ.get("TEXT_TO_JSON_SAMPLE_SHUFFLE_BUFFER")
            or config.get("sample_shuffle_buffer")
            or 8192
        ),
        seed=int(os.environ.get("TEXT_TO_JSON_SEED") or config.get("seed") or 42),
        max_samples=int(os.environ.get("TEXT_TO_JSON_MAX_SAMPLES") or config.get("max_samples") or 0),
    )


_MISSING = object()


def _nested_value(row: Mapping[str, Any], field: str) -> Any:
    """Resolve a dot-delimited mapping path without treating falsey values as missing."""
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _normalise_target(value: Any, *, must_be_object: bool) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_reject_nonstandard_json_constant)
        except (json.JSONDecodeError, ValueError):
            return None
    if must_be_object:
        if not isinstance(value, Mapping):
            return None
    elif not isinstance(value, (Mapping, list)):
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def make_row(
    source_row: Mapping[str, Any],
    *,
    source_path: str,
    config: TextToJsonConfig,
) -> dict[str, Any] | None:
    """Convert one paired text/JSON record to the exact Swift messages row."""
    if config.status_field:
        status = _nested_value(source_row, config.status_field)
        if status is _MISSING or str(status) not in config.allowed_statuses:
            return None
    else:
        status = ""

    input_value = _nested_value(source_row, config.input_field)
    target_value = _nested_value(source_row, config.target_field)
    if input_value is _MISSING or target_value is _MISSING or not isinstance(input_value, str):
        return None
    input_text = input_value.strip()
    if not input_text:
        return None
    target = _normalise_target(target_value, must_be_object=config.target_must_be_object)
    if target is None:
        return None

    row_id = _nested_value(source_row, config.id_field)
    if row_id is _MISSING:
        row_id = ""
    return {
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": config.user_prompt_template.replace("{input}", input_text),
            },
            {"role": "assistant", "content": target},
        ],
        "channel": "text_to_json",
        "meta": {
            "jsonl_path": source_path,
            "source_name": config.source_name,
            "row_id": str(row_id),
            "status": str(status),
        },
    }


def _iter_text_to_json_source(
    path_payload: Mapping[str, Any],
    *,
    config: TextToJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
    data_rank: int | None = None,
    data_world_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    paths = list(path_payload["paths"])
    if data_rank is None or data_world_size is None:
        data_rank, data_world_size = data_stream_shard_context()

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
            raise RuntimeError(f"No valid text-to-JSON rows found in {len(pass_paths)} JSONL paths")
        pass_index += 1


def _build_hf_stream(
    paths: Sequence[str],
    *,
    config: TextToJsonConfig,
    repeat: bool,
    shuffle_paths: bool,
    max_samples: int,
):
    from datasets import Features, IterableDataset, Value

    data_rank, data_world_size = data_stream_shard_context()
    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "channel": Value("string"),
            "meta": {
                "jsonl_path": Value("string"),
                "source_name": Value("string"),
                "row_id": Value("string"),
                "status": Value("string"),
            },
        }
    )
    return IterableDataset.from_generator(
        _iter_text_to_json_source,
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


class TextToJsonLoader:
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
            raise ValueError("text-to-JSON requires --streaming true")
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
                raise ValueError("No text-to-JSON validation JSONL paths resolved")
            dataset = _build_hf_stream(
                val_paths,
                config=config,
                repeat=False,
                shuffle_paths=False,
                max_samples=config.val_max_samples,
            )
        else:
            if not train_paths:
                raise ValueError("No text-to-JSON training JSONL paths resolved after val split")
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

    class SwiftTextToJsonLoader(TextToJsonLoader, BaseDatasetLoader):
        pass

    config = load_runtime_config(resolve_paths=False)
    for name, tags in (
        (config.dataset_name, ["text", "sft", "structured-json"]),
        (config.val_dataset_name, ["text", "sft", "structured-json", "validation"]),
    ):
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                preprocess_func=MessagesPreprocessor(),
                loader=SwiftTextToJsonLoader,
                tags=tags,
            ),
            exist_ok=True,
        )
    print(f"[dlms] text-to-JSON stream enabled: dataset={config.dataset_name}", flush=True)
    return config.dataset_name, config.val_dataset_name


def _swift_is_available() -> bool:
    try:
        return importlib.util.find_spec("swift") is not None
    except (ImportError, ValueError):
        return False


if _swift_is_available():
    register_with_swift()
