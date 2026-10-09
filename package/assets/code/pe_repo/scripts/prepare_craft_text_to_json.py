#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3==1.43.55",
#     "pyarrow==25.0.0",
# ]
# ///
"""Build the paired CRAFT text-to-JSON baseline dataset.

The prompt source is a Parquet sidecar keyed by ``sample_id``. The target
source is the canonical Short-v4 JSONL corpus used by image-to-JSON training.
This tool performs the expensive join once, validates the source identity,
materializes the exact Short-v4 ``full_json`` target view, and publishes
rank-shardable JSONL files for ``dlms.datasets.text_to_json``.

Baseline defaults are intentionally strict:

* input: ``original_user_prompt`` (no fallback captions);
* target: Short-v4 ``full_json`` including optional visible ``text``;
* geometry: source ``yxyx_1000`` converted to ``xyxy_1000``;
* split: deterministic 1% validation split grouped by ``sample_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import sys
import tempfile
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the proven target projection without registering its legacy Swift
# dataset when this preparation script happens to run in a trainer image.
_old_library_only = os.environ.get("DLMS_CRAFT_MIXED4_LIBRARY_ONLY")
os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = "true"
try:
    from dlms.datasets.craft_mixed4 import TaskSpec, annotation_view
finally:
    if _old_library_only is None:
        os.environ.pop("DLMS_CRAFT_MIXED4_LIBRARY_ONLY", None)
    else:
        os.environ["DLMS_CRAFT_MIXED4_LIBRARY_ONLY"] = _old_library_only


DEFAULT_PROMPT_ROOT = (
    "s3://core-cn-oss.canva.com/usr/hongdexiang/reve2_json_prompt/cluster_sample_1m_user_prompt_v1_20260801"
)
DEFAULT_CRAFT_MANIFEST = (
    "s3://core-cn-oss.canva.com/usr/joeyzhang/comm/design_layout_swift/datasets/"
    "craft_token_compress_gemini35_flash_lite_v6_canonical_v2_20260729/"
    "short/manifest.json"
)
PROMPT_FIELDS = ("original_user_prompt", "latest_user_prompt", "fallback_text")
EXPECTED_SELECTED_PROMPTS = {
    "original_user_prompt": 518_988,
    "latest_user_prompt": 518_988,
    "fallback_text": 231_346,
}
EXPECTED_PROMPT_ROWS = 1_000_000
EXPECTED_CRAFT_ROWS = 998_685
EXPECTED_CRAFT_DROPS = EXPECTED_PROMPT_ROWS - EXPECTED_CRAFT_ROWS
DATASET_SCHEMA_VERSION = 1

FULL_JSON_TASK = TaskSpec(
    name="full_json",
    probability=1.0,
    system_prompt=(
        "You convert a user's design request into a complete structured design description. "
        "Return only valid JSON in the requested full schema."
    ),
    user_prompt="{input}",
    layout_fields=("type", "label", "parent", "bbox", "text", "desc", "color_palette"),
    include_high_level_description=True,
    include_style_description=True,
    include_background=True,
)


class PreparationError(RuntimeError):
    """The paired dataset could not be prepared without violating its contract."""


@dataclass(frozen=True, slots=True)
class PromptRecord:
    sample_id: str
    text: str
    logical_source: str
    source_shard: str
    source_row: int
    prompt_resolution: str


@dataclass(frozen=True, slots=True)
class PromptPart:
    rows: int
    selected: tuple[PromptRecord, ...]
    missing: int
    source_counts: Counter[str]
    resolution_counts: Counter[str]


@dataclass(frozen=True, slots=True)
class ShardInfo:
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    prompt_root: str
    craft_manifest: str
    output: str
    prompt_field: str = "original_user_prompt"
    validation_basis_points: int = 100
    split_seed: int = 42
    rows_per_shard: int = 4096
    download_workers: int = 16
    upload_workers: int = 8
    staging_dir: str = ""
    max_craft_paths: int = 0
    cardinality_check: bool = True


_S3_CLIENT: Any | None = None


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3

        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _split_s3_uri(path: str) -> tuple[str, str]:
    if not _is_s3(path):
        raise ValueError(f"not an S3 URI: {path}")
    remainder = path[5:]
    bucket, separator, key = remainder.partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"invalid S3 URI: {path}")
    return bucket, key


def _read_bytes(path: str) -> bytes:
    if _is_s3(path):
        bucket, key = _split_s3_uri(path)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(path).read_bytes()


def _join_output_uri(root: str, relative_path: str) -> str:
    if _is_s3(root):
        return f"{root.rstrip('/')}/{relative_path}"
    return str((Path(root) / relative_path).resolve())


def _resolve_manifest_entry(manifest_path: str, entry: str) -> str:
    if _is_s3(entry) or Path(entry).is_absolute():
        return entry
    if _is_s3(manifest_path):
        bucket, key = _split_s3_uri(manifest_path)
        return f"s3://{bucket}/{posixpath.normpath(posixpath.join(posixpath.dirname(key), entry))}"
    return str((Path(manifest_path).parent / entry).resolve())


def load_manifest_paths(manifest_path: str) -> list[str]:
    document = json.loads(_read_bytes(manifest_path))
    if isinstance(document, list):
        entries = document
    elif isinstance(document, Mapping):
        entries = document.get("paths") or document.get("shards") or document.get("jsonl_paths") or []
    else:
        raise PreparationError(f"{manifest_path}: manifest must be a JSON object or list")

    paths: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, Mapping):
            value = entry.get("path") or entry.get("s3_path")
        else:
            value = None
        if not isinstance(value, str) or not value:
            raise PreparationError(f"{manifest_path}: unsupported manifest entry: {entry!r}")
        paths.append(_resolve_manifest_entry(manifest_path, value))
    if not paths:
        raise PreparationError(f"{manifest_path}: no JSONL paths found")
    return paths


def discover_prompt_paths(prompt_root: str) -> list[str]:
    if prompt_root.endswith(".parquet"):
        return [prompt_root]
    if _is_s3(prompt_root):
        bucket, key = _split_s3_uri(prompt_root.rstrip("/") + "/final")
        prefix = key.rstrip("/") + "/"
        paginator = _s3_client().get_paginator("list_objects_v2")
        paths = [
            f"s3://{bucket}/{item['Key']}"
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for item in page.get("Contents", [])
            if item["Key"].endswith(".parquet")
        ]
    else:
        root = Path(prompt_root)
        candidate = root / "final" if (root / "final").is_dir() else root
        paths = [str(path.resolve()) for path in candidate.glob("*.parquet") if path.is_file()]
    paths = sorted(paths)
    if not paths:
        raise PreparationError(f"{prompt_root}: no final Parquet parts found")
    return paths


def _load_prompt_part(path: str, prompt_field: str) -> PromptPart:
    import pyarrow.parquet as pq

    columns = [
        "sample_id",
        prompt_field,
        "logical_source",
        "source_shard",
        "source_row",
        "prompt_resolution",
    ]
    try:
        table = (
            pq.read_table(io.BytesIO(_read_bytes(path)), columns=columns)
            if _is_s3(path)
            else pq.read_table(path, columns=columns)
        )
    except Exception as exc:
        raise PreparationError(f"failed to read prompt Parquet {path}: {exc}") from exc

    selected: list[PromptRecord] = []
    missing = 0
    source_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    for row in table.to_pylist():
        sample_id = row.get("sample_id")
        value = row.get(prompt_field)
        if not isinstance(sample_id, str) or not sample_id:
            raise PreparationError(f"{path}: prompt row has an invalid sample_id")
        if not isinstance(value, str) or not value.strip():
            missing += 1
            continue
        source_shard = row.get("source_shard")
        source_row = row.get("source_row")
        if not isinstance(source_shard, str) or not isinstance(source_row, int):
            raise PreparationError(f"{path}: {sample_id} has an invalid source key")
        logical_source = str(row.get("logical_source") or "")
        prompt_resolution = str(row.get("prompt_resolution") or "")
        selected.append(
            PromptRecord(
                sample_id=sample_id,
                text=value.strip(),
                logical_source=logical_source,
                source_shard=source_shard,
                source_row=source_row,
                prompt_resolution=prompt_resolution,
            )
        )
        source_counts[logical_source] += 1
        resolution_counts[prompt_resolution] += 1
    return PromptPart(
        rows=table.num_rows,
        selected=tuple(selected),
        missing=missing,
        source_counts=source_counts,
        resolution_counts=resolution_counts,
    )


T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(
    function: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
) -> Iterator[R]:
    """Yield parallel results in input order with bounded completed-result memory."""
    if workers <= 1:
        for item in items:
            yield function(item)
        return

    window = max(workers * 2, 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[int, Future[R]] = {}
        next_submit = 0
        next_yield = 0
        while next_submit < min(window, len(items)):
            futures[next_submit] = executor.submit(function, items[next_submit])
            next_submit += 1
        while next_yield < len(items):
            future = futures.pop(next_yield)
            yield future.result()
            next_yield += 1
            if next_submit < len(items):
                futures[next_submit] = executor.submit(function, items[next_submit])
                next_submit += 1


def load_prompt_index(
    prompt_paths: Sequence[str],
    *,
    prompt_field: str,
    workers: int,
) -> tuple[dict[str, PromptRecord], dict[str, Any]]:
    index: dict[str, PromptRecord] = {}
    total_rows = 0
    missing = 0
    source_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    loader = lambda path: _load_prompt_part(path, prompt_field)  # noqa: E731
    for part_number, part in enumerate(
        ordered_parallel_map(loader, list(prompt_paths), workers=workers), start=1
    ):
        total_rows += part.rows
        missing += part.missing
        source_counts.update(part.source_counts)
        resolution_counts.update(part.resolution_counts)
        for record in part.selected:
            if record.sample_id in index:
                raise PreparationError(f"duplicate selected prompt sample_id: {record.sample_id}")
            index[record.sample_id] = record
        print(
            f"[prepare-text-json] prompt parts={part_number}/{len(prompt_paths)} "
            f"rows={total_rows} selected={len(index)}",
            flush=True,
        )
    stats = {
        "rows": total_rows,
        "selected_rows": len(index),
        "missing_selected_field": missing,
        "source_counts": dict(sorted(source_counts.items())),
        "prompt_resolution_counts": dict(sorted(resolution_counts.items())),
    }
    return index, stats


def _read_jsonl_part(path: str) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(_read_bytes(path).splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise PreparationError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return path, rows


def _source_key_matches(prompt: PromptRecord, craft_row: Mapping[str, Any]) -> bool:
    source = craft_row.get("source")
    return bool(
        isinstance(source, Mapping)
        and source.get("source_shard") == prompt.source_shard
        and source.get("source_row") == prompt.source_row
    )


def build_training_row(prompt: PromptRecord, craft_row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = craft_row.get("sample_id")
    if sample_id != prompt.sample_id:
        raise PreparationError(f"join identity mismatch: prompt={prompt.sample_id}, target={sample_id}")
    if not _source_key_matches(prompt, craft_row):
        raise PreparationError(f"source key mismatch for {prompt.sample_id}")
    final_caption = craft_row.get("final_caption")
    annotation = final_caption.get("annotation") if isinstance(final_caption, Mapping) else None
    if not isinstance(annotation, Mapping):
        raise PreparationError(f"missing final_caption.annotation for {prompt.sample_id}")
    target = annotation_view(annotation, FULL_JSON_TASK)
    return {
        "id": prompt.sample_id,
        "input_text": prompt.text,
        "target_json": target,
        "logical_source": prompt.logical_source,
        "prompt_resolution": prompt.prompt_resolution,
    }


def split_name(sample_id: str, *, validation_basis_points: int, seed: int) -> str:
    if not 0 <= validation_basis_points <= 10_000:
        raise ValueError("validation_basis_points must be in [0, 10000]")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "validation" if bucket < validation_basis_points else "train"


class JsonlShardWriter:
    def __init__(self, root: Path, split: str, rows_per_shard: int) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.root = root
        self.split = split
        self.rows_per_shard = rows_per_shard
        self.directory = root / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards: list[ShardInfo] = []
        self._handle: Any | None = None
        self._digest: Any | None = None
        self._path: Path | None = None
        self._rows = 0
        self._bytes = 0

    @property
    def total_rows(self) -> int:
        return sum(shard.rows for shard in self.shards) + self._rows

    def _open(self) -> None:
        index = len(self.shards)
        self._path = self.directory / f"part-{index:05d}.jsonl"
        self._handle = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._rows = 0
        self._bytes = 0

    def write(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            self._open()
        payload = (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        self._handle.write(payload)
        self._digest.update(payload)
        self._rows += 1
        self._bytes += len(payload)
        if self._rows >= self.rows_per_shard:
            self._close_shard()

    def _close_shard(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        assert self._path is not None and self._digest is not None
        relative = self._path.relative_to(self.root).as_posix()
        self.shards.append(
            ShardInfo(
                relative_path=relative,
                rows=self._rows,
                bytes=self._bytes,
                sha256=self._digest.hexdigest(),
            )
        )
        self._handle = None
        self._digest = None
        self._path = None
        self._rows = 0
        self._bytes = 0

    def close(self) -> None:
        self._close_shard()


def _ensure_output_available(output: str) -> None:
    if _is_s3(output):
        bucket, key = _split_s3_uri(output.rstrip("/") + "/placeholder")
        prefix = posixpath.dirname(key).rstrip("/") + "/"
        response = _s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if response.get("KeyCount", 0):
            raise PreparationError(f"refusing to publish into non-empty S3 prefix: {output}")
        return
    path = Path(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise PreparationError(f"refusing to write into non-empty output path: {output}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_manifest(
    *,
    split: str,
    shards: Sequence[ShardInfo],
    published_root: str,
    dataset_revision: str,
) -> dict[str, Any]:
    return {
        "format": "paired text-to-JSON JSONL",
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_revision": dataset_revision,
        "split": split,
        "rows": sum(shard.rows for shard in shards),
        "bytes": sum(shard.bytes for shard in shards),
        "paths": [_join_output_uri(published_root, shard.relative_path) for shard in shards],
        "shards": [
            {
                "path": _join_output_uri(published_root, shard.relative_path),
                "rows": shard.rows,
                "bytes": shard.bytes,
                "sha256": shard.sha256,
            }
            for shard in shards
        ],
    }


def _validate_cardinality(
    *,
    config: PrepareConfig,
    prompt_stats: Mapping[str, Any],
    craft_rows: int,
    craft_completed_rows: int,
    paired_rows: int,
) -> None:
    if not config.cardinality_check or config.max_craft_paths:
        return
    expected_selected = EXPECTED_SELECTED_PROMPTS[config.prompt_field]
    minimum_paired = expected_selected - EXPECTED_CRAFT_DROPS
    checks = (
        (prompt_stats["rows"] == EXPECTED_PROMPT_ROWS, f"prompt rows={prompt_stats['rows']}"),
        (
            prompt_stats["selected_rows"] == expected_selected,
            f"selected prompts={prompt_stats['selected_rows']}",
        ),
        (craft_rows == EXPECTED_CRAFT_ROWS, f"CRAFT rows={craft_rows}"),
        (
            craft_completed_rows == craft_rows,
            f"CRAFT completed rows={craft_completed_rows}/{craft_rows}",
        ),
        (
            minimum_paired <= paired_rows <= expected_selected,
            f"paired rows={paired_rows}, expected [{minimum_paired}, {expected_selected}]",
        ),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise PreparationError("baseline cardinality check failed: " + "; ".join(failures))


def prepare_to_local(config: PrepareConfig, local_root: Path, *, published_root: str) -> dict[str, Any]:
    if config.prompt_field not in PROMPT_FIELDS:
        raise ValueError(f"unsupported prompt field: {config.prompt_field}")
    prompt_paths = discover_prompt_paths(config.prompt_root)
    craft_paths = load_manifest_paths(config.craft_manifest)
    if config.max_craft_paths > 0:
        craft_paths = craft_paths[: config.max_craft_paths]
    print(
        f"[prepare-text-json] prompt_parts={len(prompt_paths)} craft_parts={len(craft_paths)} "
        f"prompt_field={config.prompt_field}",
        flush=True,
    )
    prompt_index, prompt_stats = load_prompt_index(
        prompt_paths,
        prompt_field=config.prompt_field,
        workers=min(config.download_workers, len(prompt_paths)),
    )
    selected_prompt_rows = len(prompt_index)

    writers = {
        split: JsonlShardWriter(local_root, split, config.rows_per_shard) for split in ("train", "validation")
    }
    craft_rows = 0
    completed_rows = 0
    rows_without_selected_prompt = 0
    paired_rows = 0
    joined_source_counts: Counter[str] = Counter()
    joined_resolution_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    paired_sample_ids: set[str] = set()

    try:
        for part_number, (_, rows) in enumerate(
            ordered_parallel_map(
                _read_jsonl_part,
                craft_paths,
                workers=min(config.download_workers, len(craft_paths)),
            ),
            start=1,
        ):
            for craft_row in rows:
                craft_rows += 1
                if craft_row.get("status") != "completed":
                    continue
                completed_rows += 1
                sample_id = craft_row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise PreparationError("CRAFT row has an invalid sample_id")
                if sample_id in paired_sample_ids:
                    raise PreparationError(f"duplicate paired CRAFT sample_id: {sample_id}")
                prompt = prompt_index.pop(sample_id, None)
                if prompt is None:
                    rows_without_selected_prompt += 1
                    continue
                row = build_training_row(prompt, craft_row)
                split = split_name(
                    sample_id,
                    validation_basis_points=config.validation_basis_points,
                    seed=config.split_seed,
                )
                writers[split].write(row)
                paired_sample_ids.add(sample_id)
                paired_rows += 1
                split_counts[split] += 1
                joined_source_counts[prompt.logical_source] += 1
                joined_resolution_counts[prompt.prompt_resolution] += 1
            if part_number % 25 == 0 or part_number == len(craft_paths):
                print(
                    f"[prepare-text-json] CRAFT parts={part_number}/{len(craft_paths)} "
                    f"rows={craft_rows} paired={paired_rows}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    unmatched_prompt_rows = len(prompt_index)
    _validate_cardinality(
        config=config,
        prompt_stats=prompt_stats,
        craft_rows=craft_rows,
        craft_completed_rows=completed_rows,
        paired_rows=paired_rows,
    )
    train_manifest = _split_manifest(
        split="train",
        shards=writers["train"].shards,
        published_root=published_root,
        dataset_revision=f"craft-short-v4-{config.prompt_field}-full-json-v1",
    )
    validation_manifest = _split_manifest(
        split="validation",
        shards=writers["validation"].shards,
        published_root=published_root,
        dataset_revision=f"craft-short-v4-{config.prompt_field}-full-json-v1",
    )
    _write_json(local_root / "train/manifest.json", train_manifest)
    _write_json(local_root / "validation/manifest.json", validation_manifest)

    stats = {
        "prompt_rows": prompt_stats["rows"],
        "selected_prompt_rows": selected_prompt_rows,
        "craft_rows": craft_rows,
        "craft_completed_rows": completed_rows,
        "paired_rows": paired_rows,
        "train_rows": split_counts["train"],
        "validation_rows": split_counts["validation"],
        "rows_without_selected_prompt": rows_without_selected_prompt,
        "selected_prompts_without_target": unmatched_prompt_rows,
        "joined_source_counts": dict(sorted(joined_source_counts.items())),
        "joined_prompt_resolution_counts": dict(sorted(joined_resolution_counts.items())),
    }
    manifest = {
        "format": "paired text-to-JSON JSONL",
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_revision": f"craft-short-v4-{config.prompt_field}-full-json-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "prompt_root": config.prompt_root,
            "prompt_field": config.prompt_field,
            "prompt_parts": len(prompt_paths),
            "craft_manifest": config.craft_manifest,
            "craft_parts": len(craft_paths),
        },
        "target": {
            "task": FULL_JSON_TASK.name,
            "bbox_format": "xyxy_1000",
            "layout_fields": list(FULL_JSON_TASK.layout_fields),
            "include_high_level_description": True,
            "include_style_description": True,
            "include_background": True,
        },
        "split": {
            "strategy": "sha256(seed:sample_id) modulo 10000",
            "seed": config.split_seed,
            "validation_basis_points": config.validation_basis_points,
        },
        "stats": stats,
        # The training loader consumes this top-level list directly. Validation
        # remains separate so no validation shard is accidentally repeated as
        # training data.
        "paths": train_manifest["paths"],
        "train_manifest": _join_output_uri(published_root, "train/manifest.json"),
        "validation_manifest": _join_output_uri(published_root, "validation/manifest.json"),
    }
    _write_json(local_root / "manifest.json", manifest)
    success = {
        "status": "passed",
        "manifest": _join_output_uri(published_root, "manifest.json"),
        "manifest_sha256": _sha256_file(local_root / "manifest.json"),
        "paired_rows": paired_rows,
        "train_rows": split_counts["train"],
        "validation_rows": split_counts["validation"],
    }
    _write_json(local_root / "_SUCCESS.json", success)
    return manifest


def _upload_tree(local_root: Path, output: str, *, workers: int) -> None:
    bucket, output_key = _split_s3_uri(output.rstrip("/") + "/placeholder")
    prefix = posixpath.dirname(output_key).rstrip("/")
    files = [path for path in local_root.rglob("*") if path.is_file()]
    data_files = sorted(path for path in files if path.suffix == ".jsonl")
    metadata_files = sorted(path for path in files if path not in data_files and path.name != "_SUCCESS.json")
    success_file = local_root / "_SUCCESS.json"

    def upload(path: Path) -> None:
        relative = path.relative_to(local_root).as_posix()
        _s3_client().upload_file(str(path), bucket, f"{prefix}/{relative}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(upload, data_files), start=1):
            if index % 25 == 0 or index == len(data_files):
                print(
                    f"[prepare-text-json] uploaded data shards={index}/{len(data_files)}",
                    flush=True,
                )
    for path in metadata_files:
        upload(path)
    upload(success_file)


def prepare_dataset(config: PrepareConfig) -> dict[str, Any]:
    _ensure_output_available(config.output)
    if _is_s3(config.output):
        staging_parent = Path(config.staging_dir).resolve() if config.staging_dir else None
        if staging_parent is not None:
            staging_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="craft-text-json-", dir=staging_parent) as temporary:
            local_root = Path(temporary) / "dataset"
            local_root.mkdir()
            manifest = prepare_to_local(config, local_root, published_root=config.output)
            _upload_tree(local_root, config.output, workers=config.upload_workers)
            return manifest

    local_root = Path(config.output).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    try:
        return prepare_to_local(config, local_root, published_root=str(local_root))
    except Exception:
        # Keep failed local output for diagnosis, but never leave a success
        # marker that could be mistaken for a complete baseline dataset.
        success = local_root / "_SUCCESS.json"
        if success.exists():
            success.unlink()
        raise


def parse_args(argv: Sequence[str] | None = None) -> PrepareConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New local directory or empty s3:// prefix")
    parser.add_argument("--prompt-root", default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--craft-manifest", default=DEFAULT_CRAFT_MANIFEST)
    parser.add_argument("--prompt-field", choices=PROMPT_FIELDS, default="original_user_prompt")
    parser.add_argument("--validation-basis-points", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rows-per-shard", type=int, default=4096)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--staging-dir", default="")
    parser.add_argument("--max-craft-paths", type=int, default=0)
    parser.add_argument(
        "--skip-cardinality-check",
        action="store_true",
        help="Allow fixtures or partial smoke conversions; never use for the baseline publication.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.validation_basis_points <= 10_000:
        parser.error("--validation-basis-points must be in [0, 10000]")
    for name in ("rows_per_shard", "download_workers", "upload_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_craft_paths < 0:
        parser.error("--max-craft-paths must be non-negative")
    return PrepareConfig(
        prompt_root=args.prompt_root,
        craft_manifest=args.craft_manifest,
        output=args.output,
        prompt_field=args.prompt_field,
        validation_basis_points=args.validation_basis_points,
        split_seed=args.split_seed,
        rows_per_shard=args.rows_per_shard,
        download_workers=args.download_workers,
        upload_workers=args.upload_workers,
        staging_dir=args.staging_dir,
        max_craft_paths=args.max_craft_paths,
        cardinality_check=not args.skip_cardinality_check,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        manifest = prepare_dataset(config)
    except (PreparationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "output": config.output,
                "stats": manifest["stats"],
                "train_manifest": manifest["train_manifest"],
                "validation_manifest": manifest["validation_manifest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
