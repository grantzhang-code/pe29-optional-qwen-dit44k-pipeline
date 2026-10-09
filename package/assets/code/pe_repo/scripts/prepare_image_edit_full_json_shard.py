#!/usr/bin/env python3
"""Build one Qwen-ready image-edit JSONL shard from the authoritative S3 export.

The annotation export intentionally stores three facts in separate places:

* ``<annotation-root>/full/shard_XX/cases/*/target_json.with_inline_edit.json``
  contains the lossless inline-edit Target JSON;
* the shard summary points to the authoritative ``final_results.jsonl`` whose
  per-sample status decides whether a row is trainable;
* the immutable source Parquet dataset contains the real instruction,
  ``input_s3_paths`` and ``output_s3_path``.

This script joins those sources by sample ID. It never infers an instruction,
media path, upstream acceptance status, or semantic Target value. By default it
keeps only upstream ``validated`` rows and stores compact prepared rows consumed
by ``dlms/datasets/image_edit_full_json.py``; the loader materializes the pinned
system/user prompts and assistant message at training time. This script never
writes ``messages``, ``system_prompt``, or ``user_prompt`` fields into its JSONL.
Use ``--selection packaged`` for a pure-S3, whole-shard export of every case
object; those rows are honestly marked ``packaged``, never relabeled as validated.

Typical devbox run::

    python scripts/prepare_image_edit_full_json_shard.py \
      --output data/image_edit_full_json.random_shard.jsonl \
      --aws-profile prod_core-cn-readwrite \
      --output-parts 8 \
      --seed 20260804

Use ``--shard 8`` to select a specific shard. The automatic source inventory
build reads only Parquet footers and is cached beside the output; pass an
existing production inventory with ``--source-inventory`` to skip that step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import random
import re
import struct
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_full_json import (  # noqa: E402
    InlineTargetValidationError,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    canonical_inline_target,
    prompt_contract_sha256,
)


DEFAULT_ANNOTATION_ROOT = "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v4/"
DEFAULT_SOURCE_DATASET = (
    "s3://core-cn-oss.canva.com/dataset/image_edit_results/craft_v7_reverse_judged_v2_parquet"
)
FORMAT_VERSION = "image_edit_full_json_qwen_shard_v1"
PREPARED_MANIFEST_FORMAT = "image_edit_full_json_prepared_manifest_v1"
ANNOTATION_FORMAT = "target_json_inline_edit_per_case_v1"
ANNOTATION_NAME = "target_json.with_inline_edit.json"
SOURCE_COLUMNS = (
    "sample_id",
    "input_s3_paths",
    "output_s3_path",
    "instructions",
    "num_inputs",
)
CASE_KEY_RE = re.compile(rf"/cases/([^/]+)/{re.escape(ANNOTATION_NAME)}$")


def _progress(message: str) -> None:
    print(f"[prepare-image-edit] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(uri: str) -> S3Location:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return S3Location(bucket=bucket, key=key)


def _annotation_root(uri: str) -> S3Location:
    location = parse_s3_uri(uri.rstrip("/"))
    key = location.key.rstrip("/")
    if key.endswith("/full"):
        key = key[: -len("/full")]
    return S3Location(location.bucket, key)


def _join_s3(root: S3Location, suffix: str) -> S3Location:
    return S3Location(root.bucket, f"{root.key.rstrip('/')}/{suffix.lstrip('/')}")


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _open_atomic_stream(path: Path, *, overwrite: bool) -> tuple[BinaryIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return os.fdopen(descriptor, "wb"), Path(temporary_name)


def _commit_atomic_stream(stream: BinaryIO, temporary: Path, output: Path) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    os.replace(temporary, output)


def output_part_paths(output: Path, parts: int) -> list[Path]:
    """Resolve stable physical JSONL names from one logical output path."""

    if parts <= 0:
        raise ValueError("output parts must be positive")
    if parts == 1:
        return [output]
    if output.suffix != ".jsonl":
        raise ValueError("--output must end in .jsonl when --output-parts is greater than 1")
    return [
        output.with_name(f"{output.stem}.part-{index:05d}-of-{parts:05d}.jsonl") for index in range(parts)
    ]


def _join_uri_prefix(prefix: str, name: str) -> str:
    if prefix.startswith("s3://"):
        location = parse_s3_uri(prefix.rstrip("/"))
        return _join_s3(location, name).uri
    return str((Path(prefix).expanduser().resolve() / name).resolve())


def _prepared_dataset_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_format": "prepared",
        "row_format_version": FORMAT_VERSION,
        "runtime_loader": "dlms/datasets/image_edit_full_json.py",
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "image_order": ["reference frame_0", "target"],
        "selection": summary["selection"],
        "allowed_statuses": list(summary["allowed_statuses"]),
        "annotation_root": summary["annotation_root"],
        "annotation_shard_count": int(summary["annotation_shard_count"]),
        "source_dataset": summary["source_dataset"],
    }


def update_prepared_manifest(
    manifest_path: Path,
    *,
    summary: Mapping[str, Any],
    part_paths: Sequence[Path],
    part_rows: Sequence[int],
    part_sha256: Sequence[str],
    entry_prefix: str | None,
) -> dict[str, Any]:
    """Atomically upsert one annotation shard into a loader-compatible manifest."""

    if not (len(part_paths) == len(part_rows) == len(part_sha256)):
        raise ValueError("prepared manifest part metadata lengths differ")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"{manifest_path}: expected a JSON object")
            if existing.get("format_version") != PREPARED_MANIFEST_FORMAT:
                raise ValueError(f"{manifest_path}: incompatible manifest format")
            expected_contract = _prepared_dataset_contract(summary)
            if existing.get("dataset_contract") != expected_contract:
                raise ValueError(f"{manifest_path}: dataset contract differs; use a separate manifest")
            raw_shards = existing.get("shards")
            if not isinstance(raw_shards, list):
                raise ValueError(f"{manifest_path}: shards must be an array")
            shards = [
                dict(item)
                for item in raw_shards
                if int(item.get("annotation_shard", -1)) != int(summary["selected_shard"])
            ]
        else:
            shards = []

        for index, (path, rows, digest) in enumerate(zip(part_paths, part_rows, part_sha256)):
            entry_path = _join_uri_prefix(entry_prefix, path.name) if entry_prefix else str(path.resolve())
            shards.append(
                {
                    "path": entry_path,
                    "rows": int(rows),
                    "sha256": digest,
                    "annotation_shard": int(summary["selected_shard"]),
                    "annotation_shard_complete": int(summary["limit"]) == 0,
                    "part_index": index,
                    "part_count": len(part_paths),
                }
            )
        shards.sort(key=lambda item: (int(item["annotation_shard"]), int(item["part_index"])))
        annotation_shards = sorted({int(item["annotation_shard"]) for item in shards})
        expected_annotation_shards = int(summary["annotation_shard_count"])
        corpus_complete = annotation_shards == list(range(expected_annotation_shards)) and all(
            bool(item.get("annotation_shard_complete")) for item in shards
        )
        manifest = {
            "format_version": PREPARED_MANIFEST_FORMAT,
            "complete": corpus_complete,
            "dataset_contract": _prepared_dataset_contract(summary),
            # `paths` is consumed directly by the existing Swift loader.
            "paths": [str(item["path"]) for item in shards],
            "shards": shards,
            "physical_shard_count": len(shards),
            "annotation_shards": annotation_shards,
            "expected_annotation_shards": expected_annotation_shards,
            "total_rows": sum(int(item["rows"]) for item in shards),
        }
        _write_atomic(manifest_path, _json_bytes(manifest), overwrite=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return manifest


def _read_s3_bytes(client: Any, location: S3Location) -> bytes:
    return client.get_object(Bucket=location.bucket, Key=location.key)["Body"].read()


def _read_s3_json(client: Any, location: S3Location) -> dict[str, Any]:
    value = json.loads(_read_s3_bytes(client, location))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {location.uri}")
    return value


def _iter_s3_keys(client: Any, location: S3Location) -> Iterator[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=location.bucket, Prefix=location.key.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            if int(item.get("Size", 0)) > 0:
                yield item


def _shard_summaries(global_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if global_summary.get("format_version") != ANNOTATION_FORMAT:
        raise ValueError(f"unsupported annotation format: {global_summary.get('format_version')!r}")
    if not global_summary.get("complete"):
        raise ValueError("global annotation summary is not complete")
    raw = global_summary.get("shard_summaries")
    if not isinstance(raw, list) or not raw:
        raise ValueError("global summary has no shard_summaries")
    result = sorted((dict(item) for item in raw), key=lambda item: int(item["shard"]))
    expected_indices = list(range(len(result)))
    actual_indices = [int(item["shard"]) for item in result]
    if actual_indices != expected_indices:
        raise ValueError(f"shard indices are not contiguous: {actual_indices}")
    for item in result:
        if not item.get("complete") or int(item.get("expected_rows", 0)) <= 0:
            raise ValueError(f"shard {item.get('shard')} is incomplete")
    if sum(int(item["expected_rows"]) for item in result) != int(global_summary["row_count"]):
        raise ValueError("shard row counts do not match global row_count")
    return result


def choose_shard(
    summaries: Sequence[Mapping[str, Any]], *, shard: int | None, seed: int | None
) -> tuple[dict[str, Any], int, int]:
    if shard is None:
        chooser = random.SystemRandom() if seed is None else random.Random(seed)
        shard = chooser.randrange(len(summaries))
    if shard < 0 or shard >= len(summaries):
        raise ValueError(f"shard must be between 0 and {len(summaries) - 1}")
    start = sum(int(item["expected_rows"]) for item in summaries[:shard])
    selected = dict(summaries[shard])
    return selected, start, start + int(selected["expected_rows"])


def extract_acceptance_status(row: Mapping[str, Any]) -> str:
    value = row.get("status")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("status", "acceptance_tier"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    raise ValueError(f"sample {row.get('sample_id')!r} has no authoritative status")


def _iter_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _iter_jsonl_s3(client: Any, location: S3Location) -> Iterator[dict[str, Any]]:
    body = client.get_object(Bucket=location.bucket, Key=location.key)["Body"]
    for line_number, raw in enumerate(body.iter_lines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{location.uri}:{line_number}: expected a JSON object")
        yield value


def _iter_jsonl_gcs(uri: str, *, project: str | None) -> Iterator[dict[str, Any]]:
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "google-cloud-storage is required to read the authoritative gs:// results"
        ) from exc
    bucket_name, separator, blob_name = uri[5:].partition("/")
    if not bucket_name or not separator or not blob_name:
        raise ValueError(f"invalid GCS URI: {uri!r}")
    client = storage.Client(project=project)
    blob = client.bucket(bucket_name).blob(blob_name)
    with blob.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{uri}:{line_number}: expected a JSON object")
            yield value


def iter_jsonl_uri(uri: str, *, s3_client: Any, gcp_project: str | None = None) -> Iterator[dict[str, Any]]:
    if uri.startswith("s3://"):
        yield from _iter_jsonl_s3(s3_client, parse_s3_uri(uri))
    elif uri.startswith("gs://"):
        yield from _iter_jsonl_gcs(uri, project=gcp_project)
    else:
        yield from _iter_jsonl_file(Path(uri))


def load_status_map(
    uri: str,
    *,
    s3_client: Any,
    gcp_project: str | None,
    expected_rows: int,
    expected_counts: Mapping[str, Any],
) -> tuple[dict[str, str], Counter[str]]:
    by_id: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in iter_jsonl_uri(uri, s3_client=s3_client, gcp_project=gcp_project):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"status source contains invalid sample_id: {sample_id!r}")
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in status source: {sample_id}")
        status = extract_acceptance_status(row)
        by_id[sample_id] = status
        counts[status] += 1
    if len(by_id) != expected_rows:
        raise ValueError(f"status source has {len(by_id)} rows; expected {expected_rows}")
    normalized_expected = Counter({str(key): int(value) for key, value in expected_counts.items()})
    if counts != normalized_expected:
        raise ValueError(
            f"status counts changed: actual={dict(counts)}, expected={dict(normalized_expected)}"
        )
    return by_id, counts


def _parquet_footer_metadata(
    client: Any, *, bucket: str, key: str, size: int
) -> tuple[int, tuple[str, ...], int]:
    if size < 12:
        raise ValueError(f"Parquet object is too small: s3://{bucket}/{key}")
    tail = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={size - 8}-{size - 1}")["Body"].read()
    if len(tail) != 8 or tail[4:] != b"PAR1":
        raise ValueError(f"invalid Parquet footer magic: s3://{bucket}/{key}")
    metadata_size = struct.unpack("<I", tail[:4])[0]
    start = size - metadata_size - 8
    if start < 4:
        raise ValueError(f"invalid Parquet metadata size: s3://{bucket}/{key}")
    footer = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{size - 1}")["Body"].read()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("pyarrow is required for source Parquet metadata") from exc
    parquet = pq.ParquetFile(pa.BufferReader(b"PAR1" + footer))
    return parquet.metadata.num_rows, tuple(parquet.schema_arrow.names), parquet.num_row_groups


def build_source_inventory(source_dataset: S3Location, *, s3_client: Any, workers: int) -> dict[str, Any]:
    objects = sorted(
        (
            {
                "bucket": source_dataset.bucket,
                "key": str(item["Key"]),
                "bytes": int(item["Size"]),
                "etag": _clean_etag(item.get("ETag")),
            }
            for item in _iter_s3_keys(s3_client, source_dataset)
            if str(item["Key"]).endswith(".parquet")
        ),
        key=lambda item: item["key"],
    )
    if not objects:
        raise ValueError(f"no Parquet objects found under {source_dataset.uri}")

    def inspect(item: dict[str, Any]) -> dict[str, Any]:
        rows, columns, row_groups = _parquet_footer_metadata(
            s3_client,
            bucket=item["bucket"],
            key=item["key"],
            size=item["bytes"],
        )
        missing = sorted(set(SOURCE_COLUMNS).difference(columns))
        if missing:
            raise ValueError(f"s3://{item['bucket']}/{item['key']} lacks columns {missing}")
        return {**item, "rows": rows, "row_groups": row_groups}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = list(pool.map(inspect, objects))
    cursor = 0
    for index, item in enumerate(inspected):
        item["object_index"] = index
        item["global_row_start"] = cursor
        cursor += int(item["rows"])
        item["global_row_end_exclusive"] = cursor
    return {
        "format_version": "image_edit_source_parquet_inventory_v1",
        "ordering": "lexicographic_s3_key_then_parquet_row",
        "s3_uri": source_dataset.uri,
        "bucket": source_dataset.bucket,
        "object_count": len(inspected),
        "total_rows": cursor,
        "objects": inspected,
    }


def _read_json_document(uri: str, *, s3_client: Any) -> dict[str, Any]:
    if uri.startswith("s3://"):
        return _read_s3_json(s3_client, parse_s3_uri(uri))
    value = json.loads(Path(uri).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return value


def load_or_build_inventory(
    *,
    source_inventory: str | None,
    inventory_cache: Path,
    source_dataset: S3Location,
    s3_client: Any,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if source_inventory:
        inventory = _read_json_document(source_inventory, s3_client=s3_client)
    elif inventory_cache.is_file():
        inventory = _read_json_document(str(inventory_cache), s3_client=s3_client)
    else:
        inventory = build_source_inventory(source_dataset, s3_client=s3_client, workers=workers)
        _write_atomic(inventory_cache, _json_bytes(inventory), overwrite=overwrite)
    if inventory.get("ordering") != "lexicographic_s3_key_then_parquet_row":
        raise ValueError("source inventory ordering is not authoritative")
    if inventory.get("s3_uri", "").rstrip("/") != source_dataset.uri.rstrip("/"):
        raise ValueError("source inventory points to a different dataset")
    return inventory


class S3RangeReader(io.RawIOBase):
    """Seekable boto3-backed reader used by PyArrow for column-pruned reads."""

    def __init__(self, client: Any, *, bucket: str, key: str, size: int) -> None:
        super().__init__()
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = min(position, self._size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        end = self._size if size is None or size < 0 else min(self._position + size, self._size)
        if end <= self._position:
            return b""
        start = self._position
        payload = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={start}-{end - 1}",
        )["Body"].read()
        if len(payload) != end - start:
            raise IOError(
                f"short S3 range read for s3://{self._bucket}/{self._key}: {len(payload)} != {end - start}"
            )
        self._position = end
        return payload

    def readinto(self, buffer: Any) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


def _selected_inventory_objects(
    inventory: Mapping[str, Any], *, global_start: int, global_end: int
) -> list[dict[str, Any]]:
    raw = inventory.get("objects")
    if not isinstance(raw, list):
        raise ValueError("source inventory has no objects list")
    selected = [
        dict(item)
        for item in raw
        if int(item["global_row_start"]) < global_end and int(item["global_row_end_exclusive"]) > global_start
    ]
    if not selected:
        raise ValueError("source inventory does not overlap the selected shard")
    return selected


def _first_instruction(values: Any, *, context: str) -> str:
    if not isinstance(values, list):
        raise ValueError(f"{context}: instructions must be a list")
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{context}: instructions contains no usable text")


def source_row_to_metadata(
    raw: Mapping[str, Any],
    *,
    dataset_uri: str,
    parquet_uri: str,
    object_index: int,
    parquet_local_row: int,
    dataset_global_row: int,
) -> dict[str, Any]:
    context = f"{parquet_uri} row {parquet_local_row}"
    sample_id = raw.get("sample_id")
    input_paths = raw.get("input_s3_paths")
    output_path = raw.get("output_s3_path")
    num_inputs = raw.get("num_inputs")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{context}: invalid sample_id")
    if isinstance(num_inputs, bool) or not isinstance(num_inputs, int) or num_inputs != 1:
        raise ValueError(f"{context}: this two-image task requires num_inputs == 1")
    if not isinstance(input_paths, list) or len(input_paths) != 1 or not isinstance(input_paths[0], str):
        raise ValueError(f"{context}: invalid input_s3_paths")
    reference_path = input_paths[0].strip()
    parse_s3_uri(reference_path)
    if not isinstance(output_path, str):
        raise ValueError(f"{context}: invalid output_s3_path")
    target_path = output_path.strip()
    parse_s3_uri(target_path)
    if reference_path == target_path:
        raise ValueError(f"{context}: reference and target image paths must differ")
    return {
        "sample_id": sample_id,
        "instruction": _first_instruction(raw.get("instructions"), context=context),
        "reference_images": [{"reference_id": "frame_0", "s3_uri": reference_path}],
        "target_image": {"s3_uri": target_path},
        "dataset_source": {
            "dataset_s3_uri": dataset_uri,
            "parquet_s3_uri": parquet_uri,
            "parquet_object_index": object_index,
            "parquet_local_row": parquet_local_row,
            "dataset_global_row": dataset_global_row,
        },
    }


def _read_source_object(
    item: Mapping[str, Any],
    *,
    s3_client: Any,
    source_dataset_uri: str,
    global_start: int,
    global_end: int,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("pyarrow is required to read source Parquet metadata") from exc
    bucket = str(item.get("bucket") or parse_s3_uri(source_dataset_uri).bucket)
    key = str(item["key"])
    size = int(item["bytes"])
    head = s3_client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != size or _clean_etag(head.get("ETag")) != _clean_etag(item.get("etag")):
        raise ValueError(f"source Parquet changed since inventory: s3://{bucket}/{key}")
    reader = S3RangeReader(s3_client, bucket=bucket, key=key, size=size)
    parquet = pq.ParquetFile(reader)
    table = parquet.read(columns=list(SOURCE_COLUMNS), use_threads=False)
    if table.num_rows != int(item["rows"]):
        raise ValueError(f"row count changed for s3://{bucket}/{key}")
    object_start = int(item["global_row_start"])
    overlap_start = max(global_start, object_start)
    overlap_end = min(global_end, int(item["global_row_end_exclusive"]))
    local_start = overlap_start - object_start
    rows = table.slice(local_start, overlap_end - overlap_start).to_pylist()
    return [
        source_row_to_metadata(
            raw,
            dataset_uri=source_dataset_uri,
            parquet_uri=f"s3://{bucket}/{key}",
            object_index=int(item["object_index"]),
            parquet_local_row=local_start + offset,
            dataset_global_row=overlap_start + offset,
        )
        for offset, raw in enumerate(rows)
    ]


def recover_source_metadata(
    inventory: Mapping[str, Any],
    *,
    s3_client: Any,
    source_dataset_uri: str,
    global_start: int,
    global_end: int,
    workers: int,
) -> list[dict[str, Any]]:
    if int(inventory.get("total_rows", -1)) < global_end:
        raise ValueError("source inventory is shorter than the annotation corpus")
    selected = _selected_inventory_objects(inventory, global_start=global_start, global_end=global_end)

    def read(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _read_source_object(
            item,
            s3_client=s3_client,
            source_dataset_uri=source_dataset_uri,
            global_start=global_start,
            global_end=global_end,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(read, selected))
    rows = sorted(
        (row for chunk in chunks for row in chunk),
        key=lambda row: row["dataset_source"]["dataset_global_row"],
    )
    if len(rows) != global_end - global_start:
        raise ValueError(f"recovered {len(rows)} source rows; expected {global_end - global_start}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source metadata contains duplicate sample IDs")
    return rows


def list_annotation_cases(root: S3Location, *, shard: int, s3_client: Any) -> dict[str, S3Location]:
    prefix = _join_s3(root, f"full/shard_{shard:02d}/cases")
    result: dict[str, S3Location] = {}
    for item in _iter_s3_keys(s3_client, prefix):
        key = str(item["Key"])
        match = CASE_KEY_RE.search(key)
        if match is None:
            continue
        sample_id = match.group(1)
        if sample_id in result:
            raise ValueError(f"duplicate annotation case: {sample_id}")
        result[sample_id] = S3Location(root.bucket, key)
    if not result:
        raise ValueError(f"no {ANNOTATION_NAME} objects under {prefix.uri}")
    return result


def build_prepared_record(
    metadata: Mapping[str, Any], *, status: str, target_json: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "sample_id": metadata["sample_id"],
        "status": status,
        "instruction": metadata["instruction"],
        "reference_images": metadata["reference_images"],
        "target_image": metadata["target_image"],
        "target_json": target_json,
    }


def build_output_record(
    metadata: Mapping[str, Any],
    *,
    status: str,
    target_json: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one prompt-free row for runtime conversation materialization."""

    # Validate without changing the semantic object. The runtime loader is the
    # sole authority that adds system/user/assistant messages and image slots.
    canonical_inline_target(target_json)
    return build_prepared_record(metadata, status=status, target_json=target_json)


def _verify_image_uri(client: Any, uri: str) -> tuple[str, int, str]:
    location = parse_s3_uri(uri)
    head = client.head_object(Bucket=location.bucket, Key=location.key)
    size = int(head.get("ContentLength", 0))
    if size <= 0:
        raise ValueError(f"empty image object: {uri}")
    return uri, size, str(head.get("ContentType", ""))


def verify_images(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    sample_count: int,
    s3_client: Any,
    workers: int,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": mode, "objects_checked": 0}
    rows = list(metadata_rows)
    if mode == "sample":
        rows = rows[:sample_count]
    uris = []
    for row in rows:
        uris.append(row["reference_images"][0]["s3_uri"])
        uris.append(row["target_image"]["s3_uri"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda uri: _verify_image_uri(s3_client, uri), uris))
    return {
        "mode": mode,
        "objects_checked": len(results),
        "content_type_counts": dict(sorted(Counter(item[2] for item in results).items())),
        "total_bytes": sum(item[1] for item in results),
    }


def _aws_session(profile: str | None, region: str) -> tuple[Any, Any]:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("boto3 is required for S3 conversion") from exc
    session = boto3.Session(profile_name=profile or None, region_name=region)
    client = session.client(
        "s3",
        region_name=region,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, max_pool_connections=128),
    )
    return session, client


def convert(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    actual_prompt_digest = prompt_contract_sha256()
    if actual_prompt_digest != PROMPT_CONTRACT_SHA256:
        raise ValueError(
            "prompt contract drifted before prepared dataset publication: "
            f"actual={actual_prompt_digest}, declared={PROMPT_CONTRACT_SHA256}"
        )
    _, s3_client = _aws_session(args.aws_profile, args.region)
    annotation_root = _annotation_root(args.annotation_root)
    source_dataset = parse_s3_uri(args.source_dataset.rstrip("/"))
    _progress(
        f"start selection={args.selection} annotation_root={annotation_root.uri} "
        f"output={args.output.resolve()}"
    )
    global_summary_uri = _join_s3(annotation_root, "full/global_summary.json")
    global_summary = _read_s3_json(s3_client, global_summary_uri)
    summaries = _shard_summaries(global_summary)
    selected_summary, global_start, global_end = choose_shard(summaries, shard=args.shard, seed=args.seed)
    shard = int(selected_summary["shard"])
    _progress(
        f"shard={shard:02d}/{len(summaries) - 1:02d} source_rows="
        f"{int(selected_summary['expected_rows']):,} global_range=[{global_start:,},{global_end:,})"
    )
    shard_summary_uri = _join_s3(annotation_root, f"full/shard_{shard:02d}/summary.json")
    shard_summary = _read_s3_json(s3_client, shard_summary_uri)
    if shard_summary != selected_summary:
        raise ValueError(f"shard {shard} summary differs from global summary")

    allowed_statuses = tuple(dict.fromkeys(args.allowed_status))
    status_uri: str | None = None
    statuses: dict[str, str] = {}
    upstream_status_counts: Counter[str] | None = None
    if args.selection == "validated":
        status_uri = args.results_jsonl or str(shard_summary["input_uri"])
        statuses, upstream_status_counts = load_status_map(
            status_uri,
            s3_client=s3_client,
            gcp_project=args.gcp_project,
            expected_rows=int(shard_summary["expected_rows"]),
            expected_counts=shard_summary["status_counts"],
        )
        unknown_statuses = sorted(set(allowed_statuses).difference(upstream_status_counts))
        if unknown_statuses:
            raise ValueError(f"requested statuses are absent from the shard: {unknown_statuses}")

    output = args.output.resolve()
    physical_outputs = output_part_paths(output, args.output_parts)
    inventory_cache = (
        args.inventory_cache.resolve()
        if args.inventory_cache
        else output.with_name("image_edit_v4.source_inventory.json")
    )
    inventory = load_or_build_inventory(
        source_inventory=args.source_inventory,
        inventory_cache=inventory_cache,
        source_dataset=source_dataset,
        s3_client=s3_client,
        workers=args.inventory_workers,
        overwrite=args.overwrite,
    )
    if int(inventory["total_rows"]) != int(global_summary["row_count"]):
        raise ValueError(
            f"source/annotation row count mismatch: {inventory['total_rows']} != "
            f"{global_summary['row_count']}"
        )
    _progress(
        f"shard={shard:02d} source inventory ready: rows={int(inventory['total_rows']):,}; "
        "recovering exact instruction and media paths"
    )
    metadata_rows = recover_source_metadata(
        inventory,
        s3_client=s3_client,
        source_dataset_uri=source_dataset.uri,
        global_start=global_start,
        global_end=global_end,
        workers=args.parquet_workers,
    )
    _progress(f"shard={shard:02d} recovered source metadata rows={len(metadata_rows):,}")
    cases = list_annotation_cases(annotation_root, shard=shard, s3_client=s3_client)
    expected_cases = int(shard_summary["artifact_category_counts"]["cases"])
    if len(cases) != expected_cases:
        raise ValueError(f"listed {len(cases)} annotation cases; expected {expected_cases}")
    _progress(f"shard={shard:02d} listed annotation cases={len(cases):,}")
    metadata_ids = {row["sample_id"] for row in metadata_rows}
    if not set(cases).issubset(metadata_ids):
        raise ValueError(f"{len(set(cases).difference(metadata_ids))} annotation cases lack source metadata")
    if args.selection == "validated":
        if metadata_ids != set(statuses):
            raise ValueError(
                "source metadata IDs do not match authoritative final-results IDs: "
                f"metadata_only={len(metadata_ids.difference(statuses))}, "
                f"status_only={len(set(statuses).difference(metadata_ids))}"
            )
    else:
        statuses = {sample_id: "packaged" for sample_id in cases}

    case_status_counts = Counter(statuses[sample_id] for sample_id in cases)
    if case_status_counts.get("failed", 0):
        raise ValueError("failed samples unexpectedly have packageable case JSON")

    eligible_metadata = [
        row
        for row in metadata_rows
        if row["sample_id"] in cases and statuses[row["sample_id"]] in allowed_statuses
    ]
    missing_cases = [
        row["sample_id"]
        for row in metadata_rows
        if args.selection == "validated"
        and statuses[row["sample_id"]] in allowed_statuses
        and row["sample_id"] not in cases
    ]
    if missing_cases:
        raise ValueError(f"trainable statuses lack case JSON: {missing_cases[:5]}")
    if args.limit > 0:
        eligible_metadata = eligible_metadata[: args.limit]
    _progress(
        f"shard={shard:02d} eligible rows={len(eligible_metadata):,}; "
        f"verifying images mode={args.verify_images}"
    )
    image_verification = verify_images(
        eligible_metadata,
        mode=args.verify_images,
        sample_count=args.image_check_samples,
        s3_client=s3_client,
        workers=args.image_workers,
    )
    _progress(
        f"shard={shard:02d} image verification complete: "
        f"objects_checked={int(image_verification['objects_checked']):,}"
    )

    temporaries: list[Path] = []
    streams: list[BinaryIO] = []
    output_hasher = hashlib.sha256()
    part_hashers = [hashlib.sha256() for _ in physical_outputs]
    part_rows = [0 for _ in physical_outputs]
    emitted = 0
    rejection_counts: Counter[str] = Counter()
    rejection_examples: list[dict[str, str]] = []
    try:
        for physical_output in physical_outputs:
            stream, temporary = _open_atomic_stream(physical_output, overwrite=args.overwrite)
            streams.append(stream)
            temporaries.append(temporary)

        def convert_one(metadata: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
            sample_id = str(metadata["sample_id"])
            location = cases[sample_id]
            target = _read_s3_json(s3_client, location)
            try:
                result = build_output_record(
                    metadata,
                    status=statuses[sample_id],
                    target_json=target,
                )
                return result, None
            except InlineTargetValidationError as exc:
                return None, str(exc)

        conversion_started_at = time.monotonic()
        total_eligible = len(eligible_metadata)
        _progress(
            f"shard={shard:02d} conversion started: rows={total_eligible:,} "
            f"workers={args.annotation_workers} parts={len(physical_outputs)}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.annotation_workers) as pool:
            for processed, (metadata, (record, rejection)) in enumerate(
                zip(eligible_metadata, pool.map(convert_one, eligible_metadata)),
                1,
            ):
                if rejection is not None:
                    rejection_counts[rejection] += 1
                    if len(rejection_examples) < args.rejection_examples:
                        rejection_examples.append(
                            {"sample_id": str(metadata["sample_id"]), "error": rejection}
                        )
                    if args.invalid_policy == "error":
                        raise InlineTargetValidationError(f"{metadata['sample_id']}: {rejection}")
                    continue
                assert record is not None
                payload = _json_bytes(record)
                part_index = emitted % len(streams)
                streams[part_index].write(payload)
                part_hashers[part_index].update(payload)
                part_rows[part_index] += 1
                output_hasher.update(payload)
                emitted += 1
                if args.progress_every > 0 and (
                    processed % args.progress_every == 0 or processed == total_eligible
                ):
                    elapsed = max(time.monotonic() - conversion_started_at, 1e-9)
                    _progress(
                        f"shard={shard:02d} processed={processed:,}/{total_eligible:,} "
                        f"emitted={emitted:,} rejected={sum(rejection_counts.values()):,} "
                        f"rate={processed / elapsed:.1f} rows/s"
                    )
        if emitted == 0:
            raise ValueError("conversion emitted no rows")
        if any(rows == 0 for rows in part_rows):
            raise ValueError(f"--output-parts={len(part_rows)} exceeds emitted rows={emitted}")
        for stream, temporary, physical_output in zip(streams, temporaries, physical_outputs):
            _commit_atomic_stream(stream, temporary, physical_output)
        streams = []
        temporaries = []
    finally:
        for stream in streams:
            if not stream.closed:
                stream.close()
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()

    part_digests = [hasher.hexdigest() for hasher in part_hashers]

    summary = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "annotation_root": annotation_root.uri,
        "annotation_global_summary": global_summary_uri.uri,
        "annotation_shard_summary": shard_summary_uri.uri,
        "selected_shard": shard,
        "annotation_shard_count": len(summaries),
        "selection_seed": args.seed,
        "global_row_start": global_start,
        "global_row_end_exclusive": global_end,
        "selection": args.selection,
        "status_source": status_uri or "S3 case-object presence",
        "source_dataset": source_dataset.uri,
        "source_inventory": args.source_inventory or str(inventory_cache),
        "allowed_statuses": list(allowed_statuses),
        "output_format": "prepared",
        "output": str(output),
        "physical_outputs": [str(path) for path in physical_outputs],
        "physical_output_rows": part_rows,
        "physical_output_sha256": part_digests,
        "output_sha256": output_hasher.hexdigest(),
        "output_rows": emitted,
        "selected_before_schema_validation": len(eligible_metadata),
        "limit": args.limit,
        "upstream_status_counts": (
            dict(sorted(upstream_status_counts.items())) if upstream_status_counts is not None else None
        ),
        "case_status_counts": dict(sorted(case_status_counts.items())),
        "schema_rejection_count": sum(rejection_counts.values()),
        "schema_rejection_counts": dict(sorted(rejection_counts.items())),
        "schema_rejection_examples": rejection_examples,
        "image_verification": image_verification,
        "semantic_values_modified": False,
        "upstream_status_inferred": False,
        "packaged_status_means": (
            "target_json.with_inline_edit.json exists in the selected S3 shard"
            if args.selection == "packaged"
            else None
        ),
        "media_paths_inferred": False,
        "model_api_calls_added": 0,
        "qwen_contract": {
            "template": "qwen3_5",
            "materialization": "runtime_custom_loader",
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "image_order": ["reference frame_0", "target"],
            "image_placeholders": 2,
        },
    }
    summary_path = output.with_name(output.name + ".summary.json")
    manifest_path: Path | None = None
    if not args.no_manifest:
        manifest_path = (
            args.manifest.resolve()
            if args.manifest
            else output.parent / "image_edit_full_json.prepared.manifest.json"
        )
        summary["manifest"] = str(manifest_path)
        summary["manifest_entry_prefix"] = args.manifest_entry_prefix
    _write_atomic(summary_path, _json_bytes(summary), overwrite=args.overwrite)
    if manifest_path is not None:
        update_prepared_manifest(
            manifest_path,
            summary=summary,
            part_paths=physical_outputs,
            part_rows=part_rows,
            part_sha256=part_digests,
            entry_prefix=args.manifest_entry_prefix,
        )
    _progress(
        f"shard={shard:02d} complete: emitted={emitted:,} "
        f"elapsed={time.monotonic() - started_at:.1f}s summary={summary_path}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-inventory", help="Existing local or s3:// inventory JSON")
    parser.add_argument("--inventory-cache", type=Path)
    parser.add_argument("--results-jsonl", help="Override summary input_uri with local/s3:// file")
    parser.add_argument("--shard", type=int, help="0-based shard; omitted means random selection")
    parser.add_argument("--seed", type=int, help="Reproducible random shard selection")
    parser.add_argument(
        "--selection",
        choices=("validated", "packaged"),
        default="validated",
        help=(
            "validated reads authoritative per-sample status; packaged emits every S3 case "
            "without claiming upstream validation"
        ),
    )
    parser.add_argument(
        "--allowed-status",
        action="append",
        default=None,
        help="Authoritative status to include; repeat as needed (default: validated)",
    )
    parser.add_argument(
        "--output-format",
        choices=("prepared",),
        default="prepared",
        help=(
            "Compatibility flag; only prepared is supported. The JSONL stores compact "
            "source fields and the runtime loader materializes all conversation messages."
        ),
    )
    parser.add_argument(
        "--output-parts",
        type=int,
        default=1,
        help=(
            "Physical JSONL files to write round-robin. For rank-local distributed "
            "streaming, the aggregate manifest needs at least one file per DP rank."
        ),
    )
    manifest_group = parser.add_mutually_exclusive_group()
    manifest_group.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Prepared manifest to atomically upsert (default: "
            "<output-dir>/image_edit_full_json.prepared.manifest.json)"
        ),
    )
    manifest_group.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not create/update a prepared dataset manifest",
    )
    parser.add_argument(
        "--manifest-entry-prefix",
        help=(
            "Optional local directory or s3:// prefix recorded for physical shards "
            "after they are published; defaults to their local absolute paths"
        ),
    )
    parser.add_argument("--aws-profile", help="e.g. prod_core-cn-readwrite; omit for pod IRSA")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--gcp-project")
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--parquet-workers", type=int, default=16)
    parser.add_argument("--annotation-workers", type=int, default=64)
    parser.add_argument("--image-workers", type=int, default=32)
    parser.add_argument("--verify-images", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--image-check-samples", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test cap; 0 means full shard")
    parser.add_argument("--invalid-policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--rejection-examples", type=int, default=20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit a flushed progress line every N processed rows; 0 disables progress logs",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.selection == "packaged":
        if args.allowed_status is not None:
            raise ValueError("--allowed-status cannot be combined with --selection packaged")
        if args.results_jsonl is not None:
            raise ValueError("--results-jsonl cannot be combined with --selection packaged")
        args.allowed_status = ["packaged"]
    elif args.allowed_status is None:
        args.allowed_status = ["validated"]
    if not args.allowed_status or any(not value for value in args.allowed_status):
        raise ValueError("allowed statuses must be non-empty")
    for name in (
        "inventory_workers",
        "parquet_workers",
        "annotation_workers",
        "image_workers",
        "image_check_samples",
        "rejection_examples",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.output_parts <= 0:
        raise ValueError("--output-parts must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.no_manifest and args.manifest_entry_prefix is not None:
        raise ValueError("--manifest-entry-prefix cannot be combined with --no-manifest")
    if args.manifest_entry_prefix and args.manifest_entry_prefix.startswith("s3://"):
        parse_s3_uri(args.manifest_entry_prefix.rstrip("/"))


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    summary = convert(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
