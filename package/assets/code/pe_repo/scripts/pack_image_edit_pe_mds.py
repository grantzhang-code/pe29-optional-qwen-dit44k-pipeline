#!/usr/bin/env python3
"""Pack frozen image-edit JSONL and exact source-image bytes into MDS shards.

The source JSON row is preserved verbatim (apart from its JSONL newline), and
the source image is copied byte-for-byte from the row's authoritative
``source_s3_uri``.  Images are never resized, decoded/re-encoded, or replaced
with a companion path.  The resulting Mosaic MDS v2 shards can be downloaded
and cached as whole files by a StreamingDataset-compatible training reader.

Parallelism is process-level plus a bounded thread pool. Indexed UTP workers
own disjoint source-manifest shards; one worker may own multiple JSONL shards.
The elected finalizer waits for all immutable rank manifests and publishes the
root ``index.json``, audit manifest, and ``_SUCCESS`` marker last.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
from http.client import IncompleteRead
import io
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image
from urllib3.exceptions import ProtocolError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_media import read_media_bytes  # noqa: E402


FORMAT_VERSION = "image-edit-pe-mds-original-source-v1"
ALGORITHM_VERSION = "image-edit-pe-jsonl-exact-source-to-mds-v1"
SAMPLING_ALGORITHM_VERSION = "python-mt19937-global-srs-bitset-v1"
MDS_COLUMNS = {
    "row_json": "str",
    "source_image": "bytes",
    "source_sha256": "str",
}
ENCODING_SIZES: dict[str, int | None] = {
    "bytes": None,
    "str": None,
    "int64": 8,
    "float64": 8,
}
MIB = 1024 * 1024
MEDIA_READ_MAX_ATTEMPTS = 8
MEDIA_READ_BACKOFF_SECONDS = 0.25


@dataclass(frozen=True)
class SamplingPlan:
    bitset: bytes
    manifest: Mapping[str, Any]
    manifest_sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.manifest["sample_count"])

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def population_rows(self) -> int:
        return int(self.manifest["population_rows"])

    def includes(self, global_row_index: int) -> bool:
        if not 0 <= global_row_index < self.population_rows:
            raise IndexError(global_row_index)
        return bool(self.bitset[global_row_index >> 3] & (1 << (global_row_index & 7)))

    def part(self, part_index: int) -> Mapping[str, Any]:
        parts = self.manifest["source_parts"]
        if not isinstance(parts, list) or not 0 <= part_index < len(parts):
            raise IndexError(part_index)
        value = parts[part_index]
        if not isinstance(value, Mapping) or int(value["part_index"]) != part_index:
            raise ValueError("sampling plan source parts are not index-addressable")
        return value


def canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    kwargs["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.port is not None
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def normalize_s3_prefix(uri: str) -> str:
    bucket, key = parse_s3_uri(uri.rstrip("/"))
    return f"s3://{bucket}/{key}/"


def join_s3(prefix: str, relative: str) -> str:
    return normalize_s3_prefix(prefix) + relative.lstrip("/")


def validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def s3_client(*, max_pool_connections: int = 256):
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region != "us-east-1":
        raise ValueError("AWS_REGION/AWS_DEFAULT_REGION must be us-east-1")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=5,
            read_timeout=90,
            tcp_keepalive=True,
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


def maybe_get_json(client: Any, uri: str) -> dict[str, Any] | None:
    bucket, key = parse_s3_uri(uri)
    try:
        payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code", "")) in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {uri}")
    return value


def get_exact_object(client: Any, descriptor: Mapping[str, Any]) -> bytes:
    uri = descriptor.get("s3_uri") or descriptor.get("path")
    if not isinstance(uri, str):
        raise ValueError("object descriptor lacks an S3 URI")
    bucket, key = parse_s3_uri(uri)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if descriptor.get("version_id"):
        kwargs["VersionId"] = str(descriptor["version_id"])
    payload = client.get_object(**kwargs)["Body"].read()
    if descriptor.get("bytes") is not None and len(payload) != int(descriptor["bytes"]):
        raise ValueError(f"object byte count changed: {uri}")
    if descriptor.get("sha256") is not None:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != descriptor["sha256"]:
            raise ValueError(f"object SHA-256 changed: {uri}")
    return payload


def object_descriptor(
    client: Any,
    uri: str,
    *,
    sha256: str | None = None,
    sha1: str | None = None,
) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    response = client.head_object(Bucket=bucket, Key=key)
    result: dict[str, Any] = {
        "bytes": int(response["ContentLength"]),
        "etag": str(response.get("ETag", "")).strip('"'),
        "s3_uri": uri,
    }
    if response.get("VersionId"):
        result["version_id"] = str(response["VersionId"])
    if sha256 is not None:
        result["sha256"] = validate_sha256(sha256, "object SHA-256")
    if sha1 is not None:
        result["sha1"] = sha1
    return result


def put_bytes(
    client: Any,
    uri: str,
    payload: bytes,
    *,
    content_type: str,
) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha256(payload).hexdigest()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        Metadata={"algorithm": ALGORITHM_VERSION, "sha256": digest},
    )
    return object_descriptor(client, uri, sha256=digest)


def put_json(client: Any, uri: str, value: Any) -> dict[str, Any]:
    return put_bytes(client, uri, canonical_json(value, pretty=True), content_type="application/json")


def load_source_manifest(
    client: Any,
    source_prefix: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_uri = join_s3(source_prefix, "manifest.json")
    bucket, key = parse_s3_uri(manifest_uri)
    response = client.get_object(Bucket=bucket, Key=key)
    payload = response["Body"].read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != validate_sha256(expected_sha256, "source manifest SHA-256"):
        raise ValueError(f"source manifest SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("source manifest is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("source manifest outputs are missing")
    ordered = sorted(outputs, key=lambda item: int(item["part_index"]))
    if [int(item["part_index"]) for item in ordered] != list(range(len(ordered))):
        raise ValueError("source manifest part indexes are not contiguous")
    if any(
        item.get("part_count") is not None
        and int(item["part_count"]) != len(ordered)
        for item in ordered
    ):
        raise ValueError("source manifest part_count is inconsistent")
    source_bucket, source_key = parse_s3_uri(normalize_s3_prefix(source_prefix).rstrip("/"))
    normalized_outputs: list[dict[str, Any]] = []
    for raw_output in ordered:
        if not isinstance(raw_output, Mapping):
            raise ValueError("source manifest output is not an object")
        uri = raw_output.get("path") or raw_output.get("s3_uri")
        if not isinstance(uri, str):
            raise ValueError("source manifest output lacks an S3 URI")
        bucket, key = parse_s3_uri(uri)
        prefix = source_key.rstrip("/") + "/"
        if bucket != source_bucket or not key.startswith(prefix):
            raise ValueError("source manifest output escapes its dataset prefix")
        relative_key = key[len(prefix) :]
        declared_relative = raw_output.get("relative_key")
        if declared_relative is not None and str(declared_relative) != relative_key:
            raise ValueError("source manifest relative_key does not match its S3 URI")
        compression = raw_output.get("compression")
        if compression is None and relative_key.endswith(".gz"):
            compression = "gzip"
        if compression not in {None, "gzip"}:
            raise ValueError(f"unsupported source shard compression: {compression!r}")
        normalized_outputs.append(
            {
                **dict(raw_output),
                "compression": compression,
                "part_count": len(ordered),
                "path": uri,
                "relative_key": relative_key,
                "s3_uri": uri,
            }
        )
    manifest["outputs"] = normalized_outputs
    descriptor: dict[str, Any] = {
        "bytes": len(payload),
        "etag": str(response.get("ETag", "")).strip('"'),
        "path": manifest_uri,
        "sha256": actual_sha256,
    }
    if response.get("VersionId"):
        descriptor["version_id"] = str(response["VersionId"])
    return manifest, descriptor


def open_source_shard(path: Path, descriptor: Mapping[str, Any]) -> BinaryIO:
    """Open one verified JSONL object using its manifest-bound compression."""

    compression = descriptor.get("compression")
    if compression == "gzip":
        return gzip.open(path, "rb")
    if compression is None:
        return path.open("rb", buffering=8 * MIB)
    raise ValueError(f"unsupported source shard compression: {compression!r}")


def source_population_rows(source_manifest: Mapping[str, Any]) -> int:
    outputs = source_manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("source manifest outputs are missing")
    rows = [int(item["rows"]) for item in outputs]
    if any(value <= 0 for value in rows):
        raise ValueError("source manifest contains a non-positive shard row count")
    return sum(rows)


def selected_count(bitset: bytes | bytearray, start: int, count: int) -> int:
    """Count selected global positions in one half-open row range."""

    if start < 0 or count < 0 or start + count > len(bitset) * 8:
        raise ValueError("bitset count range is invalid")
    stop = start + count
    result = 0
    while start < stop and start & 7:
        result += bool(bitset[start >> 3] & (1 << (start & 7)))
        start += 1
    whole_stop = stop & ~7
    if start < whole_stop:
        result += sum(value.bit_count() for value in bitset[start >> 3 : whole_stop >> 3])
        start = whole_stop
    while start < stop:
        result += bool(bitset[start >> 3] & (1 << (start & 7)))
        start += 1
    return int(result)


def build_sampling_plan(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    sample_count: int,
    seed: int,
) -> tuple[bytes, dict[str, Any]]:
    """Build an exact global simple-random-sample plan without replacement."""

    validate_sha256(source_manifest_sha256, "source manifest SHA-256")
    population_rows = source_population_rows(source_manifest)
    if not 0 < sample_count < population_rows:
        raise ValueError(
            f"sample_count must be in (0, {population_rows}), got {sample_count}"
        )
    if seed < 0:
        raise ValueError("sampling seed must be non-negative")

    selection = bytearray((population_rows + 7) // 8)
    # random.sample(range(...)) implements an exact uniform sample without
    # replacement. The resulting immutable bitset is published and hash-pinned,
    # so workers never depend on reproducing Python's sampling implementation.
    for global_row_index in random.Random(seed).sample(range(population_rows), sample_count):
        selection[global_row_index >> 3] |= 1 << (global_row_index & 7)

    source_parts: list[dict[str, int]] = []
    offset = 0
    outputs = source_manifest["outputs"]
    for output in outputs:
        part_index = int(output["part_index"])
        if part_index != len(source_parts):
            raise ValueError("source manifest part indexes are not contiguous")
        rows = int(output["rows"])
        sampled_rows = selected_count(selection, offset, rows)
        source_parts.append(
            {
                "global_row_start": offset,
                "part_index": part_index,
                "population_rows": rows,
                "sampled_rows": sampled_rows,
            }
        )
        offset += rows
    if offset != population_rows or sum(item["sampled_rows"] for item in source_parts) != sample_count:
        raise AssertionError("sampling plan row accounting is inconsistent")

    return bytes(selection), {
        "algorithm": SAMPLING_ALGORITHM_VERSION,
        "complete": True,
        "population_rows": population_rows,
        "sample_count": sample_count,
        "seed": seed,
        "source_manifest_sha256": source_manifest_sha256,
        "source_parts": source_parts,
    }


def put_bytes_if_absent(
    client: Any,
    uri: str,
    payload: bytes,
    *,
    content_type: str,
) -> dict[str, Any]:
    """Atomically publish immutable bytes, accepting only identical retries."""

    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata={"algorithm": ALGORITHM_VERSION, "sha256": digest},
            IfNoneMatch="*",
        )
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code not in {"ConditionalRequestConflict", "PreconditionFailed"} and status not in {
            409,
            412,
        }:
            raise
        existing = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if existing != payload:
            raise ValueError(f"immutable S3 object already exists with different bytes: {uri}") from error
    descriptor = object_descriptor(client, uri, sha256=digest)
    if descriptor["bytes"] != len(payload):
        raise ValueError(f"immutable S3 object byte count changed: {uri}")
    return descriptor


def _get_json_with_descriptor(
    client: Any,
    uri: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    bucket, key = parse_s3_uri(uri)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code", "")) in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    payload = response["Body"].read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {uri}")
    descriptor: dict[str, Any] = {
        "bytes": len(payload),
        "etag": str(response.get("ETag", "")).strip('"'),
        "s3_uri": uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if response.get("VersionId"):
        descriptor["version_id"] = str(response["VersionId"])
    return value, descriptor


def _validate_sampling_plan(
    client: Any,
    *,
    plan: Mapping[str, Any],
    plan_descriptor: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    sample_count: int,
    seed: int,
    expected_bitset_sha256: str,
) -> SamplingPlan:
    population_rows = source_population_rows(source_manifest)
    if (
        plan.get("algorithm") != SAMPLING_ALGORITHM_VERSION
        or plan.get("complete") is not True
        or int(plan.get("population_rows", -1)) != population_rows
        or int(plan.get("sample_count", -1)) != sample_count
        or int(plan.get("seed", -1)) != seed
        or plan.get("source_manifest_sha256") != source_manifest_sha256
    ):
        raise ValueError("sampling plan contract mismatch")
    parts = plan.get("source_parts")
    outputs = source_manifest["outputs"]
    if not isinstance(parts, list) or len(parts) != len(outputs):
        raise ValueError("sampling plan source part coverage is invalid")
    offset = 0
    for part, output in zip(parts, outputs, strict=True):
        if not isinstance(part, Mapping):
            raise ValueError("sampling plan source part is not an object")
        rows = int(output["rows"])
        if (
            int(part.get("part_index", -1)) != int(output["part_index"])
            or int(part.get("global_row_start", -1)) != offset
            or int(part.get("population_rows", -1)) != rows
            or not 0 <= int(part.get("sampled_rows", -1)) <= rows
        ):
            raise ValueError("sampling plan source part metadata mismatch")
        offset += rows
    bitset_descriptor = plan.get("selection_bitset")
    if not isinstance(bitset_descriptor, Mapping):
        raise ValueError("sampling plan lacks its immutable selection bitset")
    if bitset_descriptor.get("sha256") != expected_bitset_sha256:
        raise ValueError("sampling plan selection bitset SHA-256 mismatch")
    bitset = get_exact_object(client, bitset_descriptor)
    if len(bitset) != (population_rows + 7) // 8:
        raise ValueError("sampling bitset byte count mismatch")
    if population_rows & 7 and bitset[-1] & ~((1 << (population_rows & 7)) - 1):
        raise ValueError("sampling bitset has selected padding positions")
    if sum(value.bit_count() for value in bitset) != sample_count:
        raise ValueError("sampling bitset selected row count mismatch")
    for part in parts:
        actual = selected_count(
            bitset,
            int(part["global_row_start"]),
            int(part["population_rows"]),
        )
        if actual != int(part["sampled_rows"]):
            raise ValueError("sampling bitset per-part row count mismatch")
    return SamplingPlan(
        bitset=bitset,
        manifest=dict(plan),
        manifest_sha256=validate_sha256(
            str(plan_descriptor.get("sha256", "")),
            "sampling plan SHA-256",
        ),
    )


def load_or_create_sampling_plan(
    client: Any,
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    output_prefix: str,
    sample_count: int,
    seed: int,
    expected_bitset_sha256: str,
    rank: int,
    wait_timeout_seconds: int,
    poll_seconds: int,
) -> SamplingPlan | None:
    if sample_count == 0:
        return None
    output_prefix = normalize_s3_prefix(output_prefix)
    plan_uri = join_s3(output_prefix, "sampling/sampling-plan.json")
    deadline = time.monotonic() + wait_timeout_seconds
    while True:
        existing = _get_json_with_descriptor(client, plan_uri)
        if existing is not None:
            plan, descriptor = existing
            return _validate_sampling_plan(
                client,
                plan=plan,
                plan_descriptor=descriptor,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                sample_count=sample_count,
                seed=seed,
                expected_bitset_sha256=expected_bitset_sha256,
            )
        if rank == 0:
            bitset, plan = build_sampling_plan(
                source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                sample_count=sample_count,
                seed=seed,
            )
            actual_bitset_sha256 = hashlib.sha256(bitset).hexdigest()
            if actual_bitset_sha256 != expected_bitset_sha256:
                raise ValueError(
                    "generated sampling bitset SHA-256 changed: "
                    f"{actual_bitset_sha256} != {expected_bitset_sha256}"
                )
            bitset_descriptor = put_bytes_if_absent(
                client,
                join_s3(output_prefix, "sampling/selection.bitset"),
                bitset,
                content_type="application/octet-stream",
            )
            plan["selection_bitset"] = bitset_descriptor
            put_bytes_if_absent(
                client,
                plan_uri,
                canonical_json(plan, pretty=True),
                content_type="application/json",
            )
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for rank 0 to publish the sampling plan")
        print(
            json.dumps(
                {
                    "event": "waiting_for_sampling_plan",
                    "rank": rank,
                    "sample_count": sample_count,
                    "seed": seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 1)))


def _encode_value(encoding: str, value: Any) -> bytes:
    if encoding == "bytes":
        if not isinstance(value, bytes):
            raise TypeError(f"bytes encoding received {type(value).__name__}")
        return value
    if encoding == "str":
        if not isinstance(value, str):
            raise TypeError(f"str encoding received {type(value).__name__}")
        return value.encode("utf-8")
    if encoding == "int64":
        return struct.pack("<q", int(value))
    if encoding == "float64":
        return struct.pack("<d", float(value))
    raise ValueError(f"unsupported MDS encoding: {encoding}")


def _decode_value(encoding: str, payload: bytes) -> Any:
    if encoding == "bytes":
        return payload
    if encoding == "str":
        return payload.decode("utf-8")
    if encoding == "int64":
        return struct.unpack("<q", payload)[0]
    if encoding == "float64":
        return struct.unpack("<d", payload)[0]
    raise ValueError(f"unsupported MDS encoding: {encoding}")


class MDSPartWriter:
    """Minimal uncompressed Mosaic MDS v2 writer with per-shard S3 upload."""

    def __init__(
        self,
        *,
        local_dir: Path,
        remote_prefix: str | None,
        size_limit: int,
        client: Any | None = None,
    ) -> None:
        if not 0 < size_limit < 2**32:
            raise ValueError("MDS size limit must be in (0, 2**32)")
        self.local_dir = local_dir
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True)
        self.remote_prefix = normalize_s3_prefix(remote_prefix) if remote_prefix else None
        self.client = client
        if self.remote_prefix and self.client is None:
            raise ValueError("remote MDS output requires an S3 client")
        self.size_limit = size_limit
        self.column_names = sorted(MDS_COLUMNS)
        self.column_encodings = [MDS_COLUMNS[name] for name in self.column_names]
        self.column_sizes = [ENCODING_SIZES[value] for value in self.column_encodings]
        self.config_data = json.dumps(
            {
                "column_encodings": self.column_encodings,
                "column_names": self.column_names,
                "column_sizes": self.column_sizes,
                "compression": None,
                "format": "mds",
                "hashes": ["sha1"],
                "size_limit": size_limit,
                "version": 2,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.samples: list[bytes] = []
        self.current_size = 8 + len(self.config_data)
        self.shards: list[dict[str, Any]] = []
        self.uploaded_objects: list[dict[str, Any]] = []
        self.total_samples = 0

    def _encode_sample(self, sample: Mapping[str, Any]) -> bytes:
        if set(sample) != set(self.column_names):
            raise ValueError("MDS sample columns differ from the frozen schema")
        variable_sizes: list[int] = []
        values: list[bytes] = []
        for name, encoding, fixed_size in zip(
            self.column_names,
            self.column_encodings,
            self.column_sizes,
            strict=True,
        ):
            encoded = _encode_value(encoding, sample[name])
            if fixed_size is None:
                if len(encoded) >= 2**32:
                    raise ValueError(f"MDS value is too large for {name}")
                variable_sizes.append(len(encoded))
            elif len(encoded) != fixed_size:
                raise RuntimeError(f"encoded size differs for {name}/{encoding}")
            values.append(encoded)
        return b"".join(struct.pack("<I", value) for value in variable_sizes) + b"".join(values)

    def write(self, sample: Mapping[str, Any]) -> None:
        encoded = self._encode_sample(sample)
        addition = 4 + len(encoded)
        if self.samples and self.current_size + addition > self.size_limit:
            self._flush()
        if 8 + len(self.config_data) + addition >= 2**32:
            raise ValueError("one MDS sample exceeds the v2 shard offset limit")
        self.samples.append(encoded)
        self.current_size += addition
        self.total_samples += 1

    def _upload(self, path: Path, sha1: str) -> None:
        if self.remote_prefix is None or self.client is None:
            return
        uri = join_s3(self.remote_prefix, path.name)
        bucket, key = parse_s3_uri(uri)
        concurrency = min(32, max(4, (os.cpu_count() or 8) // 4))
        self.client.upload_file(
            str(path),
            bucket,
            key,
            Config=TransferConfig(
                multipart_threshold=16 * MIB,
                multipart_chunksize=16 * MIB,
                max_concurrency=concurrency,
                use_threads=True,
            ),
        )
        descriptor = object_descriptor(self.client, uri, sha1=sha1)
        if descriptor["bytes"] != path.stat().st_size:
            raise ValueError(f"uploaded MDS shard size changed: {uri}")
        self.uploaded_objects.append(descriptor)
        path.unlink()

    def _flush(self) -> None:
        if not self.samples:
            return
        shard_index = len(self.shards)
        basename = f"shard.{shard_index:05d}.mds"
        count = len(self.samples)
        header_size = 4 + 4 * (count + 1) + len(self.config_data)
        offsets = [header_size]
        for sample in self.samples:
            offsets.append(offsets[-1] + len(sample))
        if offsets[-1] >= 2**32:
            raise ValueError("MDS shard exceeds the v2 offset limit")
        body = (
            struct.pack("<I", count)
            + b"".join(struct.pack("<I", value) for value in offsets)
            + self.config_data
            + b"".join(self.samples)
        )
        sha1 = hashlib.sha1(body).hexdigest()  # noqa: S324 - MDS integrity contract.
        path = self.local_dir / basename
        path.write_bytes(body)
        self.shards.append(
            {
                "column_encodings": self.column_encodings,
                "column_names": self.column_names,
                "column_sizes": self.column_sizes,
                "compression": None,
                "format": "mds",
                "hashes": ["sha1"],
                "raw_data": {
                    "basename": basename,
                    "bytes": len(body),
                    "hashes": {"sha1": sha1},
                },
                "samples": count,
                "size_limit": self.size_limit,
                "version": 2,
                "zip_data": None,
            }
        )
        self._upload(path, sha1)
        self.samples = []
        self.current_size = 8 + len(self.config_data)

    def finish(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        self._flush()
        index = {"version": 2, "shards": self.shards}
        payload = json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path = self.local_dir / "index.json"
        path.write_bytes(payload)
        descriptor = None
        if self.remote_prefix is not None and self.client is not None:
            uri = join_s3(self.remote_prefix, "index.json")
            descriptor = put_bytes(self.client, uri, payload, content_type="application/json")
            path.unlink()
        return index, descriptor


def decode_mds_sample(path: Path, descriptor: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Decode one local MDS sample; used by unit and pilot verification."""

    sample_count = int(descriptor["samples"])
    if not 0 <= index < sample_count:
        raise IndexError(index)
    payload = path.read_bytes()
    actual_count = struct.unpack_from("<I", payload, 0)[0]
    if actual_count != sample_count:
        raise ValueError("MDS sample count differs from its index")
    offsets = struct.unpack_from(f"<{sample_count + 1}I", payload, 4)
    encoded = payload[offsets[index] : offsets[index + 1]]
    names = list(descriptor["column_names"])
    encodings = list(descriptor["column_encodings"])
    sizes = list(descriptor["column_sizes"])
    variable_count = sum(value is None for value in sizes)
    variable_sizes = struct.unpack_from(f"<{variable_count}I", encoded, 0)
    variable_index = 0
    cursor = 4 * variable_count
    result: dict[str, Any] = {}
    for name, encoding, fixed_size in zip(names, encodings, sizes, strict=True):
        if fixed_size is None:
            size = variable_sizes[variable_index]
            variable_index += 1
        else:
            size = int(fixed_size)
        result[name] = _decode_value(encoding, encoded[cursor : cursor + size])
        cursor += size
    if cursor != len(encoded):
        raise ValueError("MDS sample has trailing bytes")
    return result


@dataclass(frozen=True)
class SourceRow:
    line_number: int
    row_json: str
    sample_id: str


@dataclass(frozen=True)
class SourceRowGroup:
    source_uri: str
    rows: tuple[SourceRow, ...]


@dataclass(frozen=True)
class FetchedMedia:
    payload: bytes
    sha256: str
    width: int
    height: int


def iter_source_row_groups(
    source: BinaryIO,
    *,
    context: str,
    max_rows: int | None = None,
    sampling_plan: SamplingPlan | None = None,
    global_row_start: int = 0,
) -> Iterator[SourceRowGroup]:
    """Group consecutive equal source locators without changing row order."""

    if max_rows is not None and sampling_plan is not None:
        raise ValueError("max_rows and sampling_plan are mutually exclusive")
    if global_row_start < 0:
        raise ValueError("global_row_start must be non-negative")
    current_uri: str | None = None
    current_rows: list[SourceRow] = []
    for line_number, line in enumerate(source, start=1):
        if max_rows is not None and line_number > max_rows:
            break
        if sampling_plan is not None and not sampling_plan.includes(
            global_row_start + line_number - 1
        ):
            continue
        payload = line.rstrip(b"\r\n")
        if not payload:
            raise ValueError(f"empty JSONL row at {context}:{line_number}")
        try:
            row = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON at {context}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSON object expected at {context}:{line_number}")
        source_uri = row.get("source_s3_uri")
        if not isinstance(source_uri, str) or not source_uri.startswith(("s3://", "parquet+s3://")):
            raise ValueError(f"invalid source_s3_uri at {context}:{line_number}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"missing sample_id at {context}:{line_number}")
        try:
            row_json = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"non-UTF-8 JSON at {context}:{line_number}") from error
        item = SourceRow(line_number, row_json, sample_id)
        if current_uri is None or source_uri == current_uri:
            current_uri = source_uri
            current_rows.append(item)
            continue
        yield SourceRowGroup(current_uri, tuple(current_rows))
        current_uri = source_uri
        current_rows = [item]
    if current_uri is not None:
        yield SourceRowGroup(current_uri, tuple(current_rows))


def _retryable_media_read_error(error: BaseException) -> bool:
    """Classify transient failures raised while consuming an object body."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ClientError):
            response = current.response
            status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)
            code = str(response.get("Error", {}).get("Code") or "")
            return status >= 500 or code in {
                "InternalError",
                "RequestTimeout",
                "RequestTimeoutException",
                "ServiceUnavailable",
                "SlowDown",
                "Throttling",
                "ThrottlingException",
            }
        if isinstance(
            current,
            (BotoCoreError, IncompleteRead, ProtocolError, TimeoutError, ConnectionError),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def fetch_exact_media(source_uri: str) -> FetchedMedia:
    """Read and validate exact encoded bytes from the authoritative locator."""

    for attempt in range(1, MEDIA_READ_MAX_ATTEMPTS + 1):
        try:
            payload = read_media_bytes(source_uri)
            break
        except Exception as error:
            if attempt == MEDIA_READ_MAX_ATTEMPTS or not _retryable_media_read_error(error):
                raise
            delay = min(
                8.0,
                MEDIA_READ_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            ) + random.random() * MEDIA_READ_BACKOFF_SECONDS
            time.sleep(delay)
    else:  # pragma: no cover - the loop either succeeds or raises.
        raise AssertionError("media read retry loop did not terminate")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image.verify()
    except Exception as error:
        raise ValueError(f"source image cannot be decoded: {source_uri}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"source image has invalid dimensions: {source_uri}")
    return FetchedMedia(payload, digest, width, height)


def pack_row_groups(
    groups: Iterable[SourceRowGroup],
    *,
    writer: MDSPartWriter,
    threads: int,
    max_inflight: int,
    counters: Counter[str],
) -> None:
    """Fetch groups concurrently and emit results in the original row order."""

    if threads <= 0 or max_inflight < threads:
        raise ValueError("threads must be positive and max_inflight must be >= threads")
    iterator = iter(groups)
    pending: deque[tuple[SourceRowGroup, concurrent.futures.Future[FetchedMedia]]] = deque()
    exhausted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_inflight:
                try:
                    group = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((group, pool.submit(fetch_exact_media, group.source_uri)))
            if not pending:
                continue
            group, future = pending.popleft()
            try:
                media = future.result()
            except Exception as error:
                first = group.rows[0]
                raise RuntimeError(
                    f"media fetch failed for sample_id={first.sample_id} "
                    f"line={first.line_number} uri={group.source_uri}"
                ) from error
            occurrences = len(group.rows)
            counters["unique_media_fetches"] += 1
            counters["media_bytes_fetched"] += len(media.payload)
            counters["source_image_output_bytes"] += len(media.payload) * occurrences
            counters["duplicate_occurrences_collapsed"] += occurrences - 1
            counters[
                "parquet_source_rows"
                if group.source_uri.startswith("parquet+s3://")
                else "direct_s3_source_rows"
            ] += occurrences
            for row in group.rows:
                writer.write(
                    {
                        "row_json": row.row_json,
                        "source_image": media.payload,
                        "source_sha256": media.sha256,
                    }
                )
                counters["rows_written"] += 1


def download_verified_shard(client: Any, output: Mapping[str, Any], destination: Path) -> None:
    uri = output.get("path") or output.get("s3_uri")
    if not isinstance(uri, str):
        raise ValueError("source output lacks an S3 URI")
    bucket, key = parse_s3_uri(uri)
    extra: dict[str, str] = {}
    if output.get("version_id"):
        extra["VersionId"] = str(output["version_id"])
    kwargs: dict[str, Any] = {
        "Config": TransferConfig(
            multipart_threshold=16 * MIB,
            multipart_chunksize=16 * MIB,
            max_concurrency=32,
            use_threads=True,
        )
    }
    if extra:
        kwargs["ExtraArgs"] = extra
    client.download_file(bucket, key, str(destination), **kwargs)
    if destination.stat().st_size != int(output["bytes"]):
        raise ValueError(f"source shard size changed: {uri}")
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * MIB), b""):
            digest.update(chunk)
    if digest.hexdigest() != output["sha256"]:
        raise ValueError(f"source shard SHA-256 changed: {uri}")


def part_relative(rank: int, world_size: int) -> str:
    return f"parts/part-{rank:05d}-of-{world_size:05d}"


def rank_manifest_relative(rank: int, world_size: int) -> str:
    return f"rank-manifests/part-{rank:05d}-of-{world_size:05d}.json"


def _validate_existing_rank(
    value: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    source_manifest_sha256: str,
    max_rows_per_shard: int,
    sample_count: int,
    sample_seed: int,
    sampling_plan_sha256: str,
) -> bool:
    return (
        value.get("complete") is True
        and value.get("format_version") == FORMAT_VERSION
        and value.get("algorithm_version") == ALGORITHM_VERSION
        and int(value.get("rank", -1)) == rank
        and int(value.get("world_size", -1)) == world_size
        and value.get("source_manifest_sha256") == source_manifest_sha256
        and int(value.get("max_rows_per_shard", -1)) == max_rows_per_shard
        and int(value.get("sample_count", 0)) == sample_count
        and int(value.get("sample_seed", 0)) == sample_seed
        and str(value.get("sampling_plan_sha256", "")) == sampling_plan_sha256
    )


def run_worker(args: argparse.Namespace) -> None:
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.sample_count and args.max_rows_per_shard:
        raise ValueError("sample_count and max_rows_per_shard are mutually exclusive")
    source_prefix = normalize_s3_prefix(args.source)
    output_prefix = normalize_s3_prefix(args.output)
    if source_prefix == output_prefix:
        raise ValueError("source and output prefixes must differ")
    client = s3_client()
    success = maybe_get_json(client, join_s3(output_prefix, "_SUCCESS"))
    if success is not None:
        if (
            success.get("complete") is True
            and success.get("source_manifest_sha256") == args.source_manifest_sha256
            and int(success.get("sample_count", 0)) == args.sample_count
            and int(success.get("sample_seed", 0)) == args.sample_seed
        ):
            print(json.dumps({"status": "already_complete", "rank": args.rank}), flush=True)
            return
        raise ValueError("output prefix contains a conflicting _SUCCESS marker")

    source_manifest, source_descriptor = load_source_manifest(
        client, source_prefix, args.source_manifest_sha256
    )
    population_rows = source_population_rows(source_manifest)
    if args.expected_source_rows and population_rows != args.expected_source_rows:
        raise ValueError(
            f"source population row count mismatch: {population_rows} != "
            f"{args.expected_source_rows}"
        )
    sampling_plan = load_or_create_sampling_plan(
        client,
        source_manifest=source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        output_prefix=output_prefix,
        sample_count=args.sample_count,
        seed=args.sample_seed,
        expected_bitset_sha256=args.expected_sampling_bitset_sha256,
        rank=args.rank,
        wait_timeout_seconds=args.sampling_plan_timeout_seconds,
        poll_seconds=args.sampling_plan_poll_seconds,
    )
    sampling_plan_sha256 = sampling_plan.manifest_sha256 if sampling_plan is not None else ""

    rank_manifest_uri = join_s3(output_prefix, rank_manifest_relative(args.rank, args.world_size))
    existing = maybe_get_json(client, rank_manifest_uri)
    if existing is not None:
        if _validate_existing_rank(
            existing,
            rank=args.rank,
            world_size=args.world_size,
            source_manifest_sha256=args.source_manifest_sha256,
            max_rows_per_shard=args.max_rows_per_shard,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            sampling_plan_sha256=sampling_plan_sha256,
        ):
            print(
                json.dumps({"status": "rank_already_complete", "rank": args.rank}),
                flush=True,
            )
            return
        raise ValueError(f"rank {args.rank} has a conflicting completion manifest")

    outputs = source_manifest["outputs"]
    if args.world_size > len(outputs):
        raise ValueError("world_size cannot exceed the source shard count")
    assigned = [item for item in outputs if int(item["part_index"]) % args.world_size == args.rank]
    if not assigned:
        raise ValueError(f"rank {args.rank} has no assigned source shards")

    threads = args.threads or (os.cpu_count() or 8)
    max_inflight = args.max_inflight or max(threads * 2, threads)
    work_parent = Path(args.work_dir)
    work_parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    source_shards: list[dict[str, Any]] = []
    part_prefix = join_s3(output_prefix, part_relative(args.rank, args.world_size))
    with tempfile.TemporaryDirectory(
        prefix=f"image-edit-pe-mds-{args.rank:05d}-", dir=work_parent
    ) as temporary:
        temporary_path = Path(temporary)
        writer = MDSPartWriter(
            local_dir=temporary_path / "mds",
            remote_prefix=part_prefix,
            size_limit=args.size_limit_mb * MIB,
            client=client,
        )
        for ordinal, output in enumerate(assigned, start=1):
            input_path = temporary_path / Path(str(output["relative_key"])).name
            print(
                json.dumps(
                    {
                        "event": "source_shard_start",
                        "rank": args.rank,
                        "ordinal": ordinal,
                        "assigned": len(assigned),
                        "part_index": int(output["part_index"]),
                        "threads": threads,
                        "max_inflight": max_inflight,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            download_verified_shard(client, output, input_path)
            before = counters["rows_written"]
            max_rows = args.max_rows_per_shard or None
            plan_part = (
                sampling_plan.part(int(output["part_index"]))
                if sampling_plan is not None
                else None
            )
            with open_source_shard(input_path, output) as source:
                groups = iter_source_row_groups(
                    source,
                    context=str(output["path"]),
                    max_rows=max_rows,
                    sampling_plan=sampling_plan,
                    global_row_start=(
                        int(plan_part["global_row_start"]) if plan_part is not None else 0
                    ),
                )
                pack_row_groups(
                    groups,
                    writer=writer,
                    threads=threads,
                    max_inflight=max_inflight,
                    counters=counters,
                )
            written = counters["rows_written"] - before
            if plan_part is not None:
                expected = int(plan_part["sampled_rows"])
            else:
                expected = min(int(output["rows"]), max_rows) if max_rows else int(output["rows"])
            if written != expected:
                raise ValueError(f"source shard row count changed: wrote {written}, expected {expected}")
            counters["source_rows_scanned"] += int(output["rows"])
            counters["source_rows_selected"] += written
            source_shards.append(
                {
                    "bytes": int(output["bytes"]),
                    "part_index": int(output["part_index"]),
                    "path": str(output["path"]),
                    "rows_available": int(output["rows"]),
                    "rows_written": written,
                    "sha256": str(output["sha256"]),
                    "version_id": str(output.get("version_id", "")),
                }
            )
            input_path.unlink()
            print(
                json.dumps(
                    {
                        "event": "source_shard_complete",
                        "rank": args.rank,
                        "part_index": int(output["part_index"]),
                        "rows_written": counters["rows_written"],
                        "unique_media_fetches": counters["unique_media_fetches"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        index, index_descriptor = writer.finish()

    if index_descriptor is None:
        raise AssertionError("remote MDS index was not published")
    if writer.total_samples != counters["rows_written"]:
        raise ValueError("MDS writer sample count differs from processed rows")
    if sum(int(item["samples"]) for item in index["shards"]) != writer.total_samples:
        raise ValueError("part index sample count differs from processed rows")
    rank_manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "complete": True,
        "counters": dict(sorted(counters.items())),
        "format": {
            "columns": MDS_COLUMNS,
            "compression": None,
            "image_encoding": "original encoded bytes copied byte-for-byte",
            "image_transform": "none",
            "mds_version": 2,
            "size_limit_bytes": args.size_limit_mb * MIB,
        },
        "format_version": FORMAT_VERSION,
        "max_inflight": max_inflight,
        "max_rows_per_shard": args.max_rows_per_shard,
        "sample_count": args.sample_count,
        "sample_seed": args.sample_seed,
        "sampling_bitset_sha256": args.expected_sampling_bitset_sha256,
        "sampling_plan_sha256": sampling_plan_sha256,
        "sampling": (
            {
                "algorithm": SAMPLING_ALGORITHM_VERSION,
                "manifest_sha256": sampling_plan_sha256,
                "manifest_uri": join_s3(output_prefix, "sampling/sampling-plan.json"),
                "population_rows": sampling_plan.population_rows,
                "sample_count": sampling_plan.sample_count,
                "seed": sampling_plan.seed,
            }
            if sampling_plan is not None
            else None
        ),
        "mds_index": index_descriptor,
        "mds_objects": writer.uploaded_objects,
        "mds_shards": len(index["shards"]),
        "output_prefix": output_prefix,
        "part_prefix": part_prefix,
        "rank": args.rank,
        "source_manifest": source_descriptor,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_shards": source_shards,
        "threads": threads,
        "world_size": args.world_size,
    }
    descriptor = put_json(client, rank_manifest_uri, rank_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "rank": args.rank,
                "rows": counters["rows_written"],
                "mds_shards": len(index["shards"]),
                "rank_manifest": descriptor,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def load_rank_manifest(
    client: Any,
    *,
    output_prefix: str,
    rank: int,
    world_size: int,
    source_manifest_sha256: str,
    sample_count: int,
    sample_seed: int,
    sampling_plan_sha256: str,
) -> dict[str, Any] | None:
    uri = join_s3(output_prefix, rank_manifest_relative(rank, world_size))
    value = maybe_get_json(client, uri)
    if value is None:
        return None
    if not _validate_existing_rank(
        value,
        rank=rank,
        world_size=world_size,
        source_manifest_sha256=source_manifest_sha256,
        max_rows_per_shard=0,
        sample_count=sample_count,
        sample_seed=sample_seed,
        sampling_plan_sha256=sampling_plan_sha256,
    ):
        raise ValueError(f"rank {rank} completion manifest is incompatible")
    return value


def merge_part_indexes(
    parts: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    root_shards: list[dict[str, Any]] = []
    for part_relative_path, index in parts:
        if index.get("version") != 2 or not isinstance(index.get("shards"), list):
            raise ValueError(f"invalid MDS part index: {part_relative_path}")
        for source_shard in index["shards"]:
            shard = json.loads(json.dumps(source_shard))
            for data_key in ("raw_data", "zip_data"):
                data = shard.get(data_key)
                if data is None:
                    continue
                basename = data.get("basename")
                if not isinstance(basename, str) or "/" in basename:
                    raise ValueError("part MDS basename must be a plain filename")
                data["basename"] = f"{part_relative_path.strip('/')}/{basename}"
            root_shards.append(shard)
    return {"version": 2, "shards": root_shards}


def run_finalize(args: argparse.Namespace) -> None:
    source_prefix = normalize_s3_prefix(args.source)
    output_prefix = normalize_s3_prefix(args.output)
    client = s3_client()
    existing = maybe_get_json(client, join_s3(output_prefix, "_SUCCESS"))
    if existing is not None:
        if (
            existing.get("complete") is True
            and existing.get("source_manifest_sha256") == args.source_manifest_sha256
            and int(existing.get("sample_count", 0)) == args.sample_count
            and int(existing.get("sample_seed", 0)) == args.sample_seed
        ):
            print(json.dumps({"status": "already_complete"}), flush=True)
            return
        raise ValueError("output prefix contains a conflicting _SUCCESS marker")
    source_manifest, source_descriptor = load_source_manifest(
        client, source_prefix, args.source_manifest_sha256
    )
    population_rows = source_population_rows(source_manifest)
    if args.expected_source_rows and population_rows != args.expected_source_rows:
        raise ValueError(
            f"source population row count mismatch: {population_rows} != "
            f"{args.expected_source_rows}"
        )
    sampling_plan = load_or_create_sampling_plan(
        client,
        source_manifest=source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        output_prefix=output_prefix,
        sample_count=args.sample_count,
        seed=args.sample_seed,
        expected_bitset_sha256=args.expected_sampling_bitset_sha256,
        rank=0,
        wait_timeout_seconds=args.sampling_plan_timeout_seconds,
        poll_seconds=args.sampling_plan_poll_seconds,
    )
    sampling_plan_sha256 = sampling_plan.manifest_sha256 if sampling_plan is not None else ""
    if args.world_size > len(source_manifest["outputs"]):
        raise ValueError("world_size cannot exceed the source shard count")
    rank_manifests: list[dict[str, Any]] = []
    parts: list[tuple[str, Mapping[str, Any]]] = []
    total_mds_bytes = 0
    total_rows = 0
    aggregate_counters: Counter[str] = Counter()
    covered_parts: list[int] = []
    for rank in range(args.world_size):
        rank_manifest = load_rank_manifest(
            client,
            output_prefix=output_prefix,
            rank=rank,
            world_size=args.world_size,
            source_manifest_sha256=args.source_manifest_sha256,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            sampling_plan_sha256=sampling_plan_sha256,
        )
        if rank_manifest is None:
            raise RuntimeError(f"rank {rank} is incomplete")
        rank_manifests.append(rank_manifest)
        for key, value in rank_manifest["counters"].items():
            aggregate_counters[key] += int(value)
        total_rows += int(rank_manifest["counters"]["rows_written"])
        total_mds_bytes += sum(int(item["bytes"]) for item in rank_manifest["mds_objects"])
        covered_parts.extend(int(item["part_index"]) for item in rank_manifest["source_shards"])
        index_payload = get_exact_object(client, rank_manifest["mds_index"])
        part_index = json.loads(index_payload)
        expected = int(rank_manifest["counters"]["rows_written"])
        actual = sum(int(item["samples"]) for item in part_index["shards"])
        if actual != expected:
            raise ValueError(f"rank {rank} MDS index rows differ: {actual} != {expected}")
        parts.append((part_relative(rank, args.world_size), part_index))

    expected_rows = args.sample_count or population_rows
    if total_rows != expected_rows:
        raise ValueError(f"root row count differs: {total_rows} != {expected_rows}")
    if sorted(covered_parts) != list(range(len(source_manifest["outputs"]))):
        raise ValueError("source part coverage is incomplete or duplicated")
    root_index = merge_part_indexes(parts)
    if sum(int(item["samples"]) for item in root_index["shards"]) != expected_rows:
        raise ValueError("root MDS index sample count differs from source")
    root_index_payload = json.dumps(root_index, sort_keys=True, separators=(",", ":")).encode("utf-8")
    index_descriptor = put_bytes(
        client,
        join_s3(output_prefix, "index.json"),
        root_index_payload,
        content_type="application/json",
    )
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "complete": True,
        "counters": dict(sorted(aggregate_counters.items())),
        "format": {
            "columns": MDS_COLUMNS,
            "compression": None,
            "image_encoding": "original encoded bytes copied byte-for-byte",
            "image_transform": "none",
            "mds_version": 2,
        },
        "format_version": FORMAT_VERSION,
        "mds_index": index_descriptor,
        "mds_shards": len(root_index["shards"]),
        "output_prefix": output_prefix,
        "rank_manifests": len(rank_manifests),
        "sampling": (
            {
                "algorithm": SAMPLING_ALGORITHM_VERSION,
                "manifest_sha256": sampling_plan_sha256,
                "manifest_uri": join_s3(output_prefix, "sampling/sampling-plan.json"),
                "population_rows": population_rows,
                "sample_count": expected_rows,
                "seed": args.sample_seed,
                "selection_bitset": sampling_plan.manifest["selection_bitset"],
            }
            if sampling_plan is not None
            else None
        ),
        "source_dataset": {
            "format_version": source_manifest.get("format_version"),
            "manifest": source_descriptor,
            "prefix": source_prefix,
            "rows": expected_rows,
            "rows_available": population_rows,
            "schema_version": source_manifest.get("schema_version"),
            "shards": len(source_manifest["outputs"]),
        },
        "storage": {
            "mds_data_bytes": total_mds_bytes,
            "source_jsonl_bytes": sum(int(item["bytes"]) for item in source_manifest["outputs"]),
        },
        "world_size": args.world_size,
    }
    manifest_descriptor = put_json(client, join_s3(output_prefix, "manifest.json"), manifest)
    success = {
        "complete": True,
        "format_version": FORMAT_VERSION,
        "index": index_descriptor,
        "manifest": manifest_descriptor,
        "rows": expected_rows,
        "sample_count": args.sample_count,
        "sample_seed": args.sample_seed,
        "sampling_bitset_sha256": args.expected_sampling_bitset_sha256,
        "sampling_plan_sha256": sampling_plan_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    success_descriptor = put_json(client, join_s3(output_prefix, "_SUCCESS"), success)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": expected_rows,
                "mds_shards": len(root_index["shards"]),
                "success": success_descriptor,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_wait_finalize(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.wait_timeout_seconds
    client = s3_client()
    source_manifest, _source_descriptor = load_source_manifest(
        client,
        normalize_s3_prefix(args.source),
        args.source_manifest_sha256,
    )
    population_rows = source_population_rows(source_manifest)
    if args.expected_source_rows and population_rows != args.expected_source_rows:
        raise ValueError(
            f"source population row count mismatch: {population_rows} != "
            f"{args.expected_source_rows}"
        )
    sampling_plan = load_or_create_sampling_plan(
        client,
        source_manifest=source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        output_prefix=args.output,
        sample_count=args.sample_count,
        seed=args.sample_seed,
        expected_bitset_sha256=args.expected_sampling_bitset_sha256,
        rank=0,
        wait_timeout_seconds=args.sampling_plan_timeout_seconds,
        poll_seconds=args.sampling_plan_poll_seconds,
    )
    sampling_plan_sha256 = sampling_plan.manifest_sha256 if sampling_plan is not None else ""
    while True:
        missing: list[int] = []
        for rank in range(args.world_size):
            value = load_rank_manifest(
                client,
                output_prefix=args.output,
                rank=rank,
                world_size=args.world_size,
                source_manifest_sha256=args.source_manifest_sha256,
                sample_count=args.sample_count,
                sample_seed=args.sample_seed,
                sampling_plan_sha256=sampling_plan_sha256,
            )
            if value is None:
                missing.append(rank)
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ranks: {missing[:20]}")
        print(
            json.dumps(
                {
                    "event": "waiting_for_ranks",
                    "complete": args.world_size - len(missing),
                    "world_size": args.world_size,
                    "missing_first": missing[:20],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(min(args.poll_seconds, max(deadline - time.monotonic(), 1)))
    run_finalize(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True)
    common.add_argument("--source-manifest-sha256", required=True)
    common.add_argument("--output", required=True)
    common.add_argument("--world-size", required=True, type=int)
    common.add_argument("--expected-source-rows", type=int, default=0)
    common.add_argument("--sample-count", type=int, default=0)
    common.add_argument("--sample-seed", type=int, default=0)
    common.add_argument("--expected-sampling-bitset-sha256", default="")
    common.add_argument("--sampling-plan-timeout-seconds", type=int, default=3600)
    common.add_argument("--sampling-plan-poll-seconds", type=int, default=10)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", parents=[common])
    worker.add_argument("--rank", required=True, type=int)
    worker.add_argument("--threads", type=int, default=0, help="0 uses every visible CPU")
    worker.add_argument("--max-inflight", type=int, default=0, help="0 uses 2x threads")
    worker.add_argument("--size-limit-mb", type=int, default=256)
    worker.add_argument("--max-rows-per-shard", type=int, default=0)
    worker.add_argument("--work-dir", default="/mnt/ephemeral/pcb-v2-9c4d-17000/data")
    worker.set_defaults(func=run_worker)

    finalize = subparsers.add_parser("finalize", parents=[common])
    finalize.set_defaults(func=run_finalize)

    wait_finalize = subparsers.add_parser("wait-finalize", parents=[common])
    wait_finalize.add_argument("--wait-timeout-seconds", type=int, default=172800)
    wait_finalize.add_argument("--poll-seconds", type=int, default=30)
    wait_finalize.set_defaults(func=run_wait_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size <= 0:
        raise ValueError("world_size must be positive")
    validate_sha256(args.source_manifest_sha256, "source manifest SHA-256")
    for field in (
        "threads",
        "max_inflight",
        "max_rows_per_shard",
        "expected_source_rows",
        "sample_count",
        "sample_seed",
    ):
        if hasattr(args, field) and getattr(args, field) < 0:
            raise ValueError(f"{field} must be non-negative")
    if hasattr(args, "size_limit_mb") and args.size_limit_mb <= 0:
        raise ValueError("size_limit_mb must be positive")
    if hasattr(args, "wait_timeout_seconds") and args.wait_timeout_seconds <= 0:
        raise ValueError("wait_timeout_seconds must be positive")
    if hasattr(args, "poll_seconds") and args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if args.sampling_plan_timeout_seconds <= 0 or args.sampling_plan_poll_seconds <= 0:
        raise ValueError("sampling-plan timeout and poll interval must be positive")
    if args.sample_count == 0 and args.sample_seed != 0:
        raise ValueError("sample_seed must be zero when sampling is disabled")
    if args.sample_count > 0:
        validate_sha256(
            args.expected_sampling_bitset_sha256,
            "expected sampling bitset SHA-256",
        )
    elif args.expected_sampling_bitset_sha256:
        raise ValueError("sampling bitset SHA-256 must be empty when sampling is disabled")
    args.func(args)


if __name__ == "__main__":
    main()
