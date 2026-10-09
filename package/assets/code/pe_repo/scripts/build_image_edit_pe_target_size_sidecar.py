#!/usr/bin/env python3
"""Materialize exact Target canvas sizes aligned to frozen PE MDS rows.

Each output record contains ``width``, ``height``, and SHA-256 of the exact
``row_json`` bytes stored in the corresponding MDS sample.  PE/Parquet rows
use their frozen, identity-bound ``pe_sample.dimensions``.  Direct-S3 Full72
rows read only enough of the Target image header to determine display-oriented
dimensions; this work happens once on CPU workers, never in the training
DataLoader.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import io
import json
import os
import re
import struct
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image, UnidentifiedImageError

from dlms.datasets import image_edit_media
from dlms.datasets.image_edit_media import _s3_client as media_s3_client
from scripts.pack_image_edit_pe_mds import (
    SamplingPlan,
    download_verified_shard,
    get_exact_object,
    load_source_manifest,
    normalize_s3_prefix,
    open_source_shard,
    parse_s3_uri,
    source_population_rows,
    validate_sha256,
)


FORMAT_VERSION = "image-edit-pe-target-size-sidecar-v1"
ALGORITHM_VERSION = "image-edit-pe-target-size-from-frozen-row-parquet-metadata-or-target-header-v2"
MAGIC = b"IEPTS001"
HEADER = struct.Struct("<8sI")
RECORD = struct.Struct("<II32s")
SAMPLING_ALGORITHM = "python-mt19937-global-srs-bitset-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
PREFIX_LIMITS = (64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)


def canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    kwargs["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def join_s3(prefix: str, relative: str) -> str:
    return normalize_s3_prefix(prefix) + relative.lstrip("/")


def control_s3_client(*, max_pool_connections: int = 256):
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region != "us-east-1":
        raise ValueError("AWS_REGION/AWS_DEFAULT_REGION must be us-east-1")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=5,
            read_timeout=120,
            tcp_keepalive=True,
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


def get_bytes(client: Any, uri: str, *, expected_sha256: str | None = None) -> bytes:
    bucket, key = parse_s3_uri(uri)
    payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if expected_sha256 is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != validate_sha256(expected_sha256, "expected object SHA-256"):
            raise ValueError(f"object SHA-256 mismatch for {uri}: {actual}")
    return payload


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


def put_bytes(
    client: Any,
    uri: str,
    payload: bytes,
    *,
    content_type: str,
) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha256(payload).hexdigest()
    response = client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        Metadata={"algorithm": ALGORITHM_VERSION, "sha256": digest},
    )
    result: dict[str, Any] = {
        "bytes": len(payload),
        "etag": str(response.get("ETag", "")).strip('"'),
        "path": uri,
        "sha256": digest,
    }
    if response.get("VersionId"):
        result["version_id"] = str(response["VersionId"])
    return result


def put_json(client: Any, uri: str, value: Any) -> dict[str, Any]:
    return put_bytes(
        client,
        uri,
        canonical_json(value, pretty=True),
        content_type="application/json",
    )


def _parquet_binding(uri: Any) -> tuple[str, str, str, int] | None:
    if not isinstance(uri, str):
        return None
    parsed = urlsplit(uri)
    if parsed.scheme != "parquet+s3" or not parsed.netloc:
        return None
    query = parse_qs(parsed.query, keep_blank_values=True)
    fragment = parse_qs(parsed.fragment, keep_blank_values=True)
    if set(query) != {"etag"} or len(query["etag"]) != 1:
        return None
    if "row" not in fragment or len(fragment["row"]) != 1:
        return None
    try:
        row = int(fragment["row"][0])
    except ValueError:
        return None
    if row < 0:
        return None
    return parsed.netloc, unquote(parsed.path.lstrip("/")), query["etag"][0], row


def trusted_pe_target_size(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return frozen PE output dimensions only with immutable row identity."""

    source_binding = _parquet_binding(row.get("source_s3_uri"))
    target_binding = _parquet_binding(row.get("target_s3_uri"))
    if source_binding is None or source_binding != target_binding:
        return None
    pe_sample = row.get("pe_sample")
    if not isinstance(pe_sample, Mapping):
        return None
    expected_parquet = f"s3://{target_binding[0]}/{target_binding[1]}"
    if pe_sample.get("parquet_uri") != expected_parquet:
        raise ValueError("PE dimensions do not bind the row's immutable Parquet object")
    if int(pe_sample.get("source_parquet_local_row", -1)) != target_binding[3]:
        raise ValueError("PE dimensions do not bind the row's immutable Parquet row")
    dimensions = pe_sample.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("identity-bound PE row has no dimensions list")
    outputs = [
        item
        for item in dimensions
        if isinstance(item, Mapping)
        and item.get("role") == "output"
        and item.get("index", 0) == 0
    ]
    if len(outputs) != 1:
        raise ValueError("identity-bound PE row must have exactly one output dimension")
    width = outputs[0].get("width")
    height = outputs[0].get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError("identity-bound PE target dimensions must be positive integers")
    pixels = outputs[0].get("pixels")
    if pixels is not None and (type(pixels) is not int or pixels != width * height):
        raise ValueError("identity-bound PE target pixel count is inconsistent")
    return width, height


@dataclass(frozen=True)
class FrozenParquetTargetSize:
    width: int
    height: int
    width_column: str
    height_column: str
    identity: str


def _positive_dimension(value: Any, *, field: str) -> int:
    if type(value) is int and value > 0:
        return value
    if type(value) is float and value > 0 and value.is_integer():
        return int(value)
    raise ValueError(f"frozen Parquet {field} must be a positive integer")


def frozen_parquet_target_size(uri: str) -> FrozenParquetTargetSize | None:
    """Read immutable scalar dimensions without touching the image column.

    The two frozen image-edit-v6 source profiles bind ``output_image`` to
    ``output_width/output_height`` and ``target_image`` to
    ``target_width/target_height``.  The Parquet ETag and absolute row carried
    by the media URI make these values identity-bound.  Reading the two scalar
    columns is several orders of magnitude smaller than reading the embedded
    target-image column.
    """

    if not isinstance(uri, str) or not uri.startswith("parquet+s3://"):
        return None
    locator = image_edit_media._parse_parquet_s3_uri(uri)
    dimension_columns = {
        "output_image": ("output_width", "output_height"),
        "target_image": ("target_width", "target_height"),
    }.get(locator.column)
    if dimension_columns is None:
        raise ValueError(
            f"unsupported frozen Parquet Target image column: {locator.column!r}"
        )
    client = media_s3_client(f"s3://{locator.bucket}/{locator.key}")
    info = image_edit_media._get_parquet_object_info(locator, client)
    width_column, height_column = dimension_columns
    if width_column not in info.columns or height_column not in info.columns:
        return None
    row_group = bisect.bisect_right(info.row_group_starts, locator.row) - 1
    if (
        row_group < 0
        or locator.row
        >= info.row_group_starts[row_group] + info.row_group_counts[row_group]
    ):
        raise ValueError(
            f"frozen Parquet Target row {locator.row} is outside row count "
            f"{info.total_rows}"
        )
    row_offset = locator.row - info.row_group_starts[row_group]
    width_table = image_edit_media._get_parquet_direct_path_table(
        locator,
        info,
        row_group,
        width_column,
        client,
    )
    height_table = image_edit_media._get_parquet_direct_path_table(
        locator,
        info,
        row_group,
        height_column,
        client,
    )
    width = _positive_dimension(
        width_table[width_column][row_offset].as_py(), field=width_column
    )
    height = _positive_dimension(
        height_table[height_column][row_offset].as_py(), field=height_column
    )
    identity = hashlib.sha256(
        canonical_json(
            {
                "bucket": locator.bucket,
                "etag": locator.etag,
                "height": height,
                "height_column": height_column,
                "key": locator.key,
                "row": locator.row,
                "width": width,
                "width_column": width_column,
            }
        )
    ).hexdigest()
    return FrozenParquetTargetSize(
        width=width,
        height=height,
        width_column=width_column,
        height_column=height_column,
        identity=identity,
    )


def _display_size_from_prefix(payload: bytes, *, uri: str) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as opened:
        width, height = opened.size
        encoded_format = str(opened.format or "unknown")
        exif_payload = opened.info.get("exif")
    orientation = 1
    if exif_payload:
        try:
            exif = Image.Exif()
            exif.load(exif_payload)
            orientation = int(exif.get(274, 1))
        except (AttributeError, TypeError, ValueError, SyntaxError):
            orientation = 1
    if orientation in {5, 6, 7, 8}:
        width, height = height, width
    if width <= 0 or height <= 0:
        raise ValueError(f"target image has invalid dimensions: {uri}")
    return int(width), int(height), encoded_format


@dataclass(frozen=True)
class DirectTargetSize:
    width: int
    height: int
    encoded_format: str
    bytes_read: int
    object_bytes: int
    etag: str
    version_id: str


def direct_s3_target_size(uri: str) -> DirectTargetSize:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"direct Target image must be a plain s3:// URI: {uri!r}")
    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    if not key:
        raise ValueError(f"direct Target image has an empty key: {uri!r}")
    client = media_s3_client(uri)
    pinned_version = ""
    first_etag = ""
    object_bytes = 0
    last_error: BaseException | None = None
    limits: list[int | None] = [*PREFIX_LIMITS, None]
    for limit in limits:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if pinned_version:
            kwargs["VersionId"] = pinned_version
        if limit is not None:
            kwargs["Range"] = f"bytes=0-{limit - 1}"
        response = client.get_object(**kwargs)
        payload = response["Body"].read()
        version_id = str(response.get("VersionId") or "")
        etag = str(response.get("ETag") or "").strip('"')
        if not pinned_version and version_id:
            pinned_version = version_id
        if not first_etag:
            first_etag = etag
        elif etag != first_etag:
            raise RuntimeError(f"Target object changed during header read: {uri}")
        content_range = response.get("ContentRange")
        if content_range:
            match = CONTENT_RANGE_RE.fullmatch(str(content_range))
            if match is None or int(match.group(1)) != 0:
                raise ValueError(f"invalid Target Content-Range for {uri}: {content_range!r}")
            object_bytes = int(match.group(3))
        else:
            object_bytes = len(payload)
        if not payload:
            raise ValueError(f"empty Target image: {uri}")
        try:
            width, height, encoded_format = _display_size_from_prefix(payload, uri=uri)
            return DirectTargetSize(
                width=width,
                height=height,
                encoded_format=encoded_format,
                bytes_read=len(payload),
                object_bytes=object_bytes,
                etag=etag,
                version_id=version_id or pinned_version,
            )
        except (UnidentifiedImageError, OSError, EOFError, struct.error) as error:
            last_error = error
            if len(payload) >= object_bytes:
                break
    raise ValueError(f"Target image header cannot be decoded: {uri}") from last_error


@dataclass(frozen=True)
class TargetSizeRecord:
    width: int
    height: int
    row_sha256: bytes
    method: str
    bytes_read: int
    object_bytes: int
    encoded_format: str
    target_identity: str


def resolve_target_size_record(row_json: bytes) -> TargetSizeRecord:
    try:
        row = json.loads(row_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selected source row is not valid JSON") from error
    if not isinstance(row, Mapping):
        raise ValueError("selected source row must be an object")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("selected source row has no sample_id")
    row_sha256 = hashlib.sha256(row_json).digest()
    pe_size = trusted_pe_target_size(row)
    if pe_size is not None:
        return TargetSizeRecord(
            width=pe_size[0],
            height=pe_size[1],
            row_sha256=row_sha256,
            method="frozen_pe_dimensions",
            bytes_read=0,
            object_bytes=0,
            encoded_format="parquet_metadata",
            target_identity=hashlib.sha256(
                str(row["target_s3_uri"]).encode("utf-8")
            ).hexdigest(),
        )
    target_uri = row.get("target_s3_uri")
    if not isinstance(target_uri, str):
        raise ValueError(f"sample_id={sample_id}: target_s3_uri is missing")
    parquet_size = frozen_parquet_target_size(target_uri)
    if parquet_size is not None:
        return TargetSizeRecord(
            width=parquet_size.width,
            height=parquet_size.height,
            row_sha256=row_sha256,
            method="frozen_parquet_dimensions",
            bytes_read=0,
            object_bytes=0,
            encoded_format="parquet_metadata",
            target_identity=parquet_size.identity,
        )
    direct = direct_s3_target_size(target_uri)
    identity = canonical_json(
        {
            "etag": direct.etag,
            "object_bytes": direct.object_bytes,
            "target_uri": target_uri,
            "version_id": direct.version_id,
        }
    )
    return TargetSizeRecord(
        width=direct.width,
        height=direct.height,
        row_sha256=row_sha256,
        method="direct_s3_header",
        bytes_read=direct.bytes_read,
        object_bytes=direct.object_bytes,
        encoded_format=direct.encoded_format,
        target_identity=hashlib.sha256(identity).hexdigest(),
    )


def iter_selected_row_json(
    source: Iterable[bytes],
    *,
    sampling_plan: SamplingPlan | None,
    global_row_start: int,
    context: str,
) -> Iterator[bytes]:
    for line_number, line in enumerate(source, start=1):
        if sampling_plan is not None and not sampling_plan.includes(
            global_row_start + line_number - 1
        ):
            continue
        row_json = line.rstrip(b"\r\n")
        if not row_json:
            raise ValueError(f"empty selected row at {context}:{line_number}")
        yield row_json


def resolve_in_order(
    rows: Iterable[bytes],
    *,
    threads: int,
    max_inflight: int,
) -> Iterator[TargetSizeRecord]:
    if threads < 1 or max_inflight < threads:
        raise ValueError("threads must be positive and max_inflight must be >= threads")
    iterator = iter(rows)
    pending: deque[concurrent.futures.Future[TargetSizeRecord]] = deque()
    exhausted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_inflight:
                try:
                    row_json = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.submit(resolve_target_size_record, row_json))
            if pending:
                yield pending.popleft().result()


class SidecarPartWriter:
    def __init__(
        self,
        *,
        client: Any,
        output_prefix: str,
        mds_shards: Sequence[Mapping[str, Any]],
    ) -> None:
        self.client = client
        self.output_prefix = normalize_s3_prefix(output_prefix)
        self.mds_shards = tuple(mds_shards)
        self.shard_index = 0
        self.records: list[TargetSizeRecord] = []
        self.outputs: list[dict[str, Any]] = []
        self.total_records = 0

    def write(self, record: TargetSizeRecord) -> None:
        if self.shard_index >= len(self.mds_shards):
            raise ValueError("more target-size records than MDS rows")
        self.records.append(record)
        self.total_records += 1
        if len(self.records) == int(self.mds_shards[self.shard_index]["samples"]):
            self._flush()

    def _flush(self) -> None:
        mds = self.mds_shards[self.shard_index]
        mds_basename = str(mds["raw_data"]["basename"])
        payload = bytearray(HEADER.pack(MAGIC, len(self.records)))
        for record in self.records:
            payload.extend(
                RECORD.pack(record.width, record.height, record.row_sha256)
            )
        relative_key = f"sidecars/{mds_basename}.target-size-v1.bin"
        descriptor = put_bytes(
            self.client,
            join_s3(self.output_prefix, relative_key),
            bytes(payload),
            content_type="application/octet-stream",
        )
        self.outputs.append(
            {
                **descriptor,
                "mds_basename": mds_basename,
                "mds_sha1": str(mds["raw_data"]["hashes"]["sha1"]),
                "relative_key": relative_key,
                "samples": len(self.records),
            }
        )
        self.records = []
        self.shard_index += 1

    def finish(self) -> None:
        if self.records:
            raise ValueError("target-size records end inside an MDS shard")
        if self.shard_index != len(self.mds_shards):
            raise ValueError("fewer target-size records than MDS rows")


def load_frozen_contract(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    source_manifest, source_descriptor = load_source_manifest(
        client,
        args.source,
        args.source_manifest_sha256,
    )
    population_rows = source_population_rows(source_manifest)
    if population_rows != args.expected_source_rows:
        raise ValueError(
            f"source population rows changed: {population_rows} != {args.expected_source_rows}"
        )
    mds_manifest_payload = get_bytes(
        client,
        join_s3(args.mds, "manifest.json"),
        expected_sha256=args.mds_manifest_sha256,
    )
    mds_index_payload = get_bytes(
        client,
        join_s3(args.mds, "index.json"),
        expected_sha256=args.mds_index_sha256,
    )
    mds_manifest = json.loads(mds_manifest_payload)
    mds_index = json.loads(mds_index_payload)
    sampling = mds_manifest.get("sampling")
    common_mismatch = (
        mds_manifest.get("complete") is not True
        or mds_manifest.get("output_prefix") != normalize_s3_prefix(args.mds)
        or mds_manifest.get("source_dataset", {}).get("manifest", {}).get("sha256")
        != args.source_manifest_sha256
        or mds_manifest.get("world_size") != args.world_size
        or not isinstance(mds_index.get("shards"), list)
    )
    if common_mismatch:
        raise ValueError("frozen MDS/source/sampling contract mismatch")
    sampling_plan: SamplingPlan | None
    if sampling is None:
        if (
            args.expected_rows != args.expected_source_rows
            or args.sample_seed != 0
            or args.sampling_plan_sha256
            or args.sampling_bitset_sha256
        ):
            raise ValueError("unsampled MDS requires empty sampling arguments")
        sampling_plan = None
    else:
        if (
            not isinstance(sampling, Mapping)
            or sampling.get("algorithm") != SAMPLING_ALGORITHM
            or int(sampling.get("sample_count") or 0) != args.expected_rows
            or int(sampling.get("population_rows") or 0) != args.expected_source_rows
            or int(sampling.get("seed", -1)) != args.sample_seed
            or sampling.get("manifest_sha256") != args.sampling_plan_sha256
            or sampling.get("selection_bitset", {}).get("sha256")
            != args.sampling_bitset_sha256
        ):
            raise ValueError("frozen MDS/source/sampling contract mismatch")
        plan_payload = get_bytes(
            client,
            str(sampling["manifest_uri"]),
            expected_sha256=args.sampling_plan_sha256,
        )
        plan = json.loads(plan_payload)
        bitset = get_exact_object(client, plan["selection_bitset"])
        if hashlib.sha256(bitset).hexdigest() != args.sampling_bitset_sha256:
            raise ValueError("selection bitset SHA-256 mismatch")
        sampling_plan = SamplingPlan(bitset, plan, args.sampling_plan_sha256)
        if sampling_plan.sample_count != args.expected_rows:
            raise ValueError("selection plan row count mismatch")
    return {
        "mds_index": mds_index,
        "mds_manifest": mds_manifest,
        "sampling_plan": sampling_plan,
        "source_descriptor": source_descriptor,
        "source_manifest": source_manifest,
    }


def _rank_manifest_relative(rank: int, world_size: int) -> str:
    return f"rank-manifests/part-{rank:05d}-of-{world_size:05d}.json"


def _valid_rank_manifest(
    value: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    rank: int,
) -> bool:
    return (
        value.get("complete") is True
        and value.get("format_version") == FORMAT_VERSION
        and value.get("algorithm_version") == ALGORITHM_VERSION
        and int(value.get("rank", -1)) == rank
        and int(value.get("world_size", -1)) == args.world_size
        and value.get("source_manifest_sha256") == args.source_manifest_sha256
        and value.get("mds_manifest_sha256") == args.mds_manifest_sha256
        and value.get("mds_index_sha256") == args.mds_index_sha256
        and value.get("sampling_plan_sha256") == args.sampling_plan_sha256
        and value.get("sampling_bitset_sha256") == args.sampling_bitset_sha256
    )


def run_worker(args: argparse.Namespace) -> None:
    client = control_s3_client()
    success = maybe_get_json(client, join_s3(args.output, "_SUCCESS"))
    if success is not None:
        if (
            success.get("complete") is True
            and success.get("format_version") == FORMAT_VERSION
            and success.get("mds_manifest_sha256") == args.mds_manifest_sha256
        ):
            print(json.dumps({"status": "already_complete", "rank": args.rank}))
            return
        raise ValueError("output prefix contains a conflicting _SUCCESS")
    rank_uri = join_s3(
        args.output,
        _rank_manifest_relative(args.rank, args.world_size),
    )
    existing = maybe_get_json(client, rank_uri)
    if existing is not None:
        if _valid_rank_manifest(existing, args=args, rank=args.rank):
            print(json.dumps({"status": "rank_already_complete", "rank": args.rank}))
            return
        raise ValueError(f"rank {args.rank} has a conflicting manifest")

    contract = load_frozen_contract(args, client)
    source_manifest = contract["source_manifest"]
    sampling_plan: SamplingPlan | None = contract["sampling_plan"]
    outputs = source_manifest["outputs"]
    assigned = [
        item
        for item in outputs
        if int(item["part_index"]) % args.world_size == args.rank
    ]
    part_prefix = f"parts/part-{args.rank:05d}-of-{args.world_size:05d}/"
    mds_shards = [
        item
        for item in contract["mds_index"]["shards"]
        if str(item.get("raw_data", {}).get("basename", "")).startswith(part_prefix)
    ]
    if not assigned or not mds_shards:
        raise ValueError(f"rank {args.rank} has no source or MDS shards")
    expected_rank_rows = sum(int(item["samples"]) for item in mds_shards)
    expected_selected_rows = sum(
        (
            int(sampling_plan.part(int(item["part_index"]))["sampled_rows"])
            if sampling_plan is not None
            else int(item["rows"])
        )
        for item in assigned
    )
    if expected_rank_rows != expected_selected_rows:
        raise ValueError("source sampling and MDS part row counts disagree")

    threads = args.threads or min(os.cpu_count() or 8, 64)
    max_inflight = args.max_inflight or max(threads * 4, threads)
    counters: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    target_identity_digest = hashlib.sha256()
    row_binding_digest = hashlib.sha256()
    writer = SidecarPartWriter(
        client=client,
        output_prefix=args.output,
        mds_shards=mds_shards,
    )
    work_parent = Path(args.work_dir)
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"image-edit-target-size-{args.rank:05d}-",
        dir=work_parent,
    ) as temporary:
        temporary_path = Path(temporary)
        for source_output in assigned:
            input_path = temporary_path / Path(str(source_output["relative_key"])).name
            download_verified_shard(client, source_output, input_path)
            part = (
                sampling_plan.part(int(source_output["part_index"]))
                if sampling_plan is not None
                else None
            )
            with open_source_shard(input_path, source_output) as source:
                rows = iter_selected_row_json(
                    source,
                    sampling_plan=sampling_plan,
                    global_row_start=int(part["global_row_start"]) if part is not None else 0,
                    context=str(source_output["path"]),
                )
                before = counters["rows"]
                for record in resolve_in_order(
                    rows,
                    threads=threads,
                    max_inflight=max_inflight,
                ):
                    writer.write(record)
                    counters["rows"] += 1
                    counters[f"method_{record.method}"] += 1
                    counters["target_header_bytes_read"] += record.bytes_read
                    counters["target_object_bytes"] += record.object_bytes
                    formats[record.encoded_format] += 1
                    target_identity_digest.update(bytes.fromhex(record.target_identity))
                    row_binding_digest.update(record.row_sha256)
            selected = counters["rows"] - before
            expected_part_rows = (
                int(part["sampled_rows"])
                if part is not None
                else int(source_output["rows"])
            )
            if selected != expected_part_rows:
                raise ValueError(
                    f"source part {source_output['part_index']} selected {selected} rows; "
                    f"expected {expected_part_rows}"
                )
            input_path.unlink()
            print(
                json.dumps(
                    {
                        "event": "source_part_complete",
                        "part_index": source_output["part_index"],
                        "rank": args.rank,
                        "rows": counters["rows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    writer.finish()
    if counters["rows"] != expected_rank_rows:
        raise ValueError("rank sidecar rows do not match MDS")
    rank_manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "complete": True,
        "counters": dict(sorted(counters.items())),
        "encoded_formats": dict(sorted(formats.items())),
        "format_version": FORMAT_VERSION,
        "mds_index_sha256": args.mds_index_sha256,
        "mds_manifest_sha256": args.mds_manifest_sha256,
        "outputs": writer.outputs,
        "rank": args.rank,
        "row_binding_digest": row_binding_digest.hexdigest(),
        "sampling_bitset_sha256": args.sampling_bitset_sha256,
        "sampling_plan_sha256": args.sampling_plan_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_parts": [int(item["part_index"]) for item in assigned],
        "target_identity_digest": target_identity_digest.hexdigest(),
        "world_size": args.world_size,
    }
    descriptor = put_json(client, rank_uri, rank_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "rank": args.rank,
                "rows": counters["rows"],
                "sidecars": len(writer.outputs),
                "rank_manifest": descriptor,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def finalize(args: argparse.Namespace) -> None:
    client = control_s3_client()
    contract = load_frozen_contract(args, client)
    root_shards = contract["mds_index"]["shards"]
    rank_manifests: list[Mapping[str, Any]] = []
    deadline = time.monotonic() + args.wait_timeout_seconds
    while True:
        rank_manifests = []
        missing: list[int] = []
        for rank in range(args.world_size):
            value = maybe_get_json(
                client,
                join_s3(args.output, _rank_manifest_relative(rank, args.world_size)),
            )
            if value is None:
                missing.append(rank)
            elif not _valid_rank_manifest(value, args=args, rank=rank):
                raise ValueError(f"rank {rank} manifest is incompatible")
            else:
                rank_manifests.append(value)
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
        time.sleep(args.poll_seconds)

    by_mds: dict[str, Mapping[str, Any]] = {}
    counters: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    for rank_manifest in rank_manifests:
        for key, value in rank_manifest["counters"].items():
            counters[str(key)] += int(value)
        for key, value in rank_manifest["encoded_formats"].items():
            formats[str(key)] += int(value)
        for output in rank_manifest["outputs"]:
            basename = str(output["mds_basename"])
            if basename in by_mds:
                raise ValueError(f"duplicate sidecar for MDS shard {basename}")
            by_mds[basename] = output
    ordered_outputs: list[Mapping[str, Any]] = []
    for shard in root_shards:
        basename = str(shard["raw_data"]["basename"])
        output = by_mds.get(basename)
        if (
            output is None
            or int(output["samples"]) != int(shard["samples"])
            or output["mds_sha1"] != shard["raw_data"]["hashes"]["sha1"]
        ):
            raise ValueError(f"missing or invalid sidecar for MDS shard {basename}")
        ordered_outputs.append(output)
    if counters["rows"] != args.expected_rows or len(ordered_outputs) != len(root_shards):
        raise ValueError("root sidecar coverage is incomplete")

    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "complete": True,
        "counters": dict(sorted(counters.items())),
        "encoded_formats": dict(sorted(formats.items())),
        "format_version": FORMAT_VERSION,
        "mds_shards": len(root_shards),
        "output_prefix": normalize_s3_prefix(args.output),
        "outputs": ordered_outputs,
        "record_format": {
            "endianness": "little",
            "header": "8-byte magic + uint32 sample count",
            "header_bytes": HEADER.size,
            "magic_ascii": MAGIC.decode("ascii"),
            "record": "uint32 width + uint32 height + 32-byte row SHA-256",
            "record_bytes": RECORD.size,
            "row_binding": "sha256(exact MDS row_json UTF-8 bytes)",
        },
        "rows": args.expected_rows,
        "sampling": (
            {
                "algorithm": SAMPLING_ALGORITHM,
                "bitset_sha256": args.sampling_bitset_sha256,
                "plan_sha256": args.sampling_plan_sha256,
                "seed": args.sample_seed,
            }
            if contract["sampling_plan"] is not None
            else None
        ),
        "source_dataset": {
            "manifest_sha256": args.source_manifest_sha256,
            "prefix": normalize_s3_prefix(args.source),
            "rows": args.expected_source_rows,
        },
        "source_mds": {
            "index_sha256": args.mds_index_sha256,
            "manifest_sha256": args.mds_manifest_sha256,
            "prefix": normalize_s3_prefix(args.mds),
            "rows": args.expected_rows,
            "shards": len(root_shards),
        },
        "world_size": args.world_size,
    }
    manifest_descriptor = put_json(
        client,
        join_s3(args.output, "manifest.json"),
        manifest,
    )
    success = {
        "complete": True,
        "format_version": FORMAT_VERSION,
        "manifest_sha256": manifest_descriptor["sha256"],
        "mds_index_sha256": args.mds_index_sha256,
        "mds_manifest_sha256": args.mds_manifest_sha256,
        "mds_shards": len(root_shards),
        "rows": args.expected_rows,
    }
    success_descriptor = put_json(
        client,
        join_s3(args.output, "_SUCCESS"),
        success,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": args.expected_rows,
                "mds_shards": len(root_shards),
                "manifest": manifest_descriptor,
                "success": success_descriptor,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finalize"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--mds", required=True)
    parser.add_argument("--mds-manifest-sha256", required=True)
    parser.add_argument("--mds-index-sha256", required=True)
    parser.add_argument("--sampling-plan-sha256", default="")
    parser.add_argument("--sampling-bitset-sha256", default="")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--work-dir", default="/mnt/ephemeral/pcb-v2-9c4d-17000/data")
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "source_manifest_sha256",
        "mds_manifest_sha256",
        "mds_index_sha256",
    ):
        validate_sha256(getattr(args, field), field)
    if bool(args.sampling_plan_sha256) != bool(args.sampling_bitset_sha256):
        raise ValueError("sampling plan and bitset hashes must be supplied together")
    for field in ("sampling_plan_sha256", "sampling_bitset_sha256"):
        if getattr(args, field):
            validate_sha256(getattr(args, field), field)
    if args.world_size != 64:
        raise ValueError("this sidecar must reproduce the frozen 64-rank MDS partitioning")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.expected_rows <= 0 or args.expected_source_rows < args.expected_rows:
        raise ValueError("expected row counts are invalid")
    if args.sample_seed < 0 or args.wait_timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("seed/wait/poll values are invalid")
    if args.threads < 0 or args.max_inflight < 0:
        raise ValueError("thread settings must be non-negative")
    if args.command == "worker":
        run_worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
