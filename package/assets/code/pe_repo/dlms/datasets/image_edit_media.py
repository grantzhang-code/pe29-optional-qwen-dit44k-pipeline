"""Resolve image-edit media for offline PE dataset packaging.

The image-edit-v6 manifests contain both ordinary ``s3://`` objects and
immutable image cells embedded in Parquet objects, represented as
``parquet+s3://...?etag=...#row=...&column=...`` locators. The offline MDS
packager resolves both forms once, before H200 training, so the hot DataLoader
path never performs one S3 request per sample.
"""

from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict, namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import Image, ImageOps


_tls = threading.local()
_DESIGN_GENERATION_BUCKET = "design-generation-core-oss.canva.com"
_DESIGN_GENERATION_GETTY_PREFIX = "datasets/getty-images/"
_DESIGN_GENERATION_ACCOUNT_ID = "051687089423"
_ParquetImageLocator = namedtuple("_ParquetImageLocator", ("bucket", "key", "etag", "row", "column", "index"))
_ParquetObjectInfo = namedtuple(
    "_ParquetObjectInfo",
    ("size", "columns", "row_group_starts", "row_group_counts", "total_rows"),
)


class _LoadFlight:
    """One process-local immutable-object load shared by concurrent readers."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None


_parquet_cache_lock = threading.Lock()
_parquet_object_info_cache: dict[tuple[str, str, str], _ParquetObjectInfo] = {}
_parquet_object_info_flights: dict[tuple[str, str, str], _LoadFlight] = {}
_parquet_row_group_cache: OrderedDict[tuple[str, str, str, int, str], tuple[Any, int]] = OrderedDict()
_parquet_row_group_flights: dict[tuple[str, str, str, int, str], _LoadFlight] = {}
_parquet_row_group_cache_bytes = 0
_parquet_direct_path_cache: dict[tuple[str, str, str, int, str], Any] = {}
_parquet_direct_path_flights: dict[tuple[str, str, str, int, str], _LoadFlight] = {}
_parquet_cache_counters = {
    "direct_path_fallbacks": 0,
    "direct_path_hits": 0,
    "direct_path_misses": 0,
    "direct_path_waits": 0,
    "object_info_hits": 0,
    "object_info_misses": 0,
    "object_info_waits": 0,
    "row_group_bypasses": 0,
    "row_group_evictions": 0,
    "row_group_hits": 0,
    "row_group_misses": 0,
    "row_group_waits": 0,
}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _s3_client_config(*, max_pool_connections: int | None = None):
    from botocore.config import Config

    if max_pool_connections is None:
        max_pool_connections = _positive_int_env("T2J_S3_MAX_POOL_CONNECTIONS", 32)
    elif max_pool_connections < 1:
        raise ValueError("max_pool_connections must be positive")

    return Config(
        # Media payloads are small enough that a silent 60-second socket wait
        # is a pathological tail, not useful work. Fail the individual attempt
        # sooner while retaining adaptive retries so a transient S3 issue does
        # not fail a tightly coupled 256-GPU training job.
        connect_timeout=_positive_int_env("T2J_S3_CONNECT_TIMEOUT_SECONDS", 5),
        read_timeout=_positive_int_env("T2J_S3_READ_TIMEOUT_SECONDS", 30),
        tcp_keepalive=_bool_env("T2J_S3_TCP_KEEPALIVE", True),
        retries={
            "max_attempts": _positive_int_env("T2J_S3_MAX_ATTEMPTS", 5),
            "mode": "adaptive",
        },
        max_pool_connections=max(max_pool_connections, 2),
    )


def _access_grant_target(uri: str) -> tuple[str, str] | None:
    """Return the fixed Access Grants scope required by a known media URI."""

    try:
        bucket, key = _parse_s3_uri(uri)
    except ValueError:
        return None
    if bucket == _DESIGN_GENERATION_BUCKET and key.startswith(_DESIGN_GENERATION_GETTY_PREFIX):
        return (
            _DESIGN_GENERATION_ACCOUNT_ID,
            f"s3://{_DESIGN_GENERATION_BUCKET}/{_DESIGN_GENERATION_GETTY_PREFIX}",
        )
    return None


def _refreshable_access_grant_s3_client(*, account_id: str, target: str):
    """Create an S3 client whose scoped Access Grants credentials auto-renew."""

    import boto3
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

    def fetch() -> dict[str, str]:
        response = (
            boto3.Session()
            .client("s3control", region_name=region)
            .get_data_access(
                AccountId=account_id,
                Target=target,
                Permission="READ",
                Privilege="Default",
            )
        )
        credentials = response["Credentials"]
        expiration = credentials.get("Expiration")
        if isinstance(expiration, datetime):
            expiry_time = expiration.astimezone(timezone.utc).isoformat()
        else:
            expiry_time = (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat()
        return {
            "access_key": credentials["AccessKeyId"],
            "secret_key": credentials["SecretAccessKey"],
            "token": credentials["SessionToken"],
            "expiry_time": expiry_time,
        }

    botocore_session = get_session()
    botocore_session._credentials = RefreshableCredentials.create_from_metadata(  # noqa: SLF001
        metadata=fetch(),
        refresh_using=fetch,
        method="s3-access-grants",
    )
    botocore_session.set_config_variable("region", region)
    return boto3.Session(botocore_session=botocore_session).client(
        "s3",
        region_name=region,
        config=_s3_client_config(),
    )


def _s3_client(uri: str):
    """Return a per-thread S3 client, routing known cross-account media."""

    bucket, _key = _parse_s3_uri(uri)
    grant = _access_grant_target(uri)
    identity = (bucket, grant)
    clients = getattr(_tls, "s3_clients", None)
    if clients is None:
        clients = {}
        _tls.s3_clients = clients
    if identity not in clients:
        import boto3

        if grant is None:
            clients[identity] = boto3.client(
                "s3",
                region_name=(
                    os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
                ),
                config=_s3_client_config(),
            )
        else:
            account_id, target = grant
            clients[identity] = _refreshable_access_grant_s3_client(
                account_id=account_id,
                target=target,
            )
    return clients[identity]


def preflight_known_s3_access_grants() -> list[str]:
    """Acquire every known cross-account grant once without exposing credentials."""

    target = f"s3://{_DESIGN_GENERATION_BUCKET}/{_DESIGN_GENERATION_GETTY_PREFIX}"
    client = _refreshable_access_grant_s3_client(
        account_id=_DESIGN_GENERATION_ACCOUNT_ID,
        target=target,
    )
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return [target]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
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
        raise ValueError(f"invalid S3 image URI: {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def _parse_parquet_s3_uri(uri: str) -> _ParquetImageLocator:
    """Parse the frozen image-cell locator used by image-edit-v6."""

    if not isinstance(uri, str) or uri != uri.strip():
        raise ValueError(f"invalid parquet+s3 image URI: {uri!r}")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "parquet+s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.username is not None
        or parsed.port is not None
    ):
        raise ValueError(f"invalid parquet+s3 image URI: {uri!r}")
    try:
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        fragment = parse_qs(parsed.fragment, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ValueError(f"invalid parquet+s3 image URI: {uri!r}") from exc

    if set(query) != {"etag"} or len(query["etag"]) != 1 or not query["etag"][0]:
        raise ValueError("parquet+s3 image URI requires exactly one non-empty etag")
    if not set(fragment).issubset({"row", "column", "index"}) or not {
        "row",
        "column",
    }.issubset(fragment):
        raise ValueError("parquet+s3 image URI has invalid fragment fields")
    if any(len(values) != 1 for values in fragment.values()):
        raise ValueError("parquet+s3 image URI fragment fields must be unique")

    try:
        row = int(fragment["row"][0])
    except ValueError as exc:
        raise ValueError("parquet+s3 image row must be an integer") from exc
    if row < 0 or str(row) != fragment["row"][0]:
        raise ValueError("parquet+s3 image row must be canonical and non-negative")

    column = fragment["column"][0]
    raw_index = fragment.get("index")
    index: int | None = None
    if raw_index is not None:
        try:
            index = int(raw_index[0])
        except ValueError as exc:
            raise ValueError("parquet+s3 image index must be an integer") from exc
        if index < 0 or str(index) != raw_index[0]:
            raise ValueError("parquet+s3 image index must be canonical and non-negative")

    if column == "input_images":
        if index != 0:
            raise ValueError("input_images parquet+s3 URI must use index=0")
    elif column in {"image", "output_image", "source_image", "target_image"}:
        if index is not None:
            raise ValueError(f"{column} parquet+s3 URI must not use index")
    else:
        raise ValueError(f"unsupported parquet+s3 image column: {column!r}")

    key = unquote(parsed.path.lstrip("/"))
    if not key.endswith(".parquet"):
        raise ValueError("parquet+s3 image object must end with .parquet")
    return _ParquetImageLocator(
        bucket=parsed.netloc,
        key=key,
        etag=query["etag"][0],
        row=row,
        column=column,
        index=index,
    )


class _S3RangeReader(io.RawIOBase):
    """Seekable file facade backed by ETag-guarded S3 range requests."""

    def __init__(self, client, *, bucket: str, key: str, etag: str, size: int) -> None:
        super().__init__()
        self.client = client
        self.bucket = bucket
        self.key = key
        self.etag = etag
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative S3 range-reader position")
        self.position = min(position, self.size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size if size is None or size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        start = self.position
        payload = self.client.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={start}-{end - 1}",
            IfMatch=self.etag,
        )["Body"].read()
        if len(payload) != end - start:
            raise IOError(
                f"short S3 range read for s3://{self.bucket}/{self.key}: {len(payload)} != {end - start}"
            )
        self.position = end
        return payload

    def readinto(self, buffer) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


def _load_parquet_object_info(locator: _ParquetImageLocator, client) -> _ParquetObjectInfo:
    import pyarrow.parquet as pq

    head = client.head_object(Bucket=locator.bucket, Key=locator.key)
    actual_etag = str(head.get("ETag") or "").strip('"')
    if actual_etag != locator.etag:
        raise ValueError(
            f"parquet+s3 ETag changed for s3://{locator.bucket}/{locator.key}: "
            f"{actual_etag!r} != {locator.etag!r}"
        )
    size = int(head.get("ContentLength") or 0)
    if size < 1:
        raise ValueError(f"empty parquet+s3 object: s3://{locator.bucket}/{locator.key}")

    parquet = pq.ParquetFile(
        _S3RangeReader(
            client,
            bucket=locator.bucket,
            key=locator.key,
            etag=locator.etag,
            size=size,
        )
    )
    starts: list[int] = []
    counts: list[int] = []
    cursor = 0
    for index in range(parquet.metadata.num_row_groups):
        starts.append(cursor)
        count = int(parquet.metadata.row_group(index).num_rows)
        counts.append(count)
        cursor += count
    return _ParquetObjectInfo(
        size=size,
        columns=tuple(parquet.schema_arrow.names),
        row_group_starts=tuple(starts),
        row_group_counts=tuple(counts),
        total_rows=cursor,
    )


def _get_parquet_object_info(locator: _ParquetImageLocator, client) -> _ParquetObjectInfo:
    key = (locator.bucket, locator.key, locator.etag)
    with _parquet_cache_lock:
        cached = _parquet_object_info_cache.get(key)
        if cached is not None:
            _parquet_cache_counters["object_info_hits"] += 1
            return cached
        flight = _parquet_object_info_flights.get(key)
        if flight is None:
            flight = _LoadFlight()
            _parquet_object_info_flights[key] = flight
            _parquet_cache_counters["object_info_misses"] += 1
            leader = True
        else:
            _parquet_cache_counters["object_info_waits"] += 1
            leader = False

    if not leader:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return flight.value

    try:
        value = _load_parquet_object_info(locator, client)
    except BaseException as exc:
        with _parquet_cache_lock:
            flight.error = exc
            _parquet_object_info_flights.pop(key, None)
            flight.event.set()
        raise
    with _parquet_cache_lock:
        _parquet_object_info_cache[key] = value
        flight.value = value
        _parquet_object_info_flights.pop(key, None)
        flight.event.set()
    return value


def _load_parquet_row_group(
    locator: _ParquetImageLocator,
    info: _ParquetObjectInfo,
    row_group: int,
    client,
):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(
        _S3RangeReader(
            client,
            bucket=locator.bucket,
            key=locator.key,
            etag=locator.etag,
            size=info.size,
        )
    )
    return parquet.read_row_group(row_group, columns=[locator.column], use_threads=False)


def _get_parquet_row_group(
    locator: _ParquetImageLocator,
    info: _ParquetObjectInfo,
    row_group: int,
    client,
):
    global _parquet_row_group_cache_bytes

    limit = _nonnegative_int_env("T2J_PARQUET_ROW_GROUP_CACHE_BYTES", 0)
    if limit == 0:
        with _parquet_cache_lock:
            _parquet_cache_counters["row_group_bypasses"] += 1
        return _load_parquet_row_group(locator, info, row_group, client)

    key = (locator.bucket, locator.key, locator.etag, row_group, locator.column)
    with _parquet_cache_lock:
        cached = _parquet_row_group_cache.pop(key, None)
        if cached is not None:
            _parquet_row_group_cache[key] = cached
            _parquet_cache_counters["row_group_hits"] += 1
            return cached[0]
        flight = _parquet_row_group_flights.get(key)
        if flight is None:
            flight = _LoadFlight()
            _parquet_row_group_flights[key] = flight
            _parquet_cache_counters["row_group_misses"] += 1
            leader = True
        else:
            _parquet_cache_counters["row_group_waits"] += 1
            leader = False

    if not leader:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return flight.value

    try:
        value = _load_parquet_row_group(locator, info, row_group, client)
    except BaseException as exc:
        with _parquet_cache_lock:
            flight.error = exc
            _parquet_row_group_flights.pop(key, None)
            flight.event.set()
        raise

    value_bytes = max(int(getattr(value, "nbytes", 0) or 0), 1)
    with _parquet_cache_lock:
        flight.value = value
        if value_bytes <= limit:
            while _parquet_row_group_cache and _parquet_row_group_cache_bytes + value_bytes > limit:
                _old_key, (_old_value, old_bytes) = _parquet_row_group_cache.popitem(last=False)
                _parquet_row_group_cache_bytes -= old_bytes
                _parquet_cache_counters["row_group_evictions"] += 1
            _parquet_row_group_cache[key] = (value, value_bytes)
            _parquet_row_group_cache_bytes += value_bytes
        _parquet_row_group_flights.pop(key, None)
        flight.event.set()
    return value


def _get_parquet_direct_path_table(
    locator: _ParquetImageLocator,
    info: _ParquetObjectInfo,
    row_group: int,
    path_column: str,
    client,
):
    """Cache the tiny companion-path column independently of image payloads."""

    key = (locator.bucket, locator.key, locator.etag, row_group, path_column)
    with _parquet_cache_lock:
        cached = _parquet_direct_path_cache.get(key)
        if cached is not None:
            _parquet_cache_counters["direct_path_hits"] += 1
            return cached
        flight = _parquet_direct_path_flights.get(key)
        if flight is None:
            flight = _LoadFlight()
            _parquet_direct_path_flights[key] = flight
            _parquet_cache_counters["direct_path_misses"] += 1
            leader = True
        else:
            _parquet_cache_counters["direct_path_waits"] += 1
            leader = False

    if not leader:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return flight.value

    path_locator = locator._replace(column=path_column, index=None)
    try:
        value = _load_parquet_row_group(path_locator, info, row_group, client)
    except BaseException as exc:
        with _parquet_cache_lock:
            flight.error = exc
            _parquet_direct_path_flights.pop(key, None)
            flight.event.set()
        raise

    with _parquet_cache_lock:
        _parquet_direct_path_cache[key] = value
        flight.value = value
        _parquet_direct_path_flights.pop(key, None)
        flight.event.set()
    return value


def resolve_parquet_direct_media_uri(uri: str, *, client=None) -> str | None:
    """Resolve an embedded Parquet cell to its companion direct S3 object.

    Canonical PE Parquet stores both embedded image bytes and the source object
    path. Reading a single embedded value otherwise downloads the complete
    image column chunk (about 117 MiB for a representative 500-row shard).
    Companion path columns are only a few KiB and are cached per worker.

    Some legacy schemas use non-S3 ``direct://`` locators or have no companion
    path column. Those rows deliberately return ``None`` so callers can retain
    the immutable embedded-cell fallback.
    """

    locator = _parse_parquet_s3_uri(uri)
    companion_columns = {
        "input_images": ("input_s3_paths", True),
        "output_image": ("output_s3_path", False),
        "source_image": ("source_image_path", False),
        "target_image": ("target_image_path", False),
    }
    path_column, path_is_list = companion_columns[locator.column]
    client = client or _s3_client(f"s3://{locator.bucket}/{locator.key}")
    info = _get_parquet_object_info(locator, client)
    if path_column not in info.columns:
        with _parquet_cache_lock:
            _parquet_cache_counters["direct_path_fallbacks"] += 1
        return None

    import bisect

    row_group = bisect.bisect_right(info.row_group_starts, locator.row) - 1
    if row_group < 0 or locator.row >= info.row_group_starts[row_group] + info.row_group_counts[row_group]:
        raise ValueError(f"parquet+s3 row {locator.row} is outside row count {info.total_rows}")
    row_offset = locator.row - info.row_group_starts[row_group]
    table = _get_parquet_direct_path_table(locator, info, row_group, path_column, client)
    value = table[path_column][row_offset].as_py()
    if path_is_list:
        if not isinstance(value, list) or locator.index is None or locator.index >= len(value):
            # A nullable or shorter companion-path list means the direct
            # source is unavailable for this row.  The embedded image cell is
            # still authoritative and must remain a valid fallback.
            with _parquet_cache_lock:
                _parquet_cache_counters["direct_path_fallbacks"] += 1
            return None
        value = value[locator.index]
    if not isinstance(value, str) or not value.startswith("s3://"):
        with _parquet_cache_lock:
            _parquet_cache_counters["direct_path_fallbacks"] += 1
        return None
    try:
        bucket, _key = _parse_s3_uri(value)
    except ValueError:
        # Several legacy companion columns contain pseudo-S3 identifiers with
        # fragments (for example ``...parquet#getty_id``). They are not
        # fetchable S3 objects, so retain the immutable embedded-cell fallback.
        with _parquet_cache_lock:
            _parquet_cache_counters["direct_path_fallbacks"] += 1
        return None
    # The model-trainer role reads Core CN buckets directly. The
    # design-generation bucket is cross-account and only its registered Getty
    # prefix has an Access Grant in this runtime; other prefixes must retain
    # the embedded Parquet-cell fallback instead of failing with AccessDenied.
    if bucket == _DESIGN_GENERATION_BUCKET and _access_grant_target(value) is None:
        with _parquet_cache_lock:
            _parquet_cache_counters["direct_path_fallbacks"] += 1
        return None
    return value


def parquet_media_cache_stats() -> dict[str, int]:
    """Return process-local immutable-Parquet cache counters for audit reports."""

    with _parquet_cache_lock:
        return {
            **_parquet_cache_counters,
            "configured_limit_bytes": _nonnegative_int_env("T2J_PARQUET_ROW_GROUP_CACHE_BYTES", 0),
            "object_info_entries": len(_parquet_object_info_cache),
            "row_group_entries": len(_parquet_row_group_cache),
            "row_group_resident_bytes": _parquet_row_group_cache_bytes,
        }


def _reset_parquet_media_caches_for_tests() -> None:
    global _parquet_row_group_cache_bytes

    with _parquet_cache_lock:
        if _parquet_object_info_flights or _parquet_row_group_flights or _parquet_direct_path_flights:
            raise RuntimeError("cannot reset Parquet media caches while loads are active")
        _parquet_object_info_cache.clear()
        _parquet_row_group_cache.clear()
        _parquet_direct_path_cache.clear()
        _parquet_row_group_cache_bytes = 0
        for key in _parquet_cache_counters:
            _parquet_cache_counters[key] = 0


def _read_parquet_s3_bytes(uri: str, *, client=None) -> bytes:
    import bisect

    locator = _parse_parquet_s3_uri(uri)
    client = client or _s3_client(f"s3://{locator.bucket}/{locator.key}")
    info = _get_parquet_object_info(locator, client)
    if locator.column not in info.columns:
        raise ValueError(f"parquet+s3 object lacks column {locator.column!r}")

    row_group = bisect.bisect_right(info.row_group_starts, locator.row) - 1
    if row_group < 0 or locator.row >= info.row_group_starts[row_group] + info.row_group_counts[row_group]:
        raise ValueError(f"parquet+s3 row {locator.row} is outside row count {info.total_rows}")
    row_offset = locator.row - info.row_group_starts[row_group]

    table = _get_parquet_row_group(locator, info, row_group, client)
    cell: Any = table[locator.column][row_offset].as_py()
    if locator.column == "input_images":
        if not isinstance(cell, list) or locator.index is None or locator.index >= len(cell):
            raise ValueError("parquet+s3 input_images cell/index is invalid")
        cell = cell[locator.index]
    if isinstance(cell, dict):
        cell = cell.get("bytes")
    if not isinstance(cell, (bytes, bytearray, memoryview)) or not cell:
        raise ValueError("parquet+s3 image cell does not contain bytes")
    return bytes(cell)


def read_media_bytes(uri: str, *, client=None) -> bytes:
    """Read one local, direct-S3, or frozen-Parquet image payload."""

    if not isinstance(uri, str) or not uri.strip() or uri != uri.strip():
        raise ValueError(f"invalid image URI/path: {uri!r}")
    if uri.startswith("parquet+s3://"):
        return _read_parquet_s3_bytes(uri, client=client)
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        response = (client or _s3_client(uri)).get_object(Bucket=bucket, Key=key)
        payload = response["Body"].read()
        if not payload:
            raise ValueError(f"empty S3 image object: {uri}")
        return payload
    payload = Path(uri).read_bytes()
    if not payload:
        raise ValueError(f"empty local image: {uri}")
    return payload


def decode_rgb_image_bytes(payload: bytes, *, context: str = "embedded image") -> Image.Image:
    """Fully decode immutable encoded bytes into display-oriented RGB pixels."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{context} must contain non-empty bytes")
    with Image.open(io.BytesIO(payload)) as opened:
        try:
            image = ImageOps.exif_transpose(opened)
        except (AttributeError, TypeError, ValueError, SyntaxError):
            # Match decode_image_size and the audited conversion policy: an
            # otherwise decodable image with malformed EXIF is orientation=1.
            # The conversion manifest records how often this fallback occurs.
            opened.load()
            image = opened
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"decoded image has invalid dimensions: {context}")
        return image.convert("RGB").copy()


def decode_rgb_image(uri: str, *, client=None) -> Image.Image:
    """Fully decode an image and return the display-oriented RGB pixels."""

    return decode_rgb_image_bytes(
        read_media_bytes(uri, client=client),
        context=uri,
    )


def decode_image_size(uri: str, *, client=None) -> tuple[int, int]:
    """Validate a target image and return display-oriented ``(width, height)``.

    This deliberately does not accept external width/height metadata and has no
    reference-image fallback: the value placed in the prompt comes from the
    target file named by the row itself.  ``Image.verify`` checks the encoded
    payload without allocating a full decoded pixel buffer.  That distinction is
    important here because every training sample needs only the target canvas
    size, while decoding target pixels in every DataLoader worker was a large,
    unnecessary host-RAM spike.
    """

    payload = read_media_bytes(uri, client=client)
    with Image.open(io.BytesIO(payload)) as opened:
        width, height = opened.size
        # Read the raw EXIF block from already-parsed container metadata.
        # Calling ``getexif`` on PNG can force ``load()`` and defeat the
        # no-full-pixel-decode guarantee.
        exif_payload = opened.info.get("exif")
        opened.verify()
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
        raise ValueError(f"decoded target image has invalid dimensions: {uri}")
    return int(width), int(height)


__all__ = [
    "decode_image_size",
    "decode_rgb_image",
    "decode_rgb_image_bytes",
    "preflight_known_s3_access_grants",
    "read_media_bytes",
    "resolve_parquet_direct_media_uri",
]
