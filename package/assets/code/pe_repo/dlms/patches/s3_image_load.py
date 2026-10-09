"""Let ms-swift load images from direct S3 or frozen Parquet image cells.

Ports design-layout-swift/configs/config_craft_hierarchy_jsonl_s3.py's monkeypatch of
`swift.template.vision_utils.load_file`: direct ``s3://`` objects are fetched as
bytes.  Immutable ``parquet+s3://...?etag=...#row=...&column=...`` locators are
resolved with ETag-guarded range reads and column-pruned PyArrow access.  Other
paths defer to the original loader.

S3 auth uses pod IRSA — the launcher exports AWS_ROLE_ARN / AWS_WEB_IDENTITY_TOKEN_FILE
/ AWS_REGION so boto3's default chain authenticates.

Inject at train time via `--custom_register_path .../dlms/patches/s3_image_load.py`
(self-applies on import). Only needed for image-mode=s3; the local-image dataset does
not need it.
"""
from __future__ import annotations

import io
import os
import threading
from collections import namedtuple
from urllib.parse import parse_qs, unquote, urlsplit

_APPLIED = False
_NOT_FOUND_ERROR_CODES = frozenset({"404", "NoSuchKey", "NoSuchObject", "NotFound"})
try:
    _MISSING_LOG_LIMIT = max(int(os.environ.get("S3_IMAGE_MISSING_LOG_LIMIT", "100")), 0)
except ValueError:
    _MISSING_LOG_LIMIT = 100
_missing_log_lock = threading.Lock()
_missing_image_uris: set[str] = set()
_missing_log_suppressed = False
# One boto3 S3 client per process (creating a client per image is ~seconds and was the
# packing-precompute bottleneck: ~1.4 img/s even at num_proc=32). Cache it, thread-local
# so concurrent workers don't share a non-threadsafe client.
_tls = threading.local()


_ParquetImageLocator = namedtuple(
    "_ParquetImageLocator", ("bucket", "key", "etag", "row", "column", "index")
)


def _parse_parquet_s3_uri(uri: str) -> _ParquetImageLocator:
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
        fragment = parse_qs(
            parsed.fragment, keep_blank_values=True, strict_parsing=True
        )
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
    elif column in {"output_image", "source_image", "target_image"}:
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
    def __init__(
        self, client, *, bucket: str, key: str, etag: str, size: int
    ) -> None:
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
                f"short S3 range read for s3://{self.bucket}/{self.key}: "
                f"{len(payload)} != {end - start}"
            )
        self.position = end
        return payload

    def readinto(self, buffer) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


def _s3_client():
    cli = getattr(_tls, "cli", None)
    if cli is None:
        import boto3
        from botocore.config import Config
        cli = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"},
                          max_pool_connections=32),
        )
        _tls.cli = cli
    return cli


def _s3_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    error = response.get("Error")
    return str(error.get("Code", "")) if isinstance(error, dict) else ""


def _log_missing_image(uri: str, error_code: str) -> None:
    """Print each missing object once per process, with a bounded log volume."""
    global _missing_log_suppressed
    with _missing_log_lock:
        if uri in _missing_image_uris:
            return
        _missing_image_uris.add(uri)
        if len(_missing_image_uris) <= _MISSING_LOG_LIMIT:
            print(
                f"[dlms] S3 image MISSING: {uri} error_code={error_code}; "
                "non-strict dataset processing will skip this sample",
                flush=True,
            )
        elif not _missing_log_suppressed:
            print(
                f"[dlms] S3 image MISSING: further unique-URI logs suppressed after "
                f"{_MISSING_LOG_LIMIT} entries in this process",
                flush=True,
            )
            _missing_log_suppressed = True


def _read_s3_bytes(uri: str) -> bytes:
    if uri.startswith("parquet+s3://"):
        return _read_parquet_s3_bytes(uri)
    b, _, k = uri[len("s3://"):].partition("/")
    try:
        return _s3_client().get_object(Bucket=b, Key=k)["Body"].read()
    except Exception as exc:
        error_code = _s3_error_code(exc)
        if error_code in _NOT_FOUND_ERROR_CODES:
            _log_missing_image(uri, error_code)
        raise


def _read_parquet_s3_bytes(uri: str) -> bytes:
    import pyarrow.parquet as pq

    locator = _parse_parquet_s3_uri(uri)
    client = _s3_client()
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
    reader = _S3RangeReader(
        client,
        bucket=locator.bucket,
        key=locator.key,
        etag=locator.etag,
        size=size,
    )
    parquet = pq.ParquetFile(reader)
    if locator.column not in parquet.schema_arrow.names:
        raise ValueError(f"parquet+s3 object lacks column {locator.column!r}")
    cursor = 0
    row_group = -1
    row_offset = -1
    for index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(index).num_rows
        if cursor <= locator.row < cursor + count:
            row_group = index
            row_offset = locator.row - cursor
            break
        cursor += count
    if row_group < 0:
        raise ValueError(f"parquet+s3 row {locator.row} is outside row count {cursor}")
    table = parquet.read_row_group(
        row_group, columns=[locator.column], use_threads=False
    )
    cell = table[locator.column][row_offset].as_py()
    if locator.column == "input_images":
        if not isinstance(cell, list) or locator.index is None or locator.index >= len(cell):
            raise ValueError("parquet+s3 input_images cell/index is invalid")
        cell = cell[locator.index]
    if not isinstance(cell, (bytes, bytearray, memoryview)) or not cell:
        raise ValueError("parquet+s3 image cell does not contain bytes")
    return bytes(cell)


def apply() -> bool:
    global _APPLIED
    from swift.template import vision_utils
    if getattr(vision_utils.load_file, "_dlms_s3_patched", False):
        return True
    _orig = vision_utils.load_file

    def _load_file_s3(path):
        if isinstance(path, str) and path.strip().startswith(("s3://", "parquet+s3://")):
            return io.BytesIO(_read_s3_bytes(path.strip()))
        return _orig(path)

    _load_file_s3._dlms_s3_patched = True
    vision_utils.load_file = _load_file_s3
    _APPLIED = True
    print("[dlms] s3 image loader patched (swift.template.vision_utils.load_file)", flush=True)
    return True


if os.environ.get("DLMS_S3_IMAGE_LOAD_NO_APPLY") != "1":
    apply()
