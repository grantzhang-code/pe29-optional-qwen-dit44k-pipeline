from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol


SHA256_METADATA_KEY = "sha256"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, inconsistent, or unsafe to use."""


@dataclass(frozen=True, slots=True)
class ObjectStat:
    size: int
    metadata: dict[str, str]
    etag: str = ""


class ObjectStore(Protocol):
    def exists(self, uri: str) -> bool: ...

    def read_bytes(self, uri: str) -> bytes: ...

    def stat(self, uri: str) -> ObjectStat: ...

    def iter_uris(self, prefix_uri: str) -> Iterable[str]: ...

    def put_bytes(
        self,
        uri: str,
        data: bytes,
        *,
        metadata: dict[str, str] | None = None,
        if_absent: bool = False,
    ) -> bool: ...

    def upload_file(
        self,
        source: Path,
        uri: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None: ...

    def download_file(self, uri: str, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class S3Location:
    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str) -> S3Location:
        if not uri.startswith("s3://"):
            raise CheckpointError(f"Expected an s3:// URI, got {uri!r}")
        bucket, separator, key = uri[5:].partition("/")
        key = key.lstrip("/")
        if not bucket or not separator or not key:
            raise CheckpointError(f"S3 URI must include a bucket and key: {uri!r}")
        return cls(bucket=bucket, key=key)

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def join(self, *parts: str) -> S3Location:
        suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
        base = self.key.rstrip("/")
        return S3Location(self.bucket, f"{base}/{suffix}" if suffix else base)


class Boto3ObjectStore:
    """Small boto3 adapter kept behind a protocol so tests never need AWS."""

    def __init__(
        self,
        *,
        profile: str | None = None,
        region: str = "us-east-1",
        max_concurrency: int | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised in the production image
            raise CheckpointError("boto3 is required for S3 checkpoint operations") from exc
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("s3", region_name=region)
        self.transfer_config = None
        if max_concurrency is not None:
            if max_concurrency < 1:
                raise CheckpointError(
                    f"S3 transfer max_concurrency must be positive, got {max_concurrency}"
                )
            from boto3.s3.transfer import TransferConfig

            # Callers already run one process per checkpoint shard. Limiting the
            # per-process transfer manager avoids multiplying boto3's default ten
            # threads by every local training rank.
            self.transfer_config = TransferConfig(max_concurrency=max_concurrency)

    @staticmethod
    def _parts(uri: str) -> tuple[str, str]:
        location = S3Location.parse(uri)
        return location.bucket, location.key

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = self._parts(uri)
        return self.client.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")["Body"].read()

    def exists(self, uri: str) -> bool:
        bucket, key = self._parts(uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            if str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def stat(self, uri: str) -> ObjectStat:
        bucket, key = self._parts(uri)
        response = self.client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        return ObjectStat(
            size=int(response.get("ContentLength") or 0),
            metadata={str(k).lower(): str(v) for k, v in (response.get("Metadata") or {}).items()},
            etag=str(response.get("ETag") or "").strip('"'),
        )

    def iter_uris(self, prefix_uri: str) -> Iterable[str]:
        location = S3Location.parse(prefix_uri)
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=location.bucket, Prefix=location.key):
            for item in page.get("Contents", []):
                yield f"s3://{location.bucket}/{item['Key']}"

    def put_bytes(
        self,
        uri: str,
        data: bytes,
        *,
        metadata: dict[str, str] | None = None,
        if_absent: bool = False,
    ) -> bool:
        bucket, key = self._parts(uri)
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": data,
            "Metadata": metadata or {},
        }
        if if_absent:
            kwargs["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**kwargs)
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            if if_absent and str(error.get("Code")) in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                return False
            raise
        return True

    def upload_file(
        self,
        source: Path,
        uri: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        bucket, key = self._parts(uri)
        kwargs: dict[str, Any] = {"ExtraArgs": {"Metadata": metadata or {}}}
        if self.transfer_config is not None:
            kwargs["Config"] = self.transfer_config
        self.client.upload_file(str(source), bucket, key, **kwargs)

    def download_file(self, uri: str, destination: Path) -> None:
        bucket, key = self._parts(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if self.transfer_config is not None:
            kwargs["Config"] = self.transfer_config
        self.client.download_file(bucket, key, str(destination), **kwargs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def parse_json_bytes(data: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected a JSON object in {source}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CheckpointError(f"Unsafe checkpoint-relative path: {raw!r}")
    return path


def local_path(root: Path, relative: str) -> Path:
    safe = safe_relative_path(relative)
    resolved_root = root.resolve()
    candidate = (root / Path(*safe.parts)).resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise CheckpointError(f"Checkpoint path escapes destination root: {relative!r}")
    return candidate


def validate_file_entries(manifest: dict[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CheckpointError(f"Manifest has no files: {source}")
    result: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise CheckpointError(f"Invalid file record in {source}: {item!r}")
        path = str(item.get("path") or "")
        safe_relative_path(path)
        if path in result:
            raise CheckpointError(f"Duplicate file record {path!r} in {source}")
        size = item.get("size")
        sha256 = item.get("sha256")
        if not isinstance(size, int) or size <= 0:
            raise CheckpointError(f"Invalid size for {path!r} in {source}: {size!r}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise CheckpointError(f"Invalid SHA-256 for {path!r} in {source}")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise CheckpointError(f"Invalid SHA-256 for {path!r} in {source}") from exc
        result[path] = item
    if manifest.get("file_count") != len(result):
        raise CheckpointError(
            f"Manifest file_count mismatch in {source}: declared={manifest.get('file_count')!r} actual={len(result)}"
        )
    total_bytes = sum(int(item["size"]) for item in result.values())
    if manifest.get("total_bytes") != total_bytes:
        raise CheckpointError(
            f"Manifest total_bytes mismatch in {source}: declared={manifest.get('total_bytes')!r} actual={total_bytes}"
        )
    return result


def verify_object_stat(
    store: ObjectStore,
    uri: str,
    *,
    size: int,
    sha256: str,
    require_sha256_metadata: bool = True,
) -> None:
    stat = store.stat(uri)
    actual_sha256 = stat.metadata.get(SHA256_METADATA_KEY)
    if stat.size != size:
        raise CheckpointError(f"Object size mismatch for {uri}: expected={size} actual={stat.size}")
    if require_sha256_metadata and actual_sha256 != sha256:
        raise CheckpointError(
            f"Object SHA-256 metadata mismatch for {uri}: expected={sha256} actual={actual_sha256!r}"
        )


def download_verified(
    store: ObjectStore,
    uri: str,
    destination: Path,
    *,
    size: int,
    sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        store.download_file(uri, temporary)
        actual_size = temporary.stat().st_size
        actual_sha256 = sha256_file(temporary)
        if actual_size != size or actual_sha256 != sha256:
            raise CheckpointError(
                f"Downloaded object failed verification for {uri}: "
                f"size={actual_size}/{size}, sha256={actual_sha256}/{sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def upload_verified(store: ObjectStore, source: Path, uri: str) -> tuple[int, str]:
    before = source.stat()
    digest = sha256_file(source)
    size = before.st_size
    store.upload_file(source, uri, metadata={SHA256_METADATA_KEY: digest})
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CheckpointError(f"Local checkpoint file changed while publishing: {source}")
    verify_object_stat(store, uri, size=size, sha256=digest)
    return size, digest
