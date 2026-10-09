"""Immutable S3 artifacts for the PCB v2 caption-conditioned edit evaluation."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from botocore.exceptions import ClientError


ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/"
    "evaluations/pe9be-pcbv2-gpt25-raw-eagle64-20260928/"
)
STEPS = (10000, 12000, 14000, 16000)


def parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get(client, uri: str) -> bytes:
    bucket, key = parts(uri)
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def maybe_get(client, uri: str) -> bytes | None:
    try:
        return get(client, uri)
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def put_immutable(client, uri: str, payload: bytes, content_type: str = "application/json") -> None:
    old = maybe_get(client, uri)
    if old is not None:
        if old != payload:
            raise ValueError(f"existing immutable artifact changed: {uri}")
        return
    bucket, key = parts(uri)
    try:
        client.put_object(
            Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*",
            ContentType=content_type, Metadata={"sha256": sha(payload)},
        )
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) != "PreconditionFailed":
            raise
        if get(client, uri) != payload:
            raise ValueError(f"concurrent immutable artifact changed: {uri}") from error
