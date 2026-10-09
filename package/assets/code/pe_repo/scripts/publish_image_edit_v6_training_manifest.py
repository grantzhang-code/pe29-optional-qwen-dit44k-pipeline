#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
# ]
# ///
"""Publish the final image-edit-v6 training manifest without unsafe overwrites."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTENT_TYPE = "application/json"
NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
CONDITIONAL_CONFLICT_CODES = {
    "409",
    "412",
    "ConditionalRequestConflict",
    "PreconditionFailed",
}


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(value: str) -> S3Location:
    if not value.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI: {value}")
    bucket_key = value[5:]
    bucket, separator, key = bucket_key.partition("/")
    if not bucket or not separator or not key.strip("/"):
        raise ValueError(f"incomplete S3 URI: {value}")
    return S3Location(bucket=bucket, key=key.strip("/"))


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return ""
    error = response.get("Error")
    if not isinstance(error, Mapping):
        return ""
    return str(error.get("Code") or "")


def _s3_client(aws_profile: str, region: str):
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=aws_profile, region_name=region).client(
        "s3",
        region_name=region,
        config=Config(retries={"max_attempts": 12, "mode": "adaptive"}),
    )


def _head_or_none(client: Any, location: S3Location) -> Mapping[str, Any] | None:
    try:
        return client.head_object(Bucket=location.bucket, Key=location.key)
    except Exception as exc:
        if _error_code(exc) in NOT_FOUND_CODES:
            return None
        raise


def _manifest_pins(payload: bytes, source: Path) -> tuple[str, int, dict[str, int]]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping) or document.get("complete") is not True:
        raise ValueError(f"{source}: final manifest must be a complete JSON object")

    total_rows = document.get("total_rows")
    if type(total_rows) is not int or total_rows < 1:
        raise ValueError(f"{source}: total_rows must be a positive integer")
    raw_counts = document.get("status_counts")
    if not isinstance(raw_counts, Mapping) or not raw_counts:
        raise ValueError(f"{source}: status_counts must be a non-empty object")

    status_counts: dict[str, int] = {}
    for status, count in raw_counts.items():
        if not isinstance(status, str) or not status:
            raise ValueError(f"{source}: status_counts keys must be non-empty strings")
        if type(count) is not int or count < 0:
            raise ValueError(f"{source}: status_counts values must be non-negative integers")
        status_counts[status] = count
    if sum(status_counts.values()) != total_rows:
        raise ValueError(
            f"{source}: status_counts sum differs from total_rows "
            f"({sum(status_counts.values())} != {total_rows})"
        )
    return hashlib.sha256(payload).hexdigest(), total_rows, dict(sorted(status_counts.items()))


def _verified_head(
    head: Mapping[str, Any],
    location: S3Location,
    *,
    expected_bytes: int,
    expected_sha256: str,
    conflict: bool,
) -> str:
    actual_bytes = int(head.get("ContentLength", -1))
    actual_sha256 = str((head.get("Metadata") or {}).get("sha256") or "")
    actual_content_type = str(head.get("ContentType") or "")
    version_id = str(head.get("VersionId") or "")
    differences = []
    if actual_bytes != expected_bytes:
        differences.append(f"bytes={actual_bytes}/{expected_bytes}")
    if actual_sha256 != expected_sha256:
        differences.append(f"sha256={actual_sha256}/{expected_sha256}")
    if actual_content_type != CONTENT_TYPE:
        differences.append(f"content_type={actual_content_type!r}/{CONTENT_TYPE!r}")
    if not version_id or version_id == "null":
        differences.append("VersionId=missing")
    if differences:
        prefix = (
            "refusing to overwrite a mismatched S3 object"
            if conflict
            else "published S3 object failed identity verification"
        )
        raise ValueError(f"{prefix}: {location.uri}; {' '.join(differences)}")
    return version_id


def publish_training_manifest(
    manifest_path: Path,
    destination_uri: str,
    *,
    aws_profile: str,
    region: str,
    client: Any | None = None,
) -> dict[str, Any]:
    source = manifest_path.expanduser().resolve(strict=True)
    payload = source.read_bytes()
    manifest_sha256, total_rows, status_counts = _manifest_pins(payload, source)
    destination = parse_s3_uri(destination_uri)
    s3 = client if client is not None else _s3_client(aws_profile, region)

    existing = _head_or_none(s3, destination)
    if existing is not None:
        version_id = _verified_head(
            existing,
            destination,
            expected_bytes=len(payload),
            expected_sha256=manifest_sha256,
            conflict=True,
        )
    else:
        try:
            put_result = s3.put_object(
                Bucket=destination.bucket,
                Key=destination.key,
                Body=payload,
                ContentType=CONTENT_TYPE,
                Metadata={"sha256": manifest_sha256},
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _error_code(exc) not in CONDITIONAL_CONFLICT_CODES:
                raise
            raced = _head_or_none(s3, destination)
            if raced is None:
                raise RuntimeError(
                    f"conditional S3 publish conflicted but object is missing: {destination.uri}"
                ) from exc
            version_id = _verified_head(
                raced,
                destination,
                expected_bytes=len(payload),
                expected_sha256=manifest_sha256,
                conflict=True,
            )
        else:
            put_version_id = str((put_result or {}).get("VersionId") or "")
            head_args: dict[str, Any] = {
                "Bucket": destination.bucket,
                "Key": destination.key,
            }
            if put_version_id and put_version_id != "null":
                head_args["VersionId"] = put_version_id
            published = s3.head_object(**head_args)
            version_id = _verified_head(
                published,
                destination,
                expected_bytes=len(payload),
                expected_sha256=manifest_sha256,
                conflict=False,
            )
            if put_version_id and version_id != put_version_id:
                raise ValueError(
                    f"published S3 VersionId changed: {destination.uri}; "
                    f"head={version_id!r} put={put_version_id!r}"
                )

    return {
        "manifest_sha256": manifest_sha256,
        "version_id": version_id,
        "required_total_rows": total_rows,
        "status_counts": status_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination-uri", "--s3-uri", dest="destination_uri", required=True)
    parser.add_argument("--aws-profile", "--profile", dest="aws_profile", required=True)
    parser.add_argument("--region", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pins = publish_training_manifest(
        args.manifest,
        args.destination_uri,
        aws_profile=args.aws_profile,
        region=args.region,
    )
    print(json.dumps(pins, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
