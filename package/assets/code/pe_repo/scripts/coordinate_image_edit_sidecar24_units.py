#!/usr/bin/env python3
"""Build once or wait for the committed sidecar24 inference unit catalog.

Replica zero scans the frozen 24-dataset source catalog, publishes the
canonical unit catalog, and writes a commit marker last.  Other replicas wait
for that marker and restore exactly the same catalog.  Every downloaded object
is checked against both its S3 ``sha256`` metadata and the local canonical
validators before inference can start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts import image_edit_sidecar24_catalog as source_catalog_module  # noqa: E402
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module  # noqa: E402


COMMIT_FORMAT = "image_edit_sidecar24_unit_catalog_commit_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JSON_CONTENT_TYPE = "application/json"
COMMIT_FIELDS = frozenset(
    {
        "format_version",
        "status",
        "created_at",
        "source_catalog_uri",
        "source_catalog_sha256",
        "unit_catalog_uri",
        "unit_catalog_sha256",
        "dataset_count",
        "total_selected_rows",
        "total_units",
    }
)


class CoordinationError(ValueError):
    """A remote catalog or commit marker does not satisfy the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
        raise CoordinationError(f"invalid S3 root: {uri!r}")
    return parsed.netloc, key


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key, suffix) if value)}"


def _missing_s3_error(error: BaseException) -> bool:
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _clean_content_type(value: Any) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def read_verified_bytes(
    client: Any,
    uri: str,
    *,
    missing_ok: bool = False,
) -> bytes | None:
    bucket, key = parse_s3_uri(uri)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except BaseException as exc:
        if missing_ok and _missing_s3_error(exc):
            return None
        raise
    payload = response["Body"].read()
    actual_sha256 = sha256_bytes(payload)
    metadata = response.get("Metadata")
    if not isinstance(metadata, Mapping) or metadata.get("sha256") != actual_sha256:
        raise CoordinationError(f"S3 sha256 metadata is missing or wrong: {uri}")
    if _clean_content_type(response.get("ContentType")) != JSON_CONTENT_TYPE:
        raise CoordinationError(f"S3 ContentType is not {JSON_CONTENT_TYPE}: {uri}")
    return payload


def read_verified_json(
    client: Any,
    uri: str,
    *,
    missing_ok: bool = False,
) -> dict[str, Any] | None:
    payload = read_verified_bytes(client, uri, missing_ok=missing_ok)
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinationError(f"invalid JSON at {uri}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoordinationError(f"expected a JSON object at {uri}")
    return value


def upload_verified_bytes(client: Any, payload: bytes, uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    digest = sha256_bytes(payload)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=JSON_CONTENT_TYPE,
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head.get("ContentLength", -1)) != len(payload):
        raise IOError(f"uploaded object size verification failed: {uri}")
    if not isinstance(head.get("Metadata"), Mapping) or head["Metadata"].get("sha256") != digest:
        raise IOError(f"uploaded object digest verification failed: {uri}")
    if _clean_content_type(head.get("ContentType")) != JSON_CONTENT_TYPE:
        raise IOError(f"uploaded object ContentType verification failed: {uri}")
    return digest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_commit(
    value: Any,
    *,
    output_s3: str,
    source_catalog_sha256: str,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoordinationError("unit catalog commit must be an object")
    missing = sorted(COMMIT_FIELDS.difference(value))
    unknown = sorted(set(value).difference(COMMIT_FIELDS))
    if missing or unknown:
        raise CoordinationError(
            f"unit catalog commit fields are invalid: missing={missing} unknown={unknown}"
        )
    dataset_count = (
        source_catalog_module.expected_dataset_count(source_catalog)
        if source_catalog is not None
        else source_catalog_module.EXPECTED_DATASET_COUNT
    )
    total_selected_rows = (
        source_catalog_module.expected_total_selected_rows(source_catalog)
        if source_catalog is not None
        else unit_catalog_module.EXPECTED_TOTAL_SELECTED_ROWS
    )
    expected_scalars = {
        "format_version": COMMIT_FORMAT,
        "status": "complete",
        "source_catalog_uri": join_s3(output_s3, "_inputs", "source_catalog.json"),
        "source_catalog_sha256": source_catalog_sha256,
        "unit_catalog_uri": join_s3(output_s3, "_inputs", "unit_catalog.json"),
        "dataset_count": dataset_count,
        "total_selected_rows": total_selected_rows,
    }
    for field, expected in expected_scalars.items():
        if value[field] != expected:
            raise CoordinationError(
                f"unit catalog commit {field}={value[field]!r}, expected={expected!r}"
            )
    if not isinstance(value["created_at"], str) or not value["created_at"].strip():
        raise CoordinationError("unit catalog commit created_at must be non-empty")
    if not isinstance(value["unit_catalog_sha256"], str) or not SHA256_RE.fullmatch(
        value["unit_catalog_sha256"]
    ):
        raise CoordinationError("unit catalog commit digest is invalid")
    if type(value["total_units"]) is not int or value["total_units"] <= 0:
        raise CoordinationError("unit catalog commit total_units must be positive")
    return value


def _load_source_catalog(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise CoordinationError("expected source catalog SHA-256 must be 64 lowercase hex")
    value = source_catalog_module.load_catalog(path)
    payload = source_catalog_module.canonical_catalog_bytes(value)
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise CoordinationError(
            "source catalog SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return value, payload


def restore_committed(
    client: Any,
    commit: Mapping[str, Any],
    *,
    source_catalog: Mapping[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    source_payload = read_verified_bytes(client, str(commit["source_catalog_uri"]))
    assert source_payload is not None
    if sha256_bytes(source_payload) != commit["source_catalog_sha256"]:
        raise CoordinationError("committed source catalog object digest does not match marker")
    try:
        remote_source = json.loads(source_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinationError("committed source catalog is invalid JSON") from exc
    source_catalog_module.validate_catalog(remote_source)
    if source_catalog_module.canonical_catalog_sha256(remote_source) != commit[
        "source_catalog_sha256"
    ]:
        raise CoordinationError("committed source catalog is not the expected canonical catalog")

    unit_payload = read_verified_bytes(client, str(commit["unit_catalog_uri"]))
    assert unit_payload is not None
    if sha256_bytes(unit_payload) != commit["unit_catalog_sha256"]:
        raise CoordinationError("committed unit catalog object digest does not match marker")
    unit_path = work_dir / "unit_catalog.json"
    _atomic_write(unit_path, unit_payload)
    unit_catalog = unit_catalog_module.load_unit_catalog(
        unit_path,
        source_catalog=source_catalog,
    )
    canonical_digest = unit_catalog_module.canonical_unit_catalog_sha256(
        unit_catalog,
        source_catalog=source_catalog,
    )
    if canonical_digest != commit["unit_catalog_sha256"]:
        raise CoordinationError(
            "unit catalog canonical SHA-256 does not match committed object SHA-256"
        )
    if unit_catalog["total_units"] != commit["total_units"]:
        raise CoordinationError("unit catalog total_units does not match commit marker")
    lock = dict(commit)
    lock.update(
        {
            "local_source_catalog": str(work_dir / "source_catalog.json"),
            "local_unit_catalog": str(unit_path),
        }
    )
    _atomic_write(work_dir / "lock.json", canonical_json_bytes(lock))
    return lock


def wait_for_commit(
    client: Any,
    *,
    commit_uri: str,
    wait_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    while True:
        value = read_verified_json(client, commit_uri, missing_ok=True)
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {commit_uri}")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _s3_client(profile: str | None, region: str) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("boto3 is required for S3 unit catalog coordination") from exc
    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=128,
        ),
    )


def run(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    source_catalog, source_payload = _load_source_catalog(
        args.catalog,
        args.expected_source_catalog_sha256,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    local_source_path = args.work_dir / "source_catalog.json"
    _atomic_write(local_source_path, source_payload)
    client = client or _s3_client(args.aws_profile, args.region)
    source_uri = join_s3(args.output_s3, "_inputs", "source_catalog.json")
    unit_uri = join_s3(args.output_s3, "_inputs", "unit_catalog.json")
    commit_uri = join_s3(args.output_s3, "_inputs", "unit_catalog.success.json")

    existing = read_verified_json(client, commit_uri, missing_ok=True)
    if existing is not None:
        commit = validate_commit(
            existing,
            output_s3=args.output_s3,
            source_catalog_sha256=args.expected_source_catalog_sha256,
            source_catalog=source_catalog,
        )
        return restore_committed(
            client,
            commit,
            source_catalog=source_catalog,
            work_dir=args.work_dir,
        )

    if args.mode == "wait":
        commit = wait_for_commit(
            client,
            commit_uri=commit_uri,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
        )
        validate_commit(
            commit,
            output_s3=args.output_s3,
            source_catalog_sha256=args.expected_source_catalog_sha256,
            source_catalog=source_catalog,
        )
        return restore_committed(
            client,
            commit,
            source_catalog=source_catalog,
            work_dir=args.work_dir,
        )

    local_unit_path = args.work_dir / "unit_catalog.building.json"
    planner_args = argparse.Namespace(
        catalog=local_source_path,
        output=local_unit_path,
        work_dir=args.work_dir / "planner",
        aws_profile=args.aws_profile,
        region=args.region,
        inventory_workers=args.inventory_workers,
        scan_workers=args.scan_workers,
        max_selected_rows=unit_catalog_module.max_selected_rows_limit(source_catalog),
        rebuild_inventories=False,
        overwrite=True,
    )
    unit_catalog_module.run(planner_args)
    unit_catalog = unit_catalog_module.load_unit_catalog(
        local_unit_path,
        source_catalog=source_catalog,
    )
    unit_payload = unit_catalog_module.canonical_unit_catalog_bytes(
        unit_catalog,
        source_catalog=source_catalog,
    )
    unit_digest = sha256_bytes(unit_payload)
    upload_verified_bytes(client, source_payload, source_uri)
    uploaded_unit_digest = upload_verified_bytes(client, unit_payload, unit_uri)
    if uploaded_unit_digest != unit_digest:
        raise AssertionError("uploaded unit digest changed unexpectedly")

    commit = {
        "format_version": COMMIT_FORMAT,
        "status": "complete",
        "created_at": utc_now(),
        "source_catalog_uri": source_uri,
        "source_catalog_sha256": args.expected_source_catalog_sha256,
        "unit_catalog_uri": unit_uri,
        "unit_catalog_sha256": unit_digest,
        "dataset_count": unit_catalog["dataset_count"],
        "total_selected_rows": unit_catalog["total_selected_rows"],
        "total_units": unit_catalog["total_units"],
    }
    validate_commit(
        commit,
        output_s3=args.output_s3,
        source_catalog_sha256=args.expected_source_catalog_sha256,
        source_catalog=source_catalog,
    )
    upload_verified_bytes(client, canonical_json_bytes(commit), commit_uri)
    return restore_committed(
        client,
        commit,
        source_catalog=source_catalog,
        work_dir=args.work_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "wait"), required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected-source-catalog-sha256", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--aws-profile")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--wait-seconds", type=int, default=43_200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    parse_s3_uri(args.output_s3)
    if SHA256_RE.fullmatch(args.expected_source_catalog_sha256) is None:
        parser.error("--expected-source-catalog-sha256 must be 64 lowercase hex characters")
    for name in ("inventory_workers", "scan_workers", "wait_seconds", "poll_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lock = run(args)
    print(json.dumps(lock, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
