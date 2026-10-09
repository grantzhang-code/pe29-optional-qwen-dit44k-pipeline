#!/usr/bin/env python3
"""Wait for and lock a fully verified shared trained-MTP HF artifact.

The artifact publisher commits ``_SUCCESS`` last.  This command waits only for
that immutable marker, then delegates to the canonical MCore-to-HF artifact
reader to validate the marker, manifest, namespace, and every remote file
record.  It additionally binds the artifact to caller-supplied checkpoint and
initial-model lineage before writing a deterministic lock JSON document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    write_json_atomic,
)
from scripts.mtp_eval.hf_artifact import (
    EXPECTED_MTP_TENSOR_COUNT,
    EXPECTED_NON_MTP_TENSOR_COUNT,
    EXPECTED_TENSOR_COUNT,
    RemoteArtifact,
    SUCCESS_NAME,
    _read_remote,
)


DEFAULT_WAIT_SECONDS = 60 * 60
DEFAULT_POLL_SECONDS = 30.0
LOCK_VERSION = 1
LOCK_KIND = "mcore-hf-artifact-lock"
EXPECTED_VARIANT = "trained_mtp"


def _normalise_root(uri: str, *, label: str) -> str:
    try:
        location = S3Location.parse(uri)
    except (AttributeError, CheckpointError) as exc:
        raise CheckpointError(f"Invalid {label}: {uri!r}") from exc
    return S3Location(location.bucket, location.key.rstrip("/")).uri + "/"


def _success_uri(artifact_s3: str) -> str:
    location = S3Location.parse(artifact_s3)
    return S3Location(location.bucket, location.key.rstrip("/")).join(SUCCESS_NAME).uri


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointError(f"Invalid {label}: {value!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CheckpointError(f"Invalid {label}: {value!r}") from exc
    return value


def _require_equal(*, label: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise CheckpointError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def _lineage_lock(
    artifact: RemoteArtifact,
    *,
    artifact_s3: str,
    expected_source_checkpoint_s3: str,
    expected_source_step: int,
    expected_source_checkpoint_manifest_sha256: str,
    expected_initial_artifact_s3: str,
    expected_initial_artifact_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate caller pins against an already fully verified remote artifact."""

    manifest = artifact.manifest
    _require_equal(label="artifact root URI", expected=artifact_s3, actual=artifact.root_uri)
    _require_equal(label="artifact variant", expected=EXPECTED_VARIANT, actual=manifest.get("variant"))

    source = manifest.get("source_mcore_checkpoint")
    if not isinstance(source, Mapping):
        raise CheckpointError("Artifact has no source_mcore_checkpoint lineage")
    actual_source_uri = _normalise_root(str(source.get("uri") or ""), label="source checkpoint URI")
    _require_equal(
        label="source checkpoint URI",
        expected=expected_source_checkpoint_s3,
        actual=actual_source_uri,
    )
    _require_equal(label="source checkpoint step", expected=expected_source_step, actual=source.get("step"))
    _require_equal(
        label="source checkpoint manifest SHA-256",
        expected=expected_source_checkpoint_manifest_sha256,
        actual=source.get("manifest_sha256"),
    )

    initial = manifest.get("initial_mtp_artifact")
    if not isinstance(initial, Mapping):
        raise CheckpointError("Artifact has no initial_mtp_artifact lineage")
    actual_initial_uri = _normalise_root(str(initial.get("uri") or ""), label="initial artifact URI")
    _require_equal(
        label="initial artifact URI",
        expected=expected_initial_artifact_s3,
        actual=actual_initial_uri,
    )
    _require_equal(
        label="initial artifact manifest SHA-256",
        expected=expected_initial_artifact_manifest_sha256,
        actual=initial.get("manifest_sha256"),
    )

    coverage = manifest.get("tensor_coverage")
    if not isinstance(coverage, Mapping):
        raise CheckpointError("Artifact has no tensor_coverage")
    expected_coverage = {
        "expected": EXPECTED_TENSOR_COUNT,
        "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
        "mtp": EXPECTED_MTP_TENSOR_COUNT,
        "final": EXPECTED_TENSOR_COUNT,
    }
    for field, expected in expected_coverage.items():
        _require_equal(
            label=f"tensor coverage {field}",
            expected=expected,
            actual=coverage.get(field),
        )

    artifact_manifest_sha256 = _require_sha256(
        artifact.manifest_sha256,
        label="artifact manifest SHA-256",
    )
    marker_manifest_uri = artifact.marker.get("manifest_uri")
    attempt_id = artifact.marker.get("attempt_id")
    if not isinstance(marker_manifest_uri, str) or not marker_manifest_uri:
        raise CheckpointError("Artifact _SUCCESS has no manifest_uri")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise CheckpointError("Artifact _SUCCESS has no attempt_id")

    return {
        "version": LOCK_VERSION,
        "kind": LOCK_KIND,
        "status": "complete",
        "artifact_s3": artifact.root_uri,
        "success_uri": _success_uri(artifact.root_uri),
        "attempt_id": attempt_id,
        "manifest_uri": marker_manifest_uri,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "variant": EXPECTED_VARIANT,
        "source_mcore_checkpoint": {
            "uri": actual_source_uri,
            "step": source["step"],
            "manifest_sha256": source["manifest_sha256"],
        },
        "initial_mtp_artifact": {
            "uri": actual_initial_uri,
            "manifest_sha256": initial["manifest_sha256"],
        },
        "tensor_coverage": expected_coverage,
        "file_count": len(artifact.entries),
        "total_bytes": sum(int(record["size"]) for record in artifact.entries.values()),
        "verified": True,
    }


def wait_for_artifact(
    store: ObjectStore,
    artifact_s3: str,
    *,
    expected_source_checkpoint_s3: str,
    expected_source_step: int,
    expected_source_checkpoint_manifest_sha256: str,
    expected_initial_artifact_s3: str,
    expected_initial_artifact_manifest_sha256: str,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait for ``_SUCCESS``, fully verify the artifact, and return its lock."""

    if wait_seconds < 0 or poll_seconds <= 0:
        raise CheckpointError("wait_seconds must be non-negative and poll_seconds must be positive")
    if isinstance(expected_source_step, bool) or expected_source_step <= 0:
        raise CheckpointError(f"expected source step must be positive, got {expected_source_step!r}")

    artifact_root = _normalise_root(artifact_s3, label="artifact URI")
    source_root = _normalise_root(
        expected_source_checkpoint_s3,
        label="expected source checkpoint URI",
    )
    initial_root = _normalise_root(
        expected_initial_artifact_s3,
        label="expected initial artifact URI",
    )
    source_digest = _require_sha256(
        expected_source_checkpoint_manifest_sha256,
        label="expected source checkpoint manifest SHA-256",
    )
    initial_digest = _require_sha256(
        expected_initial_artifact_manifest_sha256,
        label="expected initial artifact manifest SHA-256",
    )

    success_uri = _success_uri(artifact_root)
    deadline = monotonic() + wait_seconds
    while not store.exists(success_uri):
        now = monotonic()
        if now >= deadline:
            raise CheckpointError(f"Timed out waiting for shared HF artifact _SUCCESS: {success_uri}")
        sleep(min(poll_seconds, max(0.0, deadline - now)))

    # The commit marker is immutable and published last.  Once it exists, any
    # validation or lineage failure is terminal rather than a condition to poll.
    artifact = _read_remote(store, artifact_root, verify_objects=True)
    return _lineage_lock(
        artifact,
        artifact_s3=artifact_root,
        expected_source_checkpoint_s3=source_root,
        expected_source_step=expected_source_step,
        expected_source_checkpoint_manifest_sha256=source_digest,
        expected_initial_artifact_s3=initial_root,
        expected_initial_artifact_manifest_sha256=initial_digest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-s3", required=True)
    parser.add_argument(
        "--expected-source-checkpoint-s3",
        "--expected-source-checkpoint-uri",
        dest="expected_source_checkpoint_s3",
        required=True,
    )
    parser.add_argument(
        "--expected-source-step",
        "--expected-source-checkpoint-step",
        dest="expected_source_step",
        type=int,
        required=True,
    )
    parser.add_argument("--expected-source-checkpoint-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-initial-artifact-s3",
        "--expected-initial-artifact-uri",
        "--expected-initial-mtp-artifact-s3",
        dest="expected_initial_artifact_s3",
        required=True,
    )
    parser.add_argument(
        "--expected-initial-artifact-manifest-sha256",
        "--expected-initial-mtp-artifact-manifest-sha256",
        dest="expected_initial_artifact_manifest_sha256",
        required=True,
    )
    parser.add_argument(
        "--wait-seconds",
        "--wait",
        dest="wait_seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
    )
    parser.add_argument(
        "--poll-seconds",
        "--poll",
        dest="poll_seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    store = Boto3ObjectStore(profile=args.profile, region=args.region)
    lock = wait_for_artifact(
        store,
        args.artifact_s3,
        expected_source_checkpoint_s3=args.expected_source_checkpoint_s3,
        expected_source_step=args.expected_source_step,
        expected_source_checkpoint_manifest_sha256=(
            args.expected_source_checkpoint_manifest_sha256
        ),
        expected_initial_artifact_s3=args.expected_initial_artifact_s3,
        expected_initial_artifact_manifest_sha256=(
            args.expected_initial_artifact_manifest_sha256
        ),
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )
    write_json_atomic(args.output_json, lock)
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lock = run(args)
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
