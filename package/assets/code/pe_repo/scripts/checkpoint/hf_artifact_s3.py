#!/usr/bin/env python3
"""Atomically publish and restore a verified HF model artifact.

The S3 prefix is committed only after every immutable attempt object and the
``conversion_manifest.json`` have been verified. Consumers must restore through
this tool: a bare collection of uploaded weight files is never considered valid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    canonical_json_bytes,
    download_verified,
    local_path,
    parse_json_bytes,
    sha256_bytes,
    sha256_file,
    upload_verified,
    validate_file_entries,
    verify_object_stat,
    write_json_atomic,
)
from scripts.checkpoint.fsdp_to_hf import (
    EXPECTED_INITIAL_TENSOR_COUNT,
    EXPECTED_MTP_TENSOR_COUNT,
)


PROTOCOL_VERSION = 1
SUCCESS_NAME = "_SUCCESS"
MANIFEST_NAME = "conversion_manifest.json"
ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LEGACY_MANIFEST_KIND = "fsdp-model-only-to-hf"
GRAFT_MANIFEST_KIND = "hf-mtp-graft"
BASE_SNAPSHOT_MANIFEST_KIND = "hf-base-snapshot"
LEGACY_COMMIT_KIND = "fsdp-model-only-to-hf-commit"
GENERIC_COMMIT_KIND = "hf-model-artifact-commit"


def _commit_kind(manifest_kind: str) -> str:
    if manifest_kind == LEGACY_MANIFEST_KIND:
        return LEGACY_COMMIT_KIND
    if manifest_kind in {GRAFT_MANIFEST_KIND, BASE_SNAPSHOT_MANIFEST_KIND}:
        return GENERIC_COMMIT_KIND
    raise CheckpointError(f"Unsupported HF artifact manifest kind: {manifest_kind!r}")


def _validate_digest(value: Any, *, field: str, source: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointError(f"Invalid {field} in {source}: {value!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CheckpointError(f"Invalid {field} in {source}: {value!r}") from exc


def _root(uri: str) -> S3Location:
    location = S3Location.parse(uri)
    return S3Location(location.bucket, location.key.rstrip("/"))


def _success_uri(uri: str) -> str:
    return _root(uri).join(SUCCESS_NAME).uri


def _manifest_uri(uri: str, attempt_id: str) -> str:
    return _root(uri).join("_attempts", attempt_id, MANIFEST_NAME).uri


def _validate_attempt_id(attempt_id: str) -> None:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise CheckpointError(
            "attempt_id must be 1-128 characters containing only letters, digits, '.', '_', or '-'"
        )


def _validate_conversion_manifest(manifest: dict[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    expected = {
        "version": PROTOCOL_VERSION,
        "status": "complete",
        "optimizer_scheduler_rng_global_step_restored": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"Conversion manifest {key} mismatch in {source}: "
                f"expected={value!r} actual={manifest.get(key)!r}"
            )
    manifest_kind = manifest.get("kind")
    if manifest_kind not in {
        LEGACY_MANIFEST_KIND,
        GRAFT_MANIFEST_KIND,
        BASE_SNAPSHOT_MANIFEST_KIND,
    }:
        raise CheckpointError(f"Unsupported HF artifact manifest kind in {source}: {manifest_kind!r}")
    source_step = manifest.get("source_step")
    expected_source_step = 0 if manifest_kind == BASE_SNAPSHOT_MANIFEST_KIND else None
    valid_source_step = (
        type(source_step) is int and source_step == expected_source_step
        if expected_source_step is not None
        else type(source_step) is int and source_step > 0
    )
    if not valid_source_step:
        raise CheckpointError(f"Invalid source_step in {source}: {source_step!r}")
    if manifest_kind == LEGACY_MANIFEST_KIND:
        digest_fields = ("source_checkpoint_manifest_sha256", "initial_model_manifest_sha256")
    elif manifest_kind == GRAFT_MANIFEST_KIND:
        digest_fields = ("trained_model_index_sha256", "initial_model_index_sha256")
    else:
        digest_fields = ("source_model_index_sha256",)
        repository = manifest.get("source_hf_repository")
        revision = manifest.get("source_hf_revision")
        if (
            not isinstance(repository, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository)
            is None
        ):
            raise CheckpointError(f"Invalid source_hf_repository in {source}: {repository!r}")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise CheckpointError(f"Invalid source_hf_revision in {source}: {revision!r}")
    for field in digest_fields:
        _validate_digest(manifest.get(field), field=field, source=source)
    coverage = manifest.get("key_coverage")
    if not isinstance(coverage, dict):
        raise CheckpointError(f"Missing key_coverage in {source}")
    expected_keys = coverage.get("expected")
    final_keys = coverage.get("final")
    mtp_keys = coverage.get("mtp")
    if manifest_kind == LEGACY_MANIFEST_KIND:
        backbone_keys = coverage.get("merged_non_mtp")
        valid = (
            expected_keys == EXPECTED_INITIAL_TENSOR_COUNT
            and mtp_keys == EXPECTED_MTP_TENSOR_COUNT
        )
    else:
        contract = manifest.get("model_contract")
        if not isinstance(contract, dict):
            raise CheckpointError(f"Missing model_contract in {source}")
        expected_contract = contract.get("expected_tensor_count")
        backbone_contract = contract.get("expected_backbone_tensor_count")
        mtp_contract = contract.get("expected_mtp_tensor_count")
        if not all(isinstance(value, int) and value >= 0 for value in (
            expected_contract, backbone_contract, mtp_contract
        )):
            raise CheckpointError(f"Invalid model_contract counts in {source}: {contract!r}")
        for field in ("tensor_layout_sha256", "backbone_layout_sha256", "mtp_layout_sha256"):
            _validate_digest(contract.get(field), field=f"model_contract.{field}", source=source)
        backbone_keys = coverage.get("backbone")
        valid = (
            expected_keys == expected_contract
            and backbone_keys == backbone_contract
            and mtp_keys == mtp_contract
            and backbone_contract + mtp_contract == expected_contract
        )
    if (
        not all(isinstance(value, int) for value in (expected_keys, final_keys, mtp_keys, backbone_keys))
        or not valid
        or expected_keys != final_keys
        or backbone_keys + mtp_keys != final_keys
    ):
        raise CheckpointError(f"Invalid exact/MTP key coverage in {source}: {coverage!r}")
    return validate_file_entries(manifest, source=source)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes(), source=str(path))
    except OSError as exc:
        raise CheckpointError(f"Cannot read {path}: {exc}") from exc


def validate_local_artifact(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if root.is_symlink():
        raise CheckpointError(f"HF artifact root must not be a symlink: {root}")
    root = root.resolve()
    marker_path = root / SUCCESS_NAME
    manifest_path = root / MANIFEST_NAME
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    entries = _validate_conversion_manifest(manifest, source=str(manifest_path))
    expected = {
        "version": PROTOCOL_VERSION,
        "kind": _commit_kind(str(manifest["kind"])),
        "status": "complete",
        "source_step": manifest["source_step"],
        "manifest": MANIFEST_NAME,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Local artifact marker {key} mismatch in {marker_path}: "
                f"expected={value!r} actual={marker.get(key)!r}"
            )
    actual_digest = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != actual_digest:
        raise CheckpointError(
            f"Local conversion manifest digest mismatch: "
            f"marker={marker.get('manifest_sha256')!r} actual={actual_digest}"
        )
    for relative, record in entries.items():
        path = local_path(root, relative)
        if not path.is_file():
            raise CheckpointError(f"Converted artifact is missing {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record["size"] or digest != record["sha256"]:
            raise CheckpointError(
                f"Converted artifact file mismatch for {path}: "
                f"size={size}/{record['size']} sha256={digest}/{record['sha256']}"
            )
    if manifest["kind"] in {GRAFT_MANIFEST_KIND, BASE_SNAPSHOT_MANIFEST_KIND}:
        expected_paths = set(entries) | {MANIFEST_NAME, SUCCESS_NAME}
        actual_paths: set[str] = set()
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise CheckpointError(f"Grafted HF artifact contains a non-regular entry: {path}")
            actual_paths.add(path.name)
        if actual_paths != expected_paths:
            raise CheckpointError(
                "Grafted HF artifact file coverage mismatch: "
                f"missing={sorted(expected_paths - actual_paths)} "
                f"unexpected={sorted(actual_paths - expected_paths)}"
            )
        # Imported lazily to keep the generic artifact protocol usable without
        # introducing module-import cycles in artifact builders.
        if manifest["kind"] == GRAFT_MANIFEST_KIND:
            from scripts.checkpoint.hf_mtp_graft import validate_graft_layout

            validate_graft_layout(root, manifest)
        else:
            from scripts.checkpoint.hf_base_snapshot import validate_base_snapshot_layout

            validate_base_snapshot_layout(root, manifest, verified_entries=entries)
    return manifest, entries


def _put_json_if_absent(store: ObjectStore, uri: str, value: dict[str, Any]) -> str:
    data = canonical_json_bytes(value)
    digest = sha256_bytes(data)
    created = store.put_bytes(uri, data, metadata={"sha256": digest}, if_absent=True)
    if not created and store.read_bytes(uri) != data:
        raise CheckpointError(f"Immutable object already exists with different content: {uri}")
    verify_object_stat(store, uri, size=len(data), sha256=digest)
    return digest


def publish_artifact(
    store: ObjectStore,
    artifact_dir: Path,
    artifact_s3: str,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    _validate_attempt_id(attempt_id)
    artifact_dir = artifact_dir.resolve()
    source_manifest, entries = validate_local_artifact(artifact_dir)
    attempt_root = _root(artifact_s3).join("_attempts", attempt_id)
    remote_records: list[dict[str, Any]] = []
    for relative, record in sorted(entries.items()):
        object_uri = attempt_root.join("files", relative).uri
        size, digest = upload_verified(store, local_path(artifact_dir, relative), object_uri)
        if (size, digest) != (record["size"], record["sha256"]):
            raise CheckpointError(f"Converted artifact changed while publishing: {relative}")
        remote_records.append({**record, "object_uri": object_uri})

    remote_manifest = dict(source_manifest)
    remote_manifest.update(
        {
            "storage_protocol": "atomic-s3-artifact-v1",
            "attempt_id": attempt_id,
            "artifact_s3": _root(artifact_s3).uri + "/",
            "files": remote_records,
        }
    )
    manifest_uri = _manifest_uri(artifact_s3, attempt_id)
    manifest_digest = _put_json_if_absent(store, manifest_uri, remote_manifest)
    marker = {
        "version": PROTOCOL_VERSION,
        "kind": _commit_kind(str(source_manifest["kind"])),
        "status": "complete",
        "source_step": source_manifest["source_step"],
        "attempt_id": attempt_id,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "file_count": len(remote_records),
        "total_bytes": sum(record["size"] for record in remote_records),
    }
    success_uri = _success_uri(artifact_s3)
    _put_json_if_absent(store, success_uri, marker)
    committed_marker, _ = _read_remote(store, artifact_s3, verify_objects=True)
    if committed_marker.get("attempt_id") != attempt_id:
        raise CheckpointError(
            f"Artifact was already committed by attempt {committed_marker.get('attempt_id')!r}"
        )
    return marker


def _read_remote(
    store: ObjectStore, artifact_s3: str, *, verify_objects: bool
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    success_uri = _success_uri(artifact_s3)
    success_data = store.read_bytes(success_uri)
    marker = parse_json_bytes(success_data, source=success_uri)
    verify_object_stat(store, success_uri, size=len(success_data), sha256=sha256_bytes(success_data))
    expected = {"version": PROTOCOL_VERSION, "status": "complete"}
    for key, value in expected.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Remote artifact marker {key} mismatch: expected={value!r} actual={marker.get(key)!r}"
            )
    manifest_uri = marker.get("manifest_uri")
    if not isinstance(manifest_uri, str):
        raise CheckpointError(f"Invalid manifest_uri in {success_uri}")
    manifest_location = S3Location.parse(manifest_uri)
    expected_prefix = _root(artifact_s3).join("_attempts").key.rstrip("/") + "/"
    if manifest_location.bucket != _root(artifact_s3).bucket or not manifest_location.key.startswith(
        expected_prefix
    ):
        raise CheckpointError(f"Artifact marker points outside its namespace: {manifest_uri}")
    manifest_data = store.read_bytes(manifest_uri)
    manifest_digest = sha256_bytes(manifest_data)
    verify_object_stat(store, manifest_uri, size=len(manifest_data), sha256=manifest_digest)
    if marker.get("manifest_sha256") != manifest_digest:
        raise CheckpointError(
            f"Remote conversion manifest digest mismatch: marker={marker.get('manifest_sha256')!r} "
            f"actual={manifest_digest}"
        )
    manifest = parse_json_bytes(manifest_data, source=manifest_uri)
    entries = _validate_conversion_manifest(manifest, source=manifest_uri)
    expected_commit_kind = _commit_kind(str(manifest["kind"]))
    if marker.get("kind") != expected_commit_kind:
        raise CheckpointError(
            f"Remote artifact marker kind mismatch: expected={expected_commit_kind!r} "
            f"actual={marker.get('kind')!r}"
        )
    if marker.get("source_step") != manifest["source_step"]:
        raise CheckpointError(
            f"Remote source_step mismatch: marker={marker.get('source_step')!r} "
            f"manifest={manifest['source_step']!r}"
        )
    if marker.get("file_count") != len(entries) or marker.get("total_bytes") != sum(
        record["size"] for record in entries.values()
    ):
        raise CheckpointError(f"Remote marker totals do not match {manifest_uri}")
    for relative, record in entries.items():
        object_uri = record.get("object_uri")
        if not isinstance(object_uri, str):
            raise CheckpointError(f"Missing object_uri for {relative!r} in {manifest_uri}")
        object_location = S3Location.parse(object_uri)
        attempt_prefix = manifest_location.key.rsplit("/", 1)[0] + "/files/"
        if object_location.bucket != manifest_location.bucket or not object_location.key.startswith(
            attempt_prefix
        ):
            raise CheckpointError(f"File object points outside its immutable attempt: {object_uri}")
        if verify_objects:
            verify_object_stat(store, object_uri, size=record["size"], sha256=record["sha256"])
    return marker, entries


def inspect_artifact(store: ObjectStore, artifact_s3: str) -> dict[str, Any]:
    marker, entries = _read_remote(store, artifact_s3, verify_objects=True)
    return {
        "artifact_s3": _root(artifact_s3).uri + "/",
        "source_step": marker["source_step"],
        "attempt_id": marker["attempt_id"],
        "file_count": len(entries),
        "total_bytes": sum(record["size"] for record in entries.values()),
        "verified": True,
    }


def restore_artifact(store: ObjectStore, artifact_s3: str, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite existing restore directory: {output_dir}")
    marker, entries = _read_remote(store, artifact_s3, verify_objects=True)
    manifest_data = store.read_bytes(marker["manifest_uri"])
    manifest = parse_json_bytes(manifest_data, source=marker["manifest_uri"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.restore-", dir=output_dir.parent))
    try:
        for relative, record in entries.items():
            download_verified(
                store,
                record["object_uri"],
                local_path(temporary, relative),
                size=record["size"],
                sha256=record["sha256"],
            )
        write_json_atomic(temporary / MANIFEST_NAME, manifest)
        write_json_atomic(
            temporary / SUCCESS_NAME,
            {
                "version": PROTOCOL_VERSION,
                "kind": marker["kind"],
                "status": "complete",
                "source_step": marker["source_step"],
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(temporary / MANIFEST_NAME),
            },
        )
        validate_local_artifact(temporary)
        os.replace(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "artifact_s3": _root(artifact_s3).uri + "/",
        "output_dir": str(output_dir),
        "attempt_id": marker["attempt_id"],
        "file_count": len(entries),
        "verified": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--artifact-dir", type=Path, required=True)
    publish.add_argument("--artifact-s3", required=True)
    publish.add_argument("--attempt-id", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--artifact-s3", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--artifact-s3", required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = Boto3ObjectStore(profile=args.profile, region=args.region)
        if args.command == "publish":
            result = publish_artifact(
                store,
                args.artifact_dir,
                args.artifact_s3,
                attempt_id=args.attempt_id,
            )
        elif args.command == "inspect":
            result = inspect_artifact(store, args.artifact_s3)
        else:
            result = restore_artifact(store, args.artifact_s3, args.output_dir)
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
