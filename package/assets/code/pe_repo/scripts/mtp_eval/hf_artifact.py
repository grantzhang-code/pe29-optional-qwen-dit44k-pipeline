#!/usr/bin/env python3
"""Integrity-checked HF artifacts produced from native MCore checkpoints.

The trained artifact owns one immutable copy of the exported HF files.  An
``original_mtp`` variant references the trained non-MTP objects and the pinned
bootstrap artifact's dedicated MTP shard, so deriving it does not upload or
copy the 100+ GB backbone a second time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    canonical_json_bytes,
    download_verified,
    local_path,
    parse_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    upload_verified,
    utc_now,
    validate_file_entries,
    verify_object_stat,
    write_json_atomic,
)


PROTOCOL_VERSION = 1
ARTIFACT_KIND = "mcore-model-to-hf"
COMMIT_KIND = "mcore-model-to-hf-commit"
MANIFEST_NAME = "mcore_to_hf_manifest.json"
SUCCESS_NAME = "_SUCCESS"
INDEX_NAME = "model.safetensors.index.json"
TRAINED_MTP_FILE = "model-trained-mtp-00001-of-00001.safetensors"
ORIGINAL_MTP_FILE = "model-original-mtp-00001-of-00001.safetensors"
INITIAL_REFERENCE_NAME = "initial_mtp_reference.json"
INITIAL_REFERENCE_KIND = "selective-initial-mtp-reference"
INITIAL_REFERENCE_MODE = "metadata-mtp"
INITIAL_METADATA_DIR = "model-metadata"
INITIAL_REFERENCE_DIR = "reference"
LEGACY_MANIFEST_NAME = "conversion_manifest.json"
LEGACY_INITIAL_KIND = "fsdp-model-only-to-hf"
GRAFT_INITIAL_KIND = "hf-mtp-graft"
INITIAL_COMMIT_KINDS = frozenset({"fsdp-model-only-to-hf-commit", "hf-model-artifact-commit"})

# Keep the model directory passed to ms-swift deliberately free of a weight
# index.  Swift 4.4.2 uses these files only to construct the config/processor
# when ``load_model=False``; the complete MCore checkpoint supplies all model
# tensors.  A Swift/mcore-bridge version change must be audited before this
# allow-list is changed or the selective mode is enabled there.
INITIAL_MODEL_METADATA_FILES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)

# Qwen3.5-4B exact HF tensor contract.  Keeping this explicit is intentional:
# a successful export must contain the complete 4B backbone as well as the
# trained one-layer MTP head, rather than silently accepting a partial model.
EXPECTED_TENSOR_COUNT = 738
EXPECTED_NON_MTP_TENSOR_COUNT = 723
EXPECTED_MTP_TENSOR_COUNT = 15
MTP_NUM_LAYERS = 1

MTP_KEY_RE = re.compile(r"^mtp\.")
ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VALID_VARIANTS = {"trained_mtp", "original_mtp"}


@dataclass(frozen=True, slots=True)
class TensorSlice:
    key: str
    path: Path
    filename: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def parameters(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True, slots=True)
class WeightLayout:
    index: dict[str, Any]
    weight_map: dict[str, str]
    tensors: dict[str, TensorSlice]
    physical_by_file: dict[str, dict[str, TensorSlice]]


@dataclass(frozen=True, slots=True)
class RemoteArtifact:
    root_uri: str
    marker: dict[str, Any]
    manifest: dict[str, Any]
    manifest_sha256: str
    entries: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class InitialReference:
    artifact_s3: str
    attempt_id: str
    manifest_uri: str
    manifest_sha256: str
    metadata_dir: Path
    reference_dir: Path
    weight_map: dict[str, str]
    non_mtp_keys: frozenset[str]
    mtp_keys: frozenset[str]
    mtp_file: str
    mtp_tensor_sha256: dict[str, str]
    downloaded_files: tuple[str, ...]
    skipped_weight_files: tuple[str, ...]
    downloaded_bytes: int
    skipped_bytes: int


def _root(uri: str) -> S3Location:
    location = S3Location.parse(uri)
    return S3Location(location.bucket, location.key.rstrip("/"))


def _success_uri(uri: str) -> str:
    return _root(uri).join(SUCCESS_NAME).uri


def _manifest_uri(uri: str, attempt_id: str) -> str:
    return _root(uri).join("_attempts", attempt_id, MANIFEST_NAME).uri


def _normalise_s3(uri: str) -> str:
    return _root(uri).uri + "/"


def _validate_attempt_id(attempt_id: str) -> None:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise CheckpointError(
            "attempt_id must be 1-128 characters containing only letters, digits, '.', '_', or '-'"
        )


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_legacy_conversion_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    """Validate the committed bootstrap artifact without trusting its paths.

    This mirrors the legacy FSDP-to-HF protocol, but also checks the immutable
    remote identity fields added at publish time.  The stricter validation is
    important for selective restore: the skipped backbone payload is trusted
    through this manifest and its object HEAD, not through a local read.
    """

    expected = {
        "version": 1,
        "status": "complete",
        "optimizer_scheduler_rng_global_step_restored": False,
        "storage_protocol": "atomic-s3-artifact-v1",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"Initial artifact manifest {key} mismatch in {source}: "
                f"expected={value!r} actual={manifest.get(key)!r}"
            )
    source_step = manifest.get("source_step")
    if not isinstance(source_step, int) or source_step <= 0:
        raise CheckpointError(f"Invalid initial artifact source_step in {source}: {source_step!r}")
    manifest_kind = manifest.get("kind")
    if manifest_kind == LEGACY_INITIAL_KIND:
        digest_fields = ("source_checkpoint_manifest_sha256", "initial_model_manifest_sha256")
    elif manifest_kind == GRAFT_INITIAL_KIND:
        digest_fields = ("trained_model_index_sha256", "initial_model_index_sha256")
    else:
        raise CheckpointError(f"Unsupported initial artifact kind in {source}: {manifest_kind!r}")
    for field in digest_fields:
        if not _valid_sha256(manifest.get(field)):
            raise CheckpointError(f"Invalid {field} in {source}: {manifest.get(field)!r}")

    coverage = manifest.get("key_coverage")
    if not isinstance(coverage, dict):
        raise CheckpointError(f"Missing initial artifact key_coverage in {source}")
    expected_count = coverage.get("expected")
    non_mtp_count = coverage.get("merged_non_mtp" if manifest_kind == LEGACY_INITIAL_KIND else "backbone")
    mtp_count = coverage.get("mtp")
    final_count = coverage.get("final")
    if (
        expected_count != EXPECTED_TENSOR_COUNT
        or non_mtp_count != EXPECTED_NON_MTP_TENSOR_COUNT
        or mtp_count != EXPECTED_MTP_TENSOR_COUNT
        or final_count != EXPECTED_TENSOR_COUNT
        or non_mtp_count + mtp_count != final_count
    ):
        raise CheckpointError(f"Invalid initial artifact key_coverage in {source}: {coverage!r}")
    if manifest_kind == GRAFT_INITIAL_KIND:
        contract = manifest.get("model_contract")
        expected_contract = {
            "expected_tensor_count": EXPECTED_TENSOR_COUNT,
            "expected_backbone_tensor_count": EXPECTED_NON_MTP_TENSOR_COUNT,
            "expected_mtp_tensor_count": EXPECTED_MTP_TENSOR_COUNT,
        }
        if not isinstance(contract, dict) or any(
            contract.get(key) != value for key, value in expected_contract.items()
        ):
            raise CheckpointError(f"Invalid initial artifact model_contract in {source}: {contract!r}")
        for field in ("tensor_layout_sha256", "backbone_layout_sha256", "mtp_layout_sha256"):
            if not _valid_sha256(contract.get(field)):
                raise CheckpointError(f"Invalid model_contract.{field} in {source}")
    _initial_mtp_file(manifest, source=source)
    return validate_file_entries(manifest, source=source)


def _initial_mtp_file(manifest: Mapping[str, Any], *, source: str) -> str:
    coverage = manifest.get("key_coverage")
    if not isinstance(coverage, Mapping):
        raise CheckpointError(f"Missing initial artifact key_coverage in {source}")
    if manifest.get("kind") == GRAFT_INITIAL_KIND:
        filename = coverage.get("extracted_mtp_file")
    else:
        values = coverage.get("copied_mtp_files")
        filename = values[0] if isinstance(values, list) and len(values) == 1 else None
    if not isinstance(filename, str):
        raise CheckpointError(f"Invalid initial MTP filename in {source}: {filename!r}")
    safe_relative_path(filename)
    return filename


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"Cannot read {path}: {exc}") from exc
    return parse_json_bytes(data, source=str(path))


def _read_safetensors(path: Path) -> dict[str, TensorSlice]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise CheckpointError(f"Truncated safetensors header in {path}")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 0 or header_length > 512 * 1024 * 1024:
                raise CheckpointError(f"Unreasonable safetensors header length in {path}: {header_length}")
            raw_header = handle.read(header_length)
    except OSError as exc:
        raise CheckpointError(f"Cannot read safetensors file {path}: {exc}") from exc
    if len(raw_header) != header_length:
        raise CheckpointError(f"Truncated safetensors JSON header in {path}")
    header = parse_json_bytes(raw_header, source=str(path))
    payload_start = 8 + header_length
    payload_size = size - payload_start
    if payload_size < 0:
        raise CheckpointError(f"Safetensors payload starts beyond EOF in {path}")

    result: dict[str, TensorSlice] = {}
    intervals: list[tuple[int, int, str]] = []
    for key, value in header.items():
        if key == "__metadata__":
            if not isinstance(value, dict):
                raise CheckpointError(f"Invalid __metadata__ entry in {path}")
            continue
        if not isinstance(value, dict):
            raise CheckpointError(f"Invalid safetensors entry {key!r} in {path}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise CheckpointError(f"Invalid tensor metadata for {key!r} in {path}")
        start, end = offsets
        intervals.append((start, end, key))
        result[key] = TensorSlice(
            key=key,
            path=path,
            filename=path.name,
            dtype=dtype,
            shape=tuple(shape),
            start=payload_start + start,
            end=payload_start + end,
        )

    cursor = 0
    for start, end, key in sorted(intervals):
        if start != cursor:
            raise CheckpointError(
                f"Non-contiguous or overlapping safetensors payload before {key!r} in {path}: "
                f"expected_offset={cursor} actual={start}"
            )
        cursor = end
    if cursor != payload_size:
        raise CheckpointError(f"Unindexed payload bytes in {path}: indexed={cursor} payload={payload_size}")
    return result


def read_weight_layout(root: Path) -> WeightLayout:
    root = root.resolve()
    index_path = root / INDEX_NAME
    index = _read_json_file(index_path)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise CheckpointError(f"Invalid weight_map in {index_path}")
    weight_map: dict[str, str] = {}
    for raw_key, raw_filename in raw_weight_map.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise CheckpointError(f"Invalid tensor key in {index_path}: {raw_key!r}")
        if not isinstance(raw_filename, str):
            raise CheckpointError(f"Invalid shard name for {raw_key!r} in {index_path}")
        relative = safe_relative_path(raw_filename)
        if len(relative.parts) != 1:
            raise CheckpointError(f"Weight index must reference files in its own directory: {raw_filename!r}")
        weight_map[raw_key] = raw_filename

    referenced_files = set(weight_map.values())
    actual_files = {path.name for path in root.glob("*.safetensors") if path.is_file()}
    if actual_files != referenced_files:
        raise CheckpointError(
            "Safetensors file/index coverage mismatch: "
            f"unreferenced={sorted(actual_files - referenced_files)} "
            f"missing={sorted(referenced_files - actual_files)}"
        )

    physical_by_file: dict[str, dict[str, TensorSlice]] = {}
    physical_owner: dict[str, str] = {}
    for filename in sorted(referenced_files):
        tensors = _read_safetensors(root / filename)
        physical_by_file[filename] = tensors
        for key in tensors:
            previous = physical_owner.get(key)
            if previous is not None:
                raise CheckpointError(f"Duplicate physical tensor {key!r} in {previous!r} and {filename!r}")
            physical_owner[key] = filename

    if set(physical_owner) != set(weight_map):
        raise CheckpointError(
            "Weight index/physical tensor coverage mismatch: "
            f"missing={sorted(set(weight_map) - set(physical_owner))[:20]} "
            f"unindexed={sorted(set(physical_owner) - set(weight_map))[:20]}"
        )
    for key, filename in weight_map.items():
        if physical_owner[key] != filename:
            raise CheckpointError(
                f"Weight index maps {key!r} to {filename!r}, but it is stored in {physical_owner[key]!r}"
            )
    tensors = {key: physical_by_file[filename][key] for key, filename in weight_map.items()}
    return WeightLayout(index, weight_map, tensors, physical_by_file)


def _copy_slice(source: TensorSlice, destination: Any, *, chunk_size: int = 16 * 1024 * 1024) -> None:
    remaining = source.size
    try:
        with source.path.open("rb") as handle:
            handle.seek(source.start)
            while remaining:
                data = handle.read(min(chunk_size, remaining))
                if not data:
                    raise CheckpointError(f"Unexpected EOF while copying {source.key!r} from {source.path}")
                destination.write(data)
                remaining -= len(data)
    except OSError as exc:
        raise CheckpointError(f"Cannot copy {source.key!r} from {source.path}: {exc}") from exc


def _write_safetensors(path: Path, tensors: dict[str, TensorSlice]) -> None:
    if not tensors:
        raise CheckpointError(f"Refusing to create empty safetensors file: {path}")
    offset = 0
    header: dict[str, Any] = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        header[key] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.size],
        }
        offset += tensor.size
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for key in sorted(tensors):
                _copy_slice(tensors[key], handle)
            handle.flush()
            os.fsync(handle.fileno())
        _read_safetensors(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_slice(tensor: TensorSlice, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = tensor.size
    try:
        with tensor.path.open("rb") as handle:
            handle.seek(tensor.start)
            while remaining:
                data = handle.read(min(chunk_size, remaining))
                if not data:
                    raise CheckpointError(f"Unexpected EOF while hashing {tensor.key!r} in {tensor.path}")
                digest.update(data)
                remaining -= len(data)
    except OSError as exc:
        raise CheckpointError(f"Cannot hash {tensor.key!r} in {tensor.path}: {exc}") from exc
    return digest.hexdigest()


def mtp_tensor_hashes(layout: WeightLayout) -> dict[str, str]:
    return {
        key: _sha256_slice(layout.tensors[key]) for key in sorted(layout.weight_map) if MTP_KEY_RE.match(key)
    }


def _validate_initial_weight_index(
    index: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    *,
    source: str,
    expected_mtp_file: str,
) -> tuple[dict[str, str], frozenset[str], frozenset[str], str, frozenset[str]]:
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise CheckpointError(f"Initial artifact index has no weight_map in {source}")
    weight_map: dict[str, str] = {}
    for key, filename in raw_weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str):
            raise CheckpointError(f"Invalid initial weight_map entry in {source}: {key!r}")
        path = safe_relative_path(filename)
        if len(path.parts) != 1:
            raise CheckpointError(
                f"Initial weight shard must be a one-level filename in {source}: {filename!r}"
            )
        weight_map[key] = filename

    all_keys = set(weight_map)
    mtp_keys = frozenset(key for key in all_keys if MTP_KEY_RE.match(key))
    non_mtp_keys = frozenset(all_keys - set(mtp_keys))
    if (
        len(all_keys) != EXPECTED_TENSOR_COUNT
        or len(non_mtp_keys) != EXPECTED_NON_MTP_TENSOR_COUNT
        or len(mtp_keys) != EXPECTED_MTP_TENSOR_COUNT
    ):
        raise CheckpointError(
            f"Initial index tensor coverage mismatch in {source}: "
            f"total={len(all_keys)}/{EXPECTED_TENSOR_COUNT} "
            f"non_mtp={len(non_mtp_keys)}/{EXPECTED_NON_MTP_TENSOR_COUNT} "
            f"mtp={len(mtp_keys)}/{EXPECTED_MTP_TENSOR_COUNT}"
        )
    referenced_files = frozenset(weight_map.values())
    missing_records = referenced_files - set(entries)
    if missing_records:
        raise CheckpointError(
            f"Initial index references files absent from the committed manifest: {sorted(missing_records)}"
        )
    mtp_files = {weight_map[key] for key in mtp_keys}
    if mtp_files != {expected_mtp_file}:
        raise CheckpointError(
            f"Initial index MTP shard mismatch in {source}: "
            f"expected={expected_mtp_file!r} actual={sorted(mtp_files)}"
        )
    if any(weight_map[key] == expected_mtp_file for key in non_mtp_keys):
        raise CheckpointError(f"Initial MTP shard {expected_mtp_file!r} is not dedicated in {source}")
    return weight_map, non_mtp_keys, mtp_keys, expected_mtp_file, referenced_files


def _selective_reference_paths(root: Path) -> tuple[Path, Path]:
    return root / INITIAL_METADATA_DIR, root / INITIAL_REFERENCE_DIR


def validate_initial_metadata_mtp_reference(
    root: Path,
    *,
    expected_artifact_s3: str,
    expected_manifest_sha256: str,
) -> InitialReference:
    """Validate a local metadata-only model and its isolated MTP reference."""

    root = Path(os.path.abspath(os.fspath(root)))
    if root.is_symlink() or not root.is_dir():
        raise CheckpointError(f"Invalid selective initial-reference directory: {root}")
    if not _valid_sha256(expected_manifest_sha256):
        raise CheckpointError(
            f"Invalid expected initial artifact manifest SHA-256: {expected_manifest_sha256!r}"
        )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CheckpointError(f"Refusing symlink in selective initial reference: {path}")

    metadata_dir, reference_dir = _selective_reference_paths(root)
    if not metadata_dir.is_dir() or not reference_dir.is_dir():
        raise CheckpointError(f"Selective initial reference lacks metadata/reference directories: {root}")
    marker = _read_json_file(root / INITIAL_REFERENCE_NAME)
    expected_marker = {
        "version": PROTOCOL_VERSION,
        "kind": INITIAL_REFERENCE_KIND,
        "mode": INITIAL_REFERENCE_MODE,
        "artifact_s3": _normalise_s3(expected_artifact_s3),
        "manifest_sha256": expected_manifest_sha256,
        "metadata_dir": INITIAL_METADATA_DIR,
        "reference_dir": INITIAL_REFERENCE_DIR,
        "index_file": INDEX_NAME,
    }
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Selective initial reference {key} mismatch in {root}: "
                f"expected={value!r} actual={marker.get(key)!r}"
            )
    attempt_id = marker.get("attempt_id")
    manifest_uri = marker.get("manifest_uri")
    if not isinstance(attempt_id, str) or not isinstance(manifest_uri, str):
        raise CheckpointError(f"Selective initial reference has incomplete lineage: {root}")
    _validate_attempt_id(attempt_id)
    expected_manifest_uri = (
        _root(expected_artifact_s3).join("_attempts", attempt_id, LEGACY_MANIFEST_NAME).uri
    )
    if manifest_uri != expected_manifest_uri:
        raise CheckpointError(f"Selective initial reference manifest escapes its lineage: {manifest_uri}")

    manifest_path = reference_dir / LEGACY_MANIFEST_NAME
    success_path = reference_dir / SUCCESS_NAME
    if not manifest_path.is_file() or not success_path.is_file():
        raise CheckpointError(f"Selective initial reference lacks committed metadata: {root}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise CheckpointError(f"Selective initial manifest content mismatch: {manifest_path}")
    success_sha256 = marker.get("success_sha256")
    if not _valid_sha256(success_sha256) or sha256_file(success_path) != success_sha256:
        raise CheckpointError(f"Selective initial commit marker content mismatch: {success_path}")
    remote_marker = _read_json_file(success_path)
    if (
        remote_marker.get("attempt_id") != attempt_id
        or remote_marker.get("manifest_uri") != manifest_uri
        or remote_marker.get("manifest_sha256") != expected_manifest_sha256
        or remote_marker.get("version") != 1
        or remote_marker.get("kind") not in INITIAL_COMMIT_KINDS
        or remote_marker.get("status") != "complete"
    ):
        raise CheckpointError("Selective initial reference does not match its committed marker")

    manifest = _read_json_file(manifest_path)
    entries = _validate_legacy_conversion_manifest(manifest, source=str(manifest_path))
    if (
        manifest.get("attempt_id") != attempt_id
        or manifest.get("artifact_s3") != _normalise_s3(expected_artifact_s3)
        or remote_marker.get("source_step") != manifest.get("source_step")
        or remote_marker.get("file_count") != len(entries)
        or remote_marker.get("total_bytes") != sum(record["size"] for record in entries.values())
    ):
        raise CheckpointError("Selective initial reference marker/manifest identity mismatch")

    metadata_files = marker.get("metadata_files")
    if metadata_files != sorted(INITIAL_MODEL_METADATA_FILES):
        raise CheckpointError(f"Selective initial metadata allow-list mismatch: {metadata_files!r}")
    index_record = entries.get(INDEX_NAME)
    if index_record is None:
        raise CheckpointError(f"Committed initial artifact is missing {INDEX_NAME}")
    index_path = reference_dir / INDEX_NAME
    index = _read_json_file(index_path)
    expected_mtp_file = _initial_mtp_file(manifest, source=str(manifest_path))
    weight_map, non_mtp_keys, mtp_keys, mtp_file, referenced_files = _validate_initial_weight_index(
        index,
        entries,
        source=str(index_path),
        expected_mtp_file=expected_mtp_file,
    )
    non_mtp_weight_files = frozenset(referenced_files - {mtp_file})
    if marker.get("mtp_file") != mtp_file:
        raise CheckpointError("Selective initial MTP filename does not match the committed index")
    if marker.get("tensor_coverage") != {
        "expected": EXPECTED_TENSOR_COUNT,
        "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
        "mtp": EXPECTED_MTP_TENSOR_COUNT,
    }:
        raise CheckpointError("Selective initial tensor coverage marker is invalid")
    expected_entry_paths = INITIAL_MODEL_METADATA_FILES | {INDEX_NAME, mtp_file} | non_mtp_weight_files
    if set(entries) != expected_entry_paths:
        raise CheckpointError(
            "Initial artifact file allow-list mismatch: "
            f"missing={sorted(expected_entry_paths - set(entries))} "
            f"extra={sorted(set(entries) - expected_entry_paths)}"
        )

    downloaded_files = tuple(sorted(INITIAL_MODEL_METADATA_FILES | {INDEX_NAME, mtp_file}))
    skipped_weight_files = tuple(sorted(non_mtp_weight_files))
    if marker.get("downloaded_files") != list(downloaded_files):
        raise CheckpointError("Selective initial downloaded-file set mismatch")
    if marker.get("skipped_weight_files") != list(skipped_weight_files):
        raise CheckpointError("Selective initial skipped-weight set mismatch")

    for relative in downloaded_files:
        destination = (
            metadata_dir / relative if relative in INITIAL_MODEL_METADATA_FILES else reference_dir / relative
        )
        record = entries[relative]
        if not destination.is_file():
            raise CheckpointError(f"Selective initial reference is missing {destination}")
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if size != record["size"] or digest != record["sha256"]:
            raise CheckpointError(
                f"Selective initial file mismatch for {destination}: "
                f"size={size}/{record['size']} sha256={digest}/{record['sha256']}"
            )
    for relative in skipped_weight_files:
        if (metadata_dir / relative).exists() or (reference_dir / relative).exists():
            raise CheckpointError(f"Skipped initial backbone file is unexpectedly local: {relative}")

    expected_local_files = {
        INITIAL_REFERENCE_NAME,
        *{f"{INITIAL_METADATA_DIR}/{item}" for item in INITIAL_MODEL_METADATA_FILES},
        f"{INITIAL_REFERENCE_DIR}/{INDEX_NAME}",
        f"{INITIAL_REFERENCE_DIR}/{mtp_file}",
        f"{INITIAL_REFERENCE_DIR}/{LEGACY_MANIFEST_NAME}",
        f"{INITIAL_REFERENCE_DIR}/{SUCCESS_NAME}",
    }
    actual_local_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_local_files != expected_local_files:
        raise CheckpointError(
            "Selective initial local file set mismatch: "
            f"missing={sorted(expected_local_files - actual_local_files)} "
            f"extra={sorted(actual_local_files - expected_local_files)}"
        )

    physical_mtp = _read_safetensors(reference_dir / mtp_file)
    if set(physical_mtp) != set(mtp_keys):
        raise CheckpointError(
            "Initial dedicated MTP shard/index mismatch: "
            f"missing={sorted(set(mtp_keys) - set(physical_mtp))} "
            f"extra={sorted(set(physical_mtp) - set(mtp_keys))}"
        )
    mtp_hashes = {key: _sha256_slice(physical_mtp[key]) for key in sorted(mtp_keys)}
    if marker.get("mtp_file_sha256") != entries[mtp_file]["sha256"]:
        raise CheckpointError("Selective initial MTP shard SHA-256 lineage mismatch")
    if marker.get("mtp_tensor_sha256") != mtp_hashes:
        raise CheckpointError("Selective initial MTP tensor hashes do not match its marker")

    downloaded_bytes = sum(entries[path]["size"] for path in downloaded_files)
    skipped_bytes = sum(entries[path]["size"] for path in skipped_weight_files)
    if marker.get("downloaded_bytes") != downloaded_bytes or marker.get("skipped_bytes") != skipped_bytes:
        raise CheckpointError("Selective initial byte totals do not match its committed manifest")
    return InitialReference(
        artifact_s3=_normalise_s3(expected_artifact_s3),
        attempt_id=attempt_id,
        manifest_uri=manifest_uri,
        manifest_sha256=expected_manifest_sha256,
        metadata_dir=metadata_dir,
        reference_dir=reference_dir,
        weight_map=weight_map,
        non_mtp_keys=non_mtp_keys,
        mtp_keys=mtp_keys,
        mtp_file=mtp_file,
        mtp_tensor_sha256=mtp_hashes,
        downloaded_files=downloaded_files,
        skipped_weight_files=skipped_weight_files,
        downloaded_bytes=downloaded_bytes,
        skipped_bytes=skipped_bytes,
    )


def restore_initial_metadata_mtp_reference(
    store: ObjectStore,
    artifact_s3: str,
    output_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> InitialReference:
    """Restore processor metadata plus the committed initial MTP reference.

    Every committed object is HEAD-verified before the non-MTP backbone is
    skipped.  Downloaded payloads are content-hashed, staged under one parent,
    validated, and atomically installed.
    """

    if not _valid_sha256(expected_manifest_sha256):
        raise CheckpointError(
            f"Invalid expected initial artifact manifest SHA-256: {expected_manifest_sha256!r}"
        )
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir.is_symlink() or output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite selective initial reference: {output_dir}")

    # verify_objects=True is intentional even though only selected payloads are
    # downloaded.  The manifest's skipped backbone records remain part of the
    # immutable lineage lock.
    artifact = _read_legacy_remote(store, artifact_s3, verify_objects=True)
    if artifact.manifest_sha256 != expected_manifest_sha256:
        raise CheckpointError(
            "Pinned initial artifact manifest changed: "
            f"expected={expected_manifest_sha256} actual={artifact.manifest_sha256}"
        )
    index_record = artifact.entries.get(INDEX_NAME)
    if index_record is None:
        raise CheckpointError(f"Committed initial artifact is missing {INDEX_NAME}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.selective-", dir=output_dir.parent))
    metadata_dir, reference_dir = _selective_reference_paths(temporary)
    try:
        download_verified(
            store,
            index_record["object_uri"],
            reference_dir / INDEX_NAME,
            size=index_record["size"],
            sha256=index_record["sha256"],
        )
        index = _read_json_file(reference_dir / INDEX_NAME)
        expected_mtp_file = _initial_mtp_file(artifact.manifest, source=artifact.root_uri)
        _, _, mtp_keys, mtp_file, referenced_files = _validate_initial_weight_index(
            index,
            artifact.entries,
            source=f"{artifact.root_uri}{INDEX_NAME}",
            expected_mtp_file=expected_mtp_file,
        )
        non_mtp_weight_files = frozenset(referenced_files - {mtp_file})
        expected_entry_paths = INITIAL_MODEL_METADATA_FILES | {INDEX_NAME, mtp_file} | non_mtp_weight_files
        if set(artifact.entries) != expected_entry_paths:
            raise CheckpointError(
                "Initial artifact file allow-list mismatch: "
                f"missing={sorted(expected_entry_paths - set(artifact.entries))} "
                f"extra={sorted(set(artifact.entries) - expected_entry_paths)}"
            )

        for relative in sorted(INITIAL_MODEL_METADATA_FILES):
            record = artifact.entries[relative]
            download_verified(
                store,
                record["object_uri"],
                metadata_dir / relative,
                size=record["size"],
                sha256=record["sha256"],
            )
        mtp_record = artifact.entries[mtp_file]
        download_verified(
            store,
            mtp_record["object_uri"],
            reference_dir / mtp_file,
            size=mtp_record["size"],
            sha256=mtp_record["sha256"],
        )
        physical_mtp = _read_safetensors(reference_dir / mtp_file)
        if set(physical_mtp) != set(mtp_keys):
            raise CheckpointError(
                "Initial dedicated MTP shard/index mismatch: "
                f"missing={sorted(set(mtp_keys) - set(physical_mtp))} "
                f"extra={sorted(set(physical_mtp) - set(mtp_keys))}"
            )
        mtp_hashes = {key: _sha256_slice(physical_mtp[key]) for key in sorted(mtp_keys)}

        manifest_data = store.read_bytes(artifact.marker["manifest_uri"])
        if sha256_bytes(manifest_data) != artifact.manifest_sha256:
            raise CheckpointError("Initial artifact manifest changed during selective restore")
        success_uri = _success_uri(artifact.root_uri)
        success_data = store.read_bytes(success_uri)
        success_sha256 = sha256_bytes(success_data)
        verify_object_stat(
            store,
            success_uri,
            size=len(success_data),
            sha256=success_sha256,
        )
        _write_bytes_atomic(reference_dir / LEGACY_MANIFEST_NAME, manifest_data)
        _write_bytes_atomic(reference_dir / SUCCESS_NAME, success_data)

        downloaded_files = tuple(sorted(INITIAL_MODEL_METADATA_FILES | {INDEX_NAME, mtp_file}))
        skipped_weight_files = tuple(sorted(non_mtp_weight_files))
        write_json_atomic(
            temporary / INITIAL_REFERENCE_NAME,
            {
                "version": PROTOCOL_VERSION,
                "kind": INITIAL_REFERENCE_KIND,
                "mode": INITIAL_REFERENCE_MODE,
                "artifact_s3": artifact.root_uri,
                "attempt_id": str(artifact.marker["attempt_id"]),
                "manifest_uri": artifact.marker["manifest_uri"],
                "manifest_sha256": artifact.manifest_sha256,
                "success_sha256": success_sha256,
                "metadata_dir": INITIAL_METADATA_DIR,
                "reference_dir": INITIAL_REFERENCE_DIR,
                "metadata_files": sorted(INITIAL_MODEL_METADATA_FILES),
                "index_file": INDEX_NAME,
                "mtp_file": mtp_file,
                "mtp_file_sha256": mtp_record["sha256"],
                "mtp_tensor_sha256": mtp_hashes,
                "tensor_coverage": {
                    "expected": EXPECTED_TENSOR_COUNT,
                    "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
                    "mtp": EXPECTED_MTP_TENSOR_COUNT,
                },
                "downloaded_files": list(downloaded_files),
                "skipped_weight_files": list(skipped_weight_files),
                "downloaded_bytes": sum(artifact.entries[path]["size"] for path in downloaded_files),
                "skipped_bytes": sum(artifact.entries[path]["size"] for path in skipped_weight_files),
            },
        )
        validate_initial_metadata_mtp_reference(
            temporary,
            expected_artifact_s3=artifact_s3,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        os.replace(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return validate_initial_metadata_mtp_reference(
        output_dir,
        expected_artifact_s3=artifact_s3,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def _validate_exact_coverage(layout: WeightLayout, *, source: str) -> tuple[set[str], set[str]]:
    all_keys = set(layout.weight_map)
    mtp_keys = {key for key in all_keys if MTP_KEY_RE.match(key)}
    non_mtp_keys = all_keys - mtp_keys
    if (
        len(all_keys) != EXPECTED_TENSOR_COUNT
        or len(non_mtp_keys) != EXPECTED_NON_MTP_TENSOR_COUNT
        or len(mtp_keys) != EXPECTED_MTP_TENSOR_COUNT
    ):
        raise CheckpointError(
            f"{source} tensor coverage mismatch: total={len(all_keys)}/{EXPECTED_TENSOR_COUNT} "
            f"non_mtp={len(non_mtp_keys)}/{EXPECTED_NON_MTP_TENSOR_COUNT} "
            f"mtp={len(mtp_keys)}/{EXPECTED_MTP_TENSOR_COUNT}"
        )
    return non_mtp_keys, mtp_keys


def _require_dedicated_mtp_file(layout: WeightLayout, *, source: str) -> str:
    mtp_keys = {key for key in layout.weight_map if MTP_KEY_RE.match(key)}
    mtp_files = {layout.weight_map[key] for key in mtp_keys}
    if len(mtp_files) != 1:
        raise CheckpointError(f"{source} must use exactly one dedicated MTP shard, got {sorted(mtp_files)}")
    filename = next(iter(mtp_files))
    physical = set(layout.physical_by_file[filename])
    if physical != mtp_keys:
        raise CheckpointError(
            f"{source} MTP shard {filename!r} is not dedicated: "
            f"missing={sorted(mtp_keys - physical)} extra={sorted(physical - mtp_keys)}"
        )
    return filename


def canonicalize_trained_mtp(root: Path) -> WeightLayout:
    """Move the trained MTP tensors into one dedicated shard.

    Only shards which physically contain MTP tensors are rewritten.  Backbone
    shards that contain no MTP tensors are left byte-for-byte untouched.
    """

    root = root.resolve()
    layout = read_weight_layout(root)
    _validate_exact_coverage(layout, source="Exported HF model")
    mtp_keys = {key for key in layout.weight_map if MTP_KEY_RE.match(key)}
    mtp_source_files = {layout.weight_map[key] for key in mtp_keys}
    destination = root / TRAINED_MTP_FILE
    if destination.exists() and TRAINED_MTP_FILE not in mtp_source_files:
        raise CheckpointError(f"Refusing to overwrite unexpected trained MTP shard: {destination}")
    if TRAINED_MTP_FILE in mtp_source_files and set(layout.physical_by_file[TRAINED_MTP_FILE]) != mtp_keys:
        raise CheckpointError(
            f"Reserved trained MTP filename {TRAINED_MTP_FILE!r} already contains non-MTP tensors"
        )

    mtp_tensors = {key: layout.tensors[key] for key in mtp_keys}
    staged = root / f".{TRAINED_MTP_FILE}.staged"
    staged.unlink(missing_ok=True)
    try:
        _write_safetensors(staged, mtp_tensors)
        for filename in sorted(mtp_source_files):
            physical = layout.physical_by_file[filename]
            remaining = {key: tensor for key, tensor in physical.items() if key not in mtp_keys}
            source_path = root / filename
            if remaining:
                _write_safetensors(source_path, remaining)
            elif source_path != destination:
                source_path.unlink()
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)

    weight_map = dict(layout.weight_map)
    for key in mtp_keys:
        weight_map[key] = TRAINED_MTP_FILE
    metadata = layout.index.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["total_size"] = sum(tensor.size for tensor in layout.tensors.values())
    metadata["total_parameters"] = sum(tensor.parameters for tensor in layout.tensors.values())
    write_json_atomic(
        root / INDEX_NAME,
        {"metadata": metadata, "weight_map": dict(sorted(weight_map.items()))},
    )
    result = read_weight_layout(root)
    if _require_dedicated_mtp_file(result, source="Canonical trained model") != TRAINED_MTP_FILE:
        raise CheckpointError("Canonical trained model did not use the expected MTP filename")
    return result


def _validate_mtp_config(root: Path) -> None:
    config = _read_json_file(root / "config.json")
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config
    values = [config.get("mtp_num_hidden_layers"), config.get("num_nextn_predict_layers")]
    if MTP_NUM_LAYERS not in values:
        raise CheckpointError(
            "Exported config does not declare one MTP layer via "
            "mtp_num_hidden_layers/num_nextn_predict_layers"
        )


def _require_explicit_mtp_flag(
    command: Iterable[str],
    *,
    expected_mtp_num_layers: int = MTP_NUM_LAYERS,
    expected_mtp_shared_weights: bool = False,
) -> None:
    values = list(command)
    matches: list[str] = []
    shared_matches: list[str] = []
    for index, value in enumerate(values):
        if value in {"--mtp-num-layers", "--mtp_num_layers"}:
            if index + 1 >= len(values):
                raise CheckpointError(f"Missing value after {value}")
            matches.append(values[index + 1])
        if value in {"--mtp-shared-weights", "--mtp_shared_weights"}:
            if index + 1 >= len(values):
                raise CheckpointError(f"Missing value after {value}")
            shared_matches.append(values[index + 1].lower())
    if matches != [str(expected_mtp_num_layers)]:
        raise CheckpointError(
            "MCore-to-HF export command must explicitly contain exactly one "
            f"--mtp-num-layers {expected_mtp_num_layers}; found values={matches}"
        )
    expected_shared = ["true"] if expected_mtp_shared_weights else []
    if shared_matches != expected_shared:
        raise CheckpointError(
            "MCore-to-HF export command has an unexpected shared-head flag: "
            f"expected={expected_shared} actual={shared_matches}"
        )


def _artifact_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    excluded = {MANIFEST_NAME, SUCCESS_NAME}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointError(f"Refusing artifact symlink: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        safe_relative_path(relative)
        records.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    if not records:
        raise CheckpointError(f"HF artifact has no files under {root}")
    return records


def prepare_trained_artifact(
    artifact_dir: Path,
    initial_model_dir: Path,
    *,
    initial_reference: InitialReference | None = None,
    source_checkpoint_s3: str,
    source_checkpoint_manifest_sha256: str,
    source_step: int,
    initial_artifact_s3: str,
    initial_artifact_manifest_sha256: str,
    export_command: Iterable[str],
    mtp_num_layers: int = MTP_NUM_LAYERS,
    mtp_shared_weights: bool = False,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    initial_model_dir = initial_model_dir.resolve()
    if (artifact_dir / MANIFEST_NAME).exists() or (artifact_dir / SUCCESS_NAME).exists():
        raise CheckpointError(f"Artifact is already sealed: {artifact_dir}")
    if source_step <= 0:
        raise CheckpointError(f"source_step must be positive, got {source_step}")
    for label, digest in {
        "source_checkpoint_manifest_sha256": source_checkpoint_manifest_sha256,
        "initial_artifact_manifest_sha256": initial_artifact_manifest_sha256,
    }.items():
        if not _valid_sha256(digest):
            raise CheckpointError(f"Invalid {label}: {digest!r}")
    command = [str(value) for value in export_command]
    if mtp_num_layers <= 0:
        raise CheckpointError(f"Invalid logical MTP layer count: {mtp_num_layers}")
    if mtp_num_layers != MTP_NUM_LAYERS and not mtp_shared_weights:
        raise CheckpointError(
            "Multiple logical MTP steps require shared weights for the one-head HF contract"
        )
    _require_explicit_mtp_flag(
        command,
        expected_mtp_num_layers=mtp_num_layers,
        expected_mtp_shared_weights=mtp_shared_weights,
    )
    _validate_mtp_config(artifact_dir)

    if initial_reference is None:
        initial_layout = read_weight_layout(initial_model_dir)
        initial_non_mtp, initial_mtp = _validate_exact_coverage(initial_layout, source="Initial HF model")
        initial_mtp_file = _require_dedicated_mtp_file(initial_layout, source="Initial HF model")
        initial_hashes = mtp_tensor_hashes(initial_layout)
    else:
        # Never trust a caller-constructed dataclass.  Revalidate the sealed
        # local selective reference and bind it to the explicit remote lineage.
        validated_reference = validate_initial_metadata_mtp_reference(
            initial_reference.metadata_dir.parent,
            expected_artifact_s3=initial_artifact_s3,
            expected_manifest_sha256=initial_artifact_manifest_sha256,
        )
        if initial_model_dir != validated_reference.metadata_dir:
            raise CheckpointError(
                "Selective export --model directory must be the isolated metadata directory: "
                f"expected={validated_reference.metadata_dir} actual={initial_model_dir}"
            )
        initial_non_mtp = set(validated_reference.non_mtp_keys)
        initial_mtp = set(validated_reference.mtp_keys)
        initial_mtp_file = validated_reference.mtp_file
        initial_hashes = dict(validated_reference.mtp_tensor_sha256)

    trained_layout = canonicalize_trained_mtp(artifact_dir)
    trained_non_mtp, trained_mtp = _validate_exact_coverage(trained_layout, source="Trained HF model")
    if trained_non_mtp != initial_non_mtp or trained_mtp != initial_mtp:
        raise CheckpointError(
            "Initial/trained tensor key sets differ: "
            f"non_mtp_missing={sorted(initial_non_mtp - trained_non_mtp)[:20]} "
            f"non_mtp_extra={sorted(trained_non_mtp - initial_non_mtp)[:20]} "
            f"mtp_missing={sorted(initial_mtp - trained_mtp)} "
            f"mtp_extra={sorted(trained_mtp - initial_mtp)}"
        )
    trained_hashes = mtp_tensor_hashes(trained_layout)
    changed_keys = sorted(key for key in trained_hashes if trained_hashes[key] != initial_hashes[key])
    if not changed_keys:
        raise CheckpointError(
            "Exported MTP tensor hashes are identical to the pinned initial head; "
            "the trained MTP head was not exported"
        )

    records = _artifact_file_records(artifact_dir)
    manifest = {
        "version": PROTOCOL_VERSION,
        "kind": ARTIFACT_KIND,
        "variant": "trained_mtp",
        "status": "complete",
        "created_at": utc_now(),
        "source_mcore_checkpoint": {
            "uri": _normalise_s3(source_checkpoint_s3),
            "step": source_step,
            "manifest_sha256": source_checkpoint_manifest_sha256,
        },
        "initial_mtp_artifact": {
            "uri": _normalise_s3(initial_artifact_s3),
            "manifest_sha256": initial_artifact_manifest_sha256,
            "mtp_file": initial_mtp_file,
        },
        "export": {
            "command": command,
            "mtp_num_layers": mtp_num_layers,
            "mtp_shared_weights": mtp_shared_weights,
            "physical_mtp_num_layers": MTP_NUM_LAYERS,
        },
        "tensor_coverage": {
            "expected": EXPECTED_TENSOR_COUNT,
            "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
            "mtp": EXPECTED_MTP_TENSOR_COUNT,
            "final": len(trained_layout.weight_map),
            "mtp_files": [TRAINED_MTP_FILE],
            "mtp_tensor_sha256": trained_hashes,
            "initial_mtp_tensor_sha256": initial_hashes,
            "changed_mtp_keys": changed_keys,
        },
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    write_json_atomic(artifact_dir / MANIFEST_NAME, manifest)
    write_json_atomic(
        artifact_dir / SUCCESS_NAME,
        {
            "version": PROTOCOL_VERSION,
            "kind": COMMIT_KIND,
            "variant": "trained_mtp",
            "status": "complete",
            "source_step": source_step,
            "manifest": MANIFEST_NAME,
            "manifest_sha256": sha256_file(artifact_dir / MANIFEST_NAME),
        },
    )
    validate_local_artifact(artifact_dir)
    return manifest


def _validate_tensor_coverage(manifest: dict[str, Any], *, source: str) -> dict[str, Any]:
    coverage = manifest.get("tensor_coverage")
    if not isinstance(coverage, dict):
        raise CheckpointError(f"Missing tensor_coverage in {source}")
    expected = {
        "expected": EXPECTED_TENSOR_COUNT,
        "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
        "mtp": EXPECTED_MTP_TENSOR_COUNT,
        "final": EXPECTED_TENSOR_COUNT,
    }
    for key, value in expected.items():
        if coverage.get(key) != value:
            raise CheckpointError(
                f"Tensor coverage {key} mismatch in {source}: expected={value!r} actual={coverage.get(key)!r}"
            )
    for field in ("mtp_tensor_sha256", "initial_mtp_tensor_sha256"):
        hashes = coverage.get(field)
        if (
            not isinstance(hashes, dict)
            or len(hashes) != EXPECTED_MTP_TENSOR_COUNT
            or any(not isinstance(key, str) or MTP_KEY_RE.match(key) is None for key in hashes)
            or any(not _valid_sha256(value) for value in hashes.values())
        ):
            raise CheckpointError(f"Invalid {field} in {source}")
    if set(coverage["mtp_tensor_sha256"]) != set(coverage["initial_mtp_tensor_sha256"]):
        raise CheckpointError(f"MTP hash key sets differ in {source}")
    mtp_files = coverage.get("mtp_files")
    if not isinstance(mtp_files, list) or len(mtp_files) != 1 or not isinstance(mtp_files[0], str):
        raise CheckpointError(f"Expected exactly one MTP file in {source}")
    changed = coverage.get("changed_mtp_keys")
    if not isinstance(changed, list) or any(key not in coverage["mtp_tensor_sha256"] for key in changed):
        raise CheckpointError(f"Invalid changed_mtp_keys in {source}")
    if manifest.get("variant") == "trained_mtp":
        actual_changed = sorted(
            key
            for key, value in coverage["mtp_tensor_sha256"].items()
            if value != coverage["initial_mtp_tensor_sha256"][key]
        )
        if not actual_changed or sorted(changed) != actual_changed:
            raise CheckpointError(f"Trained MTP change set is empty or inconsistent in {source}")
    else:
        if coverage["mtp_tensor_sha256"] != coverage["initial_mtp_tensor_sha256"] or changed:
            raise CheckpointError(f"original_mtp does not exactly identify the initial head in {source}")
    return coverage


def validate_manifest(manifest: dict[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    for key, expected in {
        "version": PROTOCOL_VERSION,
        "kind": ARTIFACT_KIND,
        "status": "complete",
    }.items():
        if manifest.get(key) != expected:
            raise CheckpointError(
                f"Artifact manifest {key} mismatch in {source}: "
                f"expected={expected!r} actual={manifest.get(key)!r}"
            )
    variant = manifest.get("variant")
    if variant not in VALID_VARIANTS:
        raise CheckpointError(f"Invalid artifact variant in {source}: {variant!r}")
    source_checkpoint = manifest.get("source_mcore_checkpoint")
    if (
        not isinstance(source_checkpoint, dict)
        or not isinstance(source_checkpoint.get("step"), int)
        or source_checkpoint["step"] <= 0
        or not _valid_sha256(source_checkpoint.get("manifest_sha256"))
        or not isinstance(source_checkpoint.get("uri"), str)
    ):
        raise CheckpointError(f"Invalid source_mcore_checkpoint in {source}")
    _normalise_s3(source_checkpoint["uri"])
    initial = manifest.get("initial_mtp_artifact")
    if (
        not isinstance(initial, dict)
        or not isinstance(initial.get("uri"), str)
        or not _valid_sha256(initial.get("manifest_sha256"))
    ):
        raise CheckpointError(f"Invalid initial_mtp_artifact in {source}")
    _normalise_s3(initial["uri"])
    # Selective exports record the dedicated initial MTP shard as ``mtp_file``.
    # Artifacts produced before that optimisation record every source shard that
    # contains an MTP tensor as ``mtp_files`` instead.  Both forms are immutable
    # lineage metadata; accepting the historical plural form is required to
    # restore already-committed trained artifacts.
    mtp_file = initial.get("mtp_file")
    mtp_files = initial.get("mtp_files")
    if (mtp_file is None) == (mtp_files is None):
        raise CheckpointError(f"Invalid initial_mtp_artifact in {source}")
    if mtp_file is not None:
        if not isinstance(mtp_file, str):
            raise CheckpointError(f"Invalid initial_mtp_artifact in {source}")
        safe_relative_path(mtp_file)
    else:
        if (
            not isinstance(mtp_files, list)
            or not mtp_files
            or any(not isinstance(value, str) for value in mtp_files)
            or len(set(mtp_files)) != len(mtp_files)
        ):
            raise CheckpointError(f"Invalid initial_mtp_artifact in {source}")
        for value in mtp_files:
            safe_relative_path(value)
    export = manifest.get("export")
    logical_mtp_layers = export.get("mtp_num_layers") if isinstance(export, dict) else None
    shared_mtp = export.get("mtp_shared_weights", False) if isinstance(export, dict) else None
    physical_mtp_layers = (
        export.get("physical_mtp_num_layers", MTP_NUM_LAYERS)
        if isinstance(export, dict)
        else None
    )
    if (
        not isinstance(export, dict)
        or not isinstance(logical_mtp_layers, int)
        or logical_mtp_layers <= 0
        or not isinstance(shared_mtp, bool)
        or physical_mtp_layers != MTP_NUM_LAYERS
        or (logical_mtp_layers != MTP_NUM_LAYERS and not shared_mtp)
        or not isinstance(export.get("command"), list)
    ):
        raise CheckpointError(f"Invalid export metadata in {source}")
    _require_explicit_mtp_flag(
        export["command"],
        expected_mtp_num_layers=logical_mtp_layers,
        expected_mtp_shared_weights=shared_mtp,
    )
    _validate_tensor_coverage(manifest, source=source)
    entries = validate_file_entries(manifest, source=source)
    for required in ("config.json", INDEX_NAME):
        if required not in entries:
            raise CheckpointError(f"Artifact manifest is missing {required!r} in {source}")
    return entries


def validate_local_artifact(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    marker_path = root / SUCCESS_NAME
    manifest = _read_json_file(manifest_path)
    marker = _read_json_file(marker_path)
    entries = validate_manifest(manifest, source=str(manifest_path))
    expected_marker = {
        "version": PROTOCOL_VERSION,
        "kind": COMMIT_KIND,
        "variant": manifest["variant"],
        "status": "complete",
        "source_step": manifest["source_mcore_checkpoint"]["step"],
        "manifest": MANIFEST_NAME,
    }
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Local artifact marker {key} mismatch in {marker_path}: "
                f"expected={value!r} actual={marker.get(key)!r}"
            )
    manifest_digest = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_digest:
        raise CheckpointError(
            f"Local artifact manifest digest mismatch: marker={marker.get('manifest_sha256')!r} "
            f"actual={manifest_digest}"
        )
    for relative, record in entries.items():
        path = local_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"Artifact is missing regular file {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record["size"] or digest != record["sha256"]:
            raise CheckpointError(
                f"Artifact file mismatch for {path}: size={size}/{record['size']} "
                f"sha256={digest}/{record['sha256']}"
            )
    _validate_mtp_config(root)
    layout = read_weight_layout(root)
    _validate_exact_coverage(layout, source=f"Local {manifest['variant']} artifact")
    mtp_file = _require_dedicated_mtp_file(layout, source=f"Local {manifest['variant']} artifact")
    coverage = manifest["tensor_coverage"]
    if mtp_file != coverage["mtp_files"][0]:
        raise CheckpointError(
            f"Local MTP shard mismatch: index={mtp_file!r} manifest={coverage['mtp_files'][0]!r}"
        )
    hashes = mtp_tensor_hashes(layout)
    if hashes != coverage["mtp_tensor_sha256"]:
        raise CheckpointError("Local MTP tensor hashes do not match the artifact manifest")
    return manifest, entries


def validate_local_trained_artifact(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_mtp_num_layers: int | None = None,
    expected_mtp_shared_weights: bool | None = None,
) -> dict[str, Any]:
    """Fully validate a pinned local ``trained_mtp`` warm-start artifact."""

    artifact_dir = Path(os.path.abspath(os.fspath(artifact_dir)))
    if artifact_dir.is_symlink():
        raise CheckpointError(f"Refusing artifact directory symlink: {artifact_dir}")
    if not _valid_sha256(expected_manifest_sha256):
        raise CheckpointError(
            f"Invalid expected artifact manifest SHA-256: {expected_manifest_sha256!r}"
        )

    manifest, entries = validate_local_artifact(artifact_dir)
    manifest_sha256 = sha256_file(artifact_dir / MANIFEST_NAME)
    if manifest_sha256 != expected_manifest_sha256:
        raise CheckpointError(
            "Pinned local artifact manifest changed: "
            f"expected={expected_manifest_sha256} actual={manifest_sha256}"
        )
    if manifest["variant"] != "trained_mtp":
        raise CheckpointError(
            "Warm-start artifact must use the trained_mtp variant: "
            f"actual={manifest['variant']!r}"
        )
    export = manifest["export"]
    logical_mtp_layers = export["mtp_num_layers"]
    shared_mtp = export.get("mtp_shared_weights", False)
    physical_mtp_layers = export.get("physical_mtp_num_layers", MTP_NUM_LAYERS)
    if expected_mtp_num_layers is not None and logical_mtp_layers != expected_mtp_num_layers:
        raise CheckpointError(
            "Warm-start artifact logical MTP layer mismatch: "
            f"expected={expected_mtp_num_layers} actual={logical_mtp_layers}"
        )
    if expected_mtp_shared_weights is not None and shared_mtp != expected_mtp_shared_weights:
        raise CheckpointError(
            "Warm-start artifact shared-MTP mismatch: "
            f"expected={expected_mtp_shared_weights} actual={shared_mtp}"
        )
    return {
        "artifact_dir": str(artifact_dir),
        "variant": manifest["variant"],
        "source_step": manifest["source_mcore_checkpoint"]["step"],
        "mtp_num_layers": logical_mtp_layers,
        "mtp_shared_weights": shared_mtp,
        "physical_mtp_num_layers": physical_mtp_layers,
        "manifest_sha256": manifest_sha256,
        "file_count": len(entries),
        "total_bytes": sum(record["size"] for record in entries.values()),
        "verified": True,
    }


def _put_json_if_absent(store: ObjectStore, uri: str, value: dict[str, Any]) -> str:
    data = canonical_json_bytes(value)
    digest = sha256_bytes(data)
    created = store.put_bytes(uri, data, metadata={"sha256": digest}, if_absent=True)
    if not created and store.read_bytes(uri) != data:
        raise CheckpointError(f"Immutable JSON object already exists with different content: {uri}")
    verify_object_stat(store, uri, size=len(data), sha256=digest)
    return digest


def _put_bytes_if_absent(store: ObjectStore, uri: str, data: bytes) -> tuple[int, str]:
    digest = sha256_bytes(data)
    created = store.put_bytes(uri, data, metadata={"sha256": digest}, if_absent=True)
    if not created and store.read_bytes(uri) != data:
        raise CheckpointError(f"Immutable object already exists with different content: {uri}")
    verify_object_stat(store, uri, size=len(data), sha256=digest)
    return len(data), digest


def _read_committed_json(store: ObjectStore, uri: str) -> tuple[dict[str, Any], str]:
    data = store.read_bytes(uri)
    digest = sha256_bytes(data)
    verify_object_stat(store, uri, size=len(data), sha256=digest)
    return parse_json_bytes(data, source=uri), digest


def _read_legacy_remote(
    store: ObjectStore,
    artifact_s3: str,
    *,
    verify_objects: bool,
) -> RemoteArtifact:
    root_uri = _normalise_s3(artifact_s3)
    marker, _ = _read_committed_json(store, _success_uri(root_uri))
    expected_marker = {"version": 1, "status": "complete"}
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Initial artifact marker {key} mismatch at {root_uri}: "
                f"expected={value!r} actual={marker.get(key)!r}"
            )
    if marker.get("kind") not in INITIAL_COMMIT_KINDS:
        raise CheckpointError(
            f"Unsupported initial artifact marker kind at {root_uri}: {marker.get('kind')!r}"
        )
    attempt_id = marker.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise CheckpointError(f"Initial artifact marker lacks attempt_id: {root_uri}")
    _validate_attempt_id(attempt_id)
    manifest_uri = marker.get("manifest_uri")
    manifest_digest = marker.get("manifest_sha256")
    if not isinstance(manifest_uri, str) or not _valid_sha256(manifest_digest):
        raise CheckpointError(f"Invalid initial artifact marker: {root_uri}")
    expected_manifest_uri = _root(root_uri).join("_attempts", attempt_id, LEGACY_MANIFEST_NAME).uri
    if manifest_uri != expected_manifest_uri:
        raise CheckpointError(f"Initial artifact marker escapes its namespace: {manifest_uri}")
    manifest, actual_digest = _read_committed_json(store, manifest_uri)
    if actual_digest != manifest_digest:
        raise CheckpointError(
            f"Initial artifact manifest digest mismatch: marker={manifest_digest} actual={actual_digest}"
        )
    entries = _validate_legacy_conversion_manifest(manifest, source=manifest_uri)
    if (
        manifest.get("attempt_id") != attempt_id
        or manifest.get("artifact_s3") != root_uri
        or marker.get("source_step") != manifest.get("source_step")
        or marker.get("file_count") != len(entries)
        or marker.get("total_bytes") != sum(record["size"] for record in entries.values())
    ):
        raise CheckpointError(f"Initial artifact marker/manifest identity mismatch: {root_uri}")
    for relative, record in entries.items():
        object_uri = record.get("object_uri")
        expected_object_uri = _root(root_uri).join("_attempts", attempt_id, "files", relative).uri
        if object_uri != expected_object_uri:
            raise CheckpointError(
                f"Initial artifact record escapes its immutable attempt for {relative!r}: "
                f"expected={expected_object_uri!r} actual={object_uri!r}"
            )
        if verify_objects:
            verify_object_stat(store, object_uri, size=record["size"], sha256=record["sha256"])
    return RemoteArtifact(root_uri, marker, manifest, actual_digest, entries)


def _read_remote(
    store: ObjectStore,
    artifact_s3: str,
    *,
    verify_objects: bool,
    validate_lineage: bool = True,
) -> RemoteArtifact:
    root_uri = _normalise_s3(artifact_s3)
    marker, _ = _read_committed_json(store, _success_uri(root_uri))
    expected_marker = {
        "version": PROTOCOL_VERSION,
        "kind": COMMIT_KIND,
        "status": "complete",
    }
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise CheckpointError(
                f"Remote artifact marker {key} mismatch at {root_uri}: "
                f"expected={value!r} actual={marker.get(key)!r}"
            )
    manifest_uri = marker.get("manifest_uri")
    manifest_digest = marker.get("manifest_sha256")
    if not isinstance(manifest_uri, str) or not _valid_sha256(manifest_digest):
        raise CheckpointError(f"Invalid remote artifact marker at {root_uri}")
    manifest_location = S3Location.parse(manifest_uri)
    attempt_prefix = _root(root_uri).join("_attempts").key.rstrip("/") + "/"
    if (
        manifest_location.bucket != _root(root_uri).bucket
        or not manifest_location.key.startswith(attempt_prefix)
        or not manifest_location.key.endswith(f"/{MANIFEST_NAME}")
    ):
        raise CheckpointError(f"Remote artifact marker escapes its namespace: {manifest_uri}")
    manifest, actual_digest = _read_committed_json(store, manifest_uri)
    if actual_digest != manifest_digest:
        raise CheckpointError(
            f"Remote artifact manifest digest mismatch: marker={manifest_digest} actual={actual_digest}"
        )
    entries = validate_manifest(manifest, source=manifest_uri)
    if marker.get("variant") != manifest["variant"]:
        raise CheckpointError("Remote artifact marker/manifest variant mismatch")
    if marker.get("source_step") != manifest["source_mcore_checkpoint"]["step"]:
        raise CheckpointError("Remote artifact marker/manifest source step mismatch")
    if manifest.get("artifact_s3") != root_uri or marker.get("attempt_id") != manifest.get("attempt_id"):
        raise CheckpointError("Remote artifact identity mismatch")
    if marker.get("file_count") != len(entries) or marker.get("total_bytes") != sum(
        record["size"] for record in entries.values()
    ):
        raise CheckpointError("Remote artifact marker totals do not match its manifest")

    own_files_prefix = manifest_uri.rsplit("/", 1)[0] + "/files/"
    if manifest["variant"] == "trained_mtp":
        for relative, record in entries.items():
            object_uri = record.get("object_uri")
            if not isinstance(object_uri, str) or not object_uri.startswith(own_files_prefix):
                raise CheckpointError(f"Trained artifact object escapes its attempt for {relative!r}")
    elif validate_lineage:
        _validate_original_lineage(store, manifest, entries)

    for relative, record in entries.items():
        object_uri = record.get("object_uri")
        if not isinstance(object_uri, str):
            raise CheckpointError(f"Remote artifact record lacks object_uri for {relative!r}")
        if verify_objects:
            verify_object_stat(store, object_uri, size=record["size"], sha256=record["sha256"])
    return RemoteArtifact(root_uri, marker, manifest, actual_digest, entries)


def _validate_original_lineage(
    store: ObjectStore,
    manifest: dict[str, Any],
    entries: dict[str, dict[str, Any]],
) -> None:
    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise CheckpointError("original_mtp artifact is missing lineage")
    trained_info = lineage.get("trained_artifact")
    initial_info = lineage.get("initial_mtp_artifact")
    if not isinstance(trained_info, dict) or not isinstance(initial_info, dict):
        raise CheckpointError("original_mtp artifact lineage is incomplete")
    trained = _read_remote(
        store,
        str(trained_info.get("artifact_s3")),
        verify_objects=False,
        validate_lineage=False,
    )
    if trained.manifest.get("variant") != "trained_mtp":
        raise CheckpointError("original_mtp parent is not a trained_mtp artifact")
    if trained.manifest_sha256 != trained_info.get("manifest_sha256") or trained.marker.get(
        "attempt_id"
    ) != trained_info.get("attempt_id"):
        raise CheckpointError("original_mtp trained-artifact lineage lock mismatch")
    initial = _read_legacy_remote(
        store,
        str(initial_info.get("artifact_s3")),
        verify_objects=False,
    )
    if initial.manifest_sha256 != initial_info.get("manifest_sha256"):
        raise CheckpointError("original_mtp initial-artifact lineage lock mismatch")

    own_prefix = _manifest_uri(manifest["artifact_s3"], manifest["attempt_id"]).rsplit("/", 1)[0] + "/files/"
    for relative, record in entries.items():
        origin = record.get("origin")
        if not isinstance(origin, dict):
            raise CheckpointError(f"original_mtp record lacks origin for {relative!r}")
        kind = origin.get("kind")
        if kind == "self":
            if relative != INDEX_NAME or not str(record.get("object_uri", "")).startswith(own_prefix):
                raise CheckpointError(f"Invalid self-owned original_mtp record for {relative!r}")
            continue
        source_path = origin.get("path")
        if not isinstance(source_path, str):
            raise CheckpointError(f"Invalid lineage source path for {relative!r}")
        if kind == "trained_artifact":
            source_record = trained.entries.get(source_path)
        elif kind == "initial_mtp_artifact":
            source_record = initial.entries.get(source_path)
        else:
            raise CheckpointError(f"Unknown original_mtp origin {kind!r} for {relative!r}")
        if source_record is None:
            raise CheckpointError(f"Lineage source {source_path!r} is absent for {relative!r}")
        for field in ("size", "sha256", "object_uri"):
            if record.get(field) != source_record.get(field):
                raise CheckpointError(
                    f"Lineage record mismatch for {relative!r} field {field}: "
                    f"derived={record.get(field)!r} source={source_record.get(field)!r}"
                )


def publish_artifact(
    store: ObjectStore,
    artifact_dir: Path,
    artifact_s3: str,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    _validate_attempt_id(attempt_id)
    source_manifest, entries = validate_local_artifact(artifact_dir)
    if source_manifest["variant"] != "trained_mtp":
        raise CheckpointError(
            "Only trained_mtp local artifacts may be uploaded; derive original_mtp by immutable references"
        )
    root_uri = _normalise_s3(artifact_s3)
    attempt_root = _root(root_uri).join("_attempts", attempt_id)
    remote_records: list[dict[str, Any]] = []
    for relative, record in sorted(entries.items()):
        object_uri = attempt_root.join("files", relative).uri
        size, digest = upload_verified(store, local_path(artifact_dir, relative), object_uri)
        if (size, digest) != (record["size"], record["sha256"]):
            raise CheckpointError(f"Artifact changed while publishing: {relative}")
        remote_records.append({**record, "object_uri": object_uri})
    remote_manifest = dict(source_manifest)
    remote_manifest.update(
        {
            "storage_protocol": "atomic-mcore-hf-artifact-v1",
            "artifact_s3": root_uri,
            "attempt_id": attempt_id,
            "files": remote_records,
        }
    )
    manifest_uri = _manifest_uri(root_uri, attempt_id)
    manifest_digest = _put_json_if_absent(store, manifest_uri, remote_manifest)
    marker = {
        "version": PROTOCOL_VERSION,
        "kind": COMMIT_KIND,
        "variant": "trained_mtp",
        "status": "complete",
        "source_step": source_manifest["source_mcore_checkpoint"]["step"],
        "attempt_id": attempt_id,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "file_count": len(remote_records),
        "total_bytes": sum(record["size"] for record in remote_records),
    }
    _put_json_if_absent(store, _success_uri(root_uri), marker)
    committed = _read_remote(store, root_uri, verify_objects=True)
    if committed.marker.get("attempt_id") != attempt_id:
        raise CheckpointError(
            f"Artifact was already committed by attempt {committed.marker.get('attempt_id')!r}"
        )
    return marker


def _read_verified_record_bytes(
    store: ObjectStore,
    record: dict[str, Any],
    *,
    source: str,
) -> bytes:
    object_uri = record.get("object_uri")
    if not isinstance(object_uri, str):
        raise CheckpointError(f"Missing object_uri for {source}")
    verify_object_stat(store, object_uri, size=record["size"], sha256=record["sha256"])
    data = store.read_bytes(object_uri)
    if len(data) != record["size"] or sha256_bytes(data) != record["sha256"]:
        raise CheckpointError(f"Object content mismatch for {source}: {object_uri}")
    return data


def derive_original_mtp_local(
    store: ObjectStore,
    trained_dir: Path,
    initial_artifact_s3: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a local original-head model with hardlinked trained backbone files.

    The only weight bytes materialized again are the pinned initial artifact's
    dedicated MTP shard. Hardlink failure is fatal: silently copying the 100+ GB
    backbone would defeat the storage contract this operation promises.
    """

    trained_dir = trained_dir.resolve()
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir.is_symlink() or output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite original_mtp output: {output_dir}")
    trained_manifest, trained_entries = validate_local_artifact(trained_dir)
    if trained_manifest["variant"] != "trained_mtp":
        raise CheckpointError("Local derivation requires a trained_mtp artifact")
    initial = _read_legacy_remote(store, initial_artifact_s3, verify_objects=False)
    pinned_initial = trained_manifest["initial_mtp_artifact"]
    if (
        initial.root_uri != pinned_initial["uri"]
        or initial.manifest_sha256 != pinned_initial["manifest_sha256"]
    ):
        raise CheckpointError("Initial artifact does not match the trained artifact's provenance lock")

    initial_index_record = initial.entries.get(INDEX_NAME)
    if initial_index_record is None:
        raise CheckpointError(f"Initial artifact is missing {INDEX_NAME}")
    initial_index = parse_json_bytes(
        _read_verified_record_bytes(
            store,
            initial_index_record,
            source=f"initial {INDEX_NAME}",
        ),
        source=f"{initial.root_uri}{INDEX_NAME}",
    )
    raw_initial_map = initial_index.get("weight_map")
    if not isinstance(raw_initial_map, dict):
        raise CheckpointError("Initial artifact has an invalid weight_map")
    trained_layout = read_weight_layout(trained_dir)
    if set(raw_initial_map) != set(trained_layout.weight_map):
        raise CheckpointError("Initial and trained weight-map key sets differ")
    mtp_keys = {key for key in raw_initial_map if isinstance(key, str) and MTP_KEY_RE.match(key)}
    if len(mtp_keys) != EXPECTED_MTP_TENSOR_COUNT:
        raise CheckpointError(f"Initial artifact has {len(mtp_keys)} MTP keys, expected 15")
    initial_mtp_files = {raw_initial_map[key] for key in mtp_keys}
    if len(initial_mtp_files) != 1:
        raise CheckpointError("Initial artifact does not use one dedicated MTP shard")
    initial_source_path = str(next(iter(initial_mtp_files)))
    if any(
        filename == initial_source_path for key, filename in raw_initial_map.items() if key not in mtp_keys
    ):
        raise CheckpointError("Initial MTP shard also contains indexed non-MTP tensors")
    initial_mtp_record = initial.entries.get(initial_source_path)
    if initial_mtp_record is None:
        raise CheckpointError(f"Initial artifact manifest lacks {initial_source_path!r}")
    verify_object_stat(
        store,
        initial_mtp_record["object_uri"],
        size=initial_mtp_record["size"],
        sha256=initial_mtp_record["sha256"],
    )

    trained_mtp_files = set(trained_manifest["tensor_coverage"]["mtp_files"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.derive-", dir=output_dir.parent))
    try:
        for relative in sorted(trained_entries):
            if relative == INDEX_NAME or relative in trained_mtp_files:
                continue
            source = local_path(trained_dir, relative)
            destination = local_path(temporary, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError as exc:
                raise CheckpointError(
                    f"Cannot hardlink trained artifact file {source} to {destination}; "
                    "refusing a full backbone copy"
                ) from exc

        download_verified(
            store,
            initial_mtp_record["object_uri"],
            temporary / ORIGINAL_MTP_FILE,
            size=initial_mtp_record["size"],
            sha256=initial_mtp_record["sha256"],
        )
        derived_map = dict(trained_layout.weight_map)
        for key in mtp_keys:
            derived_map[key] = ORIGINAL_MTP_FILE
        metadata = trained_layout.index.get("metadata")
        write_json_atomic(
            temporary / INDEX_NAME,
            {
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                "weight_map": dict(sorted(derived_map.items())),
            },
        )

        records = _artifact_file_records(temporary)
        manifest = {
            key: value
            for key, value in trained_manifest.items()
            if key
            not in {
                "artifact_s3",
                "attempt_id",
                "created_at",
                "files",
                "file_count",
                "total_bytes",
                "storage_protocol",
                "lineage",
            }
        }
        manifest.update(
            {
                "variant": "original_mtp",
                "created_at": utc_now(),
                "lineage": {
                    "trained_local_manifest_sha256": sha256_file(trained_dir / MANIFEST_NAME),
                    "initial_artifact_s3": initial.root_uri,
                    "initial_manifest_sha256": initial.manifest_sha256,
                },
                "tensor_coverage": {
                    **trained_manifest["tensor_coverage"],
                    "mtp_files": [ORIGINAL_MTP_FILE],
                    "mtp_tensor_sha256": trained_manifest["tensor_coverage"]["initial_mtp_tensor_sha256"],
                    "changed_mtp_keys": [],
                },
                "file_count": len(records),
                "total_bytes": sum(record["size"] for record in records),
                "files": records,
            }
        )
        write_json_atomic(temporary / MANIFEST_NAME, manifest)
        write_json_atomic(
            temporary / SUCCESS_NAME,
            {
                "version": PROTOCOL_VERSION,
                "kind": COMMIT_KIND,
                "variant": "original_mtp",
                "status": "complete",
                "source_step": manifest["source_mcore_checkpoint"]["step"],
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(temporary / MANIFEST_NAME),
            },
        )
        validate_local_artifact(temporary)
        os.replace(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "status": "complete",
        "variant": "original_mtp",
        "trained_dir": str(trained_dir),
        "initial_artifact_s3": initial.root_uri,
        "output_dir": str(output_dir),
        "source_step": trained_manifest["source_mcore_checkpoint"]["step"],
        "backbone_storage": "hardlink",
        "downloaded_weight_files": [ORIGINAL_MTP_FILE],
    }


def derive_original_mtp_artifact(
    store: ObjectStore,
    trained_artifact_s3: str,
    initial_artifact_s3: str,
    artifact_s3: str,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    """Commit an original-head variant without duplicating trained backbone objects."""

    _validate_attempt_id(attempt_id)
    trained = _read_remote(store, trained_artifact_s3, verify_objects=True)
    if trained.manifest["variant"] != "trained_mtp":
        raise CheckpointError("The derivation parent must be a trained_mtp artifact")
    initial = _read_legacy_remote(store, initial_artifact_s3, verify_objects=False)
    pinned_initial = trained.manifest["initial_mtp_artifact"]
    if (
        _normalise_s3(initial_artifact_s3) != pinned_initial["uri"]
        or initial.manifest_sha256 != pinned_initial["manifest_sha256"]
    ):
        raise CheckpointError("Initial MTP artifact does not match the trained artifact's provenance lock")

    trained_index_data = _read_verified_record_bytes(
        store,
        trained.entries[INDEX_NAME],
        source=f"trained {INDEX_NAME}",
    )
    initial_index_record = initial.entries.get(INDEX_NAME)
    if initial_index_record is None:
        raise CheckpointError(f"Initial artifact is missing {INDEX_NAME}")
    initial_index_data = _read_verified_record_bytes(
        store,
        initial_index_record,
        source=f"initial {INDEX_NAME}",
    )
    trained_index = parse_json_bytes(trained_index_data, source=f"{trained.root_uri}{INDEX_NAME}")
    initial_index = parse_json_bytes(initial_index_data, source=f"{initial.root_uri}{INDEX_NAME}")
    trained_map = trained_index.get("weight_map")
    initial_map = initial_index.get("weight_map")
    if not isinstance(trained_map, dict) or not isinstance(initial_map, dict):
        raise CheckpointError("Trained or initial artifact has an invalid weight_map")
    if set(trained_map) != set(initial_map) or len(trained_map) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError("Trained/initial index key coverage differs during original_mtp derivation")
    mtp_keys = {key for key in trained_map if isinstance(key, str) and MTP_KEY_RE.match(key)}
    if len(mtp_keys) != EXPECTED_MTP_TENSOR_COUNT:
        raise CheckpointError(f"Unexpected MTP key count during derivation: {len(mtp_keys)}")
    initial_mtp_files = {initial_map[key] for key in mtp_keys}
    if len(initial_mtp_files) != 1:
        raise CheckpointError(f"Initial artifact does not use one dedicated MTP shard: {initial_mtp_files}")
    initial_source_path = next(iter(initial_mtp_files))
    initial_mtp_record = initial.entries.get(str(initial_source_path))
    if initial_mtp_record is None:
        raise CheckpointError(
            f"Initial MTP shard is absent from the committed manifest: {initial_source_path}"
        )
    if any(filename == initial_source_path for key, filename in initial_map.items() if key not in mtp_keys):
        raise CheckpointError("Initial MTP shard is not dedicated according to its weight index")
    verify_object_stat(
        store,
        initial_mtp_record["object_uri"],
        size=initial_mtp_record["size"],
        sha256=initial_mtp_record["sha256"],
    )

    trained_mtp_files = set(trained.manifest["tensor_coverage"]["mtp_files"])
    derived_map = dict(trained_map)
    for key in mtp_keys:
        derived_map[key] = ORIGINAL_MTP_FILE
    metadata = trained_index.get("metadata")
    derived_index_data = canonical_json_bytes(
        {
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            "weight_map": dict(sorted(derived_map.items())),
        }
    )

    root_uri = _normalise_s3(artifact_s3)
    own_index_uri = _root(root_uri).join("_attempts", attempt_id, "files", INDEX_NAME).uri
    index_size, index_digest = _put_bytes_if_absent(store, own_index_uri, derived_index_data)
    records: list[dict[str, Any]] = []
    for relative, record in sorted(trained.entries.items()):
        if relative == INDEX_NAME or relative in trained_mtp_files:
            continue
        records.append(
            {
                **record,
                "origin": {"kind": "trained_artifact", "path": relative},
            }
        )
    if ORIGINAL_MTP_FILE in {record["path"] for record in records}:
        raise CheckpointError(
            f"Derived MTP filename collides with a trained artifact file: {ORIGINAL_MTP_FILE}"
        )
    records.append(
        {
            "path": ORIGINAL_MTP_FILE,
            "size": initial_mtp_record["size"],
            "sha256": initial_mtp_record["sha256"],
            "object_uri": initial_mtp_record["object_uri"],
            "origin": {"kind": "initial_mtp_artifact", "path": str(initial_source_path)},
        }
    )
    records.append(
        {
            "path": INDEX_NAME,
            "size": index_size,
            "sha256": index_digest,
            "object_uri": own_index_uri,
            "origin": {"kind": "self", "path": INDEX_NAME},
        }
    )
    records.sort(key=lambda record: record["path"])

    manifest = {
        key: value
        for key, value in trained.manifest.items()
        if key not in {"artifact_s3", "attempt_id", "created_at", "files", "file_count", "total_bytes"}
    }
    manifest.update(
        {
            "variant": "original_mtp",
            "created_at": utc_now(),
            "storage_protocol": "atomic-mcore-hf-reference-artifact-v1",
            "artifact_s3": root_uri,
            "attempt_id": attempt_id,
            "lineage": {
                "trained_artifact": {
                    "artifact_s3": trained.root_uri,
                    "attempt_id": trained.marker["attempt_id"],
                    "manifest_uri": trained.marker["manifest_uri"],
                    "manifest_sha256": trained.manifest_sha256,
                },
                "initial_mtp_artifact": {
                    "artifact_s3": initial.root_uri,
                    "attempt_id": initial.marker.get("attempt_id"),
                    "manifest_uri": initial.marker["manifest_uri"],
                    "manifest_sha256": initial.manifest_sha256,
                },
            },
            "tensor_coverage": {
                **trained.manifest["tensor_coverage"],
                "mtp_files": [ORIGINAL_MTP_FILE],
                "mtp_tensor_sha256": trained.manifest["tensor_coverage"]["initial_mtp_tensor_sha256"],
                "changed_mtp_keys": [],
            },
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
            "files": records,
        }
    )
    manifest_uri = _manifest_uri(root_uri, attempt_id)
    manifest_digest = _put_json_if_absent(store, manifest_uri, manifest)
    marker = {
        "version": PROTOCOL_VERSION,
        "kind": COMMIT_KIND,
        "variant": "original_mtp",
        "status": "complete",
        "source_step": manifest["source_mcore_checkpoint"]["step"],
        "attempt_id": attempt_id,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "file_count": len(records),
        "total_bytes": manifest["total_bytes"],
    }
    _put_json_if_absent(store, _success_uri(root_uri), marker)
    committed = _read_remote(store, root_uri, verify_objects=True)
    if committed.marker.get("attempt_id") != attempt_id:
        raise CheckpointError(
            f"Artifact was already committed by attempt {committed.marker.get('attempt_id')!r}"
        )
    return marker


def inspect_artifact(store: ObjectStore, artifact_s3: str) -> dict[str, Any]:
    artifact = _read_remote(store, artifact_s3, verify_objects=True)
    export = artifact.manifest["export"]
    return {
        "artifact_s3": artifact.root_uri,
        "variant": artifact.manifest["variant"],
        "source_step": artifact.manifest["source_mcore_checkpoint"]["step"],
        "mtp_num_layers": export["mtp_num_layers"],
        "mtp_shared_weights": export.get("mtp_shared_weights", False),
        "physical_mtp_num_layers": export.get("physical_mtp_num_layers", MTP_NUM_LAYERS),
        "attempt_id": artifact.marker["attempt_id"],
        "manifest_sha256": artifact.manifest_sha256,
        "file_count": len(artifact.entries),
        "total_bytes": sum(record["size"] for record in artifact.entries.values()),
        "verified": True,
    }


def _install_restored_directory(temporary: Path, output_dir: Path) -> None:
    backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
    backup.rmdir()
    try:
        os.replace(output_dir, backup)
        try:
            os.replace(temporary, output_dir)
        except OSError:
            os.replace(backup, output_dir)
            raise
        shutil.rmtree(backup)
    except OSError as exc:
        raise CheckpointError(f"Failed to transactionally replace {output_dir}: {exc}") from exc


def restore_artifact(
    store: ObjectStore,
    artifact_s3: str,
    output_dir: Path,
    *,
    replace_existing: bool = False,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir.is_symlink():
        raise CheckpointError(f"Refusing restore directory symlink: {output_dir}")
    if output_dir.exists() and not replace_existing:
        raise CheckpointError(f"Refusing to overwrite existing restore directory: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise CheckpointError(f"Existing restore path is not a directory: {output_dir}")
    artifact = _read_remote(store, artifact_s3, verify_objects=True)
    if expected_manifest_sha256 is not None:
        if not _valid_sha256(expected_manifest_sha256):
            raise CheckpointError(f"Invalid expected artifact manifest SHA-256: {expected_manifest_sha256!r}")
        if artifact.manifest_sha256 != expected_manifest_sha256:
            raise CheckpointError(
                "Pinned artifact manifest changed: "
                f"expected={expected_manifest_sha256} actual={artifact.manifest_sha256}"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.restore-", dir=output_dir.parent))
    try:
        for relative, record in artifact.entries.items():
            download_verified(
                store,
                record["object_uri"],
                local_path(temporary, relative),
                size=record["size"],
                sha256=record["sha256"],
            )
        write_json_atomic(temporary / MANIFEST_NAME, artifact.manifest)
        write_json_atomic(
            temporary / SUCCESS_NAME,
            {
                "version": PROTOCOL_VERSION,
                "kind": COMMIT_KIND,
                "variant": artifact.manifest["variant"],
                "status": "complete",
                "source_step": artifact.manifest["source_mcore_checkpoint"]["step"],
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(temporary / MANIFEST_NAME),
            },
        )
        validate_local_artifact(temporary)
        if output_dir.exists():
            _install_restored_directory(temporary, output_dir)
        else:
            os.replace(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "artifact_s3": artifact.root_uri,
        "variant": artifact.manifest["variant"],
        "output_dir": str(output_dir),
        "attempt_id": artifact.marker["attempt_id"],
        "manifest_sha256": artifact.manifest_sha256,
        "file_count": len(artifact.entries),
        "verified": True,
    }


def legacy_artifact_lock(store: ObjectStore, artifact_s3: str) -> dict[str, Any]:
    """Return the immutable commit identity of the FSDP bootstrap artifact."""

    artifact = _read_legacy_remote(store, artifact_s3, verify_objects=True)
    return {
        "artifact_s3": artifact.root_uri,
        "attempt_id": artifact.marker.get("attempt_id"),
        "manifest_uri": artifact.marker["manifest_uri"],
        "manifest_sha256": artifact.manifest_sha256,
        "file_count": len(artifact.entries),
        "total_bytes": sum(record["size"] for record in artifact.entries.values()),
        "files": [
            {
                "path": relative,
                "size": record["size"],
                "sha256": record["sha256"],
                "object_uri": record["object_uri"],
            }
            for relative, record in sorted(artifact.entries.items())
        ],
        "verified": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Publish a sealed local trained_mtp artifact")
    publish.add_argument("--artifact-dir", type=Path, required=True)
    publish.add_argument("--artifact-s3", required=True)
    publish.add_argument("--attempt-id", required=True)

    derive = subparsers.add_parser(
        "derive-original-mtp",
        help="Derive original_mtp locally with hardlinks, or commit a remote reference variant",
    )
    derive.add_argument("--trained-dir", type=Path)
    derive.add_argument("--output-dir", type=Path)
    derive.add_argument("--trained-artifact-s3")
    derive.add_argument("--initial-artifact-s3", required=True)
    derive.add_argument("--artifact-s3")
    derive.add_argument("--attempt-id")

    inspect = subparsers.add_parser("inspect", help="Verify a committed MCore-to-HF artifact")
    inspect.add_argument("--artifact-s3", required=True)

    validate_local = subparsers.add_parser(
        "validate-local",
        help="Fully verify a pinned local trained_mtp warm-start artifact",
    )
    validate_local.add_argument("--artifact-dir", type=Path, required=True)
    validate_local.add_argument("--expected-manifest-sha256", required=True)
    validate_local.add_argument("--expected-mtp-num-layers", type=int)
    validate_local.add_argument(
        "--expected-mtp-shared-weights",
        choices=("true", "false"),
    )

    restore = subparsers.add_parser("restore", help="Restore a complete logical HF directory")
    restore.add_argument("--artifact-s3", required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    restore.add_argument("--replace-existing", action="store_true")
    restore.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-local":
            result = validate_local_trained_artifact(
                args.artifact_dir,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_mtp_num_layers=args.expected_mtp_num_layers,
                expected_mtp_shared_weights=(
                    args.expected_mtp_shared_weights == "true"
                    if args.expected_mtp_shared_weights is not None
                    else None
                ),
            )
        else:
            store = Boto3ObjectStore(profile=args.profile, region=args.region)
            if args.command == "publish":
                result = publish_artifact(
                    store,
                    args.artifact_dir,
                    args.artifact_s3,
                    attempt_id=args.attempt_id,
                )
            elif args.command == "derive-original-mtp":
                local_values = (args.trained_dir, args.output_dir)
                remote_values = (args.trained_artifact_s3, args.artifact_s3, args.attempt_id)
                if all(value is not None for value in local_values) and all(
                    value is None for value in remote_values
                ):
                    result = derive_original_mtp_local(
                        store,
                        args.trained_dir,
                        args.initial_artifact_s3,
                        args.output_dir,
                    )
                elif all(value is not None for value in remote_values) and all(
                    value is None for value in local_values
                ):
                    result = derive_original_mtp_artifact(
                        store,
                        args.trained_artifact_s3,
                        args.initial_artifact_s3,
                        args.artifact_s3,
                        attempt_id=args.attempt_id,
                    )
                else:
                    raise CheckpointError(
                        "derive-original-mtp requires either --trained-dir/--output-dir or "
                        "--trained-artifact-s3/--artifact-s3/--attempt-id"
                    )
            elif args.command == "inspect":
                result = inspect_artifact(store, args.artifact_s3)
            else:
                result = restore_artifact(
                    store,
                    args.artifact_s3,
                    args.output_dir,
                    replace_existing=args.replace_existing,
                    expected_manifest_sha256=args.expected_manifest_sha256,
                )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
