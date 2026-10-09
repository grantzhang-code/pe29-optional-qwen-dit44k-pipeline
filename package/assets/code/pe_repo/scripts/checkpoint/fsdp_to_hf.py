#!/usr/bin/env python3
"""Convert an explicitly pinned full-FSDP checkpoint to a complete HF artifact.

Only model shards are downloaded. Optimizer, scheduler, RNG, and Trainer state are
intentionally not converted. The source FSDP manifest and every selected object are
verified before use. The 15 MTP tensors omitted by the FSDP checkpoint are restored
from the integrity-pinned initial model without rewriting the 100+ GB merged tensor file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from scripts.checkpoint.common import (
    Boto3ObjectStore,
    CheckpointError,
    ObjectStore,
    S3Location,
    download_verified,
    local_path,
    parse_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_file_entries,
    verify_object_stat,
    write_json_atomic,
)


CHECKPOINT_WORLD_SIZE = 64
EXPECTED_INITIAL_TENSOR_COUNT = 1199
EXPECTED_MTP_TENSOR_COUNT = 15
CHECKPOINT_MANIFEST = "_FSDP_UPLOAD_COMPLETE.json"

INITIAL_MODEL_S3 = (
    "s3://core-cn-oss.canva.com/usr/joeyzhang/comm/design_layout_swift/models/"
    "qwen35-27b-mix3-lora-checkpoint-1600-merged-20260724-v1/"
)
INITIAL_MANIFEST = "_UPLOAD_COMPLETE.json"
INITIAL_MANIFEST_SHA256 = "2feed8f36b086e03e23c1a4d97589d5605b50f31f0c1339cd635ad75dae48968"

MTP_KEY_RE = re.compile(r"^mtp(?:\.|_)")
MODEL_SHARD_RE = re.compile(r"^pytorch_model_fsdp_0/__(\d+)_\d+\.distcp$")
OPTIMIZER_SHARD_RE = re.compile(r"^optimizer_0/__(\d+)_\d+\.distcp$")
RNG_RE = re.compile(r"^rng_state_(\d+)\.pth$")
METADATA_FILES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    filename: str
    size: int
    parameters: int


@dataclass(frozen=True, slots=True)
class WeightLayout:
    weight_map: dict[str, str]
    tensors: dict[str, TensorInfo]


class ArtifactSource:
    label: str

    def read_bytes(self, relative: str) -> bytes:
        raise NotImplementedError

    def verify(self, relative: str, record: dict[str, Any]) -> None:
        raise NotImplementedError

    def materialize(self, relative: str, destination: Path, record: dict[str, Any]) -> None:
        raise NotImplementedError

    def load_manifest(
        self,
        name: str,
        *,
        expected_sha256: str | None,
    ) -> tuple[dict[str, Any], str]:
        data = self.read_bytes(name)
        digest = sha256_bytes(data)
        if expected_sha256 is not None and digest != expected_sha256:
            raise CheckpointError(
                f"Manifest digest mismatch for {self.label}{name}: expected={expected_sha256} actual={digest}"
            )
        return parse_json_bytes(data, source=f"{self.label}{name}"), digest


class LocalArtifactSource(ArtifactSource):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.label = f"{self.root}/"

    def read_bytes(self, relative: str) -> bytes:
        path = local_path(self.root, relative)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise CheckpointError(f"Cannot read {path}: {exc}") from exc

    def verify(self, relative: str, record: dict[str, Any]) -> None:
        path = local_path(self.root, relative)
        if not path.is_file():
            raise CheckpointError(f"Missing artifact file: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record["size"] or digest != record["sha256"]:
            raise CheckpointError(
                f"Artifact file mismatch for {path}: size={size}/{record['size']} "
                f"sha256={digest}/{record['sha256']}"
            )

    def materialize(self, relative: str, destination: Path, record: dict[str, Any]) -> None:
        self.verify(relative, record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path(self.root, relative), destination)


class S3ArtifactSource(ArtifactSource):
    def __init__(self, store: ObjectStore, root_uri: str) -> None:
        self.store = store
        self.root = S3Location.parse(root_uri)
        self.label = f"{self.root.uri.rstrip('/')}/"

    def uri(self, relative: str) -> str:
        safe_relative_path(relative)
        return self.root.join(relative).uri

    def read_bytes(self, relative: str) -> bytes:
        return self.store.read_bytes(self.uri(relative))

    def verify(self, relative: str, record: dict[str, Any]) -> None:
        verify_object_stat(
            self.store,
            self.uri(relative),
            size=record["size"],
            sha256=record["sha256"],
        )

    def materialize(self, relative: str, destination: Path, record: dict[str, Any]) -> None:
        self.verify(relative, record)
        download_verified(
            self.store,
            self.uri(relative),
            destination,
            size=record["size"],
            sha256=record["sha256"],
        )

    def load_manifest(
        self,
        name: str,
        *,
        expected_sha256: str | None,
    ) -> tuple[dict[str, Any], str]:
        manifest, digest = super().load_manifest(name, expected_sha256=expected_sha256)
        stat = self.store.stat(self.uri(name))
        metadata_digest = stat.metadata.get("sha256")
        if metadata_digest != digest:
            raise CheckpointError(
                f"Manifest SHA-256 object metadata mismatch for {self.uri(name)}: "
                f"metadata={metadata_digest!r} actual={digest}"
            )
        return manifest, digest


def _rank_set(paths: Iterable[str], pattern: re.Pattern[str]) -> set[int]:
    result: set[int] = set()
    for path in paths:
        match = pattern.fullmatch(path)
        if match:
            result.add(int(match.group(1)))
    return result


def validate_fsdp_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
    expected_step: int,
    expected_world_size: int = CHECKPOINT_WORLD_SIZE,
) -> dict[str, dict[str, Any]]:
    expected = {
        "status": "complete",
        "kind": "fsdp-sharded-checkpoint",
        "step": expected_step,
        "world_size": expected_world_size,
        "rank_manifest_count": expected_world_size,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"FSDP manifest {key} mismatch in {source}: expected={value!r} actual={manifest.get(key)!r}"
            )
    entries = validate_file_entries(manifest, source=source)
    expected_ranks = set(range(expected_world_size))
    checks = {
        "model shard": _rank_set(entries, MODEL_SHARD_RE),
        "optimizer shard": _rank_set(entries, OPTIMIZER_SHARD_RE),
        "RNG state": _rank_set(entries, RNG_RE),
    }
    for label, actual_ranks in checks.items():
        if actual_ranks != expected_ranks:
            missing = sorted(expected_ranks - actual_ranks)
            extra = sorted(actual_ranks - expected_ranks)
            raise CheckpointError(
                f"Incomplete FSDP {label} coverage in {source}: missing={missing} extra={extra}"
            )
    required = {
        "pytorch_model_fsdp_0/.metadata",
        "optimizer_0/.metadata",
        "scheduler.pt",
        "trainer_state.json",
    }
    missing_required = sorted(required - entries.keys())
    if missing_required:
        raise CheckpointError(f"FSDP manifest is missing required files in {source}: {missing_required}")
    return entries


def validate_initial_manifest(manifest: dict[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    if manifest.get("status") != "complete" or manifest.get("kind") != "atomic-directory":
        raise CheckpointError(
            f"Initial artifact is not atomically committed in {source}: "
            f"status={manifest.get('status')!r} kind={manifest.get('kind')!r}"
        )
    entries = validate_file_entries(manifest, source=source)
    required = {"config.json", "model.safetensors.index.json"}
    missing = sorted(required - entries.keys())
    if missing:
        raise CheckpointError(f"Initial artifact manifest is missing {missing} in {source}")
    return entries


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected a JSON object in {path}")
    return value


def _safetensors_header(path: Path) -> dict[str, Any]:
    try:
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
    return parse_json_bytes(raw_header, source=str(path))


def _tensor_layout(filename: str, header: dict[str, Any]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise CheckpointError(f"Invalid safetensors entry {key!r} in {filename}")
        offsets = value.get("data_offsets")
        shape = value.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
        ):
            raise CheckpointError(f"Invalid tensor metadata for {key!r} in {filename}")
        parameters = 1
        for dimension in shape:
            parameters *= dimension
        tensors[key] = TensorInfo(filename, offsets[1] - offsets[0], parameters)
    return tensors


def read_weight_layout(root: Path) -> WeightLayout:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_file(index_path)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise CheckpointError(f"Invalid weight_map in {index_path}")
        weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
    else:
        candidates = sorted(root.glob("*.safetensors"))
        if not candidates:
            raise CheckpointError(f"No safetensors weights produced under {root}")
        weight_map = {}
        for candidate in candidates:
            for key in _tensor_layout(candidate.name, _safetensors_header(candidate)):
                if key in weight_map:
                    raise CheckpointError(f"Duplicate tensor {key!r} across safetensors files under {root}")
                weight_map[key] = candidate.name

    by_file: dict[str, dict[str, TensorInfo]] = {}
    for filename in sorted(set(weight_map.values())):
        relative = safe_relative_path(filename)
        if len(relative.parts) != 1:
            raise CheckpointError(f"Weight index must reference a file in its own directory: {filename!r}")
        path = root / filename
        if not path.is_file():
            raise CheckpointError(f"Weight index references missing file: {path}")
        by_file[filename] = _tensor_layout(filename, _safetensors_header(path))

    tensors: dict[str, TensorInfo] = {}
    for key, filename in weight_map.items():
        if key not in by_file[filename]:
            raise CheckpointError(f"Weight index maps {key!r} to {filename!r}, but the tensor is absent")
        tensors[key] = by_file[filename][key]
    return WeightLayout(weight_map=weight_map, tensors=tensors)


def read_weight_subset(root: Path, weight_map: dict[str, str], keys: set[str]) -> WeightLayout:
    """Read only files needed by ``keys`` while retaining all keys physically in those files."""
    selected_map = {key: weight_map[key] for key in keys}
    tensors: dict[str, TensorInfo] = {}
    for filename in sorted(set(selected_map.values())):
        relative = safe_relative_path(filename)
        if len(relative.parts) != 1:
            raise CheckpointError(f"Weight index must reference a file in its own directory: {filename!r}")
        path = root / filename
        if not path.is_file():
            raise CheckpointError(f"Weight index references missing file: {path}")
        for key, info in _tensor_layout(filename, _safetensors_header(path)).items():
            if key in tensors:
                raise CheckpointError(f"Duplicate tensor {key!r} in selected initial weight shards")
            tensors[key] = info
    missing = sorted(keys - tensors.keys())
    if missing:
        raise CheckpointError(f"Initial weight shards are missing indexed tensors: {missing}")
    return WeightLayout(weight_map=selected_map, tensors=tensors)


def _model_records(entries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {path: item for path, item in entries.items() if path.startswith("pytorch_model_fsdp_0/")}


def inspect_sources(
    checkpoint_source: ArtifactSource,
    initial_source: ArtifactSource,
    *,
    checkpoint_manifest_sha256: str | None,
    initial_manifest_sha256: str | None = INITIAL_MANIFEST_SHA256,
    expected_step: int,
    expected_world_size: int = CHECKPOINT_WORLD_SIZE,
    expected_tensor_count: int = EXPECTED_INITIAL_TENSOR_COUNT,
    expected_mtp_tensor_count: int = EXPECTED_MTP_TENSOR_COUNT,
) -> dict[str, Any]:
    checkpoint_manifest, checkpoint_digest = checkpoint_source.load_manifest(
        CHECKPOINT_MANIFEST, expected_sha256=checkpoint_manifest_sha256
    )
    checkpoint_entries = validate_fsdp_manifest(
        checkpoint_manifest,
        source=f"{checkpoint_source.label}{CHECKPOINT_MANIFEST}",
        expected_step=expected_step,
        expected_world_size=expected_world_size,
    )
    model_records = _model_records(checkpoint_entries)
    for path, record in model_records.items():
        checkpoint_source.verify(path, record)

    initial_manifest, initial_digest = initial_source.load_manifest(
        INITIAL_MANIFEST, expected_sha256=initial_manifest_sha256
    )
    initial_entries = validate_initial_manifest(
        initial_manifest, source=f"{initial_source.label}{INITIAL_MANIFEST}"
    )
    index_record = initial_entries["model.safetensors.index.json"]
    initial_source.verify("model.safetensors.index.json", index_record)
    initial_index = parse_json_bytes(
        initial_source.read_bytes("model.safetensors.index.json"),
        source=f"{initial_source.label}model.safetensors.index.json",
    )
    weight_map = initial_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("Initial model index has no weight_map")
    if len(weight_map) != expected_tensor_count:
        raise CheckpointError(
            f"Initial tensor count mismatch: expected={expected_tensor_count} actual={len(weight_map)}"
        )
    mtp_keys = sorted(key for key in weight_map if isinstance(key, str) and MTP_KEY_RE.search(key))
    if len(mtp_keys) != expected_mtp_tensor_count:
        raise CheckpointError(
            f"Initial MTP tensor count mismatch: expected={expected_mtp_tensor_count} actual={len(mtp_keys)}"
        )
    mtp_files = sorted({str(weight_map[key]) for key in mtp_keys})
    selected_initial = sorted(
        (METADATA_FILES & initial_entries.keys()) | {"model.safetensors.index.json", *mtp_files}
    )
    for path in selected_initial:
        initial_source.verify(path, initial_entries[path])

    return {
        "checkpoint": {
            "source": checkpoint_source.label,
            "manifest_sha256": checkpoint_digest,
            "step": expected_step,
            "world_size": expected_world_size,
            "selected_file_count": len(model_records),
            "selected_bytes": sum(record["size"] for record in model_records.values()),
        },
        "initial_model": {
            "source": initial_source.label,
            "manifest_sha256": initial_digest,
            "mtp_tensor_count": len(mtp_keys),
            "mtp_files": mtp_files,
            "selected_files": selected_initial,
        },
        "conversion": "model-only; optimizer, scheduler, RNG, and global step are not converted",
    }


def _default_merger(source: Path, destination: Path) -> None:
    try:
        from accelerate.utils.fsdp_utils import merge_fsdp_weights
    except ImportError as exc:  # pragma: no cover - production dependency
        raise CheckpointError("accelerate is required to merge FSDP distributed checkpoints") from exc
    merge_fsdp_weights(str(source), str(destination), safe_serialization=True)


def _copy_initial_files(
    initial_source: ArtifactSource,
    initial_entries: dict[str, dict[str, Any]],
    relative_paths: Iterable[str],
    destination: Path,
) -> None:
    for relative in relative_paths:
        record = initial_entries.get(relative)
        if record is None:
            raise CheckpointError(f"Initial manifest does not contain required file {relative!r}")
        initial_source.materialize(relative, local_path(destination, relative), record)


def _artifact_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"_SUCCESS", "conversion_manifest.json"}:
            continue
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def convert_checkpoint(
    checkpoint_source: ArtifactSource,
    initial_source: ArtifactSource,
    output_dir: Path,
    *,
    checkpoint_manifest_sha256: str | None,
    initial_manifest_sha256: str | None = INITIAL_MANIFEST_SHA256,
    expected_step: int,
    expected_world_size: int = CHECKPOINT_WORLD_SIZE,
    expected_tensor_count: int = EXPECTED_INITIAL_TENSOR_COUNT,
    expected_mtp_tensor_count: int = EXPECTED_MTP_TENSOR_COUNT,
    merger: Callable[[Path, Path], None] = _default_merger,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.work-", dir=output_dir.parent))
    dcp_dir = work_dir / "dcp"
    initial_dir = work_dir / "initial"
    hf_dir = work_dir / "hf"
    try:
        checkpoint_manifest, checkpoint_digest = checkpoint_source.load_manifest(
            CHECKPOINT_MANIFEST, expected_sha256=checkpoint_manifest_sha256
        )
        checkpoint_entries = validate_fsdp_manifest(
            checkpoint_manifest,
            source=f"{checkpoint_source.label}{CHECKPOINT_MANIFEST}",
            expected_step=expected_step,
            expected_world_size=expected_world_size,
        )
        for relative, record in _model_records(checkpoint_entries).items():
            checkpoint_source.materialize(relative, local_path(dcp_dir, relative), record)

        initial_manifest, initial_digest = initial_source.load_manifest(
            INITIAL_MANIFEST, expected_sha256=initial_manifest_sha256
        )
        initial_entries = validate_initial_manifest(
            initial_manifest, source=f"{initial_source.label}{INITIAL_MANIFEST}"
        )
        _copy_initial_files(
            initial_source,
            initial_entries,
            ["model.safetensors.index.json"],
            initial_dir,
        )
        initial_index = _read_json_file(initial_dir / "model.safetensors.index.json")
        raw_initial_map = initial_index.get("weight_map")
        if not isinstance(raw_initial_map, dict) or not raw_initial_map:
            raise CheckpointError("Initial model index has no weight_map")
        initial_map = {str(key): str(value) for key, value in raw_initial_map.items()}
        if len(initial_map) != expected_tensor_count:
            raise CheckpointError(
                f"Initial tensor count mismatch: expected={expected_tensor_count} actual={len(initial_map)}"
            )
        mtp_keys = {key for key in initial_map if MTP_KEY_RE.search(key)}
        if len(mtp_keys) != expected_mtp_tensor_count:
            raise CheckpointError(
                f"Initial MTP tensor count mismatch: expected={expected_mtp_tensor_count} "
                f"actual={len(mtp_keys)}"
            )
        mtp_source_files = {initial_map[key] for key in mtp_keys}
        _copy_initial_files(initial_source, initial_entries, mtp_source_files, initial_dir)
        initial_layout = read_weight_subset(initial_dir, initial_map, mtp_keys)

        merger(dcp_dir / "pytorch_model_fsdp_0", hf_dir)
        merged_layout = read_weight_layout(hf_dir)
        initial_non_mtp = set(initial_map) - mtp_keys
        merged_mtp = {key for key in merged_layout.weight_map if MTP_KEY_RE.search(key)}
        merged_non_mtp = set(merged_layout.weight_map) - merged_mtp
        missing_non_mtp = sorted(initial_non_mtp - merged_non_mtp)
        unexpected_non_mtp = sorted(merged_non_mtp - initial_non_mtp)
        if missing_non_mtp or unexpected_non_mtp:
            raise CheckpointError(
                "FSDP/HF non-MTP key coverage mismatch: "
                f"missing={missing_non_mtp[:20]} unexpected={unexpected_non_mtp[:20]}"
            )
        if merged_mtp and merged_mtp != mtp_keys:
            raise CheckpointError(
                f"FSDP merge contains only a partial MTP key set: present={sorted(merged_mtp)} "
                f"expected={sorted(mtp_keys)}"
            )

        final_map = dict(merged_layout.weight_map)
        copied_mtp_files: list[str] = []
        if not merged_mtp:
            for index, source_filename in enumerate(sorted(mtp_source_files), start=1):
                mapped_keys = {key for key, filename in initial_map.items() if filename == source_filename}
                actual_keys = {
                    key
                    for key, tensor in initial_layout.tensors.items()
                    if tensor.filename == source_filename
                }
                if mapped_keys != actual_keys or not mapped_keys <= mtp_keys:
                    raise CheckpointError(
                        f"Cannot safely reuse initial shard {source_filename!r}: it contains non-MTP or unindexed tensors"
                    )
                destination_name = f"model-initial-mtp-{index:05d}-of-{len(mtp_source_files):05d}.safetensors"
                shutil.copyfile(initial_dir / source_filename, hf_dir / destination_name)
                copied_mtp_files.append(destination_name)
                for key in mapped_keys:
                    final_map[key] = destination_name

        for metadata_name in sorted(METADATA_FILES & initial_entries.keys()):
            initial_source.materialize(metadata_name, hf_dir / metadata_name, initial_entries[metadata_name])

        tensor_info = dict(merged_layout.tensors)
        if copied_mtp_files:
            for key in mtp_keys:
                source_info = initial_layout.tensors[key]
                tensor_info[key] = TensorInfo(final_map[key], source_info.size, source_info.parameters)
        index = {
            "metadata": {
                "total_parameters": sum(tensor_info[key].parameters for key in final_map),
                "total_size": sum(tensor_info[key].size for key in final_map),
            },
            "weight_map": dict(sorted(final_map.items())),
        }
        write_json_atomic(hf_dir / "model.safetensors.index.json", index)
        final_layout = read_weight_layout(hf_dir)
        missing_final = sorted(set(initial_map) - set(final_layout.weight_map))
        unexpected_final = sorted(set(final_layout.weight_map) - set(initial_map))
        if missing_final or unexpected_final:
            raise CheckpointError(
                f"Final HF key coverage mismatch: missing={missing_final[:20]} unexpected={unexpected_final[:20]}"
            )

        records = _artifact_file_records(hf_dir)
        conversion_manifest = {
            "version": 1,
            "kind": "fsdp-model-only-to-hf",
            "status": "complete",
            "created_at": utc_now(),
            "source_checkpoint": checkpoint_source.label,
            "source_checkpoint_manifest_sha256": checkpoint_digest,
            "source_step": expected_step,
            "source_world_size": expected_world_size,
            "initial_model": initial_source.label,
            "initial_model_manifest_sha256": initial_digest,
            "optimizer_scheduler_rng_global_step_restored": False,
            "key_coverage": {
                "expected": len(initial_map),
                "merged_non_mtp": len(merged_non_mtp),
                "mtp": len(mtp_keys),
                "final": len(final_layout.weight_map),
                "copied_mtp_files": copied_mtp_files,
            },
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
            "files": records,
        }
        write_json_atomic(hf_dir / "conversion_manifest.json", conversion_manifest)
        manifest_digest = sha256_file(hf_dir / "conversion_manifest.json")
        write_json_atomic(
            hf_dir / "_SUCCESS",
            {
                "version": 1,
                "kind": "fsdp-model-only-to-hf-commit",
                "status": "complete",
                "source_step": expected_step,
                "manifest": "conversion_manifest.json",
                "manifest_sha256": manifest_digest,
            },
        )
        os.replace(hf_dir, output_dir)
        return conversion_manifest
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _remote_sources(
    store: ObjectStore,
    *,
    checkpoint_s3: str,
    initial_model_s3: str,
) -> tuple[S3ArtifactSource, S3ArtifactSource]:
    return S3ArtifactSource(store, checkpoint_s3), S3ArtifactSource(store, initial_model_s3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Optional AWS profile; pod IRSA is used when omitted")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--checkpoint-s3",
        required=True,
        help="Exact checkpoint-<step>/ S3 URI; deliberately has no floating default",
    )
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        required=True,
        help="SHA-256 of the source _FSDP_UPLOAD_COMPLETE.json",
    )
    parser.add_argument("--checkpoint-world-size", type=int, default=CHECKPOINT_WORLD_SIZE)
    parser.add_argument("--initial-model-s3", default=INITIAL_MODEL_S3)
    parser.add_argument("--initial-manifest-sha256", default=INITIAL_MANIFEST_SHA256)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "inspect", help="Validate pinned manifests and object metadata without large downloads"
    )
    convert = subparsers.add_parser(
        "convert", help="Download model shards and build the committed HF artifact"
    )
    convert.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = Boto3ObjectStore(profile=args.profile, region=args.region)
        checkpoint_source, initial_source = _remote_sources(
            store,
            checkpoint_s3=args.checkpoint_s3,
            initial_model_s3=args.initial_model_s3,
        )
        if args.command == "inspect":
            result = inspect_sources(
                checkpoint_source,
                initial_source,
                checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
                initial_manifest_sha256=args.initial_manifest_sha256,
                expected_step=args.checkpoint_step,
                expected_world_size=args.checkpoint_world_size,
            )
        else:
            result = convert_checkpoint(
                checkpoint_source,
                initial_source,
                args.output_dir,
                checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
                initial_manifest_sha256=args.initial_manifest_sha256,
                expected_step=args.checkpoint_step,
                expected_world_size=args.checkpoint_world_size,
            )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
