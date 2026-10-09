#!/usr/bin/env python3
"""Build a committed, offline-restorable artifact from a pinned HF snapshot.

This command never contacts Hugging Face or S3.  Its input must be a clean,
fully materialized model directory for an exact 40-hex Hugging Face revision.
The output uses the same local commit marker consumed by ``hf_artifact_s3``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from scripts.checkpoint.common import (
    CheckpointError,
    canonical_json_bytes,
    parse_json_bytes,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.checkpoint.hf_mtp_graft import MTP_KEY_RE, IndexedLayout, read_indexed_layout


MANIFEST_NAME = "conversion_manifest.json"
SUCCESS_NAME = "_SUCCESS"
MANIFEST_KIND = "hf-base-snapshot"
COMMIT_KIND = "hf-model-artifact-commit"
INDEX_NAME = "model.safetensors.index.json"

# A base snapshot is deliberately data-only.  In particular, arbitrary Python
# remote-code files and nested repositories are not admitted into the artifact.
ALLOWED_AUXILIARY_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
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
REQUIRED_RUNTIME_FILES = {
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
HF_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
HF_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes(), source=str(path))
    except OSError as exc:
        raise CheckpointError(f"Cannot read {path}: {exc}") from exc


def _layout_digest(layout: IndexedLayout, keys: Iterable[str]) -> str:
    value = {
        key: {
            "dtype": layout.tensors[key].dtype,
            "shape": list(layout.tensors[key].shape),
        }
        for key in sorted(keys)
    }
    return sha256_bytes(canonical_json_bytes(value))


def _validate_hf_source(
    source_dir: Path, *, allow_commit_files: bool = False
) -> tuple[Path, IndexedLayout, set[str]]:
    if source_dir.is_symlink():
        raise CheckpointError(f"HF source directory must not be a symlink: {source_dir}")
    if not source_dir.is_dir():
        raise CheckpointError(f"HF source directory does not exist: {source_dir}")
    root = source_dir.resolve()
    actual_files: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(
                f"HF source must contain only top-level regular non-symlink files: {path}"
            )
        actual_files.add(path.name)

    model_files = actual_files - ({MANIFEST_NAME, SUCCESS_NAME} if allow_commit_files else set())

    if INDEX_NAME not in model_files:
        raise CheckpointError(f"HF source is missing required tensor index: {root / INDEX_NAME}")
    missing_runtime = REQUIRED_RUNTIME_FILES - model_files
    if missing_runtime:
        raise CheckpointError(
            f"HF source is missing required runtime files: {sorted(missing_runtime)}"
        )

    layout = read_indexed_layout(root)
    shard_files = set(layout.weight_map.values())
    expected_files = {INDEX_NAME, *shard_files} | (model_files & ALLOWED_AUXILIARY_FILES)
    unexpected = model_files - expected_files
    if unexpected:
        raise CheckpointError(
            "HF source contains unindexed or unsafe extra files: " f"{sorted(unexpected)}"
        )
    missing_shards = shard_files - model_files
    if missing_shards:
        raise CheckpointError(f"HF source is missing indexed weight shards: {sorted(missing_shards)}")

    index = _read_json(root / INDEX_NAME)
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointError(f"HF tensor index has no metadata object: {root / INDEX_NAME}")
    tensor_bytes = sum(ref.size for ref in layout.tensors.values())
    if metadata.get("total_size") != tensor_bytes:
        raise CheckpointError(
            "HF tensor index total_size mismatch: "
            f"index={metadata.get('total_size')!r} tensors={tensor_bytes}"
        )
    total_parameters = metadata.get("total_parameters")
    if total_parameters is not None:
        actual_parameters = sum(ref.parameters for ref in layout.tensors.values())
        if total_parameters != actual_parameters:
            raise CheckpointError(
                "HF tensor index total_parameters mismatch: "
                f"index={total_parameters!r} tensors={actual_parameters}"
            )
    return root, layout, expected_files


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.name in {MANIFEST_NAME, SUCCESS_NAME}:
            continue
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"HF artifact contains a non-regular entry: {path}")
        records.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def validate_base_snapshot_layout(
    root: Path,
    manifest: dict[str, Any],
    *,
    verified_entries: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Recompute all structural claims in a base-snapshot manifest."""

    validated_root, layout, actual_files = _validate_hf_source(root, allow_commit_files=True)
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise CheckpointError("Base snapshot manifest is missing files")
    recorded_files = {
        record.get("path") for record in raw_entries if isinstance(record, dict)
    }
    if recorded_files != actual_files:
        raise CheckpointError(
            "Base snapshot file coverage mismatch: "
            f"missing={sorted(actual_files - recorded_files)} "
            f"unexpected={sorted(recorded_files - actual_files)}"
        )

    actual_index_sha = sha256_file(validated_root / INDEX_NAME)
    if manifest.get("source_model_index_sha256") != actual_index_sha:
        raise CheckpointError(
            "Base snapshot tensor index digest mismatch: "
            f"manifest={manifest.get('source_model_index_sha256')!r} actual={actual_index_sha}"
        )

    mtp_keys = {key for key in layout.tensors if MTP_KEY_RE.search(key)}
    backbone_keys = set(layout.tensors) - mtp_keys
    if not mtp_keys:
        raise CheckpointError("Base snapshot contains no MTP tensors")
    contract = manifest.get("model_contract")
    if not isinstance(contract, dict):
        raise CheckpointError("Base snapshot manifest is missing model_contract")
    checks = {
        "expected_tensor_count": len(layout.tensors),
        "expected_backbone_tensor_count": len(backbone_keys),
        "expected_mtp_tensor_count": len(mtp_keys),
        "tensor_layout_sha256": _layout_digest(layout, layout.tensors),
        "backbone_layout_sha256": _layout_digest(layout, backbone_keys),
        "mtp_layout_sha256": _layout_digest(layout, mtp_keys),
    }
    for field, actual in checks.items():
        if contract.get(field) != actual:
            raise CheckpointError(
                f"Base snapshot {field} mismatch: "
                f"manifest={contract.get(field)!r} actual={actual!r}"
            )

    shard_digests = manifest.get("weight_shard_sha256")
    shard_files = set(layout.weight_map.values())
    if not isinstance(shard_digests, dict) or set(shard_digests) != shard_files:
        raise CheckpointError("Base snapshot manifest has incomplete weight-shard digests")
    for filename in sorted(shard_files):
        if verified_entries is None:
            actual = sha256_file(validated_root / filename)
        else:
            record = verified_entries.get(filename)
            actual = record.get("sha256") if isinstance(record, dict) else None
        if shard_digests[filename] != actual:
            raise CheckpointError(
                f"Base snapshot weight shard digest mismatch for {filename!r}: "
                f"manifest={shard_digests[filename]!r} actual={actual}"
            )


def build_base_snapshot(
    source_dir: Path,
    output_dir: Path,
    *,
    source_hf_repository: str,
    source_hf_revision: str,
    expected_tensor_count: int,
    expected_mtp_tensor_count: int,
) -> dict[str, Any]:
    """Copy and atomically commit one exact, data-only HF base snapshot."""

    if HF_REPOSITORY_RE.fullmatch(source_hf_repository) is None:
        raise CheckpointError(f"Invalid Hugging Face repository: {source_hf_repository!r}")
    if HF_REVISION_RE.fullmatch(source_hf_revision) is None:
        raise CheckpointError(
            "Hugging Face revision must be an exact lowercase 40-hex commit: "
            f"{source_hf_revision!r}"
        )
    if expected_tensor_count <= 0:
        raise CheckpointError(f"expected_tensor_count must be positive: {expected_tensor_count}")
    if expected_mtp_tensor_count <= 0:
        raise CheckpointError(
            f"expected_mtp_tensor_count must be positive: {expected_mtp_tensor_count}"
        )
    if output_dir.is_symlink() or output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite existing output directory: {output_dir}")

    source_root, source_layout, source_files = _validate_hf_source(source_dir)
    output_dir = output_dir.resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise CheckpointError("Output directory must not be inside the HF source directory")

    mtp_keys = {key for key in source_layout.tensors if MTP_KEY_RE.search(key)}
    backbone_keys = set(source_layout.tensors) - mtp_keys
    if len(source_layout.tensors) != expected_tensor_count:
        raise CheckpointError(
            "Base snapshot tensor count mismatch: "
            f"expected={expected_tensor_count} actual={len(source_layout.tensors)}"
        )
    if len(mtp_keys) != expected_mtp_tensor_count:
        raise CheckpointError(
            "Base snapshot MTP tensor count mismatch: "
            f"expected={expected_mtp_tensor_count} actual={len(mtp_keys)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.base-", dir=output_dir.parent))
    try:
        for filename in sorted(source_files):
            shutil.copyfile(source_root / filename, work / filename)
        copied_layout = read_indexed_layout(work)
        if set(copied_layout.tensors) != set(source_layout.tensors):
            raise CheckpointError("Copied base snapshot tensor keys changed during materialization")

        records = _artifact_records(work)
        record_sha256 = {str(record["path"]): str(record["sha256"]) for record in records}
        manifest = {
            "version": 1,
            "kind": MANIFEST_KIND,
            "status": "complete",
            "created_at": utc_now(),
            "source_step": 0,
            "source_hf_repository": source_hf_repository,
            "source_hf_revision": source_hf_revision,
            "source_model_index_sha256": sha256_file(work / INDEX_NAME),
            "optimizer_scheduler_rng_global_step_restored": False,
            "model_contract": {
                "expected_tensor_count": len(copied_layout.tensors),
                "expected_backbone_tensor_count": len(backbone_keys),
                "expected_mtp_tensor_count": len(mtp_keys),
                "tensor_layout_sha256": _layout_digest(copied_layout, copied_layout.tensors),
                "backbone_layout_sha256": _layout_digest(copied_layout, backbone_keys),
                "mtp_layout_sha256": _layout_digest(copied_layout, mtp_keys),
            },
            "key_coverage": {
                "expected": expected_tensor_count,
                "backbone": len(backbone_keys),
                "mtp": len(mtp_keys),
                "final": len(copied_layout.tensors),
            },
            "weight_shard_sha256": {
                filename: record_sha256[filename]
                for filename in sorted(set(copied_layout.weight_map.values()))
            },
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
            "files": records,
        }
        write_json_atomic(work / MANIFEST_NAME, manifest)
        write_json_atomic(
            work / SUCCESS_NAME,
            {
                "version": 1,
                "kind": COMMIT_KIND,
                "status": "complete",
                "source_step": 0,
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(work / MANIFEST_NAME),
            },
        )

        from scripts.checkpoint.hf_artifact_s3 import validate_local_artifact

        validate_local_artifact(work)
        os.replace(work, output_dir)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-hf-repository", required=True)
    parser.add_argument("--source-hf-revision", required=True)
    parser.add_argument("--expected-tensor-count", type=int, required=True)
    parser.add_argument("--expected-mtp-tensor-count", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_base_snapshot(
            args.source_model_dir,
            args.output_dir,
            source_hf_repository=args.source_hf_repository,
            source_hf_revision=args.source_hf_revision,
            expected_tensor_count=args.expected_tensor_count,
            expected_mtp_tensor_count=args.expected_mtp_tensor_count,
        )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
