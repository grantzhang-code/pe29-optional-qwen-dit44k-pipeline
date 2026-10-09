#!/usr/bin/env python3
"""Build a committed HF artifact from a trained backbone plus an original MTP head.

The trained HF checkpoint must contain every non-MTP tensor and no ``mtp.*``
tensor.  The initial HF model supplies exactly the missing MTP tensors.  MTP
tensors are extracted by byte range even when their source safetensors shards
also contain ordinary model weights; no initial backbone tensor is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from scripts.checkpoint.common import (
    CheckpointError,
    canonical_json_bytes,
    local_path,
    parse_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.checkpoint.fsdp_to_hf import METADATA_FILES


MANIFEST_NAME = "conversion_manifest.json"
SUCCESS_NAME = "_SUCCESS"
MANIFEST_KIND = "hf-mtp-graft"
COMMIT_KIND = "hf-model-artifact-commit"
MTP_KEY_RE = re.compile(r"^mtp(?:\.|_)")
MTP_FILENAME = "model-original-mtp.safetensors"

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True, slots=True)
class TensorRef:
    key: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    offset_start: int
    offset_end: int
    data_start: int

    @property
    def size(self) -> int:
        return self.offset_end - self.offset_start

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class IndexedLayout:
    weight_map: dict[str, str]
    tensors: dict[str, TensorRef]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = parse_json_bytes(path.read_bytes(), source=str(path))
    except OSError as exc:
        raise CheckpointError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected JSON object in {path}")
    return value


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointError(f"Safetensors shard must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CheckpointError(f"Truncated safetensors prefix in {path}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > 512 * 1024 * 1024:
                raise CheckpointError(f"Invalid safetensors header length in {path}: {header_length}")
            raw_header = handle.read(header_length)
    except OSError as exc:
        raise CheckpointError(f"Cannot read safetensors header {path}: {exc}") from exc
    if len(raw_header) != header_length:
        raise CheckpointError(f"Truncated safetensors header in {path}")
    header = parse_json_bytes(raw_header, source=str(path))
    if not isinstance(header, dict):
        raise CheckpointError(f"Safetensors header is not an object in {path}")
    return header, 8 + header_length


def read_indexed_layout(root: Path) -> IndexedLayout:
    root = root.resolve()
    index_path = root / "model.safetensors.index.json"
    index = _read_json(index_path)
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise CheckpointError(f"Missing weight_map in {index_path}")
    weight_map: dict[str, str] = {}
    for raw_key, raw_filename in raw_map.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise CheckpointError(f"Invalid tensor key in {index_path}: {raw_key!r}")
        if not isinstance(raw_filename, str):
            raise CheckpointError(f"Invalid shard for {raw_key!r} in {index_path}")
        relative = safe_relative_path(raw_filename)
        if len(relative.parts) != 1 or relative.suffix != ".safetensors":
            raise CheckpointError(f"Weight shard must be a top-level .safetensors file: {raw_filename!r}")
        weight_map[raw_key] = raw_filename

    tensors: dict[str, TensorRef] = {}
    for filename in sorted(set(weight_map.values())):
        path = local_path(root, filename)
        header, data_start = _read_header(path)
        physical: dict[str, TensorRef] = {}
        intervals: list[tuple[int, int, str]] = []
        for key, raw in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(key, str) or not isinstance(raw, dict):
                raise CheckpointError(f"Invalid tensor entry in {path}: {key!r}")
            dtype = raw.get("dtype")
            shape = raw.get("shape")
            offsets = raw.get("data_offsets")
            if dtype not in _DTYPE_BYTES:
                raise CheckpointError(f"Unsupported dtype for {key!r} in {path}: {dtype!r}")
            if not isinstance(shape, list) or not all(
                isinstance(value, int) and value >= 0 for value in shape
            ):
                raise CheckpointError(f"Invalid shape for {key!r} in {path}: {shape!r}")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(
                isinstance(value, int) for value in offsets
            ):
                raise CheckpointError(f"Invalid offsets for {key!r} in {path}: {offsets!r}")
            start, end = offsets
            if start < 0 or end < start:
                raise CheckpointError(f"Invalid offsets for {key!r} in {path}: {offsets!r}")
            expected_size = math.prod(shape) * _DTYPE_BYTES[dtype]
            if end - start != expected_size:
                raise CheckpointError(
                    f"Tensor byte-size mismatch for {key!r} in {path}: "
                    f"offsets={end - start} dtype/shape={expected_size}"
                )
            ref = TensorRef(key, filename, dtype, tuple(shape), start, end, data_start)
            physical[key] = ref
            intervals.append((start, end, key))

        indexed_keys = {key for key, mapped in weight_map.items() if mapped == filename}
        if set(physical) != indexed_keys:
            raise CheckpointError(
                f"Index/header key mismatch for {path}: "
                f"missing={sorted(indexed_keys - set(physical))[:20]} "
                f"unindexed={sorted(set(physical) - indexed_keys)[:20]}"
            )
        cursor = 0
        for start, end, key in sorted(intervals):
            if start != cursor:
                raise CheckpointError(
                    f"Safetensors payload is not contiguous in {path}: {key!r} starts at {start}, expected {cursor}"
                )
            cursor = end
        payload_size = path.stat().st_size - data_start
        if cursor != payload_size:
            raise CheckpointError(
                f"Safetensors payload size mismatch in {path}: header={cursor} file={payload_size}"
            )
        tensors.update(physical)
    if set(tensors) != set(weight_map):
        raise CheckpointError(f"Incomplete tensor layout under {root}")
    return IndexedLayout(weight_map=weight_map, tensors=tensors)


def _layout_value(tensors: dict[str, TensorRef], keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": tensors[key].dtype, "shape": list(tensors[key].shape)}
        for key in sorted(keys)
    }


def _layout_digest(tensors: dict[str, TensorRef], keys: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(_layout_value(tensors, keys)))


def _tensor_digest(root: Path, ref: TensorRef) -> str:
    digest = hashlib.sha256()
    with local_path(root, ref.filename).open("rb") as handle:
        handle.seek(ref.data_start + ref.offset_start)
        remaining = ref.size
        while remaining:
            chunk = handle.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise CheckpointError(f"Unexpected EOF while hashing tensor {ref.key!r}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _copy_exact(source: BinaryIO, destination: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise CheckpointError(f"Unexpected EOF while extracting {size} bytes")
        destination.write(chunk)
        remaining -= len(chunk)


def extract_tensors(
    source_root: Path,
    layout: IndexedLayout,
    keys: Iterable[str],
    destination: Path,
) -> None:
    ordered = sorted(keys)
    if not ordered:
        raise CheckpointError("Refusing to create an empty safetensors shard")
    offset = 0
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    for key in ordered:
        ref = layout.tensors[key]
        header[key] = {
            "dtype": ref.dtype,
            "shape": list(ref.shape),
            "data_offsets": [offset, offset + ref.size],
        }
        offset += ref.size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        handles: dict[str, BinaryIO] = {}
        try:
            for key in ordered:
                ref = layout.tensors[key]
                handle = handles.get(ref.filename)
                if handle is None:
                    handle = local_path(source_root, ref.filename).open("rb")
                    handles[ref.filename] = handle
                handle.seek(ref.data_start + ref.offset_start)
                _copy_exact(handle, output, ref.size)
        finally:
            for handle in handles.values():
                handle.close()


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.name in {MANIFEST_NAME, SUCCESS_NAME}:
            continue
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"HF artifact must contain only top-level regular files: {path}")
        records.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def _compare_layout(
    actual: IndexedLayout,
    expected: dict[str, TensorRef],
    *,
    source: str,
) -> None:
    if set(actual.tensors) != set(expected):
        raise CheckpointError(
            f"Final key mismatch in {source}: missing={sorted(set(expected) - set(actual.tensors))[:20]} "
            f"unexpected={sorted(set(actual.tensors) - set(expected))[:20]}"
        )
    mismatches = [
        key
        for key, ref in expected.items()
        if (actual.tensors[key].dtype, actual.tensors[key].shape) != (ref.dtype, ref.shape)
    ]
    if mismatches:
        raise CheckpointError(f"Final dtype/shape mismatch in {source}: {mismatches[:20]}")


def validate_graft_layout(root: Path, manifest: dict[str, Any]) -> None:
    """Recompute the exact tensor contract recorded by a graft manifest."""
    layout = read_indexed_layout(root)
    mtp_keys = {key for key in layout.tensors if MTP_KEY_RE.search(key)}
    backbone_keys = set(layout.tensors) - mtp_keys
    contract = manifest.get("model_contract")
    if not isinstance(contract, dict):
        raise CheckpointError("Graft manifest is missing model_contract")
    checks = {
        "expected_tensor_count": len(layout.tensors),
        "expected_backbone_tensor_count": len(backbone_keys),
        "expected_mtp_tensor_count": len(mtp_keys),
        "tensor_layout_sha256": _layout_digest(layout.tensors, layout.tensors),
        "backbone_layout_sha256": _layout_digest(layout.tensors, backbone_keys),
        "mtp_layout_sha256": _layout_digest(layout.tensors, mtp_keys),
    }
    for field, actual in checks.items():
        if contract.get(field) != actual:
            raise CheckpointError(
                f"Grafted artifact {field} mismatch: manifest={contract.get(field)!r} actual={actual!r}"
            )
    extracted = manifest.get("key_coverage", {}).get("extracted_mtp_file")
    if extracted != MTP_FILENAME or {layout.weight_map[key] for key in mtp_keys} != {MTP_FILENAME}:
        raise CheckpointError("Every MTP key must resolve to the dedicated extracted MTP shard")
    if any(layout.weight_map[key] == MTP_FILENAME for key in backbone_keys):
        raise CheckpointError("The extracted MTP shard contains a backbone key")
    expected_mtp_digests = manifest.get("mtp_tensor_sha256")
    if not isinstance(expected_mtp_digests, dict) or set(expected_mtp_digests) != mtp_keys:
        raise CheckpointError("Graft manifest has incomplete MTP tensor digests")
    for key in sorted(mtp_keys):
        actual = _tensor_digest(root, layout.tensors[key])
        if expected_mtp_digests[key] != actual:
            raise CheckpointError(
                f"Grafted MTP tensor digest mismatch for {key!r}: "
                f"manifest={expected_mtp_digests[key]!r} actual={actual}"
            )
    expected_backbone_files = manifest.get("trained_backbone_file_sha256")
    backbone_files = {layout.weight_map[key] for key in backbone_keys}
    if not isinstance(expected_backbone_files, dict) or set(expected_backbone_files) != backbone_files:
        raise CheckpointError("Graft manifest has incomplete trained-backbone file digests")
    for filename in sorted(backbone_files):
        actual = sha256_file(local_path(root, filename))
        if expected_backbone_files[filename] != actual:
            raise CheckpointError(
                f"Grafted backbone shard digest mismatch for {filename!r}: "
                f"manifest={expected_backbone_files[filename]!r} actual={actual}"
            )


def graft_mtp_head(
    trained_root: Path,
    initial_root: Path,
    output_dir: Path,
    *,
    source_step: int,
    expected_tensor_count: int,
    expected_mtp_tensor_count: int,
    expected_backbone_dtype_differences: int | None = None,
) -> dict[str, Any]:
    if source_step <= 0:
        raise CheckpointError(f"source_step must be positive, got {source_step}")
    output_dir = output_dir.resolve()
    trained_root = trained_root.resolve()
    initial_root = initial_root.resolve()
    if output_dir.exists():
        raise CheckpointError(f"Refusing to overwrite existing output directory: {output_dir}")
    if trained_root == initial_root:
        raise CheckpointError("Trained and initial model directories must be different")

    trained = read_indexed_layout(trained_root)
    initial = read_indexed_layout(initial_root)
    initial_mtp = {key for key in initial.tensors if MTP_KEY_RE.search(key)}
    trained_mtp = {key for key in trained.tensors if MTP_KEY_RE.search(key)}
    if len(initial.tensors) != expected_tensor_count:
        raise CheckpointError(
            f"Initial tensor count mismatch: expected={expected_tensor_count} actual={len(initial.tensors)}"
        )
    if len(initial_mtp) != expected_mtp_tensor_count:
        raise CheckpointError(
            f"Initial MTP tensor count mismatch: expected={expected_mtp_tensor_count} actual={len(initial_mtp)}"
        )
    if trained_mtp:
        raise CheckpointError(f"Trained checkpoint unexpectedly contains MTP tensors: {sorted(trained_mtp)}")
    expected_backbone = set(initial.tensors) - initial_mtp
    if set(trained.tensors) != expected_backbone:
        raise CheckpointError(
            "Trained/initial backbone key mismatch: "
            f"missing={sorted(expected_backbone - set(trained.tensors))[:20]} "
            f"unexpected={sorted(set(trained.tensors) - expected_backbone)[:20]}"
        )
    shape_mismatches = [
        key for key in sorted(expected_backbone)
        if trained.tensors[key].shape != initial.tensors[key].shape
    ]
    if shape_mismatches:
        raise CheckpointError(f"Trained/initial backbone shape mismatch: {shape_mismatches[:20]}")
    dtype_differences = [
        {
            "key": key,
            "trained_dtype": trained.tensors[key].dtype,
            "initial_dtype": initial.tensors[key].dtype,
        }
        for key in sorted(expected_backbone)
        if trained.tensors[key].dtype != initial.tensors[key].dtype
    ]
    if (
        expected_backbone_dtype_differences is not None
        and len(dtype_differences) != expected_backbone_dtype_differences
    ):
        raise CheckpointError(
            "Backbone dtype-difference count mismatch: "
            f"expected={expected_backbone_dtype_differences} actual={len(dtype_differences)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.graft-", dir=output_dir.parent))
    try:
        for filename in sorted(set(trained.weight_map.values())):
            shutil.copy2(local_path(trained_root, filename), work / filename)
        for metadata_name in sorted(METADATA_FILES):
            source = trained_root / metadata_name
            if source.is_file() and not source.is_symlink():
                shutil.copy2(source, work / metadata_name)
        if not (work / "config.json").is_file():
            raise CheckpointError(f"Trained model has no config.json: {trained_root}")

        extract_tensors(initial_root, initial, initial_mtp, work / MTP_FILENAME)
        final_map = dict(trained.weight_map)
        final_map.update({key: MTP_FILENAME for key in initial_mtp})
        expected_layout = dict(trained.tensors)
        expected_layout.update({key: initial.tensors[key] for key in initial_mtp})
        write_json_atomic(
            work / "model.safetensors.index.json",
            {
                "metadata": {
                    "total_parameters": sum(ref.parameters for ref in expected_layout.values()),
                    "total_size": sum(ref.size for ref in expected_layout.values()),
                },
                "weight_map": dict(sorted(final_map.items())),
            },
        )
        final = read_indexed_layout(work)
        _compare_layout(final, expected_layout, source=str(work))

        records = _artifact_records(work)
        manifest = {
            "version": 1,
            "kind": MANIFEST_KIND,
            "status": "complete",
            "created_at": utc_now(),
            "source_step": source_step,
            "trained_model": str(trained_root),
            "trained_model_index_sha256": sha256_file(trained_root / "model.safetensors.index.json"),
            "initial_model": str(initial_root),
            "initial_model_index_sha256": sha256_file(initial_root / "model.safetensors.index.json"),
            "optimizer_scheduler_rng_global_step_restored": False,
            "model_contract": {
                "expected_tensor_count": len(final.tensors),
                "expected_backbone_tensor_count": len(expected_backbone),
                "expected_mtp_tensor_count": len(initial_mtp),
                "tensor_layout_sha256": _layout_digest(final.tensors, final.tensors),
                "backbone_layout_sha256": _layout_digest(final.tensors, expected_backbone),
                "mtp_layout_sha256": _layout_digest(final.tensors, initial_mtp),
            },
            "key_coverage": {
                "expected": expected_tensor_count,
                "backbone": len(expected_backbone),
                "mtp": len(initial_mtp),
                "final": len(final.tensors),
                "extracted_mtp_file": MTP_FILENAME,
            },
            "backbone_dtype_differences_from_initial": dtype_differences,
            "trained_backbone_file_sha256": {
                filename: sha256_file(local_path(trained_root, filename))
                for filename in sorted(set(trained.weight_map.values()))
            },
            "mtp_tensor_sha256": {
                key: _tensor_digest(initial_root, initial.tensors[key]) for key in sorted(initial_mtp)
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
                "source_step": source_step,
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
    parser.add_argument("--trained-model-dir", type=Path, required=True)
    parser.add_argument("--initial-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--expected-tensor-count", type=int, required=True)
    parser.add_argument("--expected-mtp-tensor-count", type=int, default=15)
    parser.add_argument("--expected-backbone-dtype-differences", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = graft_mtp_head(
            args.trained_model_dir,
            args.initial_model_dir,
            args.output_dir,
            source_step=args.source_step,
            expected_tensor_count=args.expected_tensor_count,
            expected_mtp_tensor_count=args.expected_mtp_tensor_count,
            expected_backbone_dtype_differences=args.expected_backbone_dtype_differences,
        )
    except CheckpointError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
