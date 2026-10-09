#!/usr/bin/env python3
"""Prepare one image-edit-v6 Call-2 ``final_results.jsonl`` partition.

This is deliberately a local, upload-free producer.  It validates every
published row through the runtime V6 loader contract, removes the pinned
evaluation IDs, deterministically assigns rows with ``sha256(sample_id) % N``,
and writes prompt-free training JSONL plus an atomic partition summary.

The summary is not an S3 publication.  A later publisher must upload each
object with ``sha256=<summary digest>`` object metadata, capture its immutable
VersionId, and pass those identities to
``finalize_image_edit_v6_full64_manifest.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    ANNOTATION_CONTRACT_SHA256,
    ANNOTATION_CONTRACT_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    ROW_FORMAT,
    TARGET_SCHEMA,
    ImageEditV6FullJsonConfig,
    make_row,
    prompt_contract_sha256,
)


SUMMARY_FORMAT = "image_edit_v6_training_partition_summary_v1"
SHARD_ASSIGNMENT = "sha256_sample_id_mod_parts_v1"
DEFAULT_ALLOWED_STATUSES = ("validated", "manual_review_required")
DEFAULT_GAVIN_EXCLUSION_FILES = (
    REPO_ROOT
    / "artifacts/image_edit_v4_prepared_gavin_disjoint_v1/audit/shared_validation_ids.txt",
    REPO_ROOT
    / "artifacts/image_edit_v4_prepared_gavin_disjoint_v1/audit/missing_validation_ids.txt",
    REPO_ROOT
    / "artifacts/image_edit_v4_prepared_gavin_disjoint_v1/audit/shared_test_ids.txt",
    REPO_ROOT
    / "artifacts/image_edit_v4_prepared_gavin_disjoint_v1/audit/missing_test_ids.txt",
)
GAVIN_EXCLUSION_SAMPLE_IDS_SHA256 = (
    "fdba32d13fc6ff2bc71c1cf5d7c26d38f80c07cc82ba88b81d3eff5fa8e2a8ac"
)
GAVIN_EXCLUSION_UNIQUE_SAMPLE_IDS = 25_830

PREPARED_FIELDS = (
    "sample_id",
    "status",
    "instruction",
    "source_s3_uri",
    "target_s3_uri",
    "contract_version",
    "contract_sha256",
    "bbox_coordinate_order",
    "call2_audit",
    "final_json",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_document_bytes(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
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


def sample_id_part(sample_id: str, part_count: int) -> int:
    if part_count < 1:
        raise ValueError("part_count must be positive")
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % part_count


def load_excluded_ids(paths: Sequence[Path]) -> tuple[set[str], dict[str, Any]]:
    if not paths:
        raise ValueError("at least one evaluation exclusion file is required")
    sample_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = path.read_bytes()
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: exclusion file must be UTF-8") from exc
        ids = [line.strip() for line in lines if line.strip()]
        if any(value != value.strip() for value in ids):  # pragma: no cover - defensive
            raise ValueError(f"{path}: exclusion ID contains surrounding whitespace")
        duplicate_count = len(ids) - len(set(ids))
        if duplicate_count:
            raise ValueError(f"{path}: contains {duplicate_count} duplicate IDs")
        sample_ids.update(ids)
        source_files.append(
            {
                "path": str(path),
                "rows": len(ids),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    ordered = sorted(sample_ids)
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return sample_ids, {
        "policy": "exclude_exact_sample_id_v1",
        "unique_sample_ids": len(ordered),
        "sample_ids_sha256": digest,
        "sources": source_files,
    }


@dataclass(frozen=True)
class SourceIdentity:
    local_path: Path
    uri: str
    version_id: str
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    expected_rows: int | None = None
    expected_status_counts: Mapping[str, int] | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    checkpoint_bytes: int | None = None


@dataclass(frozen=True)
class PreparePartitionOptions:
    output_dir: Path
    partition_index: int
    partition_count: int = 64
    output_parts: int = 4
    input_jsonl: Path | None = None
    call2_checkpoint: Path | None = None
    source_uri: str | None = None
    source_version_id: str | None = None
    exclude_sample_ids: tuple[Path, ...] = DEFAULT_GAVIN_EXCLUSION_FILES
    allowed_statuses: tuple[str, ...] = DEFAULT_ALLOWED_STATUSES
    invalid_policy: str = "error"
    summary_path: Path | None = None
    overwrite: bool = False
    rejection_examples: int = 20


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _first(document: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in document and document[key] not in (None, ""):
            return document[key]
    return None


def _local_path(value: str, *, relative_to: Path) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _load_checkpoint_source(
    path: Path,
    *,
    partition_index: int,
    partition_count: int,
    source_version_id: str | None,
) -> SourceIdentity:
    checkpoint_path = path.expanduser().resolve(strict=True)
    raw = checkpoint_path.read_bytes()
    document = _mapping(json.loads(raw), str(checkpoint_path))

    completion_markers = []
    if "complete" in document:
        completion_markers.append(document.get("complete") is True)
    if "status" in document:
        completion_markers.append(str(document.get("status")) in {"complete", "completed"})
    if "phase" in document:
        completion_markers.append(str(document.get("phase")) == "complete")
    if not completion_markers or not all(completion_markers):
        raise ValueError(f"{checkpoint_path}: Call-2 checkpoint is not complete")

    declared_stage = _first(document, "stage", "source_stage", "logical_stage")
    if declared_stage is not None and str(declared_stage) not in {"call2", "call2_final"}:
        raise ValueError(f"{checkpoint_path}: checkpoint is not a Call-2 final checkpoint")
    declared_partition = _first(document, "partition_index", "logical_partition")
    if declared_partition is not None and int(declared_partition) != partition_index:
        raise ValueError(f"{checkpoint_path}: partition_index does not match the CLI")
    declared_count = _first(document, "partition_count", "logical_partition_count")
    if declared_count is not None and int(declared_count) != partition_count:
        raise ValueError(f"{checkpoint_path}: partition_count does not match the CLI")

    nested_raw = document.get("final_results")
    nested = _mapping(nested_raw, "final_results") if nested_raw is not None else {}
    local_value = _first(nested, "local_path", "path") or _first(
        document, "final_results_local_path", "final_results_path"
    )
    local: Path | None = None
    if local_value is not None:
        local = _local_path(str(local_value), relative_to=checkpoint_path.parent)
    if local is None:
        sibling = checkpoint_path.parent / "final_results.jsonl"
        if sibling.is_file():
            local = sibling.resolve()
    uri_value = _first(nested, "uri", "s3_uri") or _first(
        document, "final_results_uri", "final_results_s3_uri"
    )
    if local is None and uri_value is not None:
        local = _local_path(str(uri_value), relative_to=checkpoint_path.parent)
    if local is None:
        raise ValueError(
            f"{checkpoint_path}: no local final_results.jsonl is available; "
            "this producer intentionally performs no network reads"
        )
    local = local.resolve(strict=True)
    uri = str(uri_value or local)

    expected_sha = _first(nested, "sha256") or _first(document, "final_results_sha256")
    expected_bytes = _first(nested, "bytes") or _first(document, "final_results_bytes")
    expected_rows = _first(nested, "rows") or _first(
        document, "final_results_rows", "sample_count", "expected_samples"
    )
    status_counts_raw = _first(nested, "status_counts") or document.get("status_counts")
    status_counts = None
    if status_counts_raw is not None:
        status_counts = {
            str(key): int(value)
            for key, value in _mapping(status_counts_raw, "status_counts").items()
        }
    version_id = str(
        _first(nested, "version_id")
        or _first(document, "final_results_version_id")
        or source_version_id
        or "LOCAL_ONLY_NOT_PUBLISHED"
    )
    return SourceIdentity(
        local_path=local,
        uri=uri,
        version_id=version_id,
        expected_sha256=str(expected_sha) if expected_sha is not None else None,
        expected_bytes=int(expected_bytes) if expected_bytes is not None else None,
        expected_rows=int(expected_rows) if expected_rows is not None else None,
        expected_status_counts=status_counts,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_bytes=len(raw),
    )


def resolve_source(options: PreparePartitionOptions) -> SourceIdentity:
    if (options.input_jsonl is None) == (options.call2_checkpoint is None):
        raise ValueError("provide exactly one of input_jsonl or call2_checkpoint")
    if options.call2_checkpoint is not None:
        return _load_checkpoint_source(
            options.call2_checkpoint,
            partition_index=options.partition_index,
            partition_count=options.partition_count,
            source_version_id=options.source_version_id,
        )
    assert options.input_jsonl is not None
    path = options.input_jsonl.expanduser().resolve(strict=True)
    return SourceIdentity(
        local_path=path,
        uri=str(options.source_uri or path),
        version_id=str(options.source_version_id or "LOCAL_ONLY_NOT_PUBLISHED"),
    )


def _part_paths(output_dir: Path, partition_index: int, part_count: int) -> list[Path]:
    width = max(3, len(str(part_count - 1)))
    return [
        output_dir / f"part-{index:0{width}d}-of-{part_count:0{width}d}.jsonl"
        for index in range(part_count)
    ]


def _open_temporary(path: Path) -> tuple[BinaryIO, Path]:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return os.fdopen(descriptor, "wb"), Path(name)


def _project_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in PREPARED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"prepared row is missing fields: {missing}")
    return {field: row[field] for field in PREPARED_FIELDS}


def prepare_partition(options: PreparePartitionOptions) -> dict[str, Any]:
    if prompt_contract_sha256() != PROMPT_CONTRACT_SHA256:
        raise ValueError("V6 prompt contract drifted before dataset publication")
    if options.partition_count < 1:
        raise ValueError("partition_count must be positive")
    if not 0 <= options.partition_index < options.partition_count:
        raise ValueError("partition_index is outside partition_count")
    if options.output_parts < 1:
        raise ValueError("output_parts must be positive")
    if options.invalid_policy not in {"error", "skip"}:
        raise ValueError("invalid_policy must be error or skip")
    if not options.allowed_statuses or len(set(options.allowed_statuses)) != len(
        options.allowed_statuses
    ):
        raise ValueError("allowed_statuses must be non-empty and unique")

    source = resolve_source(options)
    excluded_ids, exclusion = load_excluded_ids(options.exclude_sample_ids)
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    part_paths = _part_paths(output_dir, options.partition_index, options.output_parts)
    summary_path = (
        options.summary_path.expanduser().resolve()
        if options.summary_path is not None
        else output_dir / f"partition-{options.partition_index:03d}.summary.json"
    )
    existing = [path for path in [*part_paths, summary_path] if path.exists()]
    if existing and not options.overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    runtime_config = ImageEditV6FullJsonConfig(
        dataset_name="image_edit_v6_partition_prepare",
        val_dataset_name="image_edit_v6_partition_prepare_val",
        paths=(),
        allowed_statuses=options.allowed_statuses,
        reject_invalid_rows=True,
    )

    opened = [_open_temporary(path) for path in part_paths]
    streams = [item[0] for item in opened]
    temporary_paths = [item[1] for item in opened]
    part_hashers = [hashlib.sha256() for _ in part_paths]
    part_rows = [0 for _ in part_paths]
    part_bytes = [0 for _ in part_paths]
    source_digest = hashlib.sha256()
    source_bytes = 0
    input_rows = 0
    output_rows = 0
    excluded_rows = 0
    evaluation_excluded_rows = 0
    status_excluded_rows = 0
    rejected_rows = 0
    source_status_counts: Counter[str] = Counter()
    output_status_counts: Counter[str] = Counter()
    excluded_status_counts: Counter[str] = Counter()
    rejected_status_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    rejection_examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        with source.local_path.open("rb") as input_stream:
            for line_number, raw_line in enumerate(input_stream, 1):
                source_digest.update(raw_line)
                source_bytes += len(raw_line)
                if not raw_line.strip():
                    raise ValueError(f"{source.local_path}:{line_number}: blank lines are forbidden")
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{source.local_path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{source.local_path}:{line_number}: expected a JSON object"
                    )
                input_rows += 1
                sample_id = value.get("sample_id")
                if (
                    not isinstance(sample_id, str)
                    or not sample_id
                    or sample_id != sample_id.strip()
                ):
                    raise ValueError(
                        f"{source.local_path}:{line_number}: sample_id must be a trimmed string"
                    )
                if sample_id in seen_ids:
                    raise ValueError(
                        f"{source.local_path}:{line_number}: duplicate sample_id {sample_id!r}"
                    )
                seen_ids.add(sample_id)
                status = value.get("status")
                if not isinstance(status, str) or not status:
                    raise ValueError(f"{sample_id}: status must be a non-empty string")
                source_status_counts[status] += 1

                evaluation_excluded = sample_id in excluded_ids
                if evaluation_excluded:
                    evaluation_excluded_rows += 1
                if status not in options.allowed_statuses:
                    excluded_rows += 1
                    status_excluded_rows += 1
                    excluded_status_counts[status] += 1
                    continue
                if evaluation_excluded:
                    excluded_rows += 1
                    excluded_status_counts[status] += 1
                    continue

                try:
                    validated = make_row(
                        value,
                        source_path=f"{source.local_path}:{line_number}",
                        config=runtime_config,
                    )
                    if validated is None:  # reject_invalid_rows=True makes this defensive only.
                        raise ValueError("V6 runtime validator rejected the row")
                    prepared = _project_row(value)
                except (TypeError, ValueError) as exc:
                    reason = str(exc)
                    rejected_rows += 1
                    rejected_status_counts[status] += 1
                    rejection_reasons[reason] += 1
                    if len(rejection_examples) < options.rejection_examples:
                        rejection_examples.append(
                            {"sample_id": sample_id, "line": line_number, "reason": reason}
                        )
                    if options.invalid_policy == "error":
                        raise ValueError(f"{sample_id}: invalid V6 training row: {reason}") from exc
                    continue

                part_index = sample_id_part(sample_id, options.output_parts)
                payload = canonical_bytes(prepared) + b"\n"
                streams[part_index].write(payload)
                part_hashers[part_index].update(payload)
                part_rows[part_index] += 1
                part_bytes[part_index] += len(payload)
                output_rows += 1
                output_status_counts[status] += 1

        actual_source_sha256 = source_digest.hexdigest()
        if source.expected_sha256 is not None and actual_source_sha256 != source.expected_sha256:
            raise ValueError(
                "source final_results SHA-256 mismatch: "
                f"actual={actual_source_sha256} expected={source.expected_sha256}"
            )
        if source.expected_bytes is not None and source_bytes != source.expected_bytes:
            raise ValueError(
                f"source final_results byte count mismatch: {source_bytes} != "
                f"{source.expected_bytes}"
            )
        if source.expected_rows is not None and input_rows != source.expected_rows:
            raise ValueError(
                f"source final_results row count mismatch: {input_rows} != {source.expected_rows}"
            )
        if source.expected_status_counts is not None and dict(source_status_counts) != dict(
            source.expected_status_counts
        ):
            raise ValueError(
                "source final_results status counts mismatch: "
                f"actual={dict(source_status_counts)} "
                f"expected={dict(source.expected_status_counts)}"
            )
        if input_rows != excluded_rows + rejected_rows + output_rows:
            raise AssertionError("partition row conservation failed")
        empty_parts = [index for index, rows in enumerate(part_rows) if rows == 0]
        if empty_parts:
            raise ValueError(
                f"deterministic split produced empty output parts {empty_parts}; "
                "publication requires every physical shard to be non-empty"
            )

        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        for temporary, destination in zip(temporary_paths, part_paths, strict=True):
            os.replace(temporary, destination)

        shard_records = []
        for part_index, path in enumerate(part_paths):
            relative = f"partition-{options.partition_index:03d}/{path.name}"
            shard_records.append(
                {
                    "local_path": str(path),
                    "relative_path": relative,
                    "rows": part_rows[part_index],
                    "bytes": part_bytes[part_index],
                    "sha256": part_hashers[part_index].hexdigest(),
                    "source_partition": options.partition_index,
                    "part_index": part_index,
                    "part_count": options.output_parts,
                    "assignment": SHARD_ASSIGNMENT,
                    "publication": {
                        "required_content_type": "application/x-ndjson",
                        "required_metadata": {
                            "sha256": part_hashers[part_index].hexdigest()
                        },
                        "version_id": "PENDING_S3_UPLOAD",
                    },
                }
            )

        summary = {
            "format_version": SUMMARY_FORMAT,
            "complete": True,
            "partition_index": options.partition_index,
            "partition_count": options.partition_count,
            "source": {
                "stage": "call2_final",
                "local_path": str(source.local_path),
                "uri": source.uri,
                "version_id": source.version_id,
                "bytes": source_bytes,
                "rows": input_rows,
                "sha256": actual_source_sha256,
                "status_counts": dict(sorted(source_status_counts.items())),
                "publication": {
                    "required_content_type": "application/x-ndjson",
                    "required_metadata": {"sha256": actual_source_sha256},
                    "version_id": (
                        source.version_id
                        if source.version_id != "LOCAL_ONLY_NOT_PUBLISHED"
                        else "PENDING_S3_UPLOAD"
                    ),
                },
            },
            "input_rows": input_rows,
            "excluded_rows": excluded_rows,
            "evaluation_excluded_rows": evaluation_excluded_rows,
            "status_excluded_rows": status_excluded_rows,
            "rejected_rows": rejected_rows,
            "output_rows": output_rows,
            "source_status_counts": dict(sorted(source_status_counts.items())),
            "excluded_status_counts": dict(sorted(excluded_status_counts.items())),
            "rejected_status_counts": dict(sorted(rejected_status_counts.items())),
            "status_counts": dict(sorted(output_status_counts.items())),
            "rejection_counts": dict(sorted(rejection_reasons.items())),
            "rejection_examples": rejection_examples,
            "invalid_policy": options.invalid_policy,
            "exclusion": {
                **exclusion,
                "matched_source_rows": evaluation_excluded_rows,
                "output_overlap_rows": 0,
            },
            "sharding": {
                "assignment": SHARD_ASSIGNMENT,
                "part_count": options.output_parts,
                "all_parts_nonempty": True,
            },
            "shards": shard_records,
            "dataset_contract": {
                "row_format_version": ROW_FORMAT,
                "target_schema": TARGET_SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
                "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
                "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
                "allowed_statuses": list(options.allowed_statuses),
                "selection": "authoritative_call2_final_json",
                "source_stage": "call2_final",
                "bbox_coordinate_order": "xyxy",
            },
            "semantic_values_modified": False,
            "checkpoint": (
                {
                    "path": str(source.checkpoint_path),
                    "bytes": source.checkpoint_bytes,
                    "sha256": source.checkpoint_sha256,
                }
                if source.checkpoint_path is not None
                else None
            ),
        }
        write_atomic(summary_path, json_document_bytes(summary), overwrite=options.overwrite)
        summary["summary_path"] = str(summary_path)
        summary["summary_bytes"] = summary_path.stat().st_size
        summary["summary_sha256"] = sha256_file(summary_path)
        return summary
    except Exception:
        for stream in streams:
            if not stream.closed:
                stream.close()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--call2-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", dest="summary_path", type=Path)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, default=64)
    parser.add_argument("--output-parts", type=int, default=4)
    parser.add_argument("--source-uri")
    parser.add_argument("--source-version-id")
    parser.add_argument(
        "--exclude-sample-ids",
        action="append",
        type=Path,
        default=None,
        help=(
            "UTF-8 newline-delimited evaluation IDs; repeat for multiple files. "
            "Defaults to the four Gavin validation/test files."
        ),
    )
    parser.add_argument("--allowed-status", action="append", default=None)
    parser.add_argument("--invalid-policy", choices=("error", "skip"), default="error")
    parser.add_argument("--rejection-examples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = PreparePartitionOptions(
        output_dir=args.output_dir,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
        output_parts=args.output_parts,
        input_jsonl=args.input_jsonl,
        call2_checkpoint=args.call2_checkpoint,
        source_uri=args.source_uri,
        source_version_id=args.source_version_id,
        exclude_sample_ids=tuple(
            args.exclude_sample_ids
            if args.exclude_sample_ids is not None
            else DEFAULT_GAVIN_EXCLUSION_FILES
        ),
        allowed_statuses=tuple(args.allowed_status or DEFAULT_ALLOWED_STATUSES),
        invalid_policy=args.invalid_policy,
        summary_path=args.summary_path,
        overwrite=args.overwrite,
        rejection_examples=args.rejection_examples,
    )
    summary = prepare_partition(options)
    print(
        json.dumps(
            {
                "partition_index": summary["partition_index"],
                "input_rows": summary["input_rows"],
                "excluded_rows": summary["excluded_rows"],
                "output_rows": summary["output_rows"],
                "summary_path": summary["summary_path"],
                "summary_sha256": summary["summary_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
