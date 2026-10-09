#!/usr/bin/env python3
"""Deterministically remove global sample-ID duplicates from prepared V6 shards.

The input is an already validated/filter-audited ``partition-*/prepared`` tree.
Rows are visited in ``(partition, part, line)`` order and the first occurrence
of each ``sample_id`` is retained.  A disk-backed SQLite primary-key index keeps
memory bounded.  Row bytes are copied verbatim, so an existing exact-template
sequence audit remains valid after its row count and dataset identity are
updated, provided the row which established the recorded maximum is retained.

The complete output tree is assembled under a sibling staging directory and
renamed into place only after every shard, summary, and audit has passed its
checks.  Existing output roots are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_image_edit_v6_full64_manifest import (
    SEQUENCE_AUDIT_FORMAT,
    dataset_content_sha256_from_summaries,
)
from scripts.prepare_image_edit_v6_full_json_partition import (
    SUMMARY_FORMAT,
    canonical_bytes,
    json_document_bytes,
    sha256_file,
    write_atomic,
)


DUPLICATE_REJECTION_REASON = (
    "duplicate_sample_id_global_first_by_partition_part_line_v1"
)
DEDUPLICATION_FORMAT = "image_edit_v6_prepared_global_deduplication_v1"
SCAN_ORDER = "source_partition_asc_part_index_asc_line_number_asc"


@dataclass(frozen=True)
class DedupeOptions:
    input_root: Path
    output_root: Path
    sequence_length_audit: Path | None = None
    partition_count: int = 64
    parts_per_partition: int = 4
    rejection_examples: int = 40
    sqlite_dir: Path | None = None


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid {description} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be an object")
    return value


def _counter(value: Any) -> Counter[str]:
    if value is None:
        return Counter()
    if not isinstance(value, Mapping):
        raise ValueError("status/count fields must be objects")
    result = Counter({str(key): int(count) for key, count in value.items()})
    if any(count < 0 for count in result.values()):
        raise ValueError("status/count fields cannot contain negative values")
    return result


def _summary_paths(root: Path, partition_count: int) -> list[Path]:
    paths = sorted(root.glob("partition-*/prepared/partition-*.summary.json"))
    if len(paths) != partition_count:
        raise ValueError(
            f"expected {partition_count} partition summaries, got {len(paths)}"
        )
    return paths


def _load_summaries(
    root: Path, *, partition_count: int, parts_per_partition: int
) -> list[dict[str, Any]]:
    by_partition: dict[int, dict[str, Any]] = {}
    for path in _summary_paths(root, partition_count):
        summary = _load_object(path, "partition summary")
        if summary.get("format_version") != SUMMARY_FORMAT:
            raise ValueError(f"{path}: incompatible partition summary format")
        if summary.get("complete") is not True:
            raise ValueError(f"{path}: partition summary is incomplete")
        partition = int(summary.get("partition_index", -1))
        if partition in by_partition:
            raise ValueError(f"duplicate partition summary {partition}")
        if int(summary.get("partition_count") or 0) != partition_count:
            raise ValueError(f"{path}: partition_count differs")
        shards = summary.get("shards")
        if not isinstance(shards, list) or len(shards) != parts_per_partition:
            raise ValueError(
                f"{path}: expected {parts_per_partition} prepared shards"
            )
        indexes = {int(item.get("part_index", -1)) for item in shards}
        if indexes != set(range(parts_per_partition)):
            raise ValueError(f"{path}: prepared part indexes are incomplete")
        if any(int(item.get("source_partition", -1)) != partition for item in shards):
            raise ValueError(f"{path}: shard source_partition differs")
        if any(int(item.get("part_count") or 0) != parts_per_partition for item in shards):
            raise ValueError(f"{path}: shard part_count differs")
        by_partition[partition] = summary
    expected = set(range(partition_count))
    if set(by_partition) != expected:
        raise ValueError(
            "partition summaries are incomplete: "
            f"missing={sorted(expected - set(by_partition))} "
            f"extra={sorted(set(by_partition) - expected)}"
        )
    return [by_partition[index] for index in range(partition_count)]


def _verify_shard_identity(path: Path, shard: Mapping[str, Any]) -> None:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"prepared shard does not exist: {path}") from exc
    actual_bytes = resolved.stat().st_size
    expected_bytes = int(shard.get("bytes") or 0)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{resolved}: byte count differs from summary "
            f"({actual_bytes} != {expected_bytes})"
        )
    actual_sha256 = sha256_file(resolved)
    expected_sha256 = str(shard.get("sha256") or "")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{resolved}: SHA-256 differs from summary")


def _new_database(directory: Path | None) -> tuple[sqlite3.Connection, Path]:
    sqlite_root = directory.expanduser().resolve() if directory is not None else None
    if sqlite_root is not None:
        sqlite_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="image-edit-v6-dedupe-", suffix=".sqlite3", dir=sqlite_root
    )
    os.close(descriptor)
    path = Path(name)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        """
        CREATE TABLE sample_ids (
            sample_id TEXT PRIMARY KEY,
            source_partition INTEGER NOT NULL,
            part_index INTEGER NOT NULL,
            line_number INTEGER NOT NULL,
            row_sha256 BLOB NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection, path


def _atomic_shard(
    *,
    input_path: Path,
    staged_path: Path,
    final_path: Path,
    partition: int,
    part_index: int,
    part_count: int,
    connection: sqlite3.Connection,
    maximum_sample_id: str,
    maximum_proof: dict[str, Any],
    example_limit: int,
) -> tuple[dict[str, Any], Counter[str], Counter[str], list[dict[str, Any]]]:
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{staged_path.name}.", suffix=".tmp", dir=staged_path.parent
    )
    output_rows = 0
    output_bytes = 0
    input_rows = 0
    output_statuses: Counter[str] = Counter()
    duplicate_statuses: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output_stream, input_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip() or not raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"{input_path}:{line_number}: invalid JSONL framing"
                    )
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{input_path}:{line_number}: invalid row JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{input_path}:{line_number}: expected an object")
                sample_id = value.get("sample_id")
                if (
                    not isinstance(sample_id, str)
                    or not sample_id
                    or sample_id != sample_id.strip()
                ):
                    raise ValueError(
                        f"{input_path}:{line_number}: sample_id must be a trimmed string"
                    )
                status = value.get("status")
                if not isinstance(status, str) or not status:
                    raise ValueError(
                        f"{input_path}:{line_number}: status must be a non-empty string"
                    )
                input_rows += 1
                row_sha256 = hashlib.sha256(raw_line).digest()
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO sample_ids(
                        sample_id, source_partition, part_index, line_number, row_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sample_id, partition, part_index, line_number, row_sha256),
                )
                if cursor.rowcount == 1:
                    output_stream.write(raw_line)
                    digest.update(raw_line)
                    output_rows += 1
                    output_bytes += len(raw_line)
                    output_statuses[status] += 1
                    if sample_id == maximum_sample_id:
                        maximum_proof["seen"] = True
                    continue

                duplicate_statuses[status] += 1
                first = connection.execute(
                    """
                    SELECT source_partition, part_index, line_number, row_sha256
                    FROM sample_ids WHERE sample_id = ?
                    """,
                    (sample_id,),
                ).fetchone()
                if first is None:  # pragma: no cover - SQLite invariant
                    raise AssertionError("duplicate lookup lost its retained row")
                if sample_id == maximum_sample_id and bytes(first[3]) != row_sha256:
                    maximum_proof["payload_mismatch"] = True
                if len(examples) < example_limit:
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "reason": DUPLICATE_REJECTION_REASON,
                            "duplicate_location": {
                                "source_partition": partition,
                                "part_index": part_index,
                                "line_number": line_number,
                            },
                            "retained_location": {
                                "source_partition": int(first[0]),
                                "part_index": int(first[1]),
                                "line_number": int(first[2]),
                            },
                        }
                    )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if output_rows < 1:
            raise ValueError(
                f"global deduplication produced an empty shard: {input_path}"
            )
        os.replace(temporary_name, staged_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return (
        {
            "local_path": str(final_path),
            "relative_path": f"partition-{partition:03d}/{final_path.name}",
            "rows": output_rows,
            "bytes": output_bytes,
            "sha256": digest.hexdigest(),
            "source_partition": partition,
            "part_index": part_index,
            "part_count": part_count,
            "publication": {
                "required_content_type": "application/x-ndjson",
                "required_metadata": {"sha256": digest.hexdigest()},
                "version_id": "PENDING_S3_UPLOAD",
            },
        },
        output_statuses,
        duplicate_statuses,
        examples,
    )


def _input_audit(
    path: Path, *, summaries: Sequence[Mapping[str, Any]], total_rows: int
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    document = _load_object(path, "sequence-length audit")
    if document.get("format_version") not in (None, SEQUENCE_AUDIT_FORMAT):
        raise ValueError(f"{path}: incompatible sequence audit format")
    nested = isinstance(document.get("sequence_length_audit"), Mapping)
    audit = dict(document["sequence_length_audit"] if nested else document)
    if audit.get("complete") is not True:
        raise ValueError(f"{path}: sequence audit is incomplete")
    expected_digest = dataset_content_sha256_from_summaries(summaries)
    if audit.get("dataset_content_sha256") != expected_digest:
        raise ValueError(f"{path}: sequence audit dataset identity differs from input")
    if int(audit.get("checked_rows") or 0) != total_rows:
        raise ValueError(f"{path}: sequence audit row count differs from input")
    for field in (
        "overlength_rows",
        "assistant_truncation_rows",
        "encode_error_rows",
        "zero_supervised_assistant_rows",
        "json_eos_incomplete_rows",
    ):
        if int(audit.get(field) or 0) != 0:
            raise ValueError(
                f"{path}: cannot reuse audit with {field}={audit.get(field)!r}; "
                "run a fresh exact sequence audit after deduplication"
            )
    maximum_sample_id = audit.get("max_observed_sample_id")
    if not isinstance(maximum_sample_id, str) or not maximum_sample_id:
        raise ValueError(
            f"{path}: max_observed_sample_id is required to prove audit reuse"
        )
    if int(audit.get("max_observed_length") or 0) < 1:
        raise ValueError(f"{path}: max_observed_length is invalid")
    return document, nested, audit


def dedupe_prepared(options: DedupeOptions) -> dict[str, Any]:
    if options.partition_count < 1 or options.parts_per_partition < 1:
        raise ValueError("partition_count and parts_per_partition must be positive")
    if options.rejection_examples < 0:
        raise ValueError("rejection_examples cannot be negative")
    input_root = options.input_root.expanduser().resolve(strict=True)
    output_root = options.output_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError("input_root and output_root must differ")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(
        input_root,
        partition_count=options.partition_count,
        parts_per_partition=options.parts_per_partition,
    )
    input_total_rows = sum(int(summary.get("output_rows") or 0) for summary in summaries)
    audit_path = (
        options.sequence_length_audit.expanduser().resolve(strict=True)
        if options.sequence_length_audit is not None
        else (input_root / "sequence-length-audit.json").resolve(strict=True)
    )
    audit_document, audit_nested, audit = _input_audit(
        audit_path, summaries=summaries, total_rows=input_total_rows
    )
    maximum_sample_id = str(audit["max_observed_sample_id"])

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    connection: sqlite3.Connection | None = None
    database_path: Path | None = None
    try:
        connection, database_path = _new_database(options.sqlite_dir)
        output_summaries: list[dict[str, Any]] = []
        total_duplicates = 0
        global_duplicate_statuses: Counter[str] = Counter()
        global_examples: list[dict[str, Any]] = []
        maximum_proof = {"seen": False, "payload_mismatch": False}

        for summary in summaries:
            partition = int(summary["partition_index"])
            old_output_rows = int(summary.get("output_rows") or 0)
            old_statuses = _counter(summary.get("status_counts"))
            new_shards: list[dict[str, Any]] = []
            kept_statuses: Counter[str] = Counter()
            duplicate_statuses: Counter[str] = Counter()
            partition_examples: list[dict[str, Any]] = []
            input_shard_rows = 0

            for shard in sorted(summary["shards"], key=lambda item: int(item["part_index"])):
                part_index = int(shard["part_index"])
                input_path = Path(str(shard.get("local_path") or ""))
                _verify_shard_identity(input_path, shard)
                input_shard_rows += int(shard.get("rows") or 0)
                staged_path = (
                    staging
                    / f"partition-{partition:03d}"
                    / "prepared"
                    / f"part-{part_index:03d}-of-{options.parts_per_partition:03d}.jsonl"
                )
                final_path = (
                    output_root
                    / f"partition-{partition:03d}"
                    / "prepared"
                    / staged_path.name
                )
                record, shard_statuses, shard_duplicates, examples = _atomic_shard(
                    input_path=input_path,
                    staged_path=staged_path,
                    final_path=final_path,
                    partition=partition,
                    part_index=part_index,
                    part_count=options.parts_per_partition,
                    connection=connection,
                    maximum_sample_id=maximum_sample_id,
                    maximum_proof=maximum_proof,
                    example_limit=max(0, options.rejection_examples - len(partition_examples)),
                )
                record["assignment"] = shard.get("assignment")
                new_shards.append(record)
                kept_statuses.update(shard_statuses)
                duplicate_statuses.update(shard_duplicates)
                partition_examples.extend(examples)
                connection.commit()

            if input_shard_rows != old_output_rows:
                raise ValueError(
                    f"partition {partition}: shard rows differ from summary output_rows"
                )
            if old_statuses != kept_statuses + duplicate_statuses:
                raise ValueError(
                    f"partition {partition}: shard statuses do not conserve summary statuses"
                )
            duplicate_rows = sum(duplicate_statuses.values())
            output_rows = sum(int(item["rows"]) for item in new_shards)
            if old_output_rows != output_rows + duplicate_rows:
                raise AssertionError(
                    f"partition {partition}: deduplication row conservation failed"
                )

            rejected_statuses = _counter(summary.get("rejected_status_counts"))
            rejected_statuses.update(duplicate_statuses)
            rejection_counts = _counter(summary.get("rejection_counts"))
            rejection_counts[DUPLICATE_REJECTION_REASON] += duplicate_rows
            rejection_examples = list(summary.get("rejection_examples") or [])
            available = max(0, options.rejection_examples - len(rejection_examples))
            rejection_examples.extend(partition_examples[:available])
            new_rejected_rows = int(summary.get("rejected_rows") or 0) + duplicate_rows
            excluded_rows = int(summary.get("excluded_rows") or 0)
            input_rows = int(summary.get("input_rows") or 0)
            if input_rows != excluded_rows + new_rejected_rows + output_rows:
                raise ValueError(
                    f"partition {partition}: final row conservation failed: "
                    f"{input_rows} != {excluded_rows}+{new_rejected_rows}+{output_rows}"
                )

            new_summary = dict(summary)
            new_summary.update(
                {
                    "rejected_rows": new_rejected_rows,
                    "output_rows": output_rows,
                    "rejected_status_counts": dict(sorted(rejected_statuses.items())),
                    "status_counts": dict(sorted(kept_statuses.items())),
                    "rejection_counts": dict(sorted(rejection_counts.items())),
                    "rejection_examples": rejection_examples,
                    "shards": new_shards,
                    "duplicate_rejected_rows": duplicate_rows,
                    "duplicate_rejected_status_counts": dict(
                        sorted(duplicate_statuses.items())
                    ),
                    "deduplication": {
                        "format_version": DEDUPLICATION_FORMAT,
                        "reason": DUPLICATE_REJECTION_REASON,
                        "scan_order": SCAN_ORDER,
                        "input_prepared_rows": old_output_rows,
                        "retained_rows": output_rows,
                        "duplicate_rejected_rows": duplicate_rows,
                        "duplicate_rejected_status_counts": dict(
                            sorted(duplicate_statuses.items())
                        ),
                        "duplicate_rejection_examples": partition_examples,
                        "semantic_values_modified": False,
                    },
                    "conservation": {
                        "input_rows": input_rows,
                        "excluded_rows": excluded_rows,
                        "rejected_rows": new_rejected_rows,
                        "output_rows": output_rows,
                        "balanced": True,
                    },
                }
            )
            summary_path = (
                staging
                / f"partition-{partition:03d}"
                / "prepared"
                / f"partition-{partition:03d}.summary.json"
            )
            write_atomic(
                summary_path, json_document_bytes(new_summary), overwrite=False
            )
            output_summaries.append(new_summary)
            total_duplicates += duplicate_rows
            global_duplicate_statuses.update(duplicate_statuses)
            remaining = max(0, options.rejection_examples - len(global_examples))
            global_examples.extend(partition_examples[:remaining])

        if not maximum_proof["seen"]:
            raise ValueError(
                "recorded max_observed_sample_id is absent from the prepared dataset; "
                "run a fresh exact sequence audit after deduplication"
            )
        if maximum_proof["payload_mismatch"]:
            raise ValueError(
                "duplicate occurrences of max_observed_sample_id have different row bytes; "
                "run a fresh exact sequence audit after deduplication"
            )

        output_total_rows = sum(
            int(summary["output_rows"]) for summary in output_summaries
        )
        if input_total_rows != output_total_rows + total_duplicates:
            raise AssertionError("global deduplication row conservation failed")
        unique_rows = int(
            connection.execute("SELECT COUNT(*) FROM sample_ids").fetchone()[0]
        )
        if unique_rows != output_total_rows:
            raise AssertionError("SQLite unique-row count differs from output rows")

        output_digest = dataset_content_sha256_from_summaries(output_summaries)
        output_audit = dict(audit)
        output_audit.update(
            {
                "complete": True,
                "dataset_content_sha256": output_digest,
                "checked_rows": output_total_rows,
                "total_rows": output_total_rows,
                "pre_dedup_rows": input_total_rows,
                "duplicate_excluded_rows": total_duplicates,
                "deduplication_audit_reuse": {
                    "format_version": DEDUPLICATION_FORMAT,
                    "scan_order": SCAN_ORDER,
                    "row_bytes_modified": False,
                    "recorded_max_sample_retained": True,
                    "recorded_max_sample_duplicate_payloads_identical": True,
                },
            }
        )
        if audit_nested:
            output_audit_document = dict(audit_document)
            output_audit_document["format_version"] = SEQUENCE_AUDIT_FORMAT
            output_audit_document["sequence_length_audit"] = output_audit
        else:
            output_audit_document = output_audit
            output_audit_document["format_version"] = SEQUENCE_AUDIT_FORMAT
        output_audit_path = staging / "sequence-length-audit.json"
        write_atomic(
            output_audit_path,
            json_document_bytes(output_audit_document),
            overwrite=False,
        )

        report = {
            "format_version": DEDUPLICATION_FORMAT,
            "complete": True,
            "scan_order": SCAN_ORDER,
            "partition_count": options.partition_count,
            "parts_per_partition": options.parts_per_partition,
            "physical_shards": options.partition_count * options.parts_per_partition,
            "input_rows": input_total_rows,
            "output_rows": output_total_rows,
            "duplicate_rejected_rows": total_duplicates,
            "duplicate_rejected_status_counts": dict(
                sorted(global_duplicate_statuses.items())
            ),
            "duplicate_rejection_reason": DUPLICATE_REJECTION_REASON,
            "duplicate_rejection_examples": global_examples,
            "dataset_content_sha256": output_digest,
            "sequence_length_audit": str(output_root / "sequence-length-audit.json"),
            "sequence_length_audit_reused": True,
            "max_observed_sample_id_retained": maximum_sample_id,
            "conservation": {
                "input_rows": input_total_rows,
                "duplicate_rejected_rows": total_duplicates,
                "output_rows": output_total_rows,
                "balanced": True,
            },
        }
        write_atomic(
            staging / "dedupe-report.json",
            json_document_bytes(report),
            overwrite=False,
        )

        connection.close()
        connection = None
        database_path.unlink(missing_ok=True)
        database_path = None
        if output_root.exists():
            raise FileExistsError(f"refusing to overwrite output root: {output_root}")
        os.rename(staging, output_root)
        report["output_root"] = str(output_root)
        return report
    finally:
        if connection is not None:
            connection.close()
        if database_path is not None:
            database_path.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence-length-audit", type=Path)
    parser.add_argument("--partition-count", type=int, default=64)
    parser.add_argument("--parts-per-partition", type=int, default=4)
    parser.add_argument("--rejection-examples", type=int, default=40)
    parser.add_argument("--sqlite-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = dedupe_prepared(
        DedupeOptions(
            input_root=args.input_root,
            output_root=args.output_root,
            sequence_length_audit=args.sequence_length_audit,
            partition_count=args.partition_count,
            parts_per_partition=args.parts_per_partition,
            rejection_examples=args.rejection_examples,
            sqlite_dir=args.sqlite_dir,
        )
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
