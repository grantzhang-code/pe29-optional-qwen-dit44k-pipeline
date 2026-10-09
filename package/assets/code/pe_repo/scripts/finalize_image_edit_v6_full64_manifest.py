#!/usr/bin/env python3
"""Finalize immutable image-edit-v6 partition summaries into one manifest.

The finalizer is local and performs no upload or network request. It re-hashes
prepared JSONL while re-counting rows/statuses, checks deterministic shard
assignment, enforces global sample-ID uniqueness through SQLite, binds an exact
sequence-length audit, and requires externally captured S3 object identities
(URI, VersionId, bytes, SHA-256 and metadata SHA-256). Source JSONL can either be
re-read locally or bound directly to exact HEAD-verified S3 VersionIds.

The S3 uploader that creates ``--publication-identities`` must set object
metadata ``sha256=<record sha256>`` on *both* source and prepared shard objects
and record the VersionId returned by a successful HEAD of that exact object.
This script intentionally cannot certify publication identities itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    ANNOTATION_CONTRACT_SHA256,
    ANNOTATION_CONTRACT_VERSION,
    MANIFEST_FORMAT,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    ROW_FORMAT,
    TARGET_SCHEMA,
)
from scripts.prepare_image_edit_v6_full_json_partition import (  # noqa: E402
    DEFAULT_ALLOWED_STATUSES,
    DEFAULT_GAVIN_EXCLUSION_FILES,
    GAVIN_EXCLUSION_SAMPLE_IDS_SHA256,
    PREPARED_FIELDS,
    SHARD_ASSIGNMENT,
    SUMMARY_FORMAT,
    canonical_bytes,
    json_document_bytes,
    load_excluded_ids,
    sample_id_part,
    sha256_file,
    write_atomic,
)


PUBLICATION_IDENTITIES_FORMAT = "image_edit_v6_s3_publication_identities_v1"
SEQUENCE_AUDIT_FORMAT = "image_edit_v6_exact_sequence_length_audit_v1"
DEFAULT_MAX_LENGTH = 32_768
DEFAULT_MAX_PIXELS = 2_359_296
DEFAULT_IMAGE_MAX_TOKEN_NUM = 2_304
DEFAULT_MODEL_TYPE = "qwen3_5"
DEFAULT_TEMPLATE_MODE = "training"
DEFAULT_TRUNCATION_STRATEGY = "raise"
DEFAULT_RUNTIME_IMAGE_DIGEST = (
    "sha256:8e22fdff5171e702b82692a7b545ca3457b4845029f52be437d6e57b7dfb1f4d"
)
DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256 = (
    "8bf33b2840671ec6339c23247ad27b7a2c01461ebfaf8170b12cf1761fad6bb4"
)


@dataclass(frozen=True)
class FinalizeManifestOptions:
    summaries: tuple[Path, ...]
    sequence_length_audit: Path
    publication_identities: Path
    output: Path
    partition_count: int = 64
    parts_per_partition: int = 4
    exclude_sample_ids: tuple[Path, ...] = DEFAULT_GAVIN_EXCLUSION_FILES
    allowed_statuses: tuple[str, ...] = DEFAULT_ALLOWED_STATUSES
    required_exclusion_sha256: str = GAVIN_EXCLUSION_SAMPLE_IDS_SHA256
    max_length: int = DEFAULT_MAX_LENGTH
    max_pixels: int = DEFAULT_MAX_PIXELS
    image_max_token_num: int = DEFAULT_IMAGE_MAX_TOKEN_NUM
    model_type: str = DEFAULT_MODEL_TYPE
    template_mode: str = DEFAULT_TEMPLATE_MODE
    truncation_strategy: str = DEFAULT_TRUNCATION_STRATEGY
    add_non_thinking_prefix: bool = False
    runtime_image_digest: str = DEFAULT_RUNTIME_IMAGE_DIGEST
    model_artifact_manifest_sha256: str = DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256
    overwrite: bool = False
    sqlite_dir: Path | None = None
    reuse_head_verified_source_identities: bool = False


def _load_json(path: Path, description: str) -> tuple[Mapping[str, Any], bytes]:
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{resolved}: invalid {description} JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{resolved}: {description} must be a JSON object")
    return value, raw


def _sha256(value: str, description: str) -> str:
    if len(value) != 64 or value != value.lower():
        raise ValueError(f"{description} must be a lowercase SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{description} must be a lowercase SHA-256") from exc
    return value


def _stable_exclusion_identity(exclusion: Mapping[str, Any]) -> dict[str, Any]:
    sources_raw = exclusion.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("partition exclusion sources must be an array")
    sources = []
    for source in sources_raw:
        if not isinstance(source, Mapping):
            raise ValueError("partition exclusion source must be an object")
        sources.append(
            {
                "rows": int(source.get("rows") or 0),
                "bytes": int(source.get("bytes") or 0),
                "sha256": str(source.get("sha256") or ""),
            }
        )
    return {
        "policy": exclusion.get("policy"),
        "unique_sample_ids": int(exclusion.get("unique_sample_ids") or 0),
        "sample_ids_sha256": str(exclusion.get("sample_ids_sha256") or ""),
        "sources": sources,
    }


def _ordered_summary_shards(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for summary in sorted(summaries, key=lambda item: int(item["partition_index"])):
        raw = summary.get("shards")
        if not isinstance(raw, list):
            raise ValueError("partition summary shards must be an array")
        shards.extend(
            sorted((dict(item) for item in raw), key=lambda item: int(item["part_index"]))
        )
    return shards


def dataset_content_sha256_from_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> str:
    payload = [
        {
            "source_partition": int(item["source_partition"]),
            "part_index": int(item["part_index"]),
            "part_count": int(item["part_count"]),
            "rows": int(item["rows"]),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in _ordered_summary_shards(summaries)
    ]
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _verify_file_identity(path: Path, record: Mapping[str, Any], description: str) -> None:
    resolved = path.expanduser().resolve(strict=True)
    actual_bytes = resolved.stat().st_size
    expected_bytes = int(record.get("bytes") or 0)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{description} byte count changed: actual={actual_bytes} expected={expected_bytes}"
        )
    actual_sha = sha256_file(resolved)
    expected_sha = _sha256(str(record.get("sha256") or ""), f"{description} sha256")
    if actual_sha != expected_sha:
        raise ValueError(
            f"{description} SHA-256 changed: actual={actual_sha} expected={expected_sha}"
        )


def _verify_source_file(summary: Mapping[str, Any]) -> None:
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("partition summary source must be an object")
    path = Path(str(source.get("local_path") or ""))
    _verify_file_identity(path, source, "source final_results")
    rows = 0
    statuses: Counter[str] = Counter()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are forbidden")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            status = value.get("status")
            if not isinstance(status, str) or not status:
                raise ValueError(f"{path}:{line_number}: status is missing")
            rows += 1
            statuses[status] += 1
    if rows != int(source.get("rows") or 0):
        raise ValueError(f"{path}: source row count changed")
    if dict(sorted(statuses.items())) != dict(source.get("status_counts") or {}):
        raise ValueError(f"{path}: source status counts changed")


def _insert_sample_ids(
    connection: sqlite3.Connection,
    rows: Sequence[tuple[str, int]],
) -> None:
    if not rows:
        return
    try:
        connection.executemany(
            "INSERT INTO sample_ids(sample_id, source_partition) VALUES (?, ?)", rows
        )
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        for sample_id, partition in rows:
            existing = connection.execute(
                "SELECT source_partition FROM sample_ids WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    f"duplicate sample_id across prepared shards: {sample_id!r}; "
                    f"partitions={existing[0]},{partition}"
                ) from exc
            connection.execute(
                "INSERT INTO sample_ids(sample_id, source_partition) VALUES (?, ?)",
                (sample_id, partition),
            )
        connection.commit()


def _audit_prepared_shard(
    shard: Mapping[str, Any],
    *,
    connection: sqlite3.Connection,
    excluded_ids: set[str],
    allowed_statuses: set[str],
) -> Counter[str]:
    local_path = Path(str(shard.get("local_path") or "")).expanduser().resolve(
        strict=True
    )
    partition = int(shard.get("source_partition", -1))
    part_index = int(shard.get("part_index", -1))
    part_count = int(shard.get("part_count") or 0)
    expected_bytes = int(shard.get("bytes") or 0)
    if local_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"prepared shard {partition}/{part_index} byte count changed"
        )
    expected_sha256 = _sha256(
        str(shard.get("sha256") or ""),
        f"prepared shard {partition}/{part_index} sha256",
    )
    digest = hashlib.sha256()
    expected_rows = int(shard.get("rows") or 0)
    rows = 0
    statuses: Counter[str] = Counter()
    pending_ids: list[tuple[str, int]] = []
    with local_path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                raise ValueError(f"{local_path}:{line_number}: blank lines are forbidden")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{local_path}:{line_number}: expected an object")
            if set(value) != set(PREPARED_FIELDS):
                raise ValueError(
                    f"{local_path}:{line_number}: prepared fields differ from the frozen row contract"
                )
            sample_id = value.get("sample_id")
            status = value.get("status")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{local_path}:{line_number}: sample_id is missing")
            if sample_id in excluded_ids:
                raise ValueError(f"{local_path}:{line_number}: Gavin exclusion overlap")
            if sample_id_part(sample_id, part_count) != part_index:
                raise ValueError(
                    f"{local_path}:{line_number}: deterministic shard assignment mismatch"
                )
            if not isinstance(status, str) or status not in allowed_statuses:
                raise ValueError(f"{local_path}:{line_number}: status is not allowed")
            rows += 1
            statuses[status] += 1
            pending_ids.append((sample_id, partition))
            if len(pending_ids) >= 10_000:
                _insert_sample_ids(connection, pending_ids)
                connection.commit()
                pending_ids.clear()
    _insert_sample_ids(connection, pending_ids)
    connection.commit()
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"prepared shard {partition}/{part_index} SHA-256 changed: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    if rows != expected_rows:
        raise ValueError(
            f"{local_path}: prepared row count changed: actual={rows} expected={expected_rows}"
        )
    return statuses


def _sample_id_audit(connection: sqlite3.Connection, total_rows: int) -> dict[str, Any]:
    unique = int(connection.execute("SELECT COUNT(*) FROM sample_ids").fetchone()[0])
    if unique != total_rows:
        raise ValueError(
            f"sample-ID uniqueness row count mismatch: unique={unique} total={total_rows}"
        )
    digest = hashlib.sha256()
    first = True
    for (sample_id,) in connection.execute("SELECT sample_id FROM sample_ids ORDER BY sample_id"):
        if not first:
            digest.update(b"\n")
        digest.update(str(sample_id).encode("utf-8"))
        first = False
    return {
        "complete": True,
        "checked_rows": total_rows,
        "unique_sample_ids": unique,
        "duplicate_rows": 0,
        "ordering": "utf8_lexicographic_no_trailing_newline_v1",
        "ordered_sample_ids_sha256": digest.hexdigest(),
    }


def _publication_index(
    path: Path,
) -> tuple[dict[int, Mapping[str, Any]], dict[tuple[int, int], Mapping[str, Any]], dict[str, Any]]:
    document, raw = _load_json(path, "publication identities")
    if document.get("format_version") != PUBLICATION_IDENTITIES_FORMAT:
        raise ValueError("publication identities format_version is incompatible")
    if document.get("complete") is not True:
        raise ValueError("publication identities are incomplete")
    source_records = document.get("source_partitions")
    shard_records = document.get("shards")
    if not isinstance(source_records, list) or not isinstance(shard_records, list):
        raise ValueError("publication identities must contain source_partitions and shards")
    sources: dict[int, Mapping[str, Any]] = {}
    for item in source_records:
        if not isinstance(item, Mapping):
            raise ValueError("publication source identity must be an object")
        index = int(item.get("partition_index", -1))
        if index in sources:
            raise ValueError(f"duplicate publication source identity for partition {index}")
        sources[index] = item
    shards: dict[tuple[int, int], Mapping[str, Any]] = {}
    for item in shard_records:
        if not isinstance(item, Mapping):
            raise ValueError("publication shard identity must be an object")
        key = (int(item.get("source_partition", -1)), int(item.get("part_index", -1)))
        if key in shards:
            raise ValueError(f"duplicate publication shard identity {key}")
        shards[key] = item
    identity = {
        "path": str(path.expanduser().resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return sources, shards, identity


def _verified_publication_record(
    identity: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    uri_field: str,
    description: str,
) -> dict[str, Any]:
    uri = str(identity.get(uri_field) or "")
    version_id = str(identity.get("version_id") or "")
    sha = _sha256(str(local.get("sha256") or ""), f"{description} local sha256")
    if not uri.startswith("s3://"):
        raise ValueError(f"{description}: publication URI must be s3://")
    if version_id in {"", "null", "PENDING_S3_UPLOAD", "LOCAL_ONLY_NOT_PUBLISHED"}:
        raise ValueError(f"{description}: immutable S3 VersionId is missing")
    if int(identity.get("bytes") or 0) != int(local.get("bytes") or 0):
        raise ValueError(f"{description}: published bytes differ from the local object")
    if str(identity.get("sha256") or "") != sha:
        raise ValueError(f"{description}: published SHA-256 differs from the local object")
    if str(identity.get("metadata_sha256") or "") != sha:
        raise ValueError(
            f"{description}: upload must set S3 object metadata sha256={sha}"
        )
    if identity.get("head_verified") is not True:
        raise ValueError(f"{description}: exact VersionId must be HEAD-verified externally")
    if "rows" in identity and int(identity["rows"]) != int(local.get("rows") or 0):
        raise ValueError(f"{description}: published rows differ from the local object")
    return {
        uri_field: uri,
        "version_id": version_id,
        "bytes": int(local["bytes"]),
        "sha256": sha,
    }


def _load_sequence_audit(
    options: FinalizeManifestOptions,
    *,
    total_rows: int,
    dataset_content_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, raw = _load_json(options.sequence_length_audit, "sequence-length audit")
    nested = document.get("sequence_length_audit")
    audit = dict(nested if isinstance(nested, Mapping) else document)
    if "format_version" in document and document.get("format_version") != SEQUENCE_AUDIT_FORMAT:
        raise ValueError("sequence-length audit format_version is incompatible")
    expected = {
        "complete": True,
        "max_length": options.max_length,
        "max_pixels": options.max_pixels,
        "image_max_token_num": options.image_max_token_num,
        "model_type": options.model_type,
        "template_mode": options.template_mode,
        "truncation_strategy": options.truncation_strategy,
        "add_non_thinking_prefix": options.add_non_thinking_prefix,
        "runtime_image_digest": options.runtime_image_digest,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "model_artifact_manifest_sha256": options.model_artifact_manifest_sha256,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "dataset_content_sha256": dataset_content_sha256,
        "checked_rows": total_rows,
        "overlength_rows": 0,
        "assistant_truncation_rows": 0,
        "encode_error_rows": 0,
        "zero_supervised_assistant_rows": 0,
        "json_eos_incomplete_rows": 0,
    }
    for key, expected_value in expected.items():
        if audit.get(key) != expected_value:
            raise ValueError(
                f"sequence_length_audit.{key}={audit.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    maximum = int(audit.get("max_observed_length") or 0)
    if not 0 < maximum <= options.max_length:
        raise ValueError("sequence_length_audit.max_observed_length is invalid")
    source_identity = {
        "path": str(options.sequence_length_audit.expanduser().resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    audit["audit_source"] = source_identity
    return audit, source_identity


def _summary_documents(
    paths: Sequence[Path], partition_count: int
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    if len(paths) != partition_count:
        raise ValueError(
            f"expected exactly {partition_count} partition summaries, got {len(paths)}"
        )
    documents: list[Mapping[str, Any]] = []
    identities: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        document, raw = _load_json(path, "partition summary")
        if document.get("format_version") != SUMMARY_FORMAT or document.get("complete") is not True:
            raise ValueError(f"{path}: incomplete or incompatible partition summary")
        index = int(document.get("partition_index", -1))
        if index in seen:
            raise ValueError(f"duplicate partition summary {index}")
        seen.add(index)
        if int(document.get("partition_count") or 0) != partition_count:
            raise ValueError(f"{path}: partition_count differs")
        documents.append(document)
        identities.append(
            {
                "partition_index": index,
                "path": str(path.expanduser().resolve()),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    expected = set(range(partition_count))
    if seen != expected:
        raise ValueError(
            f"partition summaries must be exactly 0..{partition_count - 1}; "
            f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}"
        )
    documents.sort(key=lambda item: int(item["partition_index"]))
    identities.sort(key=lambda item: int(item["partition_index"]))
    return documents, identities


def finalize_manifest(options: FinalizeManifestOptions) -> dict[str, Any]:
    if options.partition_count < 1 or options.parts_per_partition < 1:
        raise ValueError("partition_count and parts_per_partition must be positive")
    if not options.allowed_statuses or len(set(options.allowed_statuses)) != len(
        options.allowed_statuses
    ):
        raise ValueError("allowed_statuses must be non-empty and unique")
    _sha256(options.required_exclusion_sha256, "required_exclusion_sha256")
    _sha256(
        options.model_artifact_manifest_sha256,
        "model_artifact_manifest_sha256",
    )
    summaries, summary_identities = _summary_documents(
        options.summaries, options.partition_count
    )
    excluded_ids, exclusion = load_excluded_ids(options.exclude_sample_ids)
    if exclusion["sample_ids_sha256"] != options.required_exclusion_sha256:
        raise ValueError(
            "evaluation exclusion digest differs from the required publication: "
            f"actual={exclusion['sample_ids_sha256']} "
            f"expected={options.required_exclusion_sha256}"
        )
    stable_exclusion = _stable_exclusion_identity(exclusion)

    source_identities, shard_identities, publication_file_identity = _publication_index(
        options.publication_identities
    )
    expected_source_keys = set(range(options.partition_count))
    expected_shard_keys = {
        (partition, part)
        for partition in range(options.partition_count)
        for part in range(options.parts_per_partition)
    }
    if set(source_identities) != expected_source_keys:
        raise ValueError("publication source identities do not cover every partition exactly")
    if set(shard_identities) != expected_shard_keys:
        raise ValueError("publication shard identities do not cover every partition/part exactly")

    sqlite_parent = (
        options.sqlite_dir.expanduser().resolve()
        if options.sqlite_dir is not None
        else options.output.expanduser().resolve().parent
    )
    sqlite_parent.mkdir(parents=True, exist_ok=True)
    descriptor, sqlite_name = tempfile.mkstemp(
        prefix=".image-edit-v6-sample-ids.", suffix=".sqlite3", dir=sqlite_parent
    )
    os.close(descriptor)
    sqlite_path = Path(sqlite_name)
    output_status_counts: Counter[str] = Counter()
    source_partition_records: list[dict[str, Any]] = []
    manifest_shards: list[dict[str, Any]] = []
    total_rows = 0
    total_evaluation_excluded = 0

    try:
        connection = sqlite3.connect(sqlite_path)
        try:
            # This database is a disposable uniqueness work index, not a
            # publication artifact. Durability fsyncs make the 7M-row audit
            # several times slower while adding no safety: an interrupted run
            # deletes the index and the manifest is written only after every
            # source/shard/SHA/row gate succeeds. Keep the authoritative input
            # objects and final manifest fully immutable, but build this index
            # with the same bounded-memory settings as the prepare deduper.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE sample_ids ("
                "sample_id TEXT PRIMARY KEY, source_partition INTEGER NOT NULL"
                ") WITHOUT ROWID"
            )
            for summary in summaries:
                partition = int(summary["partition_index"])
                invalid_policy = str(summary.get("invalid_policy") or "")
                if invalid_policy not in {"error", "skip"}:
                    raise ValueError(
                        f"partition {partition}: invalid_policy must be error or skip"
                    )
                input_rows = int(summary.get("input_rows") or 0)
                excluded_rows = int(summary.get("excluded_rows") or 0)
                rejected_rows = int(summary.get("rejected_rows") or 0)
                evaluation_excluded_rows = int(
                    summary.get("evaluation_excluded_rows") or 0
                )
                status_excluded_rows = int(summary.get("status_excluded_rows") or 0)
                output_rows = int(summary.get("output_rows") or 0)
                if (
                    input_rows != excluded_rows + rejected_rows + output_rows
                    or evaluation_excluded_rows < 0
                    or status_excluded_rows < 0
                    or status_excluded_rows > excluded_rows
                    or rejected_rows < 0
                    or (invalid_policy == "error" and rejected_rows != 0)
                    or output_rows < 1
                ):
                    raise ValueError(f"partition {partition}: row conservation failed")
                contract = summary.get("dataset_contract")
                if not isinstance(contract, Mapping):
                    raise ValueError(f"partition {partition}: dataset contract is missing")
                expected_contract = {
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
                }
                for key, expected in expected_contract.items():
                    if contract.get(key) != expected:
                        raise ValueError(
                            f"partition {partition}: dataset_contract.{key} differs"
                        )
                summary_exclusion = summary.get("exclusion")
                if not isinstance(summary_exclusion, Mapping) or _stable_exclusion_identity(
                    summary_exclusion
                ) != stable_exclusion:
                    raise ValueError(f"partition {partition}: evaluation exclusion differs")
                if int(summary_exclusion.get("output_overlap_rows") or 0) != 0:
                    raise ValueError(f"partition {partition}: exclusion overlap is non-zero")
                if int(summary_exclusion.get("matched_source_rows") or 0) != (
                    evaluation_excluded_rows
                ):
                    raise ValueError(
                        f"partition {partition}: evaluation exclusion count differs"
                    )
                if not options.reuse_head_verified_source_identities:
                    _verify_source_file(summary)

                raw_shards = summary.get("shards")
                if not isinstance(raw_shards, list) or len(raw_shards) != options.parts_per_partition:
                    raise ValueError(
                        f"partition {partition}: expected exactly {options.parts_per_partition} shards"
                    )
                raw_shards = sorted(raw_shards, key=lambda item: int(item["part_index"]))
                if [int(item.get("part_index", -1)) for item in raw_shards] != list(
                    range(options.parts_per_partition)
                ):
                    raise ValueError(f"partition {partition}: part indexes are incomplete")

                actual_partition_statuses: Counter[str] = Counter()
                actual_partition_rows = 0
                for local_shard in raw_shards:
                    if (
                        int(local_shard.get("source_partition", -1)) != partition
                        or int(local_shard.get("part_count") or 0)
                        != options.parts_per_partition
                        or local_shard.get("assignment") != SHARD_ASSIGNMENT
                        or int(local_shard.get("rows") or 0) < 1
                        or int(local_shard.get("bytes") or 0) < 1
                    ):
                        raise ValueError(f"partition {partition}: invalid local shard summary")
                    part_index = int(local_shard["part_index"])
                    statuses = _audit_prepared_shard(
                        local_shard,
                        connection=connection,
                        excluded_ids=excluded_ids,
                        allowed_statuses=set(options.allowed_statuses),
                    )
                    actual_partition_statuses.update(statuses)
                    actual_partition_rows += int(local_shard["rows"])
                    published = _verified_publication_record(
                        shard_identities[(partition, part_index)],
                        local_shard,
                        uri_field="path",
                        description=f"prepared shard {partition}/{part_index}",
                    )
                    manifest_shards.append(
                        {
                            **published,
                            "rows": int(local_shard["rows"]),
                            "source_partition": partition,
                            "part_index": part_index,
                            "part_count": options.parts_per_partition,
                        }
                    )
                declared_statuses = {
                    str(key): int(value)
                    for key, value in dict(summary.get("status_counts") or {}).items()
                }
                if (
                    actual_partition_rows != output_rows
                    or dict(sorted(actual_partition_statuses.items())) != declared_statuses
                    or sum(declared_statuses.values()) != output_rows
                ):
                    raise ValueError(f"partition {partition}: prepared row/status audit differs")
                output_status_counts.update(actual_partition_statuses)
                total_rows += output_rows
                total_evaluation_excluded += evaluation_excluded_rows

                source = summary["source"]
                published_source = _verified_publication_record(
                    source_identities[partition],
                    source,
                    uri_field="uri",
                    description=f"source final_results partition {partition}",
                )
                source_partition_records.append(
                    {
                        "partition_index": partition,
                        "stage": "call2_final",
                        "complete": True,
                        "input_rows": input_rows,
                        "excluded_rows": excluded_rows,
                        "rejected_rows": rejected_rows,
                        "output_rows": output_rows,
                        "source_status_counts": dict(summary["source_status_counts"]),
                        "output_status_counts": declared_statuses,
                        **published_source,
                    }
                )

            sample_id_audit = _sample_id_audit(connection, total_rows)
        finally:
            connection.close()

        content_digest = dataset_content_sha256_from_summaries(summaries)
        sequence_audit, sequence_audit_identity = _load_sequence_audit(
            options,
            total_rows=total_rows,
            dataset_content_sha256=content_digest,
        )
        if sum(output_status_counts.values()) != total_rows:
            raise AssertionError("global status count conservation failed")

        exclusion_contract = {
            **exclusion,
            "matched_source_rows": total_evaluation_excluded,
            "overlap_rows": 0,
        }
        manifest = {
            "format_version": MANIFEST_FORMAT,
            "complete": True,
            "physical_shard_count": len(manifest_shards),
            "total_rows": total_rows,
            "status_counts": dict(sorted(output_status_counts.items())),
            "paths": [str(item["path"]) for item in manifest_shards],
            "shards": manifest_shards,
            "source_partitions": source_partition_records,
            "sample_id_audit": sample_id_audit,
            "sequence_length_audit": sequence_audit,
            "dataset_content_sha256": content_digest,
            "dataset_contract": {
                "output_format": "prepared",
                "row_format_version": ROW_FORMAT,
                "runtime_loader": "dlms/datasets/image_edit_v6_full_json.py",
                "target_schema": TARGET_SCHEMA,
                "target_json_field": "final_json",
                "instruction_field": "instruction",
                "reference_image_field": "source_s3_uri",
                "target_image_field": "target_s3_uri",
                "audit_valid_field": "call2_audit.valid",
                "image_order": ["reference frame_0", "target"],
                "selection": "authoritative_call2_final_json",
                "source_stage": "call2_final",
                "prompt_version": PROMPT_VERSION,
                "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
                "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
                "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
                "allowed_statuses": list(options.allowed_statuses),
                "exclusion": exclusion_contract,
                "bbox_coordinate_order": {
                    "target_layout": "xyxy",
                    "reference_bbox": "xyxy",
                    "remove_reference_bbox": "xyxy",
                },
            },
            "publication_audit": {
                "complete": True,
                "method": "external_head_verified_s3_version_ids_v1",
                "publication_identities": publication_file_identity,
                "partition_summaries": summary_identities,
                "sequence_length_audit": sequence_audit_identity,
                "source_object_count": len(source_partition_records),
                "shard_object_count": len(manifest_shards),
                "required_s3_metadata_key": "sha256",
                "all_records_head_verified": True,
                "source_local_rehash": not options.reuse_head_verified_source_identities,
            },
        }
        if len(manifest_shards) != options.partition_count * options.parts_per_partition:
            raise AssertionError("physical shard count differs from partition_count * parts")
        output = options.output.expanduser().resolve()
        payload = json_document_bytes(manifest)
        write_atomic(output, payload, overwrite=options.overwrite)
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        return {
            "manifest": manifest,
            "output": str(output),
            "bytes": len(payload),
            "sha256": manifest_sha256,
            "config_pins": {
                "manifest_sha256": manifest_sha256,
                "manifest_version_id": "PENDING_MANIFEST_S3_UPLOAD_AND_HEAD",
                "required_total_rows": total_rows,
                "required_status_counts": dict(sorted(output_status_counts.items())),
            },
            "manifest_upload_contract": {
                "required_content_type": "application/json",
                "required_metadata": {"sha256": manifest_sha256},
                "capture_version_id": True,
            },
        }
    finally:
        sqlite_path.unlink(missing_ok=True)
        Path(f"{sqlite_path}-wal").unlink(missing_ok=True)
        Path(f"{sqlite_path}-shm").unlink(missing_ok=True)


def _summary_paths(explicit: Sequence[Path], directory: Path | None, pattern: str) -> tuple[Path, ...]:
    paths = [path.expanduser().resolve(strict=True) for path in explicit]
    if directory is not None:
        root = directory.expanduser().resolve(strict=True)
        paths.extend(sorted(root.glob(pattern)))
    if not paths:
        raise ValueError("provide --summary or --summary-dir")
    resolved = tuple(dict.fromkeys(path.resolve() for path in paths))
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--summary-glob", default="partition-*.summary.json")
    parser.add_argument("--sequence-length-audit", type=Path, required=True)
    parser.add_argument("--publication-identities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-count", type=int, default=64)
    parser.add_argument("--parts-per-partition", type=int, default=4)
    parser.add_argument(
        "--exclude-sample-ids", action="append", type=Path, default=None
    )
    parser.add_argument("--allowed-status", action="append", default=None)
    parser.add_argument(
        "--required-exclusion-sha256", default=GAVIN_EXCLUSION_SAMPLE_IDS_SHA256
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--image-max-token-num", type=int, default=DEFAULT_IMAGE_MAX_TOKEN_NUM
    )
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--template-mode", default=DEFAULT_TEMPLATE_MODE)
    parser.add_argument("--truncation-strategy", default=DEFAULT_TRUNCATION_STRATEGY)
    parser.add_argument(
        "--add-non-thinking-prefix", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--runtime-image-digest", default=DEFAULT_RUNTIME_IMAGE_DIGEST)
    parser.add_argument(
        "--model-artifact-manifest-sha256",
        default=DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256,
    )
    parser.add_argument("--sqlite-dir", type=Path)
    parser.add_argument(
        "--reuse-head-verified-source-identities",
        action="store_true",
        help=(
            "skip the redundant local source JSONL scan and bind source rows to "
            "the exact HEAD-verified S3 VersionIds in publication identities"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = FinalizeManifestOptions(
        summaries=_summary_paths(args.summary, args.summary_dir, args.summary_glob),
        sequence_length_audit=args.sequence_length_audit,
        publication_identities=args.publication_identities,
        output=args.output,
        partition_count=args.partition_count,
        parts_per_partition=args.parts_per_partition,
        exclude_sample_ids=tuple(
            args.exclude_sample_ids
            if args.exclude_sample_ids is not None
            else DEFAULT_GAVIN_EXCLUSION_FILES
        ),
        allowed_statuses=tuple(args.allowed_status or DEFAULT_ALLOWED_STATUSES),
        required_exclusion_sha256=args.required_exclusion_sha256,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        image_max_token_num=args.image_max_token_num,
        model_type=args.model_type,
        template_mode=args.template_mode,
        truncation_strategy=args.truncation_strategy,
        add_non_thinking_prefix=args.add_non_thinking_prefix,
        runtime_image_digest=args.runtime_image_digest,
        model_artifact_manifest_sha256=args.model_artifact_manifest_sha256,
        overwrite=args.overwrite,
        sqlite_dir=args.sqlite_dir,
        reuse_head_verified_source_identities=args.reuse_head_verified_source_identities,
    )
    report = finalize_manifest(options)
    print(
        json.dumps(
            {
                "output": report["output"],
                "bytes": report["bytes"],
                "sha256": report["sha256"],
                "config_pins": report["config_pins"],
                "manifest_upload_contract": report["manifest_upload_contract"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
