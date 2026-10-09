#!/usr/bin/env python3
"""Plan deterministic single-reference work units for the sidecar24 corpus.

This script scans only small metadata columns.  It never reads embedded image
bytes and it never performs a HEAD request for an image URI.  Rows are ordered
by catalog ordinal, lexicographic main-Parquet object key, and Parquet-local row.
For a dataset with a sidecar, ``sample_id`` is an exact multiset join. Repeated
IDs are valid only when main and sidecar contain the same multiplicity for that
ID; missing or excess matching main occurrences fail instead of being paired
ambiguously. Main IDs absent from the sidecar remain excluded.

The output is a canonical ``image_edit_sidecar24_unit_catalog_v1`` JSON file.
Each work unit belongs to exactly one physical Parquet object and contains at
most 500 selected (``num_inputs == 1``) rows.  Sparse local rows are represented
as maximally compressed ``[start, end_exclusive]`` ranges.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.image_edit_sidecar24_catalog import (  # noqa: E402
    CRAFTV0_SOURCE_OBJECT_ORDERING,
    CRAFTV0_SCHEMA_VERSION,
    EXPECTED_DATASET_COUNT,
    SCHEMA_VERSION as SOURCE_CATALOG_VERSION,
    canonical_sha256 as source_catalog_sha256,
    expected_dataset_count as source_expected_dataset_count,
    expected_total_selected_rows as source_expected_total_selected_rows,
    load_catalog,
    validate_catalog,
)
from scripts.prepare_image_edit_full_json_shard import (  # noqa: E402
    S3RangeReader,
    parse_s3_uri,
)


FORMAT_VERSION = "image_edit_sidecar24_unit_catalog_v1"
INVENTORY_FORMAT_VERSION = "image_edit_sidecar24_parquet_inventory_v1"
EXPECTED_TOTAL_SELECTED_ROWS = 11_204_822
MAX_SELECTED_ROWS_PER_UNIT = 500
CRAFTV0_MAX_SELECTED_ROWS_PER_UNIT = 1024

URI_ADAPTER = "uri_single_v1"
EMBEDDED_ADAPTER = "embedded_single_v1"
EMBEDDED_SYNTH_ADAPTER = "embedded_synth_v1"
ALLOWED_ADAPTERS = frozenset({URI_ADAPTER, EMBEDDED_ADAPTER, EMBEDDED_SYNTH_ADAPTER})

MAIN_SCAN_COLUMNS = (
    "sample_id",
    "num_inputs",
    "instructions",
    "input_s3_paths",
    "output_s3_path",
    "data_source",
    "edit_instruction",
    "source_image",
    "target_image",
)
ADAPTER_REQUIRED_COLUMNS = {
    URI_ADAPTER: frozenset(
        {
            "sample_id",
            "num_inputs",
            "instructions",
            "input_s3_paths",
            "output_s3_path",
        }
    ),
    EMBEDDED_ADAPTER: frozenset(
        {
            "sample_id",
            "num_inputs",
            "instructions",
            "data_source",
            # Footer-only checks: the planner never reads these byte columns.
            "input_images",
            "output_image",
        }
    ),
    EMBEDDED_SYNTH_ADAPTER: frozenset(
        {
            "sample_id",
            "edit_instruction",
            "source_image",
            "target_image",
        }
    ),
}

SPECIAL_EXPECTED_SELECTED_ROWS = {
    "spatial": 10_200,
    "mico": 0,
    "echo4o": 0,
    "illustration": 14_896,
    "macro": 8_703,
}


def catalog_expected_dataset_count(catalog: Mapping[str, Any]) -> int:
    if catalog.get("schema_version") == CRAFTV0_SCHEMA_VERSION:
        return source_expected_dataset_count(catalog)
    return EXPECTED_DATASET_COUNT


def catalog_expected_total_selected_rows(catalog: Mapping[str, Any]) -> int:
    if catalog.get("schema_version") == CRAFTV0_SCHEMA_VERSION:
        return source_expected_total_selected_rows(catalog)
    return EXPECTED_TOTAL_SELECTED_ROWS


def max_selected_rows_limit(value: Mapping[str, Any]) -> int:
    if (
        value.get("schema_version") == CRAFTV0_SCHEMA_VERSION
        or value.get("planning_mode") == "footer_rows_v1"
    ):
        return CRAFTV0_MAX_SELECTED_ROWS_PER_UNIT
    return MAX_SELECTED_ROWS_PER_UNIT

SELECTION_ORDERING = "catalog_ordinal_then_lexicographic_s3_key_then_parquet_row"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

UNIT_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_ordinal",
        "dataset_unit_index",
        "global_unit_index",
        "dataset_row_start",
        "dataset_row_end_exclusive",
        "global_row_start",
        "global_row_end_exclusive",
        "local_row_ranges",
        "source",
        "adapter",
        "main_s3_uri",
    }
)
SOURCE_FIELDS = frozenset({"bucket", "key", "etag", "bytes", "rows", "object_index"})
SUMMARY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_ordinal",
        "family",
        "adapter",
        "reference_policy",
        "main_s3_uri",
        "sidecar_s3_uri",
        "catalog_row_count",
        "main_rows",
        "joined_rows",
        "expected_selected_rows",
        "selected_rows",
        "unit_count",
        "global_unit_start",
        "global_unit_end_exclusive",
        "global_row_start",
        "global_row_end_exclusive",
        "main_inventory_sha256",
        "sidecar_inventory_sha256",
    }
)
TOP_LEVEL_FIELDS = frozenset(
    {
        "format_version",
        "source_catalog",
        "selection",
        "dataset_count",
        "total_selected_rows",
        "total_units",
        "datasets",
        "units",
    }
)


class UnitPlanningError(ValueError):
    """Raised for source, join, selection, or unit-catalog contract failures."""


ObjectRowReader = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class DatasetScan:
    """Lazy inputs required to plan one dataset.

    ``sidecar_sample_ids`` may be a generator.  It is consumed exactly once and
    checked against the sidecar inventory row count before main rows are read.
    """

    main_inventory: Mapping[str, Any]
    read_main_object: ObjectRowReader
    sidecar_inventory: Mapping[str, Any] | None = None
    sidecar_sample_ids: Iterable[str] | None = None


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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise UnitPlanningError(f"{location}: all field names must be strings")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise UnitPlanningError(f"{location}: {', '.join(details)}")


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnitPlanningError(f"{location} must be a non-empty string")
    return value


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise UnitPlanningError(f"{location} must be a lowercase SHA-256 hex digest")
    return value


def _normalized_s3_uri(uri: str) -> str:
    location = parse_s3_uri(uri.rstrip("/"))
    return location.uri


def expected_selected_rows(dataset: Mapping[str, Any]) -> int:
    if "selected_row_count" in dataset:
        return int(dataset["selected_row_count"])
    dataset_id = str(dataset["dataset_id"])
    if dataset_id in SPECIAL_EXPECTED_SELECTED_ROWS:
        return SPECIAL_EXPECTED_SELECTED_ROWS[dataset_id]
    return int(dataset["row_count"])


def validate_selection_expectations(catalog: Mapping[str, Any]) -> dict[str, int]:
    """Return the exact expected selected-row count for every dataset."""

    validate_catalog(catalog)
    datasets = catalog["datasets"]
    if catalog["schema_version"] == SOURCE_CATALOG_VERSION:
        ids = {str(dataset["dataset_id"]) for dataset in datasets}
        missing = sorted(set(SPECIAL_EXPECTED_SELECTED_ROWS).difference(ids))
        if missing:
            raise UnitPlanningError(f"catalog lacks required selection-rule datasets: {missing}")
    expected = {str(dataset["dataset_id"]): expected_selected_rows(dataset) for dataset in datasets}
    total = sum(expected.values())
    expected_total = catalog_expected_total_selected_rows(catalog)
    if total != expected_total:
        raise UnitPlanningError(
            f"per-dataset expected selected rows sum to {total}, expected {expected_total}"
        )
    return expected


def compress_local_row_ranges(local_rows: Sequence[int]) -> list[list[int]]:
    """Compress strictly increasing local row indices into maximal half-open ranges."""

    if not local_rows:
        return []
    ranges: list[list[int]] = []
    start = previous = local_rows[0]
    if not _is_int(start) or start < 0:
        raise UnitPlanningError("local row indices must be non-negative integers")
    for current in local_rows[1:]:
        if not _is_int(current) or current <= previous:
            raise UnitPlanningError("local row indices must be strictly increasing integers")
        if current != previous + 1:
            ranges.append([start, previous + 1])
            start = current
        previous = current
    ranges.append([start, previous + 1])
    return ranges


def _inventory_objects(
    inventory: Mapping[str, Any], *, expected_uri: str, location: str
) -> list[Mapping[str, Any]]:
    if not isinstance(inventory, Mapping):
        raise UnitPlanningError(f"{location} must be an object")
    if inventory.get("ordering") != "lexicographic_s3_key_then_parquet_row":
        raise UnitPlanningError(f"{location}.ordering is not authoritative")
    inventory_uri = inventory.get("s3_uri")
    if not isinstance(inventory_uri, str) or _normalized_s3_uri(inventory_uri) != _normalized_s3_uri(
        expected_uri
    ):
        raise UnitPlanningError(f"{location}.s3_uri does not match {expected_uri}")
    raw_objects = inventory.get("objects")
    if not isinstance(raw_objects, list):
        raise UnitPlanningError(f"{location}.objects must be an array")
    if not _is_int(inventory.get("object_count")) or inventory["object_count"] != len(raw_objects):
        raise UnitPlanningError(f"{location}.object_count does not match objects")

    expected_bucket = parse_s3_uri(expected_uri.rstrip("/")).bucket
    cursor = 0
    previous_key: str | None = None
    for object_index, item in enumerate(raw_objects):
        item_location = f"{location}.objects[{object_index}]"
        if not isinstance(item, Mapping):
            raise UnitPlanningError(f"{item_location} must be an object")
        for field in (
            "bucket",
            "key",
            "etag",
            "bytes",
            "rows",
            "object_index",
            "global_row_start",
            "global_row_end_exclusive",
        ):
            if field not in item:
                raise UnitPlanningError(f"{item_location} lacks {field!r}")
        bucket = _require_nonempty_string(item["bucket"], f"{item_location}.bucket")
        key = _require_nonempty_string(item["key"], f"{item_location}.key")
        _require_nonempty_string(item["etag"], f"{item_location}.etag")
        if bucket != expected_bucket:
            raise UnitPlanningError(f"{item_location}.bucket does not match inventory URI")
        if previous_key is not None and key <= previous_key:
            raise UnitPlanningError(f"{location}.objects are not in strict key order")
        previous_key = key
        if not _is_int(item["object_index"]) or item["object_index"] != object_index:
            raise UnitPlanningError(f"{item_location}.object_index must equal {object_index}")
        if not _is_int(item["bytes"]) or item["bytes"] <= 0:
            raise UnitPlanningError(f"{item_location}.bytes must be positive")
        if not _is_int(item["rows"]) or item["rows"] <= 0:
            raise UnitPlanningError(f"{item_location}.rows must be positive")
        if item["global_row_start"] != cursor:
            raise UnitPlanningError(f"{item_location}.global_row_start is not contiguous")
        cursor += item["rows"]
        if item["global_row_end_exclusive"] != cursor:
            raise UnitPlanningError(f"{item_location}.global_row_end_exclusive is inconsistent")

    if not _is_int(inventory.get("total_rows")) or inventory["total_rows"] != cursor:
        raise UnitPlanningError(f"{location}.total_rows does not match object rows")
    return raw_objects


def _project_inventory_object_snapshot(
    inventory: Mapping[str, Any],
    *,
    expected_uri: str,
    location: str,
    dataset: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return an inventory restricted to a catalog-frozen S3-time snapshot.

    Membership is recovered in S3 ``LastModified`` order with the key as a
    deterministic tie-breaker.  Selected objects are then restored to the
    canonical lexicographic-key row order used by inference.  The explicit
    boundary makes this fail closed if the S3 snapshot cannot be reconstructed.
    """

    objects = _inventory_objects(
        inventory,
        expected_uri=expected_uri,
        location=location,
    )
    object_count = dataset.get("source_object_count")
    if object_count is None:
        return inventory
    if not _is_int(object_count) or object_count <= 0:
        raise UnitPlanningError(f"{location} source_object_count must be a positive integer")
    if object_count > len(objects):
        raise UnitPlanningError(
            f"{location} source_object_count={object_count} exceeds "
            f"discovered object_count={len(objects)}"
        )
    if dataset.get("source_object_ordering") != CRAFTV0_SOURCE_OBJECT_ORDERING:
        raise UnitPlanningError(
            f"{location} source_object_ordering must equal "
            f"{CRAFTV0_SOURCE_OBJECT_ORDERING!r}"
        )
    for item in objects:
        modified = item.get("last_modified")
        if not isinstance(modified, str) or not modified.endswith("+00:00"):
            raise UnitPlanningError(
                f"{location} object {item.get('key')!r} lacks canonical S3 last_modified"
            )
    chronological = sorted(objects, key=lambda item: (item["last_modified"], item["key"]))
    selected = chronological[:object_count]
    boundary = selected[-1]
    if boundary["key"] != dataset.get("source_object_boundary_key"):
        raise UnitPlanningError(
            f"{location} frozen snapshot boundary key changed: "
            f"actual={boundary['key']!r} "
            f"expected={dataset.get('source_object_boundary_key')!r}"
        )
    if boundary["last_modified"] != dataset.get("source_object_boundary_last_modified"):
        raise UnitPlanningError(
            f"{location} frozen snapshot boundary LastModified changed: "
            f"actual={boundary['last_modified']!r} "
            f"expected={dataset.get('source_object_boundary_last_modified')!r}"
        )
    projected = dict(inventory)
    projected_objects = []
    cursor = 0
    for object_index, raw_item in enumerate(sorted(selected, key=lambda item: item["key"])):
        item = dict(raw_item)
        item["object_index"] = object_index
        item["global_row_start"] = cursor
        cursor += int(item["rows"])
        item["global_row_end_exclusive"] = cursor
        projected_objects.append(item)
    projected["objects"] = projected_objects
    projected["object_count"] = object_count
    projected["total_rows"] = cursor
    _inventory_objects(
        projected,
        expected_uri=expected_uri,
        location=f"{location}.frozen_prefix",
    )
    return projected


def _sidecar_join_state(
    sample_ids: Iterable[str], *, expected_rows: int, dataset_id: str
) -> dict[str, int]:
    """Return remaining sidecar multiplicities keyed by ``sample_id``.

    The planner only uses the sidecar as a row-selection membership filter; it
    never attaches sidecar payload columns to main rows. Consequently repeated
    keys are unambiguous only if *all* occurrences on both sides are selected.
    Tracking multiplicity preserves that invariant without silently choosing
    one of several same-key main rows.
    """

    state: dict[str, int] = {}
    row_count = 0
    for row_index, sample_id in enumerate(sample_ids):
        if not isinstance(sample_id, str) or not sample_id:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} sidecar row {row_index} has invalid sample_id"
            )
        state[sample_id] = state.get(sample_id, 0) + 1
        row_count += 1
    if row_count != expected_rows:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} sidecar yielded {row_count} sample IDs; "
            f"expected {expected_rows}"
        )
    return state


def _validate_uri_row(row: Mapping[str, Any], location: str) -> None:
    input_paths = row.get("input_s3_paths")
    if (
        not isinstance(input_paths, (list, tuple))
        or len(input_paths) != 1
        or not isinstance(input_paths[0], str)
        or not input_paths[0].strip()
    ):
        raise UnitPlanningError(f"{location}.input_s3_paths must contain one non-empty path")
    output_path = row.get("output_s3_path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise UnitPlanningError(f"{location}.output_s3_path must be a non-empty path")
    for field, uri in (
        ("input_s3_paths[0]", input_paths[0]),
        ("output_s3_path", output_path),
    ):
        try:
            parse_s3_uri(uri)
        except ValueError as exc:
            raise UnitPlanningError(f"{location}.{field} must be an s3:// URI") from exc


def _validate_instruction(row: Mapping[str, Any], location: str) -> None:
    raw = row.get("instructions")
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or not any(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise UnitPlanningError(
            f"{location}.instructions must contain at least one non-empty string"
        )


def _validate_synth_row(row: Mapping[str, Any], location: str) -> None:
    instruction = row.get("edit_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise UnitPlanningError(f"{location}.edit_instruction must be a non-empty string")


def _selected_by_catalog_rule(dataset: Mapping[str, Any], dataset_row_index: int) -> bool:
    if dataset.get("planning_mode") != "footer_rows_v1":
        return True
    modulus = int(dataset["selection_modulus"])
    residue = int(dataset["selection_residue"])
    return dataset_row_index % modulus == residue


def _source_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bucket": item["bucket"],
        "key": item["key"],
        "etag": item["etag"],
        "bytes": item["bytes"],
        "rows": item["rows"],
        "object_index": item["object_index"],
    }


def _ordered_object_rows(
    objects: Sequence[Mapping[str, Any]],
    reader: ObjectRowReader,
    *,
    workers: int,
) -> Iterator[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]]:
    if workers <= 0:
        raise UnitPlanningError("scan workers must be positive")
    if workers == 1:
        for item in objects:
            yield item, reader(item)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        source = iter(objects)
        pending: deque[tuple[Mapping[str, Any], concurrent.futures.Future[Sequence[Mapping[str, Any]]]]] = (
            deque()
        )
        for _ in range(workers):
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append((item, pool.submit(reader, item)))
        while pending:
            item, future = pending.popleft()
            yield item, future.result()
            try:
                next_item = next(source)
            except StopIteration:
                continue
            pending.append((next_item, pool.submit(reader, next_item)))


def plan_dataset_units(
    dataset: Mapping[str, Any],
    scan: DatasetScan,
    *,
    expected_selected: int,
    global_unit_start: int,
    global_row_start: int,
    max_selected_rows: int = MAX_SELECTED_ROWS_PER_UNIT,
    scan_workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Plan one dataset using lazy object-row reads.

    This is the main local-test seam: callers can supply in-memory inventories,
    rows, and sidecar IDs without importing boto3 or PyArrow.
    """

    dataset_id = str(dataset["dataset_id"])
    ordinal = int(dataset["ordinal"])
    adapter = str(dataset["adapter"])
    if adapter not in ALLOWED_ADAPTERS:
        raise UnitPlanningError(f"dataset {dataset_id!r} adapter must be one of {sorted(ALLOWED_ADAPTERS)}")
    if not _is_int(expected_selected) or expected_selected < 0:
        raise UnitPlanningError("expected selected rows must be a non-negative integer")
    selected_rows_limit = max_selected_rows_limit(dataset)
    if not _is_int(max_selected_rows) or not 1 <= max_selected_rows <= selected_rows_limit:
        raise UnitPlanningError(
            f"max selected rows per unit must be between 1 and {selected_rows_limit}"
        )

    main_inventory = _project_inventory_object_snapshot(
        scan.main_inventory,
        expected_uri=str(dataset["main_s3_uri"]),
        location=f"{dataset_id}.main_inventory",
        dataset=dataset,
    )
    main_objects = _inventory_objects(
        main_inventory,
        expected_uri=str(dataset["main_s3_uri"]),
        location=f"{dataset_id}.main_inventory.frozen",
    )
    _inventory_has_required_columns(
        main_inventory,
        required_columns=ADAPTER_REQUIRED_COLUMNS[adapter],
        location=f"{dataset_id}.main_inventory",
    )
    main_rows = int(main_inventory["total_rows"])
    main_inventory_digest = canonical_sha256(main_inventory)

    sidecar_uri = dataset["sidecar_s3_uri"]
    sidecar_digest: str | None
    join_state: dict[str, int] | None
    catalog_rows = int(dataset["row_count"])
    if sidecar_uri is None:
        if scan.sidecar_inventory is not None or scan.sidecar_sample_ids is not None:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} has no sidecar URI but sidecar scan inputs were provided"
            )
        if main_rows != catalog_rows:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} has no sidecar and main rows={main_rows}, "
                f"catalog rows={catalog_rows}"
            )
        sidecar_digest = None
        join_state = None
        joined_rows = main_rows
    else:
        if scan.sidecar_inventory is None or scan.sidecar_sample_ids is None:
            raise UnitPlanningError(f"dataset {dataset_id!r} requires sidecar inventory and IDs")
        _inventory_objects(
            scan.sidecar_inventory,
            expected_uri=str(sidecar_uri),
            location=f"{dataset_id}.sidecar_inventory",
        )
        _inventory_has_required_columns(
            scan.sidecar_inventory,
            required_columns=("sample_id",),
            location=f"{dataset_id}.sidecar_inventory",
        )
        if int(scan.sidecar_inventory["total_rows"]) != catalog_rows:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} sidecar inventory rows="
                f"{scan.sidecar_inventory['total_rows']}, catalog rows={catalog_rows}"
            )
        sidecar_digest = canonical_sha256(scan.sidecar_inventory)
        join_state = _sidecar_join_state(
            scan.sidecar_sample_ids,
            expected_rows=catalog_rows,
            dataset_id=dataset_id,
        )
        joined_rows = catalog_rows

    units: list[dict[str, Any]] = []
    dataset_row_cursor = 0
    dataset_unit_index = 0
    footer_rows = dataset.get("planning_mode") == "footer_rows_v1"
    if footer_rows and sidecar_uri is not None:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} footer_rows_v1 does not support a sidecar join"
        )
    object_rows: Iterable[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]] | None]]
    if footer_rows:
        object_rows = ((item, None) for item in main_objects)
    else:
        object_rows = _ordered_object_rows(
            main_objects,
            scan.read_main_object,
            workers=scan_workers,
        )

    for item, rows in object_rows:
        object_index = int(item["object_index"])
        if footer_rows:
            raw_start = int(item["global_row_start"])
            selected_local_rows = [
                local_row
                for local_row in range(int(item["rows"]))
                if _selected_by_catalog_rule(dataset, raw_start + local_row)
            ]
        else:
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                raise UnitPlanningError(
                    f"dataset {dataset_id!r} object {object_index} reader must return a row sequence"
                )
            if len(rows) != int(item["rows"]):
                raise UnitPlanningError(
                    f"dataset {dataset_id!r} object {object_index} returned {len(rows)} rows; "
                    f"inventory declares {item['rows']}"
                )
            selected_local_rows = []
            for local_row, row in enumerate(rows):
                location = f"dataset {dataset_id!r} object {object_index} row {local_row}"
                if not isinstance(row, Mapping):
                    raise UnitPlanningError(f"{location} must be an object")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise UnitPlanningError(f"{location}.sample_id must be a non-empty string")

                if join_state is not None:
                    if sample_id not in join_state:
                        continue
                    remaining = join_state[sample_id]
                    if remaining == 0:
                        raise UnitPlanningError(
                            f"dataset {dataset_id!r} main has more occurrences than sidecar for "
                            f"sample_id {sample_id!r}"
                        )
                    join_state[sample_id] = remaining - 1

                if adapter == EMBEDDED_SYNTH_ADAPTER:
                    _validate_synth_row(row, location)
                else:
                    num_inputs = row.get("num_inputs")
                    if not _is_int(num_inputs) or num_inputs < 0:
                        raise UnitPlanningError(
                            f"{location}.num_inputs must be a non-negative integer"
                        )
                    if num_inputs != 1:
                        continue
                    _validate_instruction(row, location)
                    if adapter == URI_ADAPTER:
                        _validate_uri_row(row, location)
                selected_local_rows.append(local_row)

        for chunk_start in range(0, len(selected_local_rows), max_selected_rows):
            chunk = selected_local_rows[chunk_start : chunk_start + max_selected_rows]
            selected_count = len(chunk)
            dataset_row_end = dataset_row_cursor + selected_count
            units.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_ordinal": ordinal,
                    "dataset_unit_index": dataset_unit_index,
                    "global_unit_index": global_unit_start + len(units),
                    "dataset_row_start": dataset_row_cursor,
                    "dataset_row_end_exclusive": dataset_row_end,
                    "global_row_start": global_row_start + dataset_row_cursor,
                    "global_row_end_exclusive": global_row_start + dataset_row_end,
                    "local_row_ranges": compress_local_row_ranges(chunk),
                    "source": _source_identity(item),
                    "adapter": adapter,
                    "main_s3_uri": dataset["main_s3_uri"],
                }
            )
            dataset_row_cursor = dataset_row_end
            dataset_unit_index += 1

    if join_state is not None:
        missing = [(sample_id, remaining) for sample_id, remaining in join_state.items() if remaining]
        if missing:
            missing_occurrences = sum(remaining for _, remaining in missing)
            raise UnitPlanningError(
                f"dataset {dataset_id!r} has {missing_occurrences} unmatched sidecar sample-id "
                f"occurrences across {len(missing)} keys; "
                f"examples={missing[:5]}"
            )
    if dataset_row_cursor != expected_selected:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} selected {dataset_row_cursor} rows; expected {expected_selected}"
        )

    summary = {
        "dataset_id": dataset_id,
        "dataset_ordinal": ordinal,
        "family": dataset["family"],
        "adapter": adapter,
        "reference_policy": dataset["reference_policy"],
        "main_s3_uri": dataset["main_s3_uri"],
        "sidecar_s3_uri": sidecar_uri,
        "catalog_row_count": catalog_rows,
        "main_rows": main_rows,
        "joined_rows": joined_rows,
        "expected_selected_rows": expected_selected,
        "selected_rows": dataset_row_cursor,
        "unit_count": len(units),
        "global_unit_start": global_unit_start,
        "global_unit_end_exclusive": global_unit_start + len(units),
        "global_row_start": global_row_start,
        "global_row_end_exclusive": global_row_start + dataset_row_cursor,
        "main_inventory_sha256": main_inventory_digest,
        "sidecar_inventory_sha256": sidecar_digest,
    }
    return summary, units


def selection_contract(
    max_selected_rows: int,
    *,
    footer_selection: bool = False,
) -> dict[str, Any]:
    if footer_selection:
        return {
            "predicate": "dataset_global_row_modulus_equals_frozen_residue",
            "ordering": SELECTION_ORDERING,
            "max_selected_rows_per_unit": max_selected_rows,
            "planner_io": "parquet_footer_only_no_image_bytes",
            "runtime_validation": "adapter_specific_fail_closed",
            "source_membership": "catalog_frozen_s3_time_snapshot_when_present",
        }
    contract = {
        "predicate": "num_inputs == 1",
        "ordering": SELECTION_ORDERING,
        "max_selected_rows_per_unit": max_selected_rows,
        "sidecar_join": "exact_sample_id_multiset_when_non_null_else_all_main_rows",
        "uri_adapter_validation": "valid_s3_path_fields_without_media_io",
        "embedded_adapter_media_io": "source_locator_only_no_image_bytes",
    }
    return contract


def build_unit_catalog(
    catalog: Mapping[str, Any],
    scans: Mapping[str, DatasetScan],
    *,
    max_selected_rows: int = MAX_SELECTED_ROWS_PER_UNIT,
    scan_workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Plan and strictly validate the complete frozen source catalog."""

    validate_catalog(catalog)
    expected_by_id = validate_selection_expectations(catalog)
    expected_ids = {str(dataset["dataset_id"]) for dataset in catalog["datasets"]}
    actual_ids = set(scans)
    if actual_ids != expected_ids:
        raise UnitPlanningError(
            f"dataset scans differ from catalog: missing={sorted(expected_ids - actual_ids)}, "
            f"unknown={sorted(actual_ids - expected_ids)}"
        )

    summaries: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    global_row_cursor = 0
    for dataset in catalog["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        if progress is not None:
            progress(
                f"planning dataset {int(dataset['ordinal']) + 1}/"
                f"{len(catalog['datasets'])}: {dataset_id}"
            )
        summary, dataset_units = plan_dataset_units(
            dataset,
            scans[dataset_id],
            expected_selected=expected_by_id[dataset_id],
            global_unit_start=len(units),
            global_row_start=global_row_cursor,
            max_selected_rows=max_selected_rows,
            scan_workers=scan_workers,
        )
        summaries.append(summary)
        units.extend(dataset_units)
        global_row_cursor += int(summary["selected_rows"])
        if progress is not None:
            progress(
                f"planned {dataset_id}: selected={summary['selected_rows']:,}, "
                f"units={summary['unit_count']:,}"
            )

    result = {
        "format_version": FORMAT_VERSION,
        "source_catalog": {
            "schema_version": catalog["schema_version"],
            "sha256": source_catalog_sha256(catalog),
            "dataset_count": catalog_expected_dataset_count(catalog),
            "total_rows": catalog["total_rows"],
        },
        "selection": selection_contract(
            max_selected_rows,
            footer_selection=catalog["schema_version"] == CRAFTV0_SCHEMA_VERSION,
        ),
        "dataset_count": len(summaries),
        "total_selected_rows": global_row_cursor,
        "total_units": len(units),
        "datasets": summaries,
        "units": units,
    }
    validate_unit_catalog(result, source_catalog=catalog)
    return result


def _validate_source(source: Any, *, location: str) -> None:
    if not isinstance(source, Mapping):
        raise UnitPlanningError(f"{location} must be an object")
    _require_exact_fields(source, SOURCE_FIELDS, location)
    _require_nonempty_string(source["bucket"], f"{location}.bucket")
    _require_nonempty_string(source["key"], f"{location}.key")
    _require_nonempty_string(source["etag"], f"{location}.etag")
    for field in ("bytes", "rows"):
        if not _is_int(source[field]) or source[field] <= 0:
            raise UnitPlanningError(f"{location}.{field} must be a positive integer")
    if not _is_int(source["object_index"]) or source["object_index"] < 0:
        raise UnitPlanningError(f"{location}.object_index must be non-negative")


def _validate_unit(
    unit: Any,
    *,
    location: str,
    summary: Mapping[str, Any],
    dataset_unit_index: int,
    global_unit_index: int,
    dataset_row_start: int,
    max_selected_rows: int,
) -> int:
    if not isinstance(unit, Mapping):
        raise UnitPlanningError(f"{location} must be an object")
    _require_exact_fields(unit, UNIT_FIELDS, location)
    expected_scalars = {
        "dataset_id": summary["dataset_id"],
        "dataset_ordinal": summary["dataset_ordinal"],
        "dataset_unit_index": dataset_unit_index,
        "global_unit_index": global_unit_index,
        "dataset_row_start": dataset_row_start,
        "global_row_start": summary["global_row_start"] + dataset_row_start,
        "adapter": summary["adapter"],
        "main_s3_uri": summary["main_s3_uri"],
    }
    for field, expected in expected_scalars.items():
        if unit[field] != expected:
            raise UnitPlanningError(f"{location}.{field} must equal {expected!r}, got {unit[field]!r}")

    dataset_row_end = unit["dataset_row_end_exclusive"]
    if not _is_int(dataset_row_end) or not dataset_row_start < dataset_row_end:
        raise UnitPlanningError(f"{location}.dataset_row_end_exclusive is invalid")
    selected_rows = dataset_row_end - dataset_row_start
    if selected_rows > max_selected_rows:
        raise UnitPlanningError(
            f"{location} contains {selected_rows} selected rows; max is {max_selected_rows}"
        )
    if unit["global_row_end_exclusive"] != summary["global_row_start"] + dataset_row_end:
        raise UnitPlanningError(f"{location}.global_row_end_exclusive is inconsistent")

    _validate_source(unit["source"], location=f"{location}.source")
    source_rows = unit["source"]["rows"]
    ranges = unit["local_row_ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise UnitPlanningError(f"{location}.local_row_ranges must be a non-empty array")
    range_rows = 0
    previous_end: int | None = None
    for range_index, row_range in enumerate(ranges):
        range_location = f"{location}.local_row_ranges[{range_index}]"
        if (
            not isinstance(row_range, list)
            or len(row_range) != 2
            or not all(_is_int(value) for value in row_range)
        ):
            raise UnitPlanningError(f"{range_location} must be [start, end_exclusive]")
        start, end = row_range
        if start < 0 or end <= start or end > source_rows:
            raise UnitPlanningError(f"{range_location} is outside the source object")
        if previous_end is not None and start <= previous_end:
            reason = "overlap or are adjacent instead of maximally compressed"
            raise UnitPlanningError(f"{location}.local_row_ranges {reason}")
        previous_end = end
        range_rows += end - start
    if range_rows != selected_rows:
        raise UnitPlanningError(
            f"{location}.local_row_ranges contain {range_rows} rows; unit declares {selected_rows}"
        )
    return dataset_row_end


def validate_unit_catalog(
    value: Any,
    *,
    source_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a production unit catalog against its source catalog."""

    validate_catalog(source_catalog)
    expected_by_id = validate_selection_expectations(source_catalog)
    if not isinstance(value, dict):
        raise UnitPlanningError("unit catalog must be an object")
    _require_exact_fields(value, TOP_LEVEL_FIELDS, "unit catalog")
    if value["format_version"] != FORMAT_VERSION:
        raise UnitPlanningError(f"unit catalog format must equal {FORMAT_VERSION!r}")

    expected_source_binding = {
        "schema_version": source_catalog["schema_version"],
        "sha256": source_catalog_sha256(source_catalog),
        "dataset_count": catalog_expected_dataset_count(source_catalog),
        "total_rows": source_catalog["total_rows"],
    }
    if value["source_catalog"] != expected_source_binding:
        raise UnitPlanningError("unit catalog source_catalog binding is incorrect")

    selection = value["selection"]
    if not isinstance(selection, Mapping):
        raise UnitPlanningError("unit catalog.selection must be an object")
    max_selected_rows = selection.get("max_selected_rows_per_unit")
    if (
        not _is_int(max_selected_rows)
        or not 1 <= max_selected_rows <= max_selected_rows_limit(source_catalog)
        or dict(selection)
        != selection_contract(
            max_selected_rows,
            footer_selection=(
                source_catalog["schema_version"] == CRAFTV0_SCHEMA_VERSION
            ),
        )
    ):
        raise UnitPlanningError("unit catalog.selection contract is invalid")

    summaries = value["datasets"]
    units = value["units"]
    expected_dataset_total = catalog_expected_dataset_count(source_catalog)
    if not isinstance(summaries, list) or len(summaries) != expected_dataset_total:
        raise UnitPlanningError(
            f"unit catalog.datasets must contain {expected_dataset_total} entries"
        )
    if not isinstance(units, list):
        raise UnitPlanningError("unit catalog.units must be an array")
    if value["dataset_count"] != len(summaries):
        raise UnitPlanningError("unit catalog.dataset_count is inconsistent")
    if value["total_units"] != len(units):
        raise UnitPlanningError("unit catalog.total_units is inconsistent")

    global_unit_cursor = 0
    global_row_cursor = 0
    for ordinal, (dataset, summary) in enumerate(zip(source_catalog["datasets"], summaries)):
        location = f"datasets[{ordinal}]"
        if not isinstance(summary, Mapping):
            raise UnitPlanningError(f"{location} must be an object")
        _require_exact_fields(summary, SUMMARY_FIELDS, location)
        dataset_id = dataset["dataset_id"]
        expected_summary_values = {
            "dataset_id": dataset_id,
            "dataset_ordinal": ordinal,
            "family": dataset["family"],
            "adapter": dataset["adapter"],
            "reference_policy": dataset["reference_policy"],
            "main_s3_uri": dataset["main_s3_uri"],
            "sidecar_s3_uri": dataset["sidecar_s3_uri"],
            "catalog_row_count": dataset["row_count"],
            "joined_rows": dataset["row_count"],
            "expected_selected_rows": expected_by_id[dataset_id],
            "selected_rows": expected_by_id[dataset_id],
            "global_unit_start": global_unit_cursor,
            "global_row_start": global_row_cursor,
            "global_row_end_exclusive": global_row_cursor + expected_by_id[dataset_id],
        }
        for field, expected in expected_summary_values.items():
            if summary[field] != expected:
                raise UnitPlanningError(f"{location}.{field} must equal {expected!r}, got {summary[field]!r}")
        if not _is_int(summary["main_rows"]) or summary["main_rows"] < dataset["row_count"]:
            raise UnitPlanningError(f"{location}.main_rows is too small")
        _require_sha256(summary["main_inventory_sha256"], f"{location}.main_inventory_sha256")
        if dataset["sidecar_s3_uri"] is None:
            if summary["sidecar_inventory_sha256"] is not None:
                raise UnitPlanningError(f"{location}.sidecar_inventory_sha256 must be null")
            if summary["main_rows"] != dataset["row_count"]:
                raise UnitPlanningError(f"{location}.main_rows must equal catalog rows without sidecar")
        else:
            _require_sha256(
                summary["sidecar_inventory_sha256"],
                f"{location}.sidecar_inventory_sha256",
            )

        unit_count = summary["unit_count"]
        if not _is_int(unit_count) or unit_count < 0:
            raise UnitPlanningError(f"{location}.unit_count must be non-negative")
        expected_unit_end = global_unit_cursor + unit_count
        if summary["global_unit_end_exclusive"] != expected_unit_end:
            raise UnitPlanningError(f"{location}.global_unit_end_exclusive is inconsistent")
        dataset_row_cursor = 0
        previous_object_index: int | None = None
        previous_local_end: int | None = None
        source_by_object_index: dict[int, Mapping[str, Any]] = {}
        for dataset_unit_index, unit in enumerate(units[global_unit_cursor:expected_unit_end]):
            unit_location = f"units[{global_unit_cursor + dataset_unit_index}]"
            dataset_row_cursor = _validate_unit(
                unit,
                location=unit_location,
                summary=summary,
                dataset_unit_index=dataset_unit_index,
                global_unit_index=global_unit_cursor + dataset_unit_index,
                dataset_row_start=dataset_row_cursor,
                max_selected_rows=max_selected_rows,
            )
            object_index = unit["source"]["object_index"]
            first_local_row = unit["local_row_ranges"][0][0]
            last_local_end = unit["local_row_ranges"][-1][1]
            known_source = source_by_object_index.setdefault(object_index, unit["source"])
            if dict(known_source) != dict(unit["source"]):
                raise UnitPlanningError(
                    f"{unit_location}.source changed identity for object_index={object_index}"
                )
            if previous_object_index is not None:
                if object_index < previous_object_index:
                    raise UnitPlanningError(f"{unit_location} is not in source object order")
                if (
                    object_index == previous_object_index
                    and previous_local_end is not None
                    and first_local_row < previous_local_end
                ):
                    raise UnitPlanningError(f"{unit_location} overlaps an earlier local row range")
            previous_object_index = object_index
            previous_local_end = last_local_end
        if dataset_row_cursor != summary["selected_rows"]:
            raise UnitPlanningError(f"{location} unit rows do not cover selected_rows")
        if summary["selected_rows"] == 0 and unit_count != 0:
            raise UnitPlanningError(f"{location} zero-row dataset must have zero units")
        if summary["selected_rows"] > 0 and unit_count == 0:
            raise UnitPlanningError(f"{location} selected rows have no units")

        global_unit_cursor = expected_unit_end
        global_row_cursor += summary["selected_rows"]

    if global_unit_cursor != len(units):
        raise UnitPlanningError("unit catalog contains unclaimed units")
    expected_total_selected = catalog_expected_total_selected_rows(source_catalog)
    if global_row_cursor != expected_total_selected:
        raise UnitPlanningError(
            f"unit catalog covers {global_row_cursor} rows; expected {expected_total_selected}"
        )
    if value["total_selected_rows"] != global_row_cursor:
        raise UnitPlanningError("unit catalog.total_selected_rows is inconsistent")
    return value


def canonical_unit_catalog_bytes(value: Any, *, source_catalog: Mapping[str, Any]) -> bytes:
    validate_unit_catalog(value, source_catalog=source_catalog)
    return canonical_json_bytes(value)


def canonical_unit_catalog_sha256(value: Any, *, source_catalog: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_unit_catalog_bytes(value, source_catalog=source_catalog)).hexdigest()


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _list_parquet_objects(s3_client: Any, uri: str) -> list[dict[str, Any]]:
    root = parse_s3_uri(uri.rstrip("/"))
    exact_key = root.key if root.key.endswith(".parquet") else None
    prefix = root.key if exact_key is not None else f"{root.key.rstrip('/')}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=root.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith(".parquet") and (exact_key is None or key == exact_key):
                raw_modified = item.get("LastModified")
                if raw_modified is None or not hasattr(raw_modified, "astimezone"):
                    raise UnitPlanningError(f"S3 listing lacks LastModified for s3://{root.bucket}/{key}")
                last_modified = raw_modified.astimezone(timezone.utc).replace(
                    microsecond=0
                ).isoformat()
                objects.append(
                    {
                        "bucket": root.bucket,
                        "key": key,
                        "etag": _clean_etag(item.get("ETag")),
                        "bytes": int(item["Size"]),
                        "last_modified": last_modified,
                    }
                )
    objects.sort(key=lambda item: item["key"])
    if not objects:
        raise UnitPlanningError(f"no Parquet objects found under {uri}")
    return objects


def attach_frozen_snapshot_metadata(
    inventory: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    s3_client: Any,
) -> Mapping[str, Any]:
    """Attach live S3 modification times while tolerating later appends.

    Cached footer inventories deliberately avoid repeated Parquet reads.  A
    fresh object listing is cheap and proves that every cached key still has
    the same ETag/size.  Objects appended after the frozen boundary are allowed;
    the catalog boundary must still be the Nth object in chronological order.
    """

    if "source_object_count" not in dataset:
        return inventory
    dataset_id = str(dataset["dataset_id"])
    uri = str(dataset["main_s3_uri"])
    cached = _inventory_objects(
        inventory,
        expected_uri=uri,
        location=f"{dataset_id}.cached_inventory",
    )
    listed = _list_parquet_objects(s3_client, uri)
    object_count = int(dataset["source_object_count"])
    if len(listed) < object_count:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} lists {len(listed)} objects, fewer than frozen "
            f"source_object_count={object_count}"
        )
    chronological = sorted(
        listed,
        key=lambda item: (item["last_modified"], item["key"]),
    )
    boundary = chronological[object_count - 1]
    if boundary["key"] != dataset["source_object_boundary_key"] or boundary[
        "last_modified"
    ] != dataset["source_object_boundary_last_modified"]:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} live S3 snapshot boundary changed: "
            f"actual=({boundary['last_modified']!r}, {boundary['key']!r}) "
            f"expected=({dataset['source_object_boundary_last_modified']!r}, "
            f"{dataset['source_object_boundary_key']!r})"
        )
    listed_by_key = {item["key"]: item for item in listed}
    cached_by_key = {item["key"]: item for item in cached}
    missing_frozen = [
        item["key"]
        for item in chronological[:object_count]
        if item["key"] not in cached_by_key
    ]
    if missing_frozen:
        raise UnitPlanningError(
            f"dataset {dataset_id!r} cached inventory misses frozen keys: "
            f"{missing_frozen[:5]}"
        )
    enriched_objects = []
    for item in cached:
        current = listed_by_key.get(item["key"])
        if current is None:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} cached object disappeared: {item['key']}"
            )
        if current["etag"] != item["etag"] or current["bytes"] != item["bytes"]:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} cached object identity changed: {item['key']}"
            )
        enriched_objects.append({**item, "last_modified": current["last_modified"]})
    enriched = dict(inventory)
    enriched["objects"] = enriched_objects
    return enriched


def _inspect_parquet_object(s3_client: Any, item: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("pyarrow is required to inspect Parquet") from exc
    reader = S3RangeReader(
        s3_client,
        bucket=str(item["bucket"]),
        key=str(item["key"]),
        size=int(item["bytes"]),
    )
    parquet = pq.ParquetFile(reader)
    return {
        **item,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "columns": list(parquet.schema_arrow.names),
    }


def build_parquet_inventory(
    uri: str,
    *,
    s3_client: Any,
    required_columns: Iterable[str],
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise UnitPlanningError("inventory workers must be positive")
    objects = _list_parquet_objects(s3_client, uri)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = list(pool.map(lambda item: _inspect_parquet_object(s3_client, item), objects))
    required = set(required_columns)
    cursor = 0
    for object_index, item in enumerate(inspected):
        missing = sorted(required.difference(item["columns"]))
        if missing:
            raise UnitPlanningError(f"s3://{item['bucket']}/{item['key']} lacks required columns {missing}")
        item["object_index"] = object_index
        item["global_row_start"] = cursor
        cursor += int(item["rows"])
        item["global_row_end_exclusive"] = cursor
    return {
        "format_version": INVENTORY_FORMAT_VERSION,
        "ordering": "lexicographic_s3_key_then_parquet_row",
        "s3_uri": _normalized_s3_uri(uri),
        "object_count": len(inspected),
        "total_rows": cursor,
        "objects": inspected,
    }


def _inventory_has_required_columns(
    inventory: Mapping[str, Any], *, required_columns: Iterable[str], location: str
) -> None:
    required = set(required_columns)
    objects = inventory.get("objects")
    if not isinstance(objects, list):
        raise UnitPlanningError(f"{location}.objects must be an array")
    for object_index, item in enumerate(objects):
        columns = item.get("columns") if isinstance(item, Mapping) else None
        if not isinstance(columns, list):
            raise UnitPlanningError(f"{location}.objects[{object_index}].columns must be an array")
        missing = sorted(required.difference(columns))
        if missing:
            raise UnitPlanningError(f"{location}.objects[{object_index}] lacks required columns {missing}")


def _read_parquet_columns(
    s3_client: Any,
    item: Mapping[str, Any],
    *,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("pyarrow is required to scan Parquet") from exc
    head = s3_client.head_object(Bucket=item["bucket"], Key=item["key"])
    if int(head["ContentLength"]) != item["bytes"] or _clean_etag(head.get("ETag")) != item["etag"]:
        raise UnitPlanningError(f"source object changed since inventory: s3://{item['bucket']}/{item['key']}")
    reader = S3RangeReader(
        s3_client,
        bucket=item["bucket"],
        key=item["key"],
        size=item["bytes"],
    )
    parquet = pq.ParquetFile(reader)
    table = parquet.read(columns=list(columns), use_threads=False)
    if table.num_rows != item["rows"]:
        raise UnitPlanningError(f"row count changed for s3://{item['bucket']}/{item['key']}")
    return table.to_pylist()


def iter_sidecar_sample_ids(
    inventory: Mapping[str, Any],
    *,
    s3_client: Any,
    workers: int,
) -> Iterator[str]:
    objects = inventory["objects"]

    def read(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _read_parquet_columns(s3_client, item, columns=("sample_id",))

    for _, rows in _ordered_object_rows(objects, read, workers=workers):
        for row in rows:
            yield row.get("sample_id")


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise UnitPlanningError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise UnitPlanningError(f"expected JSON object at {path}")
    return value


def load_unit_catalog(
    path: str | Path,
    *,
    source_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Load local canonical JSON and validate it against *source_catalog*."""

    catalog = _load_json(Path(path))
    return validate_unit_catalog(catalog, source_catalog=source_catalog)


def load_or_build_inventory(
    cache_path: Path,
    *,
    uri: str,
    required_columns: Iterable[str],
    s3_client: Any,
    workers: int,
    rebuild: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not rebuild:
        inventory = _load_json(cache_path)
    else:
        inventory = build_parquet_inventory(
            uri,
            s3_client=s3_client,
            required_columns=required_columns,
            workers=workers,
        )
        _atomic_write(cache_path, canonical_json_bytes(inventory), overwrite=True)
    _inventory_objects(inventory, expected_uri=uri, location=str(cache_path))
    _inventory_has_required_columns(
        inventory,
        required_columns=required_columns,
        location=str(cache_path),
    )
    return inventory


def _aws_s3_client(profile: str | None, region: str) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("boto3 is required for S3 catalog planning") from exc
    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=128,
        ),
    )


def _cache_name(dataset: Mapping[str, Any], kind: str) -> str:
    return f"{int(dataset['ordinal']):02d}-{dataset['dataset_id']}.{kind}.inventory.json"


def _progress(message: str) -> None:
    print(f"[sidecar24-units] {message}", file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    validate_selection_expectations(catalog)
    client = _aws_s3_client(args.aws_profile, args.region)
    inventory_dir = args.work_dir / "inventories"
    scans: dict[str, DatasetScan] = {}

    for dataset in catalog["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        _progress(
            f"inventory {int(dataset['ordinal']) + 1}/{len(catalog['datasets'])}: {dataset_id}"
        )
        adapter = str(dataset["adapter"])
        if adapter not in ALLOWED_ADAPTERS:
            raise UnitPlanningError(
                f"dataset {dataset_id!r} adapter must be one of {sorted(ALLOWED_ADAPTERS)}"
            )
        required_main = ADAPTER_REQUIRED_COLUMNS[adapter]
        main_inventory = load_or_build_inventory(
            inventory_dir / _cache_name(dataset, "main"),
            uri=str(dataset["main_s3_uri"]),
            required_columns=required_main,
            s3_client=client,
            workers=args.inventory_workers,
            rebuild=args.rebuild_inventories,
        )
        main_inventory = attach_frozen_snapshot_metadata(
            main_inventory,
            dataset=dataset,
            s3_client=client,
        )

        def read_main_object(
            item: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            available = set(item["columns"])
            columns = [column for column in MAIN_SCAN_COLUMNS if column in available]
            return _read_parquet_columns(client, item, columns=columns)

        sidecar_uri = dataset["sidecar_s3_uri"]
        sidecar_inventory = None
        sidecar_ids = None
        if sidecar_uri is not None:
            sidecar_inventory = load_or_build_inventory(
                inventory_dir / _cache_name(dataset, "sidecar"),
                uri=str(sidecar_uri),
                required_columns=("sample_id",),
                s3_client=client,
                workers=args.inventory_workers,
                rebuild=args.rebuild_inventories,
            )
            sidecar_ids = iter_sidecar_sample_ids(
                sidecar_inventory,
                s3_client=client,
                workers=args.scan_workers,
            )
        scans[dataset_id] = DatasetScan(
            main_inventory=main_inventory,
            read_main_object=read_main_object,
            sidecar_inventory=sidecar_inventory,
            sidecar_sample_ids=sidecar_ids,
        )

    result = build_unit_catalog(
        catalog,
        scans,
        max_selected_rows=args.max_selected_rows,
        scan_workers=args.scan_workers,
        progress=_progress,
    )
    payload = canonical_unit_catalog_bytes(result, source_catalog=catalog)
    _atomic_write(args.output, payload, overwrite=args.overwrite)
    return {
        "output": str(args.output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "datasets": result["dataset_count"],
        "units": result["total_units"],
        "selected_rows": result["total_selected_rows"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--aws-profile")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--max-selected-rows", type=int, default=MAX_SELECTED_ROWS_PER_UNIT)
    parser.add_argument("--rebuild-inventories", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.inventory_workers <= 0 or args.scan_workers <= 0:
        raise UnitPlanningError("worker counts must be positive")
    if not 1 <= args.max_selected_rows <= CRAFTV0_MAX_SELECTED_ROWS_PER_UNIT:
        raise UnitPlanningError(
            "--max-selected-rows must be between 1 and "
            f"{CRAFTV0_MAX_SELECTED_ROWS_PER_UNIT}"
        )
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
