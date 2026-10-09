#!/usr/bin/env python3
"""Validate and fingerprint the 24-dataset image-edit inference catalog.

The catalog intentionally contains only immutable dataset-level facts.  Work
unit planning and inference remain separate concerns.  The v1 JSON shape is::

    {
      "schema_version": "image_edit_sidecar24_catalog_v1",
      "total_rows": 11797395,
      "datasets": [
        {
          "dataset_id": "...",
          "ordinal": 0,
          "family": "single|multi|spatial",
          "adapter": "...",
          "reference_policy": "...",
          "main_s3_uri": "s3://bucket/prefix",
          "sidecar_s3_uri": "s3://bucket/prefix" | null,
          "row_count": 123,
          "dataset_row_offset": 0,
          "global_row_offset": 0
        }
      ]
    }

Canonical JSON uses sorted object keys, compact separators, UTF-8, and one
trailing newline.  The SHA-256 returned here is therefore independent of input
JSON whitespace and object-key order, while dataset list order remains part of
the catalog identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "image_edit_sidecar24_catalog_v1"
CRAFTV0_SCHEMA_VERSION = "image_edit_craftv0_catalog_v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, CRAFTV0_SCHEMA_VERSION})
EXPECTED_DATASET_COUNT = 24
EXPECTED_TOTAL_ROWS = 11_797_395

CHEESE_V2_DATASET_ID = "cheese_edit_pairs_v2_parquet"
CHEESE_V2_ROW_COUNT = 577_280
MISSING_MANIFEST_DATASET_ID = "design_edit_api_distill_ge4_parquet"
MISSING_MANIFEST_ROW_COUNT = 228_830

ALLOWED_FAMILIES = frozenset({"single", "multi", "spatial"})
TOP_LEVEL_FIELDS = frozenset({"schema_version", "total_rows", "datasets"})
DATASET_FIELDS = frozenset(
    {
        "dataset_id",
        "ordinal",
        "family",
        "adapter",
        "reference_policy",
        "main_s3_uri",
        "sidecar_s3_uri",
        "row_count",
        "dataset_row_offset",
        "global_row_offset",
    }
)
CRAFTV0_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "selection_commit",
        "selection_sha256",
        "total_rows",
        "total_selected_rows",
        "datasets",
    }
)
CRAFTV0_DATASET_FIELDS = DATASET_FIELDS | frozenset(
    {
        "planning_mode",
        "selection_modulus",
        "selection_residue",
        "selected_row_count",
        "global_selected_row_offset",
    }
)
CRAFTV0_SOURCE_OBJECT_ORDERING = "last_modified_then_lexicographic_s3_key"
CRAFTV0_SNAPSHOT_DATASET_FIELDS = frozenset(
    {
        "source_object_count",
        "source_object_ordering",
        "source_object_boundary_key",
        "source_object_boundary_last_modified",
    }
)
CRAFTV0_OPTIONAL_DATASET_FIELDS = CRAFTV0_SNAPSHOT_DATASET_FIELDS

_DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_S3_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")


class CatalogValidationError(ValueError):
    """Raised when a catalog does not satisfy the v1 contract."""


def _is_int(value: Any) -> bool:
    """Return true for JSON integers but not booleans."""

    return type(value) is int


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], location: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CatalogValidationError(f"{location}: all field names must be strings")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise CatalogValidationError(f"{location}: {', '.join(details)}")


def _require_fields_with_optional(
    value: dict[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    location: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CatalogValidationError(f"{location}: all field names must be strings")
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise CatalogValidationError(f"{location}: {', '.join(details)}")


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(f"{location} must be a non-empty string")
    return value


def _validate_s3_uri(value: Any, location: str) -> None:
    uri = _require_nonempty_string(value, location)
    if uri != uri.strip() or any(character.isspace() for character in uri):
        raise CatalogValidationError(f"{location} must not contain whitespace")

    parsed = urlsplit(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not _S3_BUCKET_RE.fullmatch(bucket)
        or not key
        or parsed.query
        or parsed.fragment
        or "@" in bucket
        or ":" in bucket
    ):
        raise CatalogValidationError(
            f"{location} must be an s3://bucket/non-empty-key URI without query or fragment"
        )


def _validate_dataset(entry: Any, *, list_index: int, expected_global_offset: int) -> int:
    location = f"datasets[{list_index}]"
    if not isinstance(entry, dict):
        raise CatalogValidationError(f"{location} must be an object")
    _require_exact_fields(entry, DATASET_FIELDS, location)

    dataset_id = _require_nonempty_string(entry["dataset_id"], f"{location}.dataset_id")
    if not _DATASET_ID_RE.fullmatch(dataset_id):
        raise CatalogValidationError(
            f"{location}.dataset_id must contain only lowercase letters, digits, '_' or '-'"
        )

    ordinal = entry["ordinal"]
    if not _is_int(ordinal) or ordinal != list_index:
        raise CatalogValidationError(
            f"{location}.ordinal must equal its zero-based list position {list_index}"
        )

    family = entry["family"]
    if not isinstance(family, str) or family not in ALLOWED_FAMILIES:
        raise CatalogValidationError(
            f"{location}.family must be one of {sorted(ALLOWED_FAMILIES)}, got {family!r}"
        )
    _require_nonempty_string(entry["adapter"], f"{location}.adapter")
    _require_nonempty_string(entry["reference_policy"], f"{location}.reference_policy")

    _validate_s3_uri(entry["main_s3_uri"], f"{location}.main_s3_uri")
    sidecar_s3_uri = entry["sidecar_s3_uri"]
    if sidecar_s3_uri is not None:
        _validate_s3_uri(sidecar_s3_uri, f"{location}.sidecar_s3_uri")

    row_count = entry["row_count"]
    if not _is_int(row_count) or row_count <= 0:
        raise CatalogValidationError(f"{location}.row_count must be a positive integer")

    dataset_row_offset = entry["dataset_row_offset"]
    if not _is_int(dataset_row_offset) or dataset_row_offset != 0:
        raise CatalogValidationError(f"{location}.dataset_row_offset must equal 0")

    global_row_offset = entry["global_row_offset"]
    if not _is_int(global_row_offset) or global_row_offset != expected_global_offset:
        raise CatalogValidationError(
            f"{location}.global_row_offset must equal the contiguous offset "
            f"{expected_global_offset}, got {global_row_offset!r}"
        )
    return expected_global_offset + row_count


def _selected_modulus_count(row_count: int, modulus: int, residue: int) -> int:
    if residue >= row_count:
        return 0
    return 1 + (row_count - 1 - residue) // modulus


def _validate_craftv0_dataset(
    entry: Any,
    *,
    list_index: int,
    expected_global_offset: int,
    expected_global_selected_offset: int,
) -> tuple[int, int]:
    location = f"datasets[{list_index}]"
    if not isinstance(entry, dict):
        raise CatalogValidationError(f"{location} must be an object")
    _require_fields_with_optional(
        entry,
        CRAFTV0_DATASET_FIELDS,
        CRAFTV0_OPTIONAL_DATASET_FIELDS,
        location,
    )

    base_entry = {field: entry[field] for field in DATASET_FIELDS}
    next_global_offset = _validate_dataset(
        base_entry,
        list_index=list_index,
        expected_global_offset=expected_global_offset,
    )
    if entry["planning_mode"] != "footer_rows_v1":
        raise CatalogValidationError(
            f"{location}.planning_mode must equal 'footer_rows_v1'"
        )
    snapshot_fields = CRAFTV0_SNAPSHOT_DATASET_FIELDS.intersection(entry)
    if snapshot_fields and snapshot_fields != CRAFTV0_SNAPSHOT_DATASET_FIELDS:
        missing_snapshot_fields = sorted(CRAFTV0_SNAPSHOT_DATASET_FIELDS - snapshot_fields)
        raise CatalogValidationError(
            f"{location}: source snapshot fields must be provided together; "
            f"missing {missing_snapshot_fields}"
        )
    if snapshot_fields:
        source_object_count = entry["source_object_count"]
        if not _is_int(source_object_count) or source_object_count <= 0:
            raise CatalogValidationError(
                f"{location}.source_object_count must be a positive integer"
            )
        if entry["source_object_ordering"] != CRAFTV0_SOURCE_OBJECT_ORDERING:
            raise CatalogValidationError(
                f"{location}.source_object_ordering must equal "
                f"{CRAFTV0_SOURCE_OBJECT_ORDERING!r}"
            )
        boundary_key = _require_nonempty_string(
            entry["source_object_boundary_key"],
            f"{location}.source_object_boundary_key",
        )
        main_prefix = urlsplit(entry["main_s3_uri"]).path.lstrip("/").rstrip("/") + "/"
        if not boundary_key.startswith(main_prefix) or not boundary_key.endswith(".parquet"):
            raise CatalogValidationError(
                f"{location}.source_object_boundary_key must be a Parquet key "
                "under main_s3_uri"
            )
        boundary_modified = entry["source_object_boundary_last_modified"]
        if not isinstance(boundary_modified, str) or _UTC_SECONDS_RE.fullmatch(
            boundary_modified
        ) is None:
            raise CatalogValidationError(
                f"{location}.source_object_boundary_last_modified must be an "
                "RFC3339 UTC timestamp at second precision"
            )
    modulus = entry["selection_modulus"]
    residue = entry["selection_residue"]
    if not _is_int(modulus) or modulus <= 0:
        raise CatalogValidationError(f"{location}.selection_modulus must be positive")
    if not _is_int(residue) or not 0 <= residue < modulus:
        raise CatalogValidationError(
            f"{location}.selection_residue must be in [0, selection_modulus)"
        )
    selected = entry["selected_row_count"]
    expected_selected = _selected_modulus_count(entry["row_count"], modulus, residue)
    if not _is_int(selected) or selected != expected_selected:
        raise CatalogValidationError(
            f"{location}.selected_row_count must equal {expected_selected}, got {selected!r}"
        )
    selected_offset = entry["global_selected_row_offset"]
    if not _is_int(selected_offset) or selected_offset != expected_global_selected_offset:
        raise CatalogValidationError(
            f"{location}.global_selected_row_offset must equal "
            f"{expected_global_selected_offset}, got {selected_offset!r}"
        )
    return next_global_offset, expected_global_selected_offset + selected


def _require_dataset_row_count(
    datasets_by_id: dict[str, dict[str, Any]], dataset_id: str, row_count: int
) -> None:
    try:
        entry = datasets_by_id[dataset_id]
    except KeyError as exc:
        raise CatalogValidationError(f"required dataset is missing: {dataset_id}") from exc
    if entry["row_count"] != row_count:
        raise CatalogValidationError(
            f"dataset {dataset_id!r} must contain exactly {row_count} rows, got {entry['row_count']!r}"
        )


def validate_catalog(value: Any) -> dict[str, Any]:
    """Validate *value* and return the same catalog object.

    Returning the input object makes the function convenient at JSON loading
    boundaries without silently normalizing or repairing catalog data.
    """

    if not isinstance(value, dict):
        raise CatalogValidationError("catalog must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise CatalogValidationError(
            "catalog.schema_version must be one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}, got {schema_version!r}"
        )

    if schema_version == CRAFTV0_SCHEMA_VERSION:
        _require_exact_fields(value, CRAFTV0_TOP_LEVEL_FIELDS, "catalog")
        if not isinstance(value["selection_commit"], str) or _GIT_COMMIT_RE.fullmatch(
            value["selection_commit"]
        ) is None:
            raise CatalogValidationError("catalog.selection_commit must be a 40-hex Git commit")
        if not isinstance(value["selection_sha256"], str) or _SHA256_RE.fullmatch(
            value["selection_sha256"]
        ) is None:
            raise CatalogValidationError("catalog.selection_sha256 must be a lowercase SHA-256")
    else:
        _require_exact_fields(value, TOP_LEVEL_FIELDS, "catalog")

    total_rows = value["total_rows"]
    if not _is_int(total_rows) or total_rows <= 0:
        raise CatalogValidationError("catalog.total_rows must be a positive integer")
    if schema_version == SCHEMA_VERSION and total_rows != EXPECTED_TOTAL_ROWS:
        raise CatalogValidationError(
            f"catalog.total_rows must equal {EXPECTED_TOTAL_ROWS}, got {total_rows!r}"
        )

    datasets = value["datasets"]
    if not isinstance(datasets, list):
        raise CatalogValidationError("catalog.datasets must be an array")
    if not datasets:
        raise CatalogValidationError("catalog.datasets must not be empty")
    if schema_version == SCHEMA_VERSION and len(datasets) != EXPECTED_DATASET_COUNT:
        raise CatalogValidationError(
            f"catalog.datasets must contain exactly {EXPECTED_DATASET_COUNT} entries, got {len(datasets)}"
        )

    datasets_by_id: dict[str, dict[str, Any]] = {}
    seen_ordinals: set[int] = set()
    global_offset = 0
    global_selected_offset = 0
    for list_index, entry in enumerate(datasets):
        if schema_version == CRAFTV0_SCHEMA_VERSION:
            global_offset, global_selected_offset = _validate_craftv0_dataset(
                entry,
                list_index=list_index,
                expected_global_offset=global_offset,
                expected_global_selected_offset=global_selected_offset,
            )
        else:
            global_offset = _validate_dataset(
                entry,
                list_index=list_index,
                expected_global_offset=global_offset,
            )
        dataset_id = entry["dataset_id"]
        if dataset_id in datasets_by_id:
            raise CatalogValidationError(f"duplicate dataset_id: {dataset_id!r}")
        datasets_by_id[dataset_id] = entry

        ordinal = entry["ordinal"]
        if ordinal in seen_ordinals:
            raise CatalogValidationError(f"duplicate dataset ordinal: {ordinal}")
        seen_ordinals.add(ordinal)

    if global_offset != total_rows:
        raise CatalogValidationError(
            f"dataset row counts sum to {global_offset}, expected catalog.total_rows={total_rows}"
        )

    if schema_version == CRAFTV0_SCHEMA_VERSION:
        total_selected_rows = value["total_selected_rows"]
        if not _is_int(total_selected_rows) or total_selected_rows != global_selected_offset:
            raise CatalogValidationError(
                "catalog.total_selected_rows must equal the sum of selected_row_count "
                f"({global_selected_offset}), got {total_selected_rows!r}"
            )
    else:
        _require_dataset_row_count(
            datasets_by_id,
            CHEESE_V2_DATASET_ID,
            CHEESE_V2_ROW_COUNT,
        )
        _require_dataset_row_count(
            datasets_by_id,
            MISSING_MANIFEST_DATASET_ID,
            MISSING_MANIFEST_ROW_COUNT,
        )
    return value


def expected_dataset_count(catalog: Any) -> int:
    return len(validate_catalog(catalog)["datasets"])


def expected_total_selected_rows(catalog: Any) -> int:
    validated = validate_catalog(catalog)
    if validated["schema_version"] == CRAFTV0_SCHEMA_VERSION:
        return int(validated["total_selected_rows"])
    return 11_204_822


def canonical_catalog_bytes(value: Any) -> bytes:
    """Return validated canonical UTF-8 JSON bytes, including a final newline."""

    catalog = validate_catalog(value)
    return (
        json.dumps(
            catalog,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    """Return the canonical SHA-256 hex digest for a validated catalog."""

    return hashlib.sha256(canonical_catalog_bytes(value)).hexdigest()


# The longer alias reads better at call sites that fingerprint multiple
# artifacts.  Keep ``canonical_sha256`` as the concise CLI-facing API.
canonical_catalog_sha256 = canonical_sha256


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CatalogValidationError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def load_catalog(path: str | Path) -> dict[str, Any]:
    """Load UTF-8 JSON from *path*, reject duplicate keys, and validate it."""

    catalog_path = Path(path)
    try:
        value = json.loads(
            catalog_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"could not load catalog {catalog_path}: {exc}") from exc
    return validate_catalog(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an image_edit_sidecar24_catalog_v1 JSON file and print its SHA-256."
    )
    parser.add_argument("catalog", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = load_catalog(args.catalog)
    print(canonical_sha256(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
