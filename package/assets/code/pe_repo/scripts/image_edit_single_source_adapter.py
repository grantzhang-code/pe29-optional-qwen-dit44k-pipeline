#!/usr/bin/env python3
"""Read one immutable single-reference Parquet work unit for inference.

The planner owns dataset ordering and selection.  This module deliberately
knows only the frozen unit contract and turns its selected source rows into the
existing :class:`InferenceRecord` runtime contract.  URI-backed and embedded
image datasets share the same validation and provenance rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit

from scripts.infer_image_edit_full_json import InferenceRecord, parse_s3_media_uri
from scripts.prepare_image_edit_full_json_shard import S3RangeReader


URI_SINGLE_V1 = "uri_single_v1"
EMBEDDED_SINGLE_V1 = "embedded_single_v1"
EMBEDDED_SYNTH_V1 = "embedded_synth_v1"
ALLOWED_ADAPTERS = frozenset(
    {URI_SINGLE_V1, EMBEDDED_SINGLE_V1, EMBEDDED_SYNTH_V1}
)
MISSING_INSTRUCTION_FALLBACK = (
    "Infer the intended edit from the visual difference between Reference "
    "frame_0 and Target."
)
MISSING_INSTRUCTION_FALLBACK_VERSION = "visual_difference_v1"
PARQUET_DECODE_BATCH_SIZE = 32

UNIT_FIELDS = (
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
)
SOURCE_FIELDS = ("bucket", "key", "etag", "bytes", "rows", "object_index")


class SourceAdapterError(ValueError):
    """A source unit or one of its selected Parquet rows is invalid."""


def _iter_selected_parquet_rows(
    parquet: Any,
    *,
    unit: "SourceUnit",
    columns: Sequence[str],
):
    """Yield selected rows without materializing one multi-GiB Arrow table.

    Some embedded-image shards contain more than 2 GiB of nested binary data
    in a single row group.  ``ParquetFile.read`` asks Arrow to expose that row
    group as chunked nested arrays, which pyarrow 24 rejects with
    ``Nested data conversions not implemented for chunked array outputs``.
    Record batches keep every nested conversion bounded while preserving the
    physical row order and the frozen local-row identity.
    """

    batch_start = 0
    selected_rows = 0
    ranges = unit.local_row_ranges
    range_index = 0
    for batch in parquet.iter_batches(
        batch_size=PARQUET_DECODE_BATCH_SIZE,
        columns=list(columns),
        use_threads=False,
    ):
        batch_end = batch_start + batch.num_rows
        while range_index < len(ranges) and ranges[range_index][1] <= batch_start:
            range_index += 1
        cursor = range_index
        while cursor < len(ranges) and ranges[cursor][0] < batch_end:
            start, end = ranges[cursor]
            overlap_start = max(start, batch_start)
            overlap_end = min(end, batch_end)
            if overlap_start < overlap_end:
                rows = batch.slice(
                    overlap_start - batch_start,
                    overlap_end - overlap_start,
                ).to_pylist()
                if len(rows) != overlap_end - overlap_start:
                    raise SourceAdapterError(
                        f"{unit.source.s3_uri}: selection "
                        f"[{overlap_start},{overlap_end}) decoded {len(rows)} rows"
                    )
                for offset, raw in enumerate(rows):
                    selected_rows += 1
                    yield overlap_start + offset, raw
            if end <= batch_end:
                cursor += 1
            else:
                break
        range_index = cursor
        batch_start = batch_end

    if batch_start != unit.source.rows:
        raise SourceAdapterError(
            f"{unit.source.s3_uri}: decoded rows={batch_start}, expected={unit.source.rows}"
        )
    if selected_rows != unit.selected_rows:
        raise SourceAdapterError(
            f"{unit.source.s3_uri}: selected rows={selected_rows}, "
            f"expected={unit.selected_rows}"
        )


@dataclass(frozen=True)
class SourceIdentity:
    bucket: str
    key: str
    etag: str
    bytes: int
    rows: int
    object_index: int

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class SourceUnit:
    dataset_id: str
    dataset_ordinal: int
    dataset_unit_index: int
    global_unit_index: int
    dataset_row_start: int
    dataset_row_end_exclusive: int
    global_row_start: int
    global_row_end_exclusive: int
    local_row_ranges: tuple[tuple[int, int], ...]
    source: SourceIdentity
    adapter: str
    main_s3_uri: str

    @property
    def selected_rows(self) -> int:
        return sum(end - start for start, end in self.local_row_ranges)


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceAdapterError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise SourceAdapterError(f"{field} must not have surrounding whitespace")
    return value


def _required_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SourceAdapterError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_s3_uri(value: Any, *, field: str) -> tuple[str, str]:
    uri = _required_string(value, field=field)
    parsed = urlsplit(uri)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or "@" in parsed.netloc
        or ":" in parsed.netloc
    ):
        raise SourceAdapterError(
            f"{field} must be an s3://bucket/non-empty-key URI without query or fragment"
        )
    return parsed.netloc, key.rstrip("/")


def _parse_s3_media_uri(value: Any, *, field: str) -> tuple[str, str]:
    """Validate row media while preserving every byte of the S3 object key."""

    uri = _required_string(value, field=field)
    try:
        return parse_s3_media_uri(uri)
    except ValueError as exc:
        raise SourceAdapterError(
            f"{field} must be an s3://bucket/non-empty-key media URI"
        ) from exc


def validate_unit(value: Mapping[str, Any]) -> SourceUnit:
    """Validate and freeze the planner/source-adapter boundary."""

    if not isinstance(value, Mapping):
        raise SourceAdapterError("unit must be an object")
    missing = [field for field in UNIT_FIELDS if field not in value]
    if missing:
        raise SourceAdapterError(f"unit is missing required fields {missing}")

    adapter = _required_string(value["adapter"], field="unit.adapter")
    if adapter not in ALLOWED_ADAPTERS:
        raise SourceAdapterError(
            f"unit.adapter must be one of {sorted(ALLOWED_ADAPTERS)}, got {adapter!r}"
        )

    raw_source = value["source"]
    if not isinstance(raw_source, Mapping):
        raise SourceAdapterError("unit.source must be an object")
    missing_source = [field for field in SOURCE_FIELDS if field not in raw_source]
    if missing_source:
        raise SourceAdapterError(f"unit.source is missing required fields {missing_source}")
    bucket = _required_string(raw_source["bucket"], field="unit.source.bucket")
    key = _required_string(raw_source["key"], field="unit.source.key")
    etag = _clean_etag(raw_source["etag"])
    if not etag:
        raise SourceAdapterError("unit.source.etag must be non-empty")
    source = SourceIdentity(
        bucket=bucket,
        key=key,
        etag=etag,
        bytes=_required_int(raw_source["bytes"], field="unit.source.bytes", minimum=1),
        rows=_required_int(raw_source["rows"], field="unit.source.rows", minimum=1),
        object_index=_required_int(
            raw_source["object_index"], field="unit.source.object_index"
        ),
    )
    if not source.key.endswith(".parquet"):
        raise SourceAdapterError("unit.source.key must end with .parquet")

    main_bucket, main_key = _parse_s3_uri(value["main_s3_uri"], field="unit.main_s3_uri")
    if source.bucket != main_bucket or not (
        source.key == main_key or source.key.startswith(main_key + "/")
    ):
        raise SourceAdapterError(
            f"unit source {source.s3_uri} is outside unit.main_s3_uri={value['main_s3_uri']}"
        )

    dataset_row_start = _required_int(
        value["dataset_row_start"], field="unit.dataset_row_start"
    )
    dataset_row_end = _required_int(
        value["dataset_row_end_exclusive"], field="unit.dataset_row_end_exclusive"
    )
    global_row_start = _required_int(value["global_row_start"], field="unit.global_row_start")
    global_row_end = _required_int(
        value["global_row_end_exclusive"], field="unit.global_row_end_exclusive"
    )

    raw_ranges = value["local_row_ranges"]
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise SourceAdapterError("unit.local_row_ranges must be a non-empty array")
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    for index, raw_range in enumerate(raw_ranges):
        location = f"unit.local_row_ranges[{index}]"
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise SourceAdapterError(f"{location} must be [start, end_exclusive]")
        start = _required_int(raw_range[0], field=f"{location}[0]")
        end = _required_int(raw_range[1], field=f"{location}[1]")
        if start >= end:
            raise SourceAdapterError(f"{location} must be non-empty")
        if end > source.rows:
            raise SourceAdapterError(
                f"{location} ends at {end}, beyond source rows={source.rows}"
            )
        if ranges and start < previous_end:
            raise SourceAdapterError(
                f"{location} overlaps or precedes the previous range ending at {previous_end}"
            )
        ranges.append((start, end))
        previous_end = end

    selected_rows = sum(end - start for start, end in ranges)
    if dataset_row_end - dataset_row_start != selected_rows:
        raise SourceAdapterError(
            "unit dataset row interval length does not equal flattened local_row_ranges length"
        )
    if global_row_end - global_row_start != selected_rows:
        raise SourceAdapterError(
            "unit global row interval length does not equal flattened local_row_ranges length"
        )

    return SourceUnit(
        dataset_id=_required_string(value["dataset_id"], field="unit.dataset_id"),
        dataset_ordinal=_required_int(value["dataset_ordinal"], field="unit.dataset_ordinal"),
        dataset_unit_index=_required_int(
            value["dataset_unit_index"], field="unit.dataset_unit_index"
        ),
        global_unit_index=_required_int(
            value["global_unit_index"], field="unit.global_unit_index"
        ),
        dataset_row_start=dataset_row_start,
        dataset_row_end_exclusive=dataset_row_end,
        global_row_start=global_row_start,
        global_row_end_exclusive=global_row_end,
        local_row_ranges=tuple(ranges),
        source=source,
        adapter=adapter,
        main_s3_uri=str(value["main_s3_uri"]),
    )


def _verify_head(unit: SourceUnit, *, s3_client: Any) -> None:
    source = unit.source
    head = s3_client.head_object(Bucket=source.bucket, Key=source.key)
    actual_size = head.get("ContentLength")
    actual_etag = _clean_etag(head.get("ETag"))
    if type(actual_size) is not int or actual_size != source.bytes:
        raise SourceAdapterError(
            f"{source.s3_uri}: ContentLength changed: expected={source.bytes} actual={actual_size!r}"
        )
    if actual_etag != source.etag:
        raise SourceAdapterError(
            f"{source.s3_uri}: ETag changed: expected={source.etag!r} actual={actual_etag!r}"
        )


def _instruction(raw: Any, *, context: str) -> str:
    if isinstance(raw, str):
        values: Sequence[Any] = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise SourceAdapterError(f"{context}: instructions must be a string or list of strings")
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise SourceAdapterError(f"{context}: instructions contains no non-empty instruction")


def _instruction_with_missing_fallback(
    raw: Any,
    *,
    context: str,
    field: str = "instructions",
) -> tuple[str, dict[str, Any] | None]:
    """Preserve all-row coverage when a frozen source has no usable instruction."""

    missing_reason: str | None = None
    if raw is None:
        missing_reason = "null_source_instruction"
    elif isinstance(raw, str):
        if not raw.strip():
            missing_reason = "empty_source_instruction"
    elif isinstance(raw, list):
        if not any(isinstance(value, str) and value.strip() for value in raw):
            missing_reason = "empty_source_instruction"

    if missing_reason is not None:
        return MISSING_INSTRUCTION_FALLBACK, {
            "source_instruction_fallback": {
                "version": MISSING_INSTRUCTION_FALLBACK_VERSION,
                "source_field": field,
                "reason": missing_reason,
            }
        }
    if field == "edit_instruction":
        if not isinstance(raw, str):
            raise SourceAdapterError(
                f"{context}: edit_instruction must be a non-empty string"
            )
        return raw.strip(), None
    return _instruction(raw, context=context), None


def _sample_id(raw: Any, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise SourceAdapterError(f"{context}: sample_id must be a non-empty string")
    return raw.strip()


def _require_single(raw: Mapping[str, Any], *, context: str) -> None:
    num_inputs = raw.get("num_inputs")
    if type(num_inputs) is not int or num_inputs != 1:
        raise SourceAdapterError(f"{context}: num_inputs must equal 1")


def parquet_media_identity(
    source: SourceIdentity,
    *,
    local_row: int,
    column: str,
    list_index: int | None = None,
) -> str:
    """Return a stable locator for one media cell in an immutable Parquet object."""

    encoded_key = quote(source.key, safe="/-._~")
    encoded_etag = quote(source.etag, safe="")
    identity = (
        f"parquet+s3://{source.bucket}/{encoded_key}?etag={encoded_etag}"
        f"#row={local_row}&column={quote(column, safe='_-')}"
    )
    if list_index is not None:
        identity += f"&index={list_index}"
    return identity


def _bytes_payload(value: Any, *, field: str, context: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise SourceAdapterError(f"{context}: {field} must contain image bytes")
    payload = bytes(value)
    if not payload:
        raise SourceAdapterError(f"{context}: {field} must not be empty")
    return payload


def _materialize(
    payload: bytes,
    *,
    work_dir: Path,
    source: SourceIdentity,
    local_row: int,
    role: str,
) -> Path:
    source_digest = hashlib.sha256(
        f"{source.bucket}\0{source.key}\0{source.etag}\0{source.bytes}".encode("utf-8")
    ).hexdigest()[:20]
    payload_digest = hashlib.sha256(payload).hexdigest()
    directory = work_dir / "embedded_media" / source_digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"row-{local_row:09d}.{role}-{payload_digest[:20]}.img"
    )
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != payload_digest:
            raise SourceAdapterError(f"deterministic media path has unexpected contents: {destination}")
        return destination

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _dataset_source(
    unit: SourceUnit,
    *,
    local_row: int,
    selected_ordinal: int,
) -> dict[str, Any]:
    return {
        "dataset_id": unit.dataset_id,
        "dataset_ordinal": unit.dataset_ordinal,
        "dataset_unit_index": unit.dataset_unit_index,
        "global_unit_index": unit.global_unit_index,
        "dataset_row_index": unit.dataset_row_start + selected_ordinal,
        "global_input_index": unit.global_row_start + selected_ordinal,
        "parquet_s3_uri": unit.source.s3_uri,
        "parquet_object_index": unit.source.object_index,
        "source_parquet_local_row": local_row,
    }


def _uri_record(
    raw: Mapping[str, Any],
    *,
    unit: SourceUnit,
    local_row: int,
    selected_ordinal: int,
) -> InferenceRecord:
    context = f"{unit.source.s3_uri} row {local_row}"
    _require_single(raw, context=context)
    input_paths = raw.get("input_s3_paths")
    if not isinstance(input_paths, list) or len(input_paths) != 1:
        raise SourceAdapterError(f"{context}: input_s3_paths must contain exactly one URI")
    _parse_s3_media_uri(input_paths[0], field=f"{context} input_s3_paths[0]")
    output_path = raw.get("output_s3_path")
    _parse_s3_media_uri(output_path, field=f"{context} output_s3_path")
    dataset_source = _dataset_source(
        unit, local_row=local_row, selected_ordinal=selected_ordinal
    )
    instruction, evaluation_metadata = _instruction_with_missing_fallback(
        raw.get("instructions"), context=context
    )
    return InferenceRecord(
        input_index=dataset_source["global_input_index"],
        sample_id=_sample_id(raw.get("sample_id"), context=context),
        instruction=instruction,
        reference_image=input_paths[0],
        target_image=output_path,
        evaluation_metadata=evaluation_metadata,
        dataset_source=dataset_source,
    )


def _embedded_record(
    raw: Mapping[str, Any],
    *,
    unit: SourceUnit,
    local_row: int,
    selected_ordinal: int,
    work_dir: Path,
) -> InferenceRecord:
    context = f"{unit.source.s3_uri} row {local_row}"
    _require_single(raw, context=context)
    input_images = raw.get("input_images")
    if not isinstance(input_images, list) or len(input_images) != 1:
        raise SourceAdapterError(f"{context}: input_images must contain exactly one image")
    reference_payload = _bytes_payload(
        input_images[0], field="input_images[0]", context=context
    )
    target_payload = _bytes_payload(
        raw.get("output_image"), field="output_image", context=context
    )
    reference_path = _materialize(
        reference_payload,
        work_dir=work_dir,
        source=unit.source,
        local_row=local_row,
        role="reference",
    )
    target_path = _materialize(
        target_payload,
        work_dir=work_dir,
        source=unit.source,
        local_row=local_row,
        role="target",
    )
    dataset_source = _dataset_source(
        unit, local_row=local_row, selected_ordinal=selected_ordinal
    )
    instruction, evaluation_metadata = _instruction_with_missing_fallback(
        raw.get("instructions"), context=context
    )
    return InferenceRecord(
        input_index=dataset_source["global_input_index"],
        sample_id=_sample_id(raw.get("sample_id"), context=context),
        instruction=instruction,
        reference_image=str(reference_path),
        target_image=str(target_path),
        evaluation_metadata=evaluation_metadata,
        reference_image_identity=parquet_media_identity(
            unit.source,
            local_row=local_row,
            column="input_images",
            list_index=0,
        ),
        target_image_identity=parquet_media_identity(
            unit.source,
            local_row=local_row,
            column="output_image",
        ),
        dataset_source=dataset_source,
    )


def _embedded_synth_record(
    raw: Mapping[str, Any],
    *,
    unit: SourceUnit,
    local_row: int,
    selected_ordinal: int,
    work_dir: Path,
) -> InferenceRecord:
    context = f"{unit.source.s3_uri} row {local_row}"
    instruction, evaluation_metadata = _instruction_with_missing_fallback(
        raw.get("edit_instruction"),
        context=context,
        field="edit_instruction",
    )
    reference_payload = _bytes_payload(
        raw.get("source_image"), field="source_image", context=context
    )
    target_payload = _bytes_payload(
        raw.get("target_image"), field="target_image", context=context
    )
    reference_path = _materialize(
        reference_payload,
        work_dir=work_dir,
        source=unit.source,
        local_row=local_row,
        role="reference",
    )
    target_path = _materialize(
        target_payload,
        work_dir=work_dir,
        source=unit.source,
        local_row=local_row,
        role="target",
    )
    dataset_source = _dataset_source(
        unit, local_row=local_row, selected_ordinal=selected_ordinal
    )
    return InferenceRecord(
        input_index=dataset_source["global_input_index"],
        sample_id=_sample_id(raw.get("sample_id"), context=context),
        instruction=instruction,
        reference_image=str(reference_path),
        target_image=str(target_path),
        evaluation_metadata=evaluation_metadata,
        reference_image_identity=parquet_media_identity(
            unit.source,
            local_row=local_row,
            column="source_image",
        ),
        target_image_identity=parquet_media_identity(
            unit.source,
            local_row=local_row,
            column="target_image",
        ),
        dataset_source=dataset_source,
    )


def load_unit_records(
    unit_value: Mapping[str, Any],
    *,
    s3_client: Any,
    work_dir: str | Path,
) -> list[InferenceRecord]:
    """Load exactly the selected rows from one frozen source work unit."""

    unit = validate_unit(unit_value)
    _verify_head(unit, s3_client=s3_client)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("pyarrow is required by the image-edit source adapter") from exc

    columns = ["sample_id"]
    if unit.adapter == URI_SINGLE_V1:
        columns.extend(
            ["instructions", "num_inputs", "input_s3_paths", "output_s3_path"]
        )
    elif unit.adapter == EMBEDDED_SINGLE_V1:
        columns.extend(["instructions", "num_inputs", "input_images", "output_image"])
    else:
        columns.extend(
            ["edit_instruction", "source_image", "target_image"]
        )

    reader = S3RangeReader(
        s3_client,
        bucket=unit.source.bucket,
        key=unit.source.key,
        size=unit.source.bytes,
    )
    parquet = pq.ParquetFile(reader)
    if parquet.metadata.num_rows != unit.source.rows:
        raise SourceAdapterError(
            f"{unit.source.s3_uri}: Parquet rows changed: "
            f"expected={unit.source.rows} actual={parquet.metadata.num_rows}"
        )
    missing_columns = sorted(set(columns).difference(parquet.schema_arrow.names))
    if missing_columns:
        raise SourceAdapterError(
            f"{unit.source.s3_uri}: missing required columns {missing_columns}"
        )
    work_path = Path(work_dir).expanduser().resolve()
    records: list[InferenceRecord] = []
    selected_ordinal = 0
    for local_row, raw in _iter_selected_parquet_rows(
        parquet,
        unit=unit,
        columns=columns,
    ):
        if unit.adapter == URI_SINGLE_V1:
            record = _uri_record(
                raw,
                unit=unit,
                local_row=local_row,
                selected_ordinal=selected_ordinal,
            )
        elif unit.adapter == EMBEDDED_SINGLE_V1:
            record = _embedded_record(
                raw,
                unit=unit,
                local_row=local_row,
                selected_ordinal=selected_ordinal,
                work_dir=work_path,
            )
        else:
            record = _embedded_synth_record(
                raw,
                unit=unit,
                local_row=local_row,
                selected_ordinal=selected_ordinal,
                work_dir=work_path,
            )
        records.append(record)
        selected_ordinal += 1

    if selected_ordinal != unit.selected_rows:
        raise SourceAdapterError(
            f"{unit.source.s3_uri}: selected rows={selected_ordinal}, expected={unit.selected_rows}"
        )
    if records and records[-1].input_index + 1 != unit.global_row_end_exclusive:
        raise SourceAdapterError("adapter global input mapping did not reach unit end")
    return records
