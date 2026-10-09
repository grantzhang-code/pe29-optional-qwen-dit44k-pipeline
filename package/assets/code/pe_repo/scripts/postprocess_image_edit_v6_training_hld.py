"""Postprocess prepared image-edit-v6 training JSONL shards on S3.

Each input row keeps its training record fields, replaces ``final_json`` with
the deterministic inference-time HLD/layout projection, and adds the complete
``hld_label_cleanup`` audit. Sources are read at a pinned S3 VersionId and
outputs are create-only, so retries can resume without overwriting sources.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets import image_edit_v6_hld_postprocess as hld_postprocess
from dlms.datasets.image_edit_v6_full_json import validate_v6_target

POSTPROCESS_VERSION = hld_postprocess.POSTPROCESS_VERSION
FORMAT_VERSION = "image_edit_v6_training_hld_postprocess_v3"
TRAINING_ADAPTER_VERSION = "image_edit_v6_training_hld_adapter_v3"
DEFAULT_INPUT_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
    "full64-v7-call2-final-prepared-32k-256shards-gavin-disjoint-"
    "no-recovery-dedup-v2/"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT.rstrip("/") + "-hld-postprocess-v7-fixed-v1/"
DEFAULT_PARTITIONS = 64
DEFAULT_PARTS_PER_PARTITION = 4
DEFAULT_WORKER_COUNT = 8
DEFAULT_CHUNKS_PER_OBJECT = 4
DEFAULT_PROCESS_COUNT = 128
DEFAULT_RANGE_OVERLAP_BYTES = 8 * 1024 * 1024
MAX_RANGE_OVERLAP_BYTES = 128 * 1024 * 1024

_COUNTER_FIELDS = (
    "rows",
    "changed_rows",
    "unchanged_rows",
    "hld_changed_rows",
    "layout_changed_rows",
    "small_bbox_elements_removed",
    "descendant_elements_removed",
    "invalid_label_occurrences_removed",
    "hallucinated_label_occurrences_removed",
    "bbox_removed_label_occurrences_removed",
    "hallucinated_clauses_removed",
    "long_clauses_retained_with_label_only_cleanup",
    "invalid_label_occurrences_removed_from_retained_clauses",
    "clause_numbered_label_occurrences_removed",
    "short_sentences_removed",
    "punctuation_characters_removed",
    "orphaned_conjunctions_removed",
    "layout_preservation_fallback_rows",
    "rows_with_internal_label_boundary_punctuation",
    "rows_with_emergent_invalid_labels",
    "emergent_invalid_label_occurrences_removed",
)

_PROCESS_S3_CLIENT: Any | None = None
_CORE_CLEANUP_DELETION_ARTIFACTS = hld_postprocess._cleanup_deletion_artifacts


def _s3_client() -> Any:
    global _PROCESS_S3_CLIENT
    if _PROCESS_S3_CLIENT is None:
        _PROCESS_S3_CLIENT = boto3.client("s3", region_name="us-east-1")
    return _PROCESS_S3_CLIENT


def _reset_process_s3_client() -> None:
    """Drop a parent boto client after fork before a child performs S3 I/O."""

    global _PROCESS_S3_CLIENT
    _PROCESS_S3_CLIENT = None


def _corrected_hallucinated_clause_deletion_spans(
    text: str,
    *,
    invalid_matches: list[Any],
    layout_labels: set[str],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Apply the v7 clause rule without splitting inside angle-label tokens."""

    label_spans = [(match.start(), match.end()) for match in hld_postprocess.NUMBERED_LABEL_RE.finditer(text)]
    boundary_positions = [
        match.start()
        for match in hld_postprocess._CLAUSE_BOUNDARY_RE.finditer(text)
        if not any(start <= match.start() < end for start, end in label_spans)
    ]
    clause_matches: dict[tuple[int, int], list[Any]] = {}
    for match in invalid_matches:
        left_index = bisect_left(boundary_positions, match.start())
        left_boundary = boundary_positions[left_index - 1] if left_index else -1
        right_index = bisect_left(boundary_positions, match.end())
        right_boundary = (
            boundary_positions[right_index] if right_index < len(boundary_positions) else len(text)
        )
        clause_matches.setdefault((left_boundary + 1, right_boundary), []).append(match)

    deletion_spans: list[tuple[int, int]] = []
    removed_clauses: list[dict[str, Any]] = []
    retained_clauses: list[dict[str, Any]] = []
    for (start, end), triggering_matches in sorted(clause_matches.items()):
        clause_text = text[start:end]
        labels = hld_postprocess.NUMBERED_LABEL_RE.findall(clause_text)
        non_label_word_count = len(
            hld_postprocess._NON_LABEL_WORD_RE.findall(
                hld_postprocess.NUMBERED_LABEL_RE.sub(" ", clause_text)
            )
        )
        clause_audit = {
            "start": start,
            "end": end,
            "left_boundary": text[start - 1] if start else None,
            "right_boundary": text[end] if end < len(text) else None,
            "triggering_invalid_labels": [match.group(0) for match in triggering_matches],
            "non_label_word_count": non_label_word_count,
            "numbered_label_occurrences": len(labels),
            "valid_label_occurrences": sum(label in layout_labels for label in labels),
            "invalid_label_occurrences": sum(label not in layout_labels for label in labels),
            "clause_sha256": hld_postprocess._sha256_text(clause_text),
        }
        if non_label_word_count <= hld_postprocess.MAX_NON_LABEL_WORDS_FOR_CLAUSE_DELETION:
            deletion_spans.append((start, end))
            removed_clauses.append(clause_audit | {"action": "delete_clause"})
        else:
            deletion_spans.extend((match.start(), match.end()) for match in triggering_matches)
            retained_clauses.append(clause_audit | {"action": "delete_invalid_labels_only"})

    spans = hld_postprocess._merge_spans(deletion_spans)
    emergent_label_records: list[dict[str, Any]] = []
    # Deleting an invalid nested label can join the surrounding fragments into
    # a new numbered label, for example ``<outer <invalid 2> 1>`` ->
    # ``<outer 1>``. Expand the original deletion span to cover every such
    # newly formed invalid token. The mapping keeps all deletions anchored to
    # the original HLD, so the shared v7 postprocessor can still own the audit.
    while True:
        chunks: list[str] = []
        original_offsets: list[int] = []
        cursor = 0
        for span_start, span_end in spans:
            chunks.append(text[cursor:span_start])
            original_offsets.extend(range(cursor, span_start))
            cursor = span_end
        chunks.append(text[cursor:])
        original_offsets.extend(range(cursor, len(text)))
        projected_text = "".join(chunks)
        emergent_matches = [
            match
            for match in hld_postprocess.NUMBERED_LABEL_RE.finditer(projected_text)
            if match.group(0) not in layout_labels
        ]
        if not emergent_matches:
            break

        additions: list[tuple[int, int]] = []
        for match in emergent_matches:
            start = original_offsets[match.start()]
            end = original_offsets[match.end() - 1] + 1
            additions.append((start, end))
            emergent_label_records.append(
                {
                    "projected_label_sha256": hld_postprocess._sha256_text(match.group(0)),
                    "projected_start": match.start(),
                    "projected_end": match.end(),
                    "expanded_original_start": start,
                    "expanded_original_end": end,
                }
            )
        expanded_spans = hld_postprocess._merge_spans([*spans, *additions])
        if expanded_spans == spans:
            raise hld_postprocess.HLDPostprocessError(
                "could not eliminate an emergent invalid numbered label"
            )
        spans = expanded_spans

    label_only_occurrences = sum(clause["invalid_label_occurrences"] for clause in retained_clauses)
    return spans, {
        "enabled": True,
        "rule": (
            "delete the nearest-punctuation-bounded segment when its non-label "
            "word count is at most 10; otherwise retain the segment and delete only "
            "its invalid labels; punctuation inside numbered angle-label tokens is "
            "not a clause boundary"
        ),
        "boundary_punctuation": hld_postprocess._CLAUSE_BOUNDARY_PUNCTUATION,
        "ignore_boundary_punctuation_inside_numbered_labels": True,
        "maximum_non_label_words_for_clause_deletion": (
            hld_postprocess.MAX_NON_LABEL_WORDS_FOR_CLAUSE_DELETION
        ),
        "clauses_examined": len(clause_matches),
        "clauses_removed": len(removed_clauses),
        "clauses_retained_with_label_only_cleanup": len(retained_clauses),
        "invalid_label_occurrences_removed_from_retained_clauses": (label_only_occurrences),
        "numbered_label_occurrences_removed": sum(
            clause["numbered_label_occurrences"] for clause in removed_clauses
        )
        + label_only_occurrences,
        "emergent_invalid_label_occurrences_removed": len(emergent_label_records),
        "emergent_invalid_labels": emergent_label_records,
        "removed_clauses": removed_clauses,
        "retained_clauses_with_label_only_cleanup": retained_clauses,
    }


def _has_internal_label_boundary_punctuation(text: str) -> bool:
    return any(
        any(character in hld_postprocess._CLAUSE_BOUNDARY_PUNCTUATION for character in match.group(0)[1:-1])
        for match in hld_postprocess.NUMBERED_LABEL_RE.finditer(text)
    )


def _corrected_short_complete_sentence_deletion_spans(
    text: str,
    *,
    enabled: bool,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Apply the v7 short-sentence rule without splitting label names."""

    masked_characters = list(text)
    for label_match in hld_postprocess.NUMBERED_LABEL_RE.finditer(text):
        for index in range(label_match.start(), label_match.end()):
            if masked_characters[index] in ".!?。！？":
                masked_characters[index] = " "
    boundary_masked_text = "".join(masked_characters)

    examined: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    candidate_spans: list[tuple[int, int]] = []
    if enabled:
        for match in hld_postprocess._FULL_STOP_SENTENCE_RE.finditer(boundary_masked_text):
            sentence_text = text[match.start() : match.end()]
            non_label_word_count = len(
                hld_postprocess._NON_LABEL_WORD_RE.findall(
                    hld_postprocess.NUMBERED_LABEL_RE.sub(" ", sentence_text[:-1])
                )
            )
            record = {
                "start": match.start(),
                "end": match.end(),
                "terminal": sentence_text[-1],
                "non_label_word_count": non_label_word_count,
                "sentence_sha256": hld_postprocess._sha256_text(sentence_text),
            }
            examined.append(record)
            if non_label_word_count < hld_postprocess.MIN_NON_LABEL_WORDS_FOR_COMPLETE_SENTENCE:
                candidate_spans.append((match.start(), match.end()))
                eligible.append(record | {"action": "delete_complete_sentence"})

    deletion_spans = hld_postprocess._merge_spans(candidate_spans)
    skipped_to_preserve_nonempty_hld = False
    if (
        deletion_spans
        and not hld_postprocess._cleanup_deletion_artifacts(
            hld_postprocess._delete_spans(text, deletion_spans)
        ).strip()
    ):
        deletion_spans = []
        skipped_to_preserve_nonempty_hld = True
    removed = [] if skipped_to_preserve_nonempty_hld else eligible

    return deletion_spans, {
        "enabled": True,
        "applied_after_prior_hld_deletion": enabled,
        "rule": (
            "after an earlier HLD deletion, delete each complete sentence ending "
            "in '.' or '。' when it contains fewer than 5 non-label words; full-stop "
            "characters inside numbered angle-label tokens are not sentence endings"
        ),
        "sentence_endings": ".。",
        "ignore_sentence_punctuation_inside_numbered_labels": True,
        "minimum_non_label_words_to_keep": (hld_postprocess.MIN_NON_LABEL_WORDS_FOR_COMPLETE_SENTENCE),
        "sentences_examined": len(examined),
        "sentences_eligible": len(eligible),
        "sentences_removed": len(removed),
        "skipped_to_preserve_nonempty_hld": skipped_to_preserve_nonempty_hld,
        "eligible_sentences": eligible,
        "removed_sentences": removed,
    }


def _protected_cleanup_deletion_artifacts(text: str) -> str:
    """Run v7 artifact cleanup without rewriting numbered label tokens."""

    matches = list(hld_postprocess.NUMBERED_LABEL_RE.finditer(text))
    if not matches:
        return _CORE_CLEANUP_DELETION_ARTIFACTS(text)

    marker_prefix = "HLDPROTECTEDLABELTOKEN"
    while marker_prefix in text:
        marker_prefix += "X"
    labels: list[tuple[str, str]] = []
    chunks: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        marker = f"{marker_prefix}{index}END"
        chunks.extend((text[cursor : match.start()], marker))
        labels.append((marker, match.group(0)))
        cursor = match.end()
    chunks.append(text[cursor:])
    cleaned = _CORE_CLEANUP_DELETION_ARTIFACTS("".join(chunks))
    for marker, label in labels:
        cleaned = cleaned.replace(marker, label)
    return cleaned


def _postprocess_training_target(
    final_json: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run v7 with training-data safety adaptations and revalidate."""

    original_clause_function = hld_postprocess._hallucinated_clause_deletion_spans
    original_short_sentence_function = hld_postprocess._short_complete_sentence_deletion_spans
    original_artifact_cleanup_function = hld_postprocess._cleanup_deletion_artifacts
    original_minimum_area = hld_postprocess.MIN_LAYOUT_BBOX_AREA
    used_layout_fallback = False
    default_bbox_cleanup: dict[str, Any] | None = None
    try:
        hld_postprocess._hallucinated_clause_deletion_spans = _corrected_hallucinated_clause_deletion_spans
        hld_postprocess._short_complete_sentence_deletion_spans = (
            _corrected_short_complete_sentence_deletion_spans
        )
        hld_postprocess._cleanup_deletion_artifacts = _protected_cleanup_deletion_artifacts
        processed, cleanup = hld_postprocess.postprocess_v6_hld_labels(final_json)
        if not processed["elements"]["layout"]:
            used_layout_fallback = True
            default_bbox_cleanup = cleanup["bbox_cleanup"]
            hld_postprocess.MIN_LAYOUT_BBOX_AREA = 0
            processed, cleanup = hld_postprocess.postprocess_v6_hld_labels(final_json)
    finally:
        hld_postprocess._hallucinated_clause_deletion_spans = original_clause_function
        hld_postprocess._short_complete_sentence_deletion_spans = original_short_sentence_function
        hld_postprocess._cleanup_deletion_artifacts = original_artifact_cleanup_function
        hld_postprocess.MIN_LAYOUT_BBOX_AREA = original_minimum_area

    validate_v6_target(processed)
    if used_layout_fallback:
        bbox_cleanup = cleanup["bbox_cleanup"]
        bbox_cleanup["comparison"] = "preserve_all_layout_when_default_cleanup_would_produce_empty_layout"
        bbox_cleanup["minimum_normalized_area"] = original_minimum_area
        bbox_cleanup["minimum_canvas_fraction"] = hld_postprocess.MIN_LAYOUT_BBOX_CANVAS_FRACTION
        bbox_cleanup["minimum_canvas_percent"] = 100 * hld_postprocess.MIN_LAYOUT_BBOX_CANVAS_FRACTION
        bbox_cleanup["preserved_to_avoid_empty_layout"] = True
        bbox_cleanup["default_cleanup"] = default_bbox_cleanup
    cleanup["training_adapter"] = {
        "version": TRAINING_ADAPTER_VERSION,
        "ignore_boundary_punctuation_inside_numbered_labels": True,
        "ignore_sentence_punctuation_inside_numbered_labels": True,
        "protect_numbered_labels_from_deletion_artifact_cleanup": True,
        "row_contains_internal_label_boundary_punctuation": (
            _has_internal_label_boundary_punctuation(str(final_json["high_level_description"]))
        ),
        "layout_preservation_fallback": used_layout_fallback,
        "layout_fallback_rule": (
            "if the default 0.1%-canvas bbox cleanup would remove every layout "
            "element, retain the original validated layout and continue HLD cleanup"
        ),
    }
    return processed, cleanup


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_s3_root(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected S3 URI, got {uri!r}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"expected bucket and nonempty prefix, got {uri!r}")
    return bucket, key.strip("/") + "/"


def _validate_roots(input_root: str, output_root: str) -> tuple[str, str, str, str]:
    input_bucket, input_prefix = _parse_s3_root(input_root)
    output_bucket, output_prefix = _parse_s3_root(output_root)
    if input_bucket == output_bucket and input_prefix == output_prefix:
        raise ValueError("output root must differ from input root")
    if input_bucket == output_bucket and (
        output_prefix.startswith(input_prefix) or input_prefix.startswith(output_prefix)
    ):
        raise ValueError("input and output roots must not contain one another")
    return input_bucket, input_prefix, output_bucket, output_prefix


def _relative_keys(partitions: int, parts_per_partition: int) -> list[str]:
    if partitions < 1 or parts_per_partition < 1:
        raise ValueError("partition counts must be positive")
    return [
        f"shards/partition-{partition:03d}/part-{part:03d}-of-{parts_per_partition:03d}.jsonl"
        for partition in range(partitions)
        for part in range(parts_per_partition)
    ]


def _head_source(
    bucket: str,
    prefix: str,
    relative_key: str,
) -> dict[str, Any]:
    key = prefix + relative_key
    head = _s3_client().head_object(Bucket=bucket, Key=key)
    metadata = head.get("Metadata") or {}
    sha256 = str(metadata.get("sha256") or "")
    rows = int(metadata.get("rows") or 0)
    version_id = str(head.get("VersionId") or "")
    size = int(head.get("ContentLength") or 0)
    if len(sha256) != 64 or rows < 1 or not version_id or size < 1:
        raise ValueError(
            f"source identity metadata is incomplete for s3://{bucket}/{key}: "
            f"version={version_id!r} sha256={sha256!r} rows={rows} bytes={size}"
        )
    return {
        "bucket": bucket,
        "key": key,
        "relative_key": relative_key,
        "version_id": version_id,
        "sha256": sha256,
        "rows": rows,
        "bytes": size,
    }


def _head_sources(
    bucket: str,
    prefix: str,
    relative_keys: Iterable[str],
    *,
    threads: int = 32,
) -> list[dict[str, Any]]:
    keys = list(relative_keys)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(threads, len(keys))) as pool:
        results = list(pool.map(lambda key: _head_source(bucket, prefix, key), keys))
    return sorted(results, key=lambda item: str(item["relative_key"]))


def _chunk_bounds(size: int, chunks: int) -> list[tuple[int, int]]:
    if size < 1 or chunks < 1:
        raise ValueError("size and chunks must be positive")
    return [(size * index // chunks, size * (index + 1) // chunks) for index in range(chunks)]


def _extract_complete_line_segment(
    payload: bytes,
    *,
    request_start: int,
    logical_start: int,
    logical_end: int,
    object_size: int,
) -> bytes | None:
    if logical_start == 0:
        segment_start = 0
    elif payload[:1] == b"\n":
        segment_start = 1
    else:
        first_newline = payload.find(b"\n")
        if first_newline < 0:
            return None
        segment_start = first_newline + 1

    if logical_end == object_size:
        segment_end = len(payload)
    else:
        target = logical_end - request_start
        if target <= 0 or target > len(payload):
            return None
        if payload[target - 1 : target] == b"\n":
            segment_end = target
        else:
            next_newline = payload.find(b"\n", target)
            if next_newline < 0:
                return None
            segment_end = next_newline + 1
    if segment_start > segment_end:
        raise ValueError(f"invalid line-aligned range: start={segment_start} end={segment_end}")
    return payload[segment_start:segment_end]


def _get_line_aligned_range(task: Mapping[str, Any]) -> bytes:
    start = int(task["start"])
    end = int(task["end"])
    size = int(task["source_bytes"])
    request_start = 0 if start == 0 else start - 1
    overlap = DEFAULT_RANGE_OVERLAP_BYTES
    while True:
        request_end = size - 1 if end == size else min(size - 1, end + overlap - 1)
        response = _s3_client().get_object(
            Bucket=str(task["source_bucket"]),
            Key=str(task["source_key"]),
            VersionId=str(task["source_version_id"]),
            Range=f"bytes={request_start}-{request_end}",
        )
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            body.close()
        expected = request_end - request_start + 1
        if len(payload) != expected:
            raise ValueError(
                f"short ranged read for {task['source_key']}: expected={expected} actual={len(payload)}"
            )
        segment = _extract_complete_line_segment(
            payload,
            request_start=request_start,
            logical_start=start,
            logical_end=end,
            object_size=size,
        )
        if segment is not None:
            return segment
        if request_end == size - 1 or overlap >= MAX_RANGE_OVERLAP_BYTES:
            raise ValueError(
                f"could not find a JSONL boundary within {overlap} bytes for "
                f"{task['source_key']} range {start}:{end}"
            )
        overlap = min(MAX_RANGE_OVERLAP_BYTES, overlap * 2)


def _empty_totals() -> Counter[str]:
    return Counter({field: 0 for field in _COUNTER_FIELDS})


def _accumulate_cleanup(totals: Counter[str], cleanup: Mapping[str, Any]) -> None:
    totals["rows"] += 1
    totals["changed_rows" if cleanup["changed"] else "unchanged_rows"] += 1
    totals["hld_changed_rows"] += int(bool(cleanup["hld_changed"]))
    totals["layout_changed_rows"] += int(bool(cleanup["layout_changed"]))
    bbox = cleanup["bbox_cleanup"]
    totals["small_bbox_elements_removed"] += int(bbox["small_bbox_elements_removed"])
    totals["descendant_elements_removed"] += int(bbox["descendant_elements_removed"])
    for field in (
        "invalid_label_occurrences_removed",
        "hallucinated_label_occurrences_removed",
        "bbox_removed_label_occurrences_removed",
    ):
        totals[field] += int(cleanup[field])
    clauses = cleanup["hallucinated_clause_cleanup"]
    totals["hallucinated_clauses_removed"] += int(clauses["clauses_removed"])
    totals["long_clauses_retained_with_label_only_cleanup"] += int(
        clauses["clauses_retained_with_label_only_cleanup"]
    )
    totals["invalid_label_occurrences_removed_from_retained_clauses"] += int(
        clauses["invalid_label_occurrences_removed_from_retained_clauses"]
    )
    totals["clause_numbered_label_occurrences_removed"] += int(clauses["numbered_label_occurrences_removed"])
    totals["short_sentences_removed"] += int(cleanup["short_sentence_cleanup"]["sentences_removed"])
    punctuation = cleanup["punctuation_cleanup"]
    totals["punctuation_characters_removed"] += int(punctuation["characters_removed"])
    totals["orphaned_conjunctions_removed"] += int(punctuation["orphaned_conjunctions_removed"])
    adapter = cleanup["training_adapter"]
    totals["layout_preservation_fallback_rows"] += int(bool(adapter["layout_preservation_fallback"]))
    totals["rows_with_internal_label_boundary_punctuation"] += int(
        bool(adapter["row_contains_internal_label_boundary_punctuation"])
    )
    emergent_count = int(clauses.get("emergent_invalid_label_occurrences_removed", 0))
    totals["rows_with_emergent_invalid_labels"] += int(emergent_count > 0)
    totals["emergent_invalid_label_occurrences_removed"] += emergent_count


def _process_range(task: Mapping[str, Any]) -> dict[str, Any]:
    segment = _get_line_aligned_range(task)
    source_path = Path(str(task["source_fragment_path"]))
    output_path = Path(str(task["output_fragment_path"]))
    source_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(segment)

    totals = _empty_totals()
    sample_ids: set[str] = set()
    with output_path.open("wb") as output:
        for local_line_number, line in enumerate(segment.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{task['source_key']} chunk={task['chunk_index']} line={local_line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError("training JSONL row must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("training JSONL row has invalid sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id inside range: {sample_id}")
            sample_ids.add(sample_id)
            if "hld_label_cleanup" in row:
                raise ValueError(f"source row {sample_id} already contains hld_label_cleanup")
            final_json = row.get("final_json")
            if not isinstance(final_json, dict):
                raise TypeError(f"source row {sample_id} has invalid final_json")
            processed, cleanup = _postprocess_training_target(final_json)
            result = dict(row)
            result["final_json"] = processed
            result["hld_label_cleanup"] = cleanup
            output.write(_canonical_bytes(result) + b"\n")
            _accumulate_cleanup(totals, cleanup)

    return {
        "relative_key": str(task["relative_key"]),
        "chunk_index": int(task["chunk_index"]),
        "source_fragment_path": str(source_path),
        "source_fragment_bytes": len(segment),
        "source_fragment_sha256": hashlib.sha256(segment).hexdigest(),
        "output_fragment_path": str(output_path),
        "output_fragment_bytes": output_path.stat().st_size,
        "output_fragment_sha256": _sha256_file(output_path),
        "totals": dict(totals),
    }


def _is_not_found(exc: ClientError) -> bool:
    return str(exc.response.get("Error", {}).get("Code", "")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def _get_json_if_exists(bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = _s3_client().get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_not_found(exc):
            return None
        raise
    body = response["Body"]
    try:
        value = json.loads(body.read())
    finally:
        body.close()
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at s3://{bucket}/{key}")
    return value


def _put_bytes_create_only(
    bucket: str,
    key: str,
    payload: bytes,
    *,
    content_type: str,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        return _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata=dict(metadata or {}),
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code", "")) not in {
            "412",
            "PreconditionFailed",
        }:
            raise
        response = _s3_client().get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            existing = body.read()
        finally:
            body.close()
        if existing != payload:
            raise ValueError(f"create-only collision at s3://{bucket}/{key}") from exc
        return {"VersionId": response.get("VersionId"), "reused": True}


def _put_file_create_only(
    bucket: str,
    key: str,
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    with path.open("rb") as stream:
        try:
            return _s3_client().put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentType="application/x-ndjson",
                Metadata=dict(metadata),
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code", "")) not in {
                "412",
                "PreconditionFailed",
            }:
                raise
    head = _s3_client().head_object(Bucket=bucket, Key=key)
    existing_metadata = head.get("Metadata") or {}
    if (
        int(head.get("ContentLength") or -1) != path.stat().st_size
        or str(existing_metadata.get("sha256") or "") != metadata["sha256"]
        or str(existing_metadata.get("postprocess-version") or "") != POSTPROCESS_VERSION
    ):
        raise ValueError(f"create-only collision at s3://{bucket}/{key}")
    return {"VersionId": head.get("VersionId"), "reused": True}


def _audit_relative_key(relative_key: str) -> str:
    return "_audit/shards/" + relative_key + ".audit.json"


def _validate_existing_result(
    source: Mapping[str, Any],
    *,
    output_bucket: str,
    output_prefix: str,
) -> dict[str, Any] | None:
    audit_key = output_prefix + _audit_relative_key(str(source["relative_key"]))
    audit = _get_json_if_exists(output_bucket, audit_key)
    output_key = output_prefix + str(source["relative_key"])
    try:
        output_head = _s3_client().head_object(Bucket=output_bucket, Key=output_key)
    except ClientError as exc:
        if not _is_not_found(exc):
            raise
        output_head = None
    if audit is None and output_head is None:
        return None
    if audit is None or output_head is None:
        raise ValueError(
            f"incomplete existing result for {source['relative_key']}: "
            f"audit={audit is not None} output={output_head is not None}"
        )
    expected_source = audit.get("source") or {}
    output = audit.get("output") or {}
    metadata = output_head.get("Metadata") or {}
    if (
        audit.get("format_version") != FORMAT_VERSION
        or audit.get("postprocess_version") != POSTPROCESS_VERSION
        or audit.get("complete") is not True
        or expected_source.get("version_id") != source["version_id"]
        or expected_source.get("sha256") != source["sha256"]
        or int(expected_source.get("bytes") or -1) != int(source["bytes"])
        or output.get("key") != output_key
        or int(output.get("bytes") or -1) != int(output_head["ContentLength"])
        or output.get("sha256") != metadata.get("sha256")
        or str(metadata.get("postprocess-version") or "") != POSTPROCESS_VERSION
    ):
        raise ValueError(f"existing result identity mismatch for {source['relative_key']}")
    result = dict(audit)
    result["audit_key"] = audit_key
    result["audit_version_id"] = str(audit.get("audit_version_id") or "")
    result["output_version_id"] = str(output_head.get("VersionId") or "")
    result["reused"] = True
    return result


def _merge_object(
    source: Mapping[str, Any],
    chunks: list[Mapping[str, Any]],
    *,
    object_dir: Path,
    output_bucket: str,
    output_prefix: str,
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda item: int(item["chunk_index"]))
    if [int(item["chunk_index"]) for item in ordered] != list(range(len(ordered))):
        raise ValueError(f"chunk indexes are incomplete for {source['relative_key']}")

    source_digest = hashlib.sha256()
    source_bytes = 0
    output_digest = hashlib.sha256()
    output_bytes = 0
    totals = _empty_totals()
    merged_path = object_dir / "merged.jsonl"
    with merged_path.open("wb") as merged:
        for chunk in ordered:
            source_fragment = Path(str(chunk["source_fragment_path"]))
            output_fragment = Path(str(chunk["output_fragment_path"]))
            with source_fragment.open("rb") as stream:
                for value in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    source_digest.update(value)
                    source_bytes += len(value)
            with output_fragment.open("rb") as stream:
                for value in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    merged.write(value)
                    output_digest.update(value)
                    output_bytes += len(value)
            totals.update({key: int(value) for key, value in chunk["totals"].items()})

    if (
        source_bytes != int(source["bytes"])
        or source_digest.hexdigest() != source["sha256"]
        or totals["rows"] != int(source["rows"])
    ):
        raise ValueError(
            f"source verification failed for {source['relative_key']}: "
            f"bytes={source_bytes}/{source['bytes']} "
            f"sha256={source_digest.hexdigest()}/{source['sha256']} "
            f"rows={totals['rows']}/{source['rows']}"
        )
    if totals["rows"] != totals["changed_rows"] + totals["unchanged_rows"]:
        raise ValueError(f"row accounting failed for {source['relative_key']}")

    output_key = output_prefix + str(source["relative_key"])
    audit_key = output_prefix + _audit_relative_key(str(source["relative_key"]))
    audit: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "postprocess_version": POSTPROCESS_VERSION,
        "complete": True,
        "source": {
            "bucket": source["bucket"],
            "key": source["key"],
            "version_id": source["version_id"],
            "sha256": source["sha256"],
            "bytes": int(source["bytes"]),
            "rows": int(source["rows"]),
        },
        "output": {
            "bucket": output_bucket,
            "key": output_key,
            "sha256": output_digest.hexdigest(),
            "bytes": output_bytes,
            "rows": int(totals["rows"]),
        },
        "totals": dict(totals),
        "all_rows_revalidated_with_v6_schema": True,
        "source_projection_preserved_outside_final_json_and_hld_label_cleanup": True,
    }
    audit_payload = _canonical_bytes(audit) + b"\n"
    audit_response = _put_bytes_create_only(
        output_bucket,
        audit_key,
        audit_payload,
        content_type="application/json",
        metadata={"sha256": hashlib.sha256(audit_payload).hexdigest()},
    )
    output_response = _put_file_create_only(
        output_bucket,
        output_key,
        merged_path,
        metadata={
            "sha256": output_digest.hexdigest(),
            "rows": str(totals["rows"]),
            "postprocess-version": POSTPROCESS_VERSION,
            "source-version-id": str(source["version_id"]),
            "source-sha256": str(source["sha256"]),
        },
    )
    audit["audit_key"] = audit_key
    audit["audit_version_id"] = str(audit_response.get("VersionId") or "")
    audit["output_version_id"] = str(output_response.get("VersionId") or "")
    audit["reused"] = bool(audit_response.get("reused") or output_response.get("reused"))
    return audit


def _aggregate_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals = _empty_totals()
    source_bytes = 0
    output_bytes = 0
    objects = 0
    reused_objects = 0
    for result in results:
        objects += 1
        reused_objects += int(bool(result.get("reused")))
        source_bytes += int(result["source"]["bytes"])
        output_bytes += int(result["output"]["bytes"])
        totals.update({key: int(value) for key, value in result["totals"].items()})
    return {
        "objects": objects,
        "reused_objects": reused_objects,
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "totals": dict(totals),
    }


def _worker_summary_key(output_prefix: str, worker_index: int, worker_count: int) -> str:
    return output_prefix + f"_audit/workers/worker-{worker_index:03d}-of-{worker_count:03d}.json"


def _publish_worker_summary(
    *,
    output_bucket: str,
    output_prefix: str,
    input_root: str,
    output_root: str,
    worker_index: int,
    worker_count: int,
    results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {
        "format_version": FORMAT_VERSION + "_worker_v1",
        "postprocess_version": POSTPROCESS_VERSION,
        "complete": True,
        "input_root": input_root,
        "output_root": output_root,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "aggregate": _aggregate_results(results),
        "objects": [
            {
                "relative_key": str(result["source"]["key"]).split("/shards/", 1)[-1],
                "source": result["source"],
                "output": result["output"],
                "audit_key": result["audit_key"],
                "totals": result["totals"],
            }
            for result in sorted(results, key=lambda item: str(item["source"]["key"]))
        ],
    }
    key = _worker_summary_key(output_prefix, worker_index, worker_count)
    payload = _canonical_bytes(summary) + b"\n"
    response = _put_bytes_create_only(
        output_bucket,
        key,
        payload,
        content_type="application/json",
        metadata={"sha256": hashlib.sha256(payload).hexdigest()},
    )
    summary["key"] = key
    summary["version_id"] = str(response.get("VersionId") or "")
    return summary


def _finalize(
    *,
    input_root: str,
    output_root: str,
    output_bucket: str,
    output_prefix: str,
    expected_relative_keys: list[str],
    worker_count: int,
    wait_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    summaries: list[dict[str, Any]] = []
    while True:
        summaries = []
        missing: list[str] = []
        for worker_index in range(worker_count):
            key = _worker_summary_key(output_prefix, worker_index, worker_count)
            value = _get_json_if_exists(output_bucket, key)
            if value is None:
                missing.append(key)
            else:
                summaries.append(value)
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for worker summaries: {missing}")
        print(
            f"[finalize] waiting for {len(missing)}/{worker_count} worker summaries",
            flush=True,
        )
        time.sleep(poll_seconds)

    objects: list[dict[str, Any]] = []
    for expected_index, summary in enumerate(summaries):
        if (
            summary.get("format_version") != FORMAT_VERSION + "_worker_v1"
            or summary.get("postprocess_version") != POSTPROCESS_VERSION
            or summary.get("complete") is not True
            or int(summary.get("worker_index", -1)) != expected_index
            or int(summary.get("worker_count", -1)) != worker_count
            or summary.get("input_root") != input_root
            or summary.get("output_root") != output_root
        ):
            raise ValueError(f"invalid worker summary for index {expected_index}")
        objects.extend(summary["objects"])

    expected = sorted(expected_relative_keys)
    actual = sorted("shards/" + str(item["relative_key"]) for item in objects)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"final object inventory mismatch: missing={missing} extra={extra}")

    aggregate = _aggregate_results(
        {
            "source": item["source"],
            "output": item["output"],
            "totals": item["totals"],
            "reused": False,
        }
        for item in objects
    )
    manifest = {
        "format_version": FORMAT_VERSION + "_manifest_v1",
        "postprocess_version": POSTPROCESS_VERSION,
        "complete": True,
        "input_root": input_root,
        "output_root": output_root,
        "worker_count": worker_count,
        "physical_shard_count": len(objects),
        "aggregate": aggregate,
        "objects": sorted(objects, key=lambda item: str(item["source"]["key"])),
    }
    manifest_key = output_prefix + "hld_postprocess_manifest.json"
    manifest_payload = _canonical_bytes(manifest) + b"\n"
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    response = _put_bytes_create_only(
        output_bucket,
        manifest_key,
        manifest_payload,
        content_type="application/json",
        metadata={"sha256": manifest_sha256},
    )
    success = {
        "format_version": FORMAT_VERSION + "_success_v1",
        "complete": True,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "manifest_version_id": str(response.get("VersionId") or ""),
        "postprocess_version": POSTPROCESS_VERSION,
        "rows": int(aggregate["totals"]["rows"]),
        "shards": len(objects),
    }
    success_payload = _canonical_bytes(success) + b"\n"
    _put_bytes_create_only(
        output_bucket,
        output_prefix + "_SUCCESS",
        success_payload,
        content_type="application/json",
        metadata={"sha256": hashlib.sha256(success_payload).hexdigest()},
    )
    print(
        f"[finalize] complete shards={len(objects)} rows={aggregate['totals']['rows']} "
        f"manifest=s3://{output_bucket}/{manifest_key}",
        flush=True,
    )
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_postprocess_version != POSTPROCESS_VERSION:
        raise ValueError(
            f"postprocess version drift: runtime={POSTPROCESS_VERSION!r} "
            f"expected={args.expected_postprocess_version!r}"
        )
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index is outside worker count")
    if args.worker_count != DEFAULT_WORKER_COUNT:
        raise ValueError(f"this campaign is pinned to {DEFAULT_WORKER_COUNT} workers")
    input_bucket, input_prefix, output_bucket, output_prefix = _validate_roots(
        args.input_root, args.output_root
    )
    relative_keys = _relative_keys(args.partitions, args.parts_per_partition)
    assigned_keys = [
        key for index, key in enumerate(relative_keys) if index % args.worker_count == args.worker_index
    ]
    if not assigned_keys:
        raise ValueError("worker has no assigned input shards")
    sources = _head_sources(input_bucket, input_prefix, assigned_keys)
    print(
        f"[worker {args.worker_index}/{args.worker_count}] "
        f"assigned_objects={len(sources)} source_bytes={sum(item['bytes'] for item in sources)}",
        flush=True,
    )

    staging_root = Path(args.staging_dir).resolve() / f"worker-{args.worker_index:03d}"
    staging_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending_sources: list[dict[str, Any]] = []
    for source in sources:
        existing = _validate_existing_result(
            source,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
        )
        if existing is None:
            pending_sources.append(source)
        else:
            results.append(existing)
    print(
        f"[worker {args.worker_index}] pending={len(pending_sources)} reused={len(results)}",
        flush=True,
    )

    tasks: list[dict[str, Any]] = []
    source_by_relative_key = {str(source["relative_key"]): source for source in pending_sources}
    for source_index, source in enumerate(pending_sources):
        object_dir = staging_root / f"object-{source_index:03d}"
        for chunk_index, (start, end) in enumerate(
            _chunk_bounds(int(source["bytes"]), args.chunks_per_object)
        ):
            tasks.append(
                {
                    "source_bucket": input_bucket,
                    "source_key": source["key"],
                    "source_version_id": source["version_id"],
                    "source_bytes": source["bytes"],
                    "relative_key": source["relative_key"],
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end,
                    "source_fragment_path": str(object_dir / f"source-{chunk_index:03d}.jsonl"),
                    "output_fragment_path": str(object_dir / f"output-{chunk_index:03d}.jsonl"),
                }
            )

    chunks_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in source_by_relative_key}
    if tasks:
        max_processes = min(args.process_count, len(tasks))
        print(
            f"[worker {args.worker_index}] range_tasks={len(tasks)} processes={max_processes}",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_processes,
            initializer=_reset_process_s3_client,
        ) as pool:
            futures = [pool.submit(_process_range, task) for task in tasks]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                chunk = future.result()
                chunks_by_key[str(chunk["relative_key"])].append(chunk)
                if completed % 16 == 0 or completed == len(futures):
                    print(
                        f"[worker {args.worker_index}] processed_ranges={completed}/{len(futures)}",
                        flush=True,
                    )

    for source_index, relative_key in enumerate(sorted(chunks_by_key)):
        source = source_by_relative_key[relative_key]
        object_dir = staging_root / f"object-{source_index:03d}"
        result = _merge_object(
            source,
            chunks_by_key[relative_key],
            object_dir=object_dir,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
        )
        results.append(result)
        shutil.rmtree(object_dir)
        print(
            f"[worker {args.worker_index}] committed={relative_key} "
            f"rows={result['totals']['rows']} changed={result['totals']['changed_rows']}",
            flush=True,
        )

    results.sort(key=lambda item: str(item["source"]["key"]))
    summary = _publish_worker_summary(
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        input_root=args.input_root,
        output_root=args.output_root,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        results=results,
    )
    print(
        f"[worker {args.worker_index}] complete objects={summary['aggregate']['objects']} "
        f"rows={summary['aggregate']['totals']['rows']}",
        flush=True,
    )
    if args.worker_index == 0:
        _finalize(
            input_root=args.input_root,
            output_root=args.output_root,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            expected_relative_keys=relative_keys,
            worker_count=args.worker_count,
            wait_seconds=args.finalize_wait_seconds,
            poll_seconds=args.finalize_poll_seconds,
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--parts-per-partition", type=int, default=DEFAULT_PARTS_PER_PARTITION)
    parser.add_argument("--chunks-per-object", type=int, default=DEFAULT_CHUNKS_PER_OBJECT)
    parser.add_argument("--process-count", type=int, default=DEFAULT_PROCESS_COUNT)
    parser.add_argument("--staging-dir", default="/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/hld-postprocess")
    parser.add_argument("--finalize-wait-seconds", type=int, default=7200)
    parser.add_argument("--finalize-poll-seconds", type=int, default=30)
    parser.add_argument(
        "--expected-postprocess-version",
        default=POSTPROCESS_VERSION,
    )
    args = parser.parse_args(argv)
    for field in (
        "worker_count",
        "partitions",
        "parts_per_partition",
        "chunks_per_object",
        "process_count",
        "finalize_wait_seconds",
        "finalize_poll_seconds",
    ):
        if int(getattr(args, field)) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
