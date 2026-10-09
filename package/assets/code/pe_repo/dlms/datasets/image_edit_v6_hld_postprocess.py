"""Deterministic small-bbox and HLD-label cleanup for V6 predictions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from bisect import bisect_left
from collections import Counter
from typing import Any


POSTPROCESS_VERSION = "image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7"
BBOX_COORDINATE_SCALE = 1000
BBOX_CANVAS_AREA = BBOX_COORDINATE_SCALE**2
MIN_LAYOUT_BBOX_CANVAS_FRACTION = 0.001
MIN_LAYOUT_BBOX_AREA = int(BBOX_CANVAS_AREA * MIN_LAYOUT_BBOX_CANVAS_FRACTION)
MAX_NON_LABEL_WORDS_FOR_CLAUSE_DELETION = 10
MIN_NON_LABEL_WORDS_FOR_COMPLETE_SENTENCE = 5
NUMBERED_LABEL_RE = re.compile(r"<[^<>\r\n]+ [1-9][0-9]*>")
_NON_LABEL_WORD_RE = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
_CLAUSE_BOUNDARY_PUNCTUATION = ",;:.!?，；：、。！？()（）"
_CLAUSE_BOUNDARY_RE = re.compile(
    f"[{re.escape(_CLAUSE_BOUNDARY_PUNCTUATION)}]"
)
_FULL_STOP_SENTENCE_RE = re.compile(r"[^.!?。！？]*[.。]")

_CONJUNCTION_RE = re.compile(r"\b(?:and|or)\b", re.IGNORECASE)
_PUNCTUATION = frozenset(",;:，；：、()（）.!?。！？")

_ARTIFACT_PATTERNS = {
    "double_space": re.compile(r" {2,}"),
    "space_before_punctuation": re.compile(r"\s+[,.;:!?)]"),
    "repeated_list_punctuation": re.compile(r"[,;:]\s*[,;:]"),
    "list_punctuation_before_terminal": re.compile(r"[,;:]\s*[.!?]"),
    "orphan_conjunction_before_terminal": re.compile(
        r"\b(?:and|or)\s*[.!?]", re.IGNORECASE
    ),
    "empty_parentheses": re.compile(r"\(\s*\)"),
}


class HLDPostprocessError(ValueError):
    """Raised when a prediction cannot satisfy the deterministic V6 contract."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact_counts(text: str) -> dict[str, int]:
    return {
        name: len(pattern.findall(text))
        for name, pattern in _ARTIFACT_PATTERNS.items()
    }


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _cleanup_deletion_artifacts(text: str) -> str:
    # These patterns are limited to punctuation/whitespace artifacts that can
    # be introduced at deletion joins.  No descriptive phrase is regenerated.
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[,;:，；：、]\s*[,;:，；：、](?:\s*[,;:，；：、])*", ", ", text)
    text = re.sub(r"[,;:，；：、]\s*([.!?。！？])", r"\1", text)
    text = re.sub(r"([.!?。！？])(?:\s*[.!?。！？])+", r"\1", text)
    text = re.sub(
        r"(?:[,;:，；：、]\s*)?\b(?:and|or)\s*([.!?。！？])",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*[,;:，；：、]+\s*", "", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r" {2,}", " ", text)


def _layout_element_bbox_area(
    element: dict[str, Any],
    *,
    index: int,
) -> tuple[list[int], int, int, int]:
    bbox = element.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(coordinate) is not int for coordinate in bbox)
    ):
        raise HLDPostprocessError(
            f"prediction.elements.layout[{index}].bbox must contain four integers"
        )
    x0, y0, x1, y1 = bbox
    if not (
        0 <= x0 < x1 <= BBOX_COORDINATE_SCALE
        and 0 <= y0 < y1 <= BBOX_COORDINATE_SCALE
    ):
        raise HLDPostprocessError(
            f"prediction.elements.layout[{index}].bbox must be positive XYXY/1000"
        )
    width = x1 - x0
    height = y1 - y0
    return bbox, width, height, width * height


def _filter_small_layout_elements(
    layout: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove sub-0.1%-canvas elements and descendants of removed parents."""

    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, value in enumerate(layout):
        if not isinstance(value, dict):
            raise HLDPostprocessError(
                f"prediction.elements.layout[{index}] must be an object"
            )
        label = value.get("label")
        parent = value.get("parent")
        if not isinstance(label, str):
            raise HLDPostprocessError(
                f"prediction.elements.layout[{index}].label must be a string"
            )
        if label in labels:
            raise HLDPostprocessError(
                f"prediction.elements.layout[{index}].label must be unique"
            )
        if not isinstance(parent, str):
            raise HLDPostprocessError(
                f"prediction.elements.layout[{index}].parent must be a string"
            )
        if parent != "ROOT" and parent not in labels:
            raise HLDPostprocessError(
                f"prediction.elements.layout[{index}].parent must name an earlier region"
            )
        bbox, width, height, area = _layout_element_bbox_area(value, index=index)
        labels.add(label)
        records.append(
            {
                "index": index,
                "element": value,
                "label": label,
                "parent": parent,
                "bbox": list(bbox),
                "width": width,
                "height": height,
                "normalized_area": area,
            }
        )

    direct_small_labels = {
        record["label"]
        for record in records
        if record["normalized_area"] < MIN_LAYOUT_BBOX_AREA
    }
    removed_labels: set[str] = set()
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        reason: str | None = None
        if record["label"] in direct_small_labels:
            reason = "bbox_area_below_0.1_percent"
        elif record["parent"] in removed_labels:
            # Keeping this row would leave an invalid parent reference. Delete
            # the subtree rather than silently reparenting semantic regions.
            reason = "descendant_of_removed_element"
        if reason is None:
            retained.append(copy.deepcopy(record["element"]))
            continue
        removed_labels.add(record["label"])
        removed.append(
            {
                key: record[key]
                for key in (
                    "index",
                    "label",
                    "parent",
                    "bbox",
                    "width",
                    "height",
                    "normalized_area",
                )
            }
            | {
                "canvas_fraction": record["normalized_area"] / BBOX_CANVAS_AREA,
                "canvas_percent": (
                    100 * record["normalized_area"] / BBOX_CANVAS_AREA
                ),
                "reason": reason,
            }
        )

    direct_count = sum(
        record["reason"] == "bbox_area_below_0.1_percent" for record in removed
    )
    audit = {
        "enabled": True,
        "coordinate_order": "xyxy_1000",
        "comparison": "normalized_area < minimum_normalized_area",
        "minimum_normalized_area": MIN_LAYOUT_BBOX_AREA,
        "minimum_canvas_fraction": MIN_LAYOUT_BBOX_CANVAS_FRACTION,
        "minimum_canvas_percent": 100 * MIN_LAYOUT_BBOX_CANVAS_FRACTION,
        "layout_elements_before": len(layout),
        "layout_elements_after": len(retained),
        "elements_removed": len(removed),
        "small_bbox_elements_removed": direct_count,
        "descendant_elements_removed": len(removed) - direct_count,
        "removed_labels": [record["label"] for record in removed],
        "removed_elements": removed,
    }
    return retained, audit


def _immutable_projection(prediction: dict[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(prediction)
    projection.pop("high_level_description", None)
    elements = projection.get("elements")
    if isinstance(elements, dict):
        elements.pop("layout", None)
    return projection


def _hallucinated_clause_deletion_spans(
    text: str,
    *,
    invalid_matches: list[re.Match[str]],
    layout_labels: set[str],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Delete short hallucinated segments; keep long prose and delete labels."""

    boundary_positions = [
        match.start() for match in _CLAUSE_BOUNDARY_RE.finditer(text)
    ]
    clause_matches: dict[tuple[int, int], list[re.Match[str]]] = {}
    for match in invalid_matches:
        left_index = bisect_left(boundary_positions, match.start())
        left_boundary = boundary_positions[left_index - 1] if left_index else -1
        right_index = bisect_left(boundary_positions, match.end())
        right_boundary = (
            boundary_positions[right_index]
            if right_index < len(boundary_positions)
            else len(text)
        )
        clause_matches.setdefault((left_boundary + 1, right_boundary), []).append(
            match
        )

    deletion_spans: list[tuple[int, int]] = []
    removed_clauses: list[dict[str, Any]] = []
    retained_clauses: list[dict[str, Any]] = []
    for (start, end), triggering_matches in sorted(clause_matches.items()):
        clause_text = text[start:end]
        labels = NUMBERED_LABEL_RE.findall(clause_text)
        non_label_word_count = len(
            _NON_LABEL_WORD_RE.findall(NUMBERED_LABEL_RE.sub(" ", clause_text))
        )
        clause_audit = {
            "start": start,
            "end": end,
            "left_boundary": text[start - 1] if start else None,
            "right_boundary": text[end] if end < len(text) else None,
            "triggering_invalid_labels": [
                match.group(0) for match in triggering_matches
            ],
            "non_label_word_count": non_label_word_count,
            "numbered_label_occurrences": len(labels),
            "valid_label_occurrences": sum(
                label in layout_labels for label in labels
            ),
            "invalid_label_occurrences": sum(
                label not in layout_labels for label in labels
            ),
            "clause_sha256": _sha256_text(clause_text),
        }
        if non_label_word_count <= MAX_NON_LABEL_WORDS_FOR_CLAUSE_DELETION:
            deletion_spans.append((start, end))
            removed_clauses.append(clause_audit | {"action": "delete_clause"})
        else:
            deletion_spans.extend(
                (match.start(), match.end()) for match in triggering_matches
            )
            retained_clauses.append(
                clause_audit | {"action": "delete_invalid_labels_only"}
            )

    spans = _merge_spans(deletion_spans)
    label_only_occurrences = sum(
        clause["invalid_label_occurrences"] for clause in retained_clauses
    )
    return spans, {
        "enabled": True,
        "rule": (
            "delete the nearest-punctuation-bounded segment when its non-label "
            "word count is at most 10; otherwise retain the segment and delete only "
            "its invalid labels"
        ),
        "boundary_punctuation": _CLAUSE_BOUNDARY_PUNCTUATION,
        "maximum_non_label_words_for_clause_deletion": (
            MAX_NON_LABEL_WORDS_FOR_CLAUSE_DELETION
        ),
        "clauses_examined": len(clause_matches),
        "clauses_removed": len(removed_clauses),
        "clauses_retained_with_label_only_cleanup": len(retained_clauses),
        "invalid_label_occurrences_removed_from_retained_clauses": (
            label_only_occurrences
        ),
        "numbered_label_occurrences_removed": sum(
            clause["numbered_label_occurrences"] for clause in removed_clauses
        )
        + label_only_occurrences,
        "removed_clauses": removed_clauses,
        "retained_clauses_with_label_only_cleanup": retained_clauses,
    }


def _short_complete_sentence_deletion_spans(
    text: str,
    *,
    enabled: bool,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Delete short full-stop sentences after an earlier HLD deletion."""

    examined: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    candidate_spans: list[tuple[int, int]] = []
    if enabled:
        for match in _FULL_STOP_SENTENCE_RE.finditer(text):
            sentence_text = match.group(0)
            non_label_word_count = len(
                _NON_LABEL_WORD_RE.findall(
                    NUMBERED_LABEL_RE.sub(" ", sentence_text[:-1])
                )
            )
            record = {
                "start": match.start(),
                "end": match.end(),
                "terminal": sentence_text[-1],
                "non_label_word_count": non_label_word_count,
                "sentence_sha256": _sha256_text(sentence_text),
            }
            examined.append(record)
            if non_label_word_count < MIN_NON_LABEL_WORDS_FOR_COMPLETE_SENTENCE:
                candidate_spans.append((match.start(), match.end()))
                eligible.append(record | {"action": "delete_complete_sentence"})

    deletion_spans = _merge_spans(candidate_spans)
    skipped_to_preserve_nonempty_hld = False
    if deletion_spans and not _cleanup_deletion_artifacts(
        _delete_spans(text, deletion_spans)
    ).strip():
        deletion_spans = []
        skipped_to_preserve_nonempty_hld = True
    removed = [] if skipped_to_preserve_nonempty_hld else eligible

    return deletion_spans, {
        "enabled": True,
        "applied_after_prior_hld_deletion": enabled,
        "rule": (
            "after an earlier HLD deletion, delete each complete sentence ending "
            "in '.' or '。' when it contains fewer than 5 non-label words"
        ),
        "sentence_endings": ".。",
        "minimum_non_label_words_to_keep": (
            MIN_NON_LABEL_WORDS_FOR_COMPLETE_SENTENCE
        ),
        "sentences_examined": len(examined),
        "sentences_eligible": len(eligible),
        "sentences_removed": len(removed),
        "skipped_to_preserve_nonempty_hld": skipped_to_preserve_nonempty_hld,
        "eligible_sentences": eligible,
        "removed_sentences": removed,
    }


def postprocess_v6_hld_labels(
    prediction: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Delete tiny elements, then short hallucinated clauses or long-clause labels.

    The function returns a deep-copied prediction plus a deterministic audit.
    Fields other than ``high_level_description`` and ``elements.layout`` are
    required to remain byte-equivalent under canonical JSON serialization.
    """

    if not isinstance(prediction, dict):
        raise HLDPostprocessError("prediction must be an object")
    high_level_description = prediction.get("high_level_description")
    elements = prediction.get("elements")
    layout = elements.get("layout") if isinstance(elements, dict) else None
    if not isinstance(high_level_description, str):
        raise HLDPostprocessError("prediction.high_level_description must be a string")
    if not isinstance(layout, list):
        raise HLDPostprocessError("prediction.elements.layout must be an array")

    original_layout_labels = {
        element.get("label")
        for element in layout
        if isinstance(element, dict) and isinstance(element.get("label"), str)
    }
    processed_layout, bbox_cleanup = _filter_small_layout_elements(layout)
    layout_labels = {element["label"] for element in processed_layout}
    removed_layout_labels = set(bbox_cleanup["removed_labels"])

    label_matches = list(NUMBERED_LABEL_RE.finditer(high_level_description))
    invalid_matches = [
        match for match in label_matches if match.group(0) not in layout_labels
    ]
    clause_spans, hallucinated_clause_cleanup = _hallucinated_clause_deletion_spans(
        high_level_description,
        invalid_matches=invalid_matches,
        layout_labels=layout_labels,
    )
    spans = clause_spans
    processed_hld = _cleanup_deletion_artifacts(
        _delete_spans(high_level_description, spans)
    )
    short_sentence_spans, short_sentence_cleanup = (
        _short_complete_sentence_deletion_spans(
            processed_hld,
            enabled=bool(spans),
        )
    )
    if short_sentence_spans:
        processed_hld = _cleanup_deletion_artifacts(
            _delete_spans(processed_hld, short_sentence_spans)
        )
    if spans or short_sentence_spans:
        processed_hld = processed_hld.strip()

    remaining_invalid = [
        match.group(0)
        for match in NUMBERED_LABEL_RE.finditer(processed_hld)
        if match.group(0) not in layout_labels
    ]
    if remaining_invalid:
        raise HLDPostprocessError(
            f"invalid numbered labels remain after cleanup: {remaining_invalid[:3]!r}"
        )

    processed = copy.deepcopy(prediction)
    processed["elements"]["layout"] = processed_layout
    processed["high_level_description"] = processed_hld
    original_projection = _immutable_projection(prediction)
    processed_projection = _immutable_projection(processed)
    original_projection_bytes = _canonical_bytes(original_projection)
    if original_projection_bytes != _canonical_bytes(processed_projection):
        raise HLDPostprocessError("a non-HLD/non-layout prediction field changed")

    punctuation_before = Counter(
        character for character in high_level_description if character in _PUNCTUATION
    )
    punctuation_after = Counter(
        character for character in processed_hld if character in _PUNCTUATION
    )
    punctuation_removed = {
        character: punctuation_before[character] - punctuation_after[character]
        for character in sorted(punctuation_before)
        if punctuation_before[character] > punctuation_after[character]
    }
    conjunctions_removed = max(
        0,
        len(_CONJUNCTION_RE.findall(high_level_description))
        - len(_CONJUNCTION_RE.findall(processed_hld)),
    )
    invalid_labels = [match.group(0) for match in invalid_matches]
    bbox_removed_hld_labels = [
        label for label in invalid_labels if label in removed_layout_labels
    ]
    hallucinated_hld_labels = [
        label for label in invalid_labels if label not in original_layout_labels
    ]
    audit: dict[str, Any] = {
        "version": POSTPROCESS_VERSION,
        "rule": (
            "First delete each elements.layout row whose XYXY/1000 bbox area is "
            "below 0.1% of the canvas; delete descendants whose parent was removed "
            "to preserve hierarchy. Then find every numbered HLD label absent from the "
            "remaining layout. If the segment between its nearest punctuation boundaries "
            "contains at most 10 non-label words, delete that entire segment; otherwise "
            "retain the segment and delete only the invalid label. Keep boundary "
            "punctuation and merge punctuation artifacts. After any HLD deletion, "
            "delete complete full-stop sentences containing fewer than 5 non-label "
            "words. Do not regenerate prose."
        ),
        "changed": (
            processed_hld != high_level_description
            or bbox_cleanup["elements_removed"] > 0
        ),
        "hld_changed": processed_hld != high_level_description,
        "layout_changed": bbox_cleanup["elements_removed"] > 0,
        "bbox_cleanup": bbox_cleanup,
        "hallucinated_clause_cleanup": hallucinated_clause_cleanup,
        "short_sentence_cleanup": short_sentence_cleanup,
        "layout_label_count_before": len(original_layout_labels),
        "layout_label_count": len(layout_labels),
        "invalid_label_occurrences_removed": len(invalid_labels),
        "unique_invalid_labels_removed": sorted(set(invalid_labels)),
        "hallucinated_label_occurrences_removed": len(hallucinated_hld_labels),
        "unique_hallucinated_labels_removed": sorted(
            set(hallucinated_hld_labels)
        ),
        "bbox_removed_label_occurrences_removed": len(bbox_removed_hld_labels),
        "unique_bbox_removed_labels_removed": sorted(
            set(bbox_removed_hld_labels)
        ),
        "punctuation_cleanup": {
            "enabled": True,
            "characters_removed": sum(punctuation_removed.values()),
            "removed_by_character": punctuation_removed,
            "orphaned_conjunctions_removed": conjunctions_removed,
            "artifact_counts_after": _artifact_counts(processed_hld),
        },
        "before": {
            "characters": len(high_level_description),
            "words": len(high_level_description.split()),
            "numbered_label_references": len(label_matches),
        },
        "after": {
            "characters": len(processed_hld),
            "words": len(processed_hld.split()),
            "numbered_label_references": len(
                NUMBERED_LABEL_RE.findall(processed_hld)
            ),
        },
        "original_hld_sha256": _sha256_text(high_level_description),
        "processed_hld_sha256": _sha256_text(processed_hld),
        "non_hld_non_layout_projection_sha256": hashlib.sha256(
            original_projection_bytes
        ).hexdigest(),
        "remaining_invalid_numbered_label_occurrences": 0,
    }
    return processed, audit
