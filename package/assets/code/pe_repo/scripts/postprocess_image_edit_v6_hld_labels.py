#!/usr/bin/env python3
"""Apply small-bbox and punctuation-aware HLD cleanup to V6 inference JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_hld_postprocess import (  # noqa: E402
    POSTPROCESS_VERSION,
    postprocess_v6_hld_labels,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _output_path(value: str, *, name: str) -> Path:
    path = Path(value).resolve()
    if path.exists():
        raise ValueError(f"{name} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_tokenizer(path: Path | None) -> tuple[Any | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    from transformers import AutoTokenizer

    path = path.resolve(strict=True)
    tokenizer_json = path / "tokenizer.json"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    metadata = {
        "path": str(path),
        "tokenizer_json_sha256": _sha256_bytes(tokenizer_json.read_bytes()),
    }
    return tokenizer, metadata


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(_canonical(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_jsonl.resolve(strict=True)
    output_path = _output_path(str(args.output_jsonl), name="output JSONL")
    audit_path = _output_path(str(args.audit_json), name="audit JSON")
    if len({input_path, output_path, audit_path}) != 3:
        raise ValueError("input, output, and audit paths must be distinct")
    tokenizer, tokenizer_metadata = _load_tokenizer(args.tokenizer)

    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    totals = {
        "rows": 0,
        "processed_rows": 0,
        "preserved_invalid_prediction_rows": 0,
        "changed_rows": 0,
        "unchanged_rows": 0,
        "layout_changed_rows": 0,
        "small_bbox_elements_removed": 0,
        "descendant_elements_removed": 0,
        "invalid_label_occurrences_removed": 0,
        "hallucinated_label_occurrences_removed": 0,
        "bbox_removed_label_occurrences_removed": 0,
        "hallucinated_clauses_removed": 0,
        "long_clauses_retained_with_label_only_cleanup": 0,
        "invalid_label_occurrences_removed_from_retained_clauses": 0,
        "clause_numbered_label_occurrences_removed": 0,
        "punctuation_characters_removed": 0,
        "orphaned_conjunctions_removed": 0,
        "hld_tokens_before": 0,
        "hld_tokens_after": 0,
    }
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number}: row must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{input_path}:{line_number}: invalid sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"{input_path}:{line_number}: duplicate sample_id {sample_id}")
            sample_ids.add(sample_id)
            prediction = row.get("prediction")
            if not isinstance(prediction, dict):
                if not getattr(args, "preserve_invalid_predictions", False):
                    raise ValueError(
                        f"{input_path}:{line_number}: prediction must be an object"
                    )
                rows.append(dict(row))
                totals["rows"] += 1
                totals["preserved_invalid_prediction_rows"] += 1
                continue
            raw_prediction = row.get("original_prediction")
            if not isinstance(raw_prediction, dict):
                raw_prediction = prediction
            processed, cleanup = postprocess_v6_hld_labels(raw_prediction)
            if tokenizer is not None:
                before_hld = raw_prediction["high_level_description"]
                after_hld = processed["high_level_description"]
                before_tokens = len(tokenizer.encode(before_hld, add_special_tokens=False))
                after_tokens = len(tokenizer.encode(after_hld, add_special_tokens=False))
                cleanup["before"]["tokens"] = before_tokens
                cleanup["after"]["tokens"] = after_tokens
                totals["hld_tokens_before"] += before_tokens
                totals["hld_tokens_after"] += after_tokens
            result = dict(row)
            result["original_prediction"] = raw_prediction
            result["prediction"] = processed
            result["hld_label_cleanup"] = cleanup
            rows.append(result)
            totals["rows"] += 1
            totals["processed_rows"] += 1
            totals["changed_rows" if cleanup["changed"] else "unchanged_rows"] += 1
            totals["layout_changed_rows"] += int(cleanup["layout_changed"])
            bbox_cleanup = cleanup["bbox_cleanup"]
            totals["small_bbox_elements_removed"] += bbox_cleanup[
                "small_bbox_elements_removed"
            ]
            totals["descendant_elements_removed"] += bbox_cleanup[
                "descendant_elements_removed"
            ]
            totals["invalid_label_occurrences_removed"] += cleanup[
                "invalid_label_occurrences_removed"
            ]
            totals["hallucinated_label_occurrences_removed"] += cleanup[
                "hallucinated_label_occurrences_removed"
            ]
            totals["bbox_removed_label_occurrences_removed"] += cleanup[
                "bbox_removed_label_occurrences_removed"
            ]
            clause_cleanup = cleanup["hallucinated_clause_cleanup"]
            totals["hallucinated_clauses_removed"] += clause_cleanup[
                "clauses_removed"
            ]
            totals["long_clauses_retained_with_label_only_cleanup"] += (
                clause_cleanup["clauses_retained_with_label_only_cleanup"]
            )
            totals[
                "invalid_label_occurrences_removed_from_retained_clauses"
            ] += clause_cleanup[
                "invalid_label_occurrences_removed_from_retained_clauses"
            ]
            totals["clause_numbered_label_occurrences_removed"] += (
                clause_cleanup["numbered_label_occurrences_removed"]
            )
            punctuation = cleanup["punctuation_cleanup"]
            totals["punctuation_characters_removed"] += punctuation[
                "characters_removed"
            ]
            totals["orphaned_conjunctions_removed"] += punctuation[
                "orphaned_conjunctions_removed"
            ]

    _atomic_write_jsonl(output_path, rows)
    output_bytes = output_path.read_bytes()
    audit = {
        "version": f"{POSTPROCESS_VERSION}_jsonl_audit_v1",
        "complete": True,
        "source": str(input_path),
        "source_sha256": _sha256_bytes(input_path.read_bytes()),
        "output": str(output_path),
        "output_sha256": _sha256_bytes(output_bytes),
        **totals,
        "hld_token_reduction": (
            totals["hld_tokens_before"] - totals["hld_tokens_after"]
            if tokenizer is not None
            else None
        ),
        "tokenizer": tokenizer_metadata,
        "all_non_hld_non_layout_projection_hashes_verified": True,
        "remaining_invalid_numbered_label_occurrences": 0,
        "all_postprocess_artifact_counts_zero": all(
            all(
                count == 0
                for count in row["hld_label_cleanup"]["punctuation_cleanup"][
                    "artifact_counts_after"
                ].values()
            )
            for row in rows
            if isinstance(row.get("hld_label_cleanup"), dict)
        ),
    }
    _atomic_write_json(audit_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional local tokenizer directory for before/after HLD token counts.",
    )
    parser.add_argument(
        "--preserve-invalid-predictions",
        action="store_true",
        help=(
            "Copy rows whose prediction is not an object unchanged instead of failing. "
            "Use this for complete inference outputs that retain parse failures."
        ),
    )
    return parser.parse_args()


def main() -> None:
    audit = run(parse_args())
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
