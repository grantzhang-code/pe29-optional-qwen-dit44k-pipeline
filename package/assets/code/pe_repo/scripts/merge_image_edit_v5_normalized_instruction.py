#!/usr/bin/env python3
"""Embed normalized edit instructions into parsed image-edit-v5 predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _nonempty_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def merge_row(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    """Return one enriched row without changing raw response/conversation fields."""

    result = deepcopy(_object(dict(row), context=context))
    source = _object(result.get("dataset_source"), context=f"{context}.dataset_source")
    normalized = _nonempty_text(
        source.get("normalized_edit_instruction"),
        context=f"{context}.dataset_source.normalized_edit_instruction",
    )
    prediction = _object(result.get("prediction"), context=f"{context}.prediction")
    structured = _object(
        prediction.get("structured_description"),
        context=f"{context}.prediction.structured_description",
    )
    _nonempty_text(
        structured.get("high_level_description"),
        context=f"{context}.prediction.structured_description.high_level_description",
    )
    existing = structured.get("normalized_edit_instruction")
    if existing is not None and existing != normalized:
        raise ValueError(
            f"{context}.prediction.structured_description.normalized_edit_instruction "
            "conflicts with dataset_source"
        )

    # Rebuild the object so the new field is immediately after
    # high_level_description while remaining at exactly the same JSON level.
    enriched: dict[str, Any] = {}
    for key, value in structured.items():
        if key == "normalized_edit_instruction":
            continue
        enriched[key] = value
        if key == "high_level_description":
            enriched["normalized_edit_instruction"] = normalized
    prediction["structured_description"] = enriched
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def merge_file(
    input_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    count = 0
    indices: set[int] = set()
    sample_ids: set[str] = set()
    models: Counter[str] = Counter()
    prompt_shas: Counter[str] = Counter()
    json_valid = 0
    schema_valid = 0

    try:
        with input_path.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                context = f"{input_path}:{line_number}"
                row = merge_row(json.loads(raw_line), context=context)
                input_index = row.get("input_index")
                if type(input_index) is not int or input_index < 0:
                    raise ValueError(f"{context}.input_index must be a non-negative integer")
                if input_index in indices:
                    raise ValueError(f"{context} has duplicate input_index {input_index}")
                indices.add(input_index)
                sample_id = _nonempty_text(
                    row.get("sample_id"), context=f"{context}.sample_id"
                )
                if sample_id in sample_ids:
                    raise ValueError(f"{context} has duplicate sample_id {sample_id!r}")
                sample_ids.add(sample_id)
                dataset_source = _object(
                    row.get("dataset_source"), context=f"{context}.dataset_source"
                )
                models[
                    _nonempty_text(
                        dataset_source.get("normalized_edit_instruction_model"),
                        context=(
                            f"{context}.dataset_source."
                            "normalized_edit_instruction_model"
                        ),
                    )
                ] += 1
                prompt_shas[
                    _nonempty_text(
                        dataset_source.get(
                            "normalized_edit_instruction_prompt_sha256"
                        ),
                        context=(
                            f"{context}.dataset_source."
                            "normalized_edit_instruction_prompt_sha256"
                        ),
                    )
                ] += 1
                json_valid += row.get("json_valid") is True
                schema_valid += row.get("schema_valid") is True
                destination.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"record count mismatch: actual={count} expected={expected_count}"
            )
        if indices != set(range(count)):
            raise ValueError("input_index values must cover the contiguous range 0..N-1")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "status": "complete",
        "input": input_path.name,
        "input_sha256": _sha256(input_path),
        "output": output_path.name,
        "rows": count,
        "unique_input_indices": len(indices),
        "unique_sample_ids": len(sample_ids),
        "json_valid_before_enrichment": json_valid,
        "schema_valid_before_enrichment": schema_valid,
        "embedded_normalized_edit_instructions": count,
        "normalized_instruction_models": dict(sorted(models.items())),
        "normalized_instruction_prompt_sha256": dict(sorted(prompt_shas.items())),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "raw_response_preserved": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_count is not None and args.expected_count <= 0:
        raise ValueError("expected-count must be positive")
    summary = merge_file(
        args.input,
        args.output,
        expected_count=args.expected_count,
    )
    if args.summary is not None:
        summary_path = args.summary.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_name(f".{summary_path.name}.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
