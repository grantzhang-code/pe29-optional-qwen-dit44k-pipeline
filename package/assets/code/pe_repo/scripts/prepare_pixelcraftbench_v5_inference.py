#!/usr/bin/env python3
"""Build a deterministic image-edit-v5 inference manifest from normalized labels.

The normalized instruction is retained only inside ``dataset_source`` so it is
available to the result viewer.  The model-visible instruction remains the raw
edit request, and the model receives only the ordered Source/Target images plus
that raw request through the pinned image-edit-v5 prompt contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FIELDS = (
    "sample_id",
    "dataset_index",
    "task",
    "raw_edit_instruction",
    "normalized_edit_instruction",
    "source_image_s3_uri",
    "target_image_s3_uri",
    "model",
    "prompt_sha256",
)


def _nonempty_text(row: Mapping[str, Any], field: str, *, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return value.strip()


def build_manifest_rows(
    source_rows: Iterable[Mapping[str, Any]], *, expected_count: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for line_number, source in enumerate(source_rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        if missing:
            raise ValueError(f"line {line_number}: missing fields {missing}")
        sample_id = _nonempty_text(source, "sample_id", line_number=line_number)
        if sample_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        dataset_index = source.get("dataset_index")
        if type(dataset_index) is not int or dataset_index < 0:
            raise ValueError(f"line {line_number}: dataset_index must be non-negative int")
        if dataset_index in seen_indices:
            raise ValueError(f"line {line_number}: duplicate dataset_index {dataset_index}")
        seen_indices.add(dataset_index)

        reference_image = _nonempty_text(
            source, "source_image_s3_uri", line_number=line_number
        )
        target_image = _nonempty_text(
            source, "target_image_s3_uri", line_number=line_number
        )
        if not reference_image.startswith("s3://") or not target_image.startswith("s3://"):
            raise ValueError(f"line {line_number}: both images must be explicit s3:// URIs")
        records.append(
            {
                "sample_id": sample_id,
                "instruction": _nonempty_text(
                    source, "raw_edit_instruction", line_number=line_number
                ),
                "reference_image": reference_image,
                "target_image": target_image,
                "reference_image_identity": reference_image,
                "target_image_identity": target_image,
                "dataset_source": {
                    "dataset_index": dataset_index,
                    "task": _nonempty_text(source, "task", line_number=line_number),
                    "split": str(source.get("split") or ""),
                    "normalized_edit_instruction": _nonempty_text(
                        source, "normalized_edit_instruction", line_number=line_number
                    ),
                    "normalized_edit_instruction_model": _nonempty_text(
                        source, "model", line_number=line_number
                    ),
                    "normalized_edit_instruction_prompt_sha256": _nonempty_text(
                        source, "prompt_sha256", line_number=line_number
                    ),
                },
            }
        )
    records.sort(key=lambda row: (row["dataset_source"]["dataset_index"], row["sample_id"]))
    if len(records) != expected_count:
        raise ValueError(
            f"normalized result count mismatch: actual={len(records)} expected={expected_count}"
        )
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
    temporary.replace(path)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=105)
    args = parser.parse_args(argv)
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.normalized_jsonl.resolve(strict=True)
    output = args.output_jsonl.resolve()
    if source == output:
        raise ValueError("output manifest must not overwrite normalized input")
    rows = build_manifest_rows(read_jsonl(source), expected_count=args.expected_count)
    digest = write_jsonl(output, rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(output),
                "rows": len(rows),
                "unique_sample_ids": len({row["sample_id"] for row in rows}),
                "sha256": digest,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
