#!/usr/bin/env python3
"""Strip old predictions into a label-free reference-edits TextImage2Json manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DLMS_IMAGE_EDIT_V6_PE_MDS_LIBRARY_ONLY", "true")

from dlms.datasets.image_edit_v6_pe_mds import (
    REFERENCE_EDITS_LABEL_PROJECTION_VERSION,
    REFERENCE_EDITS_PROMPT_CONTRACT_SHA256,
    REFERENCE_EDITS_PROMPT_VERSION,
    reference_edits_prompt_contract_sha256,
)


FORMAT_VERSION = "image-edit-pe-reference-edits-eval-input-v1"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))
            count += 1
    return {
        "path": path.name,
        "rows": count,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def positive_size(value: Any, *, line_number: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line_number}: target_canvas_size is missing")
    width = value.get("width")
    height = value.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError(f"line {line_number}: target_canvas_size is invalid")
    return {"width": width, "height": height}


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            if not isinstance(source, Mapping):
                raise ValueError(f"line {line_number}: source row is not an object")
            sample_id = source.get("sample_id")
            instruction = source.get("instruction")
            source_uri = source.get("source_s3_uri")
            task_type = source.get("task_type")
            selection = source.get("selection")
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"line {line_number}: invalid or duplicate sample_id")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"line {line_number}: instruction is missing")
            if not isinstance(source_uri, str) or not source_uri.startswith("s3://"):
                raise ValueError(f"line {line_number}: source_s3_uri is invalid")
            if not isinstance(task_type, str) or not task_type:
                raise ValueError(f"line {line_number}: task_type is missing")
            if not isinstance(selection, Mapping) or selection.get("training_bitset_member") is not False:
                raise ValueError(f"line {line_number}: non-overlap selection proof is missing")
            sample_ids.add(sample_id)
            task_counts[task_type] += 1
            provenance = dict(selection)
            provenance.update(
                {
                    "derived_input_source_uri": args.source_uri,
                    "derived_input_source_sha256": args.source_sha256,
                    "derived_input_source_version_id": args.source_version_id,
                    "derived_input_source_index": source.get("input_index"),
                }
            )
            rows.append(
                {
                    "format_version": FORMAT_VERSION,
                    "input_index": len(rows),
                    "sample_id": sample_id,
                    "instruction": instruction.strip(),
                    "source_s3_uri": source_uri,
                    "target_canvas_size": positive_size(
                        source.get("target_canvas_size"), line_number=line_number
                    ),
                    "task_type": task_type,
                    "selection": provenance,
                    "prompt_version": REFERENCE_EDITS_PROMPT_VERSION,
                    "prompt_contract_sha256": REFERENCE_EDITS_PROMPT_CONTRACT_SHA256,
                    "label_projection_version": REFERENCE_EDITS_LABEL_PROJECTION_VERSION,
                }
            )
    return rows, task_counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    if file_sha256(args.source_jsonl) != args.source_sha256:
        raise ValueError("source predictions SHA-256 mismatch")
    if reference_edits_prompt_contract_sha256() != REFERENCE_EDITS_PROMPT_CONTRACT_SHA256:
        raise RuntimeError("reference-edits prompt contract drifted")
    rows, task_counts = load_rows(args)
    if len(rows) != args.expected_count:
        raise ValueError(f"source row count mismatch: {len(rows)} != {args.expected_count}")
    args.output_dir.mkdir(parents=True)
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir()
    manifest = write_jsonl(args.output_dir / "manifest.jsonl", rows)
    parts: list[dict[str, Any]] = []
    for part_index in range(args.part_count):
        filename = f"part-{part_index:03d}-of-{args.part_count:03d}.jsonl"
        descriptor = write_jsonl(parts_dir / filename, rows[part_index :: args.part_count])
        descriptor.update({"part_index": part_index, "part_count": args.part_count})
        parts.append(descriptor)
    if sum(part["rows"] for part in parts) != len(rows):
        raise AssertionError("partition row accounting failed")
    summary = {
        "complete": True,
        "format_version": FORMAT_VERSION,
        "source": {
            "uri": args.source_uri,
            "version_id": args.source_version_id,
            "sha256": args.source_sha256,
            "local_path": str(args.source_jsonl),
        },
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "rows": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "task_counts": dict(sorted(task_counts.items())),
        "prompt": {
            "version": REFERENCE_EDITS_PROMPT_VERSION,
            "contract_sha256": REFERENCE_EDITS_PROMPT_CONTRACT_SHA256,
            "label_projection_version": REFERENCE_EDITS_LABEL_PROJECTION_VERSION,
        },
        "label_free_fields": sorted(rows[0]) if rows else [],
        "manifest": manifest,
        "part_count": args.part_count,
        "parts": parts,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-version-id", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--part-count", type=int, default=8)
    args = parser.parse_args(argv)
    if args.expected_count <= 0 or args.part_count <= 0:
        parser.error("counts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
