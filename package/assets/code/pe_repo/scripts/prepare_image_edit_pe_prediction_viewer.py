#!/usr/bin/env python3
"""Join label-free PE predictions to benchmark media for the offline viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TARGET_SCHEMA = "image_edit_v6_pe_sparse_edited_regions_target_size_v4"
BBOX_COORDINATE_ORDER = "xyxy_1000"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield line_number, value


def required_index(row: Mapping[str, Any], *, context: str) -> int:
    value = row.get("input_index")
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: input_index is invalid")
    return value


def required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {field} is missing")
    return value


def load_ground_truth(path: Path, *, expected_rows: int | None) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line_number, row in load_jsonl(path):
        context = f"{path}:{line_number}"
        index = required_index(row, context=context)
        if index in rows:
            raise ValueError(f"{context}: duplicate input_index {index}")
        required_text(row, "sample_id", context=context)
        required_text(row, "instruction", context=context)
        required_text(row, "source_s3_uri", context=context)
        required_text(row, "target_s3_uri", context=context)
        rows[index] = row
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"ground truth row count changed: {len(rows)} != {expected_rows}")
    return rows


def joined_row(prediction: dict[str, Any], truth: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    for field in ("sample_id", "instruction", "source_s3_uri"):
        if prediction.get(field) != truth.get(field):
            raise ValueError(f"{context}: prediction and benchmark disagree on {field}")
    if "target_s3_uri" in prediction:
        raise ValueError(f"{context}: label-free prediction unexpectedly contains target_s3_uri")
    reference_uri = required_text(prediction, "source_s3_uri", context=context)
    target_uri = required_text(truth, "target_s3_uri", context=context)
    if reference_uri == target_uri:
        raise ValueError(f"{context}: source and target image URIs are identical")
    task_type = required_text(prediction, "task_type", context=context)
    result = dict(prediction)
    result.update(
        {
            "bbox_coordinate_order": BBOX_COORDINATE_ORDER,
            "dataset_source": {
                "dataset_index": prediction["input_index"],
                "split": "train5m-complement-evaluation",
                "task": task_type,
            },
            "reference_image": reference_uri,
            "target_image": target_uri,
            "target_schema": TARGET_SCHEMA,
        }
    )
    return result


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=Path)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-ground-truth", type=int, default=1160)
    parser.add_argument("--expected-predictions", type=int)
    parser.add_argument(
        "--media-only",
        action="store_true",
        help="emit only source/target bindings so image staging can begin before inference ends",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    truth = load_ground_truth(
        args.ground_truth_jsonl.resolve(strict=True),
        expected_rows=args.expected_ground_truth,
    )
    if args.media_only:
        if args.predictions or args.expected_predictions is not None:
            raise ValueError("--media-only cannot be combined with prediction arguments")
        media_rows = [
            {
                "input_index": index,
                "reference_image": required_text(row, "source_s3_uri", context=f"ground truth {index}"),
                "sample_id": required_text(row, "sample_id", context=f"ground truth {index}"),
                "target_image": required_text(row, "target_s3_uri", context=f"ground truth {index}"),
            }
            for index, row in sorted(truth.items())
        ]
        atomic_write_jsonl(args.output_jsonl, media_rows)
        summary = {
            "complete": True,
            "ground_truth_rows": len(truth),
            "media_only": True,
            "output_jsonl": str(args.output_jsonl.resolve()),
            "output_sha256": file_sha256(args.output_jsonl),
            "prediction_rows": 0,
        }
        atomic_write_json(args.summary_json, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.predictions:
        raise ValueError("at least one --predictions file is required without --media-only")
    predictions: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for raw_path in args.predictions:
        path = raw_path.resolve(strict=True)
        before = len(predictions)
        for line_number, row in load_jsonl(path):
            context = f"{path}:{line_number}"
            index = required_index(row, context=context)
            if index in predictions:
                raise ValueError(f"{context}: duplicate prediction input_index {index}")
            if index not in truth:
                raise ValueError(f"{context}: input_index {index} is absent from benchmark")
            predictions[index] = joined_row(row, truth[index], context=context)
        sources.append(
            {
                "path": str(path),
                "rows": len(predictions) - before,
                "sha256": file_sha256(path),
            }
        )
    if args.expected_predictions is not None and len(predictions) != args.expected_predictions:
        raise ValueError(
            f"prediction row count changed: {len(predictions)} != {args.expected_predictions}"
        )
    ordered = [predictions[index] for index in sorted(predictions)]
    atomic_write_jsonl(args.output_jsonl, ordered)
    inference_errors = sum(bool(row.get("inference_error")) for row in ordered)
    summary = {
        "bbox_coordinate_order": BBOX_COORDINATE_ORDER,
        "complete": len(ordered) == len(truth),
        "ground_truth_rows": len(truth),
        "inference_errors": inference_errors,
        "json_valid": sum(row.get("json_valid") is True for row in ordered),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": file_sha256(args.output_jsonl),
        "prediction_rows": len(ordered),
        "schema_valid": sum(row.get("schema_valid") is True for row in ordered),
        "sources": sources,
        "target_schema": TARGET_SCHEMA,
    }
    atomic_write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
