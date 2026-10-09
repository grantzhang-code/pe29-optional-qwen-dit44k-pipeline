#!/usr/bin/env python3
"""Join compact Reve2 response JSON to image-edit-v5 prediction records."""

from __future__ import annotations

import argparse
from copy import deepcopy
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class _CasesJSONParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.capture = False
        self.parts: list[str] = []
        self.matches = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "script" and dict(attrs).get("id") == "cases-json":
            self.capture = True
            self.matches += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.capture:
            self.capture = False

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.parts.append(data)


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _nonempty_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            rows.append(
                _object(json.loads(raw_line), context=f"{path}:{line_number}")
            )
    return rows


def extract_reve2_cases(path: Path) -> list[dict[str, Any]]:
    parser = _CasesJSONParser()
    parser.feed(path.resolve(strict=True).read_text(encoding="utf-8"))
    if parser.matches != 1:
        raise ValueError(
            f"{path} must contain exactly one script element with id=cases-json"
        )
    value = json.loads("".join(parser.parts))
    if not isinstance(value, list):
        raise ValueError(f"{path} cases-json must contain an array")
    return [
        _object(item, context=f"{path}:cases-json[{index}]")
        for index, item in enumerate(value)
    ]


def _unique_by_sample_id(
    rows: list[dict[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = _nonempty_text(
            row.get("sample_id"), context=f"{context}[{index}].sample_id"
        )
        if sample_id in result:
            raise ValueError(f"{context} has duplicate sample_id {sample_id!r}")
        result[sample_id] = row
    return result


def _validate_layout(layout: dict[str, Any], *, context: str) -> int:
    _nonempty_text(layout.get("prompt"), context=f"{context}.prompt")
    _nonempty_text(
        layout.get("normalized_edit_instruction"),
        context=f"{context}.normalized_edit_instruction",
    )
    for dimension in ("width", "height"):
        if type(layout.get(dimension)) is not int or layout[dimension] <= 0:
            raise ValueError(f"{context}.{dimension} must be a positive integer")
    regions = layout.get("regions")
    if not isinstance(regions, list):
        raise ValueError(f"{context}.regions must be an array")
    for index, raw_region in enumerate(regions):
        region_context = f"{context}.regions[{index}]"
        region = _object(raw_region, context=region_context)
        _nonempty_text(region.get("label"), context=f"{region_context}.label")
        _nonempty_text(region.get("prompt"), context=f"{region_context}.prompt")
        bbox = _object(region.get("bbox"), context=f"{region_context}.bbox")
        if set(bbox) != {"x0", "x1", "y0", "y1"}:
            raise ValueError(f"{region_context}.bbox has unexpected keys")
        values = [bbox[key] for key in ("x0", "x1", "y0", "y1")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"{region_context}.bbox values must be numbers")
        if not (0 <= bbox["x0"] < bbox["x1"] <= 1):
            raise ValueError(f"{region_context}.bbox has invalid x bounds")
        if not (0 <= bbox["y0"] < bbox["y1"] <= 1):
            raise ValueError(f"{region_context}.bbox has invalid y bounds")
    return len(regions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def merge_files(
    image_edit_jsonl: Path,
    reve2_viewer_html: Path,
    reve2_results_jsonl: Path,
    output_path: Path,
    *,
    reve2_s3_root: str,
    expected_count: int | None = None,
) -> dict[str, Any]:
    image_edit_jsonl = image_edit_jsonl.resolve(strict=True)
    reve2_viewer_html = reve2_viewer_html.resolve(strict=True)
    reve2_results_jsonl = reve2_results_jsonl.resolve(strict=True)
    output_path = output_path.resolve()
    if not reve2_s3_root.startswith("s3://"):
        raise ValueError("reve2_s3_root must be an s3:// URI")
    reve2_s3_root = reve2_s3_root.rstrip("/")

    image_edit_rows = _load_jsonl(image_edit_jsonl)
    cases = extract_reve2_cases(reve2_viewer_html)
    results = _load_jsonl(reve2_results_jsonl)
    case_by_id = _unique_by_sample_id(cases, context="reve2_cases")
    result_by_id = _unique_by_sample_id(results, context="reve2_results")
    image_edit_by_id = _unique_by_sample_id(
        image_edit_rows, context="image_edit_rows"
    )
    expected_ids = set(image_edit_by_id)
    if set(case_by_id) != expected_ids or set(result_by_id) != expected_ids:
        raise ValueError("Reve2 and image-edit-v5 sample_id sets do not match")
    if expected_count is not None and len(image_edit_rows) != expected_count:
        raise ValueError(
            "record count mismatch: "
            f"actual={len(image_edit_rows)} expected={expected_count}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    total_regions = 0
    reve2_normalized_matches_gemini = 0
    versions: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for row_number, source_row in enumerate(image_edit_rows, start=1):
                row = deepcopy(source_row)
                sample_id = row["sample_id"]
                case = case_by_id[sample_id]
                result = result_by_id[sample_id]
                if result.get("status") != "success":
                    raise ValueError(f"Reve2 result for {sample_id} is not successful")
                if case.get("instruction") != row.get("instruction"):
                    raise ValueError(f"instruction mismatch for {sample_id}")
                dataset_source = _object(
                    row.get("dataset_source"),
                    context=f"image_edit_rows[{row_number}].dataset_source",
                )
                if not (
                    case.get("dataset_index")
                    == result.get("dataset_index")
                    == dataset_source.get("dataset_index")
                ):
                    raise ValueError(f"dataset_index mismatch for {sample_id}")
                if case.get("request_id") != result.get("request_id"):
                    raise ValueError(f"request_id mismatch for {sample_id}")
                input_path = _nonempty_text(
                    case.get("input_path"), context=f"Reve2 case {sample_id}.input_path"
                )
                output_image_path = _nonempty_text(
                    case.get("output_path"),
                    context=f"Reve2 case {sample_id}.output_path",
                )
                if not str(row.get("reference_image") or "").endswith(input_path):
                    raise ValueError(f"reference image mismatch for {sample_id}")
                if not str(row.get("target_image") or "").endswith(output_image_path):
                    raise ValueError(f"target image mismatch for {sample_id}")
                layout = _object(
                    case.get("layout"), context=f"Reve2 case {sample_id}.layout"
                )
                region_count = _validate_layout(
                    layout, context=f"Reve2 case {sample_id}.layout"
                )
                if case.get("region_count") != region_count:
                    raise ValueError(f"region_count mismatch for {sample_id}")
                total_regions += region_count
                version = _nonempty_text(
                    case.get("version"), context=f"Reve2 case {sample_id}.version"
                )
                versions.add(version)
                v5_normalized = (
                    row["prediction"]["structured_description"].get(
                        "normalized_edit_instruction"
                    )
                )
                reve2_normalized_matches_gemini += (
                    layout["normalized_edit_instruction"] == v5_normalized
                )
                response_name = f"{Path(output_image_path).stem}.json"
                row["reve2"] = {
                    "response_without_image": {
                        "content_violation": case.get("content_violation"),
                        "layout": layout,
                        "request_id": case.get("request_id"),
                        "version": version,
                        "credits_used": case.get("credits_used"),
                        "credits_remaining": case.get("credits_remaining"),
                    },
                    "omitted_response_fields": ["image"],
                    "image_uri": f"{reve2_s3_root}/{output_image_path}",
                    "response_uri": f"{reve2_s3_root}/responses/{response_name}",
                    "task": case.get("task"),
                    "dataset_index": case.get("dataset_index"),
                    "source_benchmark": case.get("source_benchmark"),
                    "elapsed_seconds": case.get("elapsed_seconds"),
                    "region_count": region_count,
                }
                destination.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "status": "complete",
        "image_edit_input": image_edit_jsonl.name,
        "image_edit_input_sha256": _sha256(image_edit_jsonl),
        "reve2_viewer_input": reve2_viewer_html.name,
        "reve2_viewer_input_sha256": _sha256(reve2_viewer_html),
        "reve2_results_input": reve2_results_jsonl.name,
        "reve2_results_input_sha256": _sha256(reve2_results_jsonl),
        "output": output_path.name,
        "output_sha256": _sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "rows": len(image_edit_rows),
        "reve2_regions": total_regions,
        "reve2_versions": sorted(versions),
        "reve2_normalized_matches_gemini": reve2_normalized_matches_gemini,
        "reve2_image_base64_omitted": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-edit-jsonl", required=True, type=Path)
    parser.add_argument("--reve2-viewer-html", required=True, type=Path)
    parser.add_argument("--reve2-results-jsonl", required=True, type=Path)
    parser.add_argument("--reve2-s3-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_count is not None and args.expected_count <= 0:
        raise ValueError("expected-count must be positive")
    summary = merge_files(
        args.image_edit_jsonl,
        args.reve2_viewer_html,
        args.reve2_results_jsonl,
        args.output,
        reve2_s3_root=args.reve2_s3_root,
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
