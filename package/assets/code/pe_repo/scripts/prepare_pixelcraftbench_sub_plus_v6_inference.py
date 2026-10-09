#!/usr/bin/env python3
"""Freeze a PixelCraftBench auto-eval export for image-edit-v6 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlsplit


FORMAT_VERSION = "pixelcraftbench_sub_plus_image_edit_v6_inference_manifest_v1"
DEFAULT_MISSING_SAMPLE_IDS = (
    "red:1524",
    "da:0da973b3-a2a5-4e53-8e1d-d1cff50a80b8",
    "pcplus:style_modification:024",
)
FORBIDDEN_LABEL_FIELDS = frozenset(
    {"target_json", "ground_truth", "annotation", "prediction", "response"}
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_s3_root(value: str) -> str:
    parsed = urlsplit(value)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
        raise ValueError(f"invalid S3 root: {value!r}")
    return f"s3://{parsed.netloc}/{key}"


def _relative_media_path(value: Any, *, directory: str, suffix: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise ValueError(f"{context} is not a safe two-component relative path: {value!r}")
    if path.parts[0] != directory or path.suffix.lower() != suffix:
        raise ValueError(
            f"{context} must match {directory}/*{suffix}, got {value!r}"
        )
    return path.as_posix()


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _load_json_object(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def prepare_manifest(
    *,
    source_manifest: Path,
    source_summary: Path,
    source_root_s3: str,
    source_manifest_s3: str,
    source_manifest_version_id: str,
    expected_source_manifest_sha256: str,
    expected_source_summary_sha256: str,
    output_jsonl: Path,
    summary_json: Path,
    expected_total_rows: int,
    expected_generated_rows: int,
    expected_missing_sample_ids: Sequence[str],
) -> dict[str, Any]:
    source_root = _require_s3_root(source_root_s3)
    manifest_uri = _require_s3_root(source_manifest_s3)
    if not source_manifest_version_id:
        raise ValueError("source_manifest_version_id must be non-empty")
    if expected_total_rows <= 0 or expected_generated_rows <= 0:
        raise ValueError("expected row counts must be positive")
    if expected_generated_rows + len(expected_missing_sample_ids) != expected_total_rows:
        raise ValueError("generated and missing counts do not add up to total rows")

    manifest_sha256 = file_sha256(source_manifest)
    source_summary_sha256 = file_sha256(source_summary)
    if manifest_sha256 != expected_source_manifest_sha256:
        raise ValueError(
            "source manifest SHA-256 mismatch: "
            f"{manifest_sha256} != {expected_source_manifest_sha256}"
        )
    if source_summary_sha256 != expected_source_summary_sha256:
        raise ValueError(
            "source summary SHA-256 mismatch: "
            f"{source_summary_sha256} != {expected_source_summary_sha256}"
        )

    export_summary = _load_json_object(source_summary, "source summary")
    expected_missing = list(expected_missing_sample_ids)
    summary_contract = {
        "total_rows": expected_total_rows,
        "reference_images": expected_total_rows,
        "generated_images": expected_generated_rows,
        "missing_generated": expected_missing,
    }
    for key, expected in summary_contract.items():
        if export_summary.get(key) != expected:
            raise ValueError(
                f"source summary {key} mismatch: {export_summary.get(key)!r} != {expected!r}"
            )

    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    missing_sample_ids: list[str] = []
    with source_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source line {line_number} is not an object")
            forbidden = FORBIDDEN_LABEL_FIELDS.intersection(value)
            if forbidden:
                raise ValueError(
                    f"source line {line_number} contains forbidden label fields: {sorted(forbidden)}"
                )
            source_index = len(seen_sample_ids)
            if value.get("dataset_index") != source_index:
                raise ValueError(
                    f"source line {line_number} dataset_index is not contiguous: "
                    f"{value.get('dataset_index')!r} != {source_index}"
                )
            sample_id = _nonempty_string(value.get("sample_id"), f"source line {line_number} sample_id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            instruction = _nonempty_string(
                value.get("instruction"), f"source line {line_number} instruction"
            )
            reference_path = _relative_media_path(
                value.get("reference_image"),
                directory="reference_images_webp",
                suffix=".webp",
                context=f"source line {line_number} reference_image",
            )
            generated_exists = value.get("generated_exists")
            if type(generated_exists) is not bool:
                raise ValueError(f"source line {line_number} generated_exists must be boolean")
            if not generated_exists:
                if value.get("generated_image") not in (None, ""):
                    raise ValueError(
                        f"source line {line_number} declares a missing generated image but has a path"
                    )
                missing_sample_ids.append(sample_id)
                continue
            generated_path = _relative_media_path(
                value.get("generated_image"),
                directory="generated_images_by_sample",
                suffix=".png",
                context=f"source line {line_number} generated_image",
            )
            if PurePosixPath(reference_path).stem != PurePosixPath(generated_path).stem:
                raise ValueError(
                    f"source line {line_number} reference/target stems do not match: "
                    f"{reference_path!r} vs {generated_path!r}"
                )
            autoeval = {
                key: value[key]
                for key in (
                    "prompt_compliance",
                    "visual_naturalness",
                    "physical_detail",
                    "redbench_avg",
                    "reasoning",
                )
                if key in value
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "instruction": instruction,
                    "reference_image": f"{source_root}/{reference_path}",
                    "target_image": f"{source_root}/{generated_path}",
                    "dataset_source": {
                        "format_version": FORMAT_VERSION,
                        "source_manifest_s3": manifest_uri,
                        "source_manifest_version_id": source_manifest_version_id,
                        "source_manifest_sha256": manifest_sha256,
                        "source_dataset_index": source_index,
                        "task": value.get("task"),
                        "source_benchmark": value.get("source_benchmark"),
                        "autoeval": autoeval,
                    },
                }
            )

    if len(seen_sample_ids) != expected_total_rows:
        raise ValueError(
            f"source row count mismatch: {len(seen_sample_ids)} != {expected_total_rows}"
        )
    if len(rows) != expected_generated_rows:
        raise ValueError(f"selected row count mismatch: {len(rows)} != {expected_generated_rows}")
    if missing_sample_ids != expected_missing:
        raise ValueError(
            f"missing sample IDs changed: {missing_sample_ids!r} != {expected_missing!r}"
        )

    manifest_payload = b"".join(canonical_json_bytes(row) for row in rows)
    output_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    ordered_ids_sha256 = hashlib.sha256(
        canonical_json_bytes([row["sample_id"] for row in rows])
    ).hexdigest()
    atomic_write(output_jsonl, manifest_payload)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "source": {
            "root_s3": source_root,
            "manifest_s3": manifest_uri,
            "manifest_version_id": source_manifest_version_id,
            "manifest_bytes": source_manifest.stat().st_size,
            "manifest_sha256": manifest_sha256,
            "summary_sha256": source_summary_sha256,
            "total_rows": expected_total_rows,
        },
        "selection": {
            "policy": "generated_exists_true_in_source_order_v1",
            "selected_rows": len(rows),
            "missing_generated_rows": len(missing_sample_ids),
            "missing_sample_ids": missing_sample_ids,
            "unique_sample_ids": len({row["sample_id"] for row in rows}),
            "ordered_sample_ids_sha256": ordered_ids_sha256,
        },
        "output": {
            "path": str(output_jsonl),
            "rows": len(rows),
            "bytes": len(manifest_payload),
            "sha256": output_sha256,
        },
    }
    atomic_write(summary_json, json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--source-root-s3", required=True)
    parser.add_argument("--source-manifest-s3", required=True)
    parser.add_argument("--source-manifest-version-id", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-total-rows", type=int, default=1514)
    parser.add_argument("--expected-generated-rows", type=int, default=1511)
    parser.add_argument(
        "--expected-missing-sample-id",
        action="append",
        dest="expected_missing_sample_ids",
    )
    args = parser.parse_args(argv)
    if args.expected_missing_sample_ids is None:
        args.expected_missing_sample_ids = list(DEFAULT_MISSING_SAMPLE_IDS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_manifest(
        source_manifest=args.source_manifest,
        source_summary=args.source_summary,
        source_root_s3=args.source_root_s3,
        source_manifest_s3=args.source_manifest_s3,
        source_manifest_version_id=args.source_manifest_version_id,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_source_summary_sha256=args.expected_source_summary_sha256,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        expected_total_rows=args.expected_total_rows,
        expected_generated_rows=args.expected_generated_rows,
        expected_missing_sample_ids=args.expected_missing_sample_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
