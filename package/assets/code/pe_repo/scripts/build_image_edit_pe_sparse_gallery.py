#!/usr/bin/env python3
"""Build a static gallery for real image-edit-v6 PE sparse-label rows."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageFile


os.environ.setdefault("DLMS_IMAGE_EDIT_V6_PE_MDS_LIBRARY_ONLY", "true")

from dlms.datasets.image_edit_media import (  # noqa: E402
    decode_rgb_image,
    parquet_media_cache_stats,
    preflight_known_s3_access_grants,
)
from dlms.datasets.image_edit_v6_pe_mds import (  # noqa: E402
    LABEL_PROJECTION_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    canonical_sparse_edit_target,
    canonical_v6_target,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
EXPECTED_CLEAN_MODEL = "gemini-3.7-flash"
DEFAULT_TEMPLATE = Path(__file__).with_name("image_edit_pe_sparse_gallery.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A real JSONL range sample. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--expected-total", type=int, default=572)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--thumbnail-max-edge", type=int, default=900)
    parser.add_argument("--webp-quality", type=int, default=78)
    parser.add_argument("--aws-profile", default="prod_core-cn-readwrite")
    parser.add_argument("--aws-region", default="us-east-1")
    return parser.parse_args()


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options) + "\n"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_input_spec(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"--input must be NAME=PATH, got {raw!r}")
    return name.strip(), Path(path).expanduser().resolve()


def load_complete_range_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read complete objects and tolerate only a truncated final range line."""

    raw_lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    trailing_invalid_bytes = 0
    nonempty_indexes = [index for index, line in enumerate(raw_lines) if line.strip()]
    last_nonempty = nonempty_indexes[-1] if nonempty_indexes else -1
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != last_nonempty:
                raise RuntimeError(f"{path}:{index + 1}: malformed non-final JSONL row")
            trailing_invalid_bytes = len(line)
            continue
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{index + 1}: JSONL row is not an object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{path}: no complete JSON objects")
    return rows, trailing_invalid_bytes


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(type(item) not in {int, float} or not math.isfinite(item) for item in value):
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return [x1, y1, x2, y2]


def box_color(index: int) -> str:
    hue = (index * 137.508 + 18) % 360
    return f"hsl({hue:.1f} 92% 66%)"


def edit_instruction(region: dict[str, Any]) -> str:
    edit = region.get("edit")
    if not isinstance(edit, dict) or not isinstance(edit.get("instruction"), str):
        raise RuntimeError(f"region {region.get('label')!r} lacks edit.instruction")
    return edit["instruction"]


def target_and_reference_boxes(
    sparse_layout: list[dict[str, Any]],
    original_by_label: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    target_boxes: list[dict[str, Any]] = []
    reference_boxes: list[dict[str, Any]] = []
    parent_rewrites = 0
    for index, region in enumerate(sparse_layout):
        bbox = valid_bbox(region.get("bbox"))
        if bbox is None:
            raise RuntimeError(f"invalid sparse target bbox: {region.get('bbox')!r}")
        label = str(region["label"])
        original = original_by_label[label]
        rewritten = original["parent"] != region["parent"]
        if rewritten:
            if region["parent"] != "ROOT":
                raise RuntimeError(f"unexpected parent rewrite for {label!r}")
            parent_rewrites += 1
        key = f"E{index + 1:02d}"
        color = box_color(index)
        instruction = edit_instruction(region)
        target_boxes.append(
            {
                "bbox": bbox,
                "color": color,
                "description": str(region.get("desc") or ""),
                "edit_instruction": instruction,
                "key": key,
                "kind": "edited",
                "label": label,
                "original_parent": str(original["parent"]),
                "parent": str(region["parent"]),
                "parent_rewritten": rewritten,
                "type": str(region.get("type") or ""),
            }
        )
        edit = region["edit"]
        references = edit.get("references") if isinstance(edit.get("references"), list) else []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("reference_id") != "frame_0":
                continue
            reference_bbox = valid_bbox(reference.get("reference_bbox"))
            if reference_bbox is None:
                continue
            reference_boxes.append(
                {
                    "bbox": reference_bbox,
                    "color": color,
                    "edit_instruction": instruction,
                    "key": key,
                    "kind": "reference",
                    "label": label,
                }
            )
    return target_boxes, reference_boxes, parent_rewrites


def keep_boxes(original_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for region in original_layout:
        if edit_instruction(region) != "keep":
            continue
        bbox = valid_bbox(region.get("bbox"))
        if bbox is None:
            raise RuntimeError(f"invalid keep target bbox: {region.get('bbox')!r}")
        boxes.append(
            {
                "bbox": bbox,
                "color": "#8b96a8",
                "edit_instruction": "keep (deleted from training label)",
                "key": f"K{len(boxes) + 1:02d}",
                "kind": "keep",
                "label": str(region["label"]),
                "parent": str(region["parent"]),
            }
        )
    return boxes


def remove_boxes(document: dict[str, Any]) -> list[dict[str, Any]]:
    elements = document["elements"]
    removes = elements.get("remove") if isinstance(elements.get("remove"), list) else []
    boxes: list[dict[str, Any]] = []
    for index, removal in enumerate(removes):
        if not isinstance(removal, dict):
            continue
        instruction = str(removal.get("instruction") or "remove")
        references = removal.get("references") if isinstance(removal.get("references"), list) else []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("reference_id") != "frame_0":
                continue
            bbox = valid_bbox(reference.get("reference_bbox"))
            if bbox is None:
                continue
            boxes.append(
                {
                    "bbox": bbox,
                    "color": "#ff5578",
                    "edit_instruction": instruction,
                    "key": f"R{index + 1:02d}",
                    "kind": "remove",
                    "label": "remove",
                }
            )
    return boxes


def validate_source_row(row: dict[str, Any], *, context: str) -> None:
    audit = row.get("call2_audit")
    if not isinstance(audit, dict) or audit.get("valid") is not True or audit.get("errors") != []:
        raise RuntimeError(f"{context}: row did not pass call2 audit")
    if row.get("schema_valid") is False:
        raise RuntimeError(f"{context}: schema_valid is false")
    for field in ("sample_id", "instruction", "source_s3_uri", "target_s3_uri"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise RuntimeError(f"{context}: missing {field}")


def prepare_row(row: dict[str, Any], *, dataset: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    context = f"{dataset}[{index}]"
    validate_source_row(row, context=context)
    original = json.loads(canonical_v6_target(row.get("final_json")))
    sparse = json.loads(canonical_sparse_edit_target(row.get("final_json")))
    original_layout = original["elements"]["layout"]
    sparse_layout = sparse["elements"]["layout"]
    original_by_label = {str(region["label"]): region for region in original_layout}
    if len(original_by_label) != len(original_layout):
        raise RuntimeError(f"{context}: duplicate layout labels")
    if any(edit_instruction(region) == "keep" for region in sparse_layout):
        raise RuntimeError(f"{context}: sparse projection retained a keep region")
    expected_labels = {
        str(region["label"]) for region in original_layout if edit_instruction(region) != "keep"
    }
    if {str(region["label"]) for region in sparse_layout} != expected_labels:
        raise RuntimeError(f"{context}: sparse projection label set mismatch")

    edited, references, parent_rewrites = target_and_reference_boxes(sparse_layout, original_by_label)
    deleted_keep = keep_boxes(original_layout)
    removals = remove_boxes(sparse)
    source_uri = row["source_s3_uri"].strip()
    target_uri = row["target_s3_uri"].strip()
    if source_uri == target_uri:
        raise RuntimeError(f"{context}: Source and Target URI are identical")

    record_name = f"{index:06d}.json"
    entry = {
        "dataset": dataset,
        "index": index,
        "instruction": row["instruction"].strip(),
        "keep_box_count": len(deleted_keep),
        "keep_boxes": deleted_keep,
        "original_layout_count": len(original_layout),
        "parent_rewrite_count": parent_rewrites,
        "record_url": f"records/{record_name}",
        "reference_box_count": len(references),
        "reference_boxes": references,
        "remove_box_count": len(removals),
        "remove_boxes": removals,
        "sample_id": row["sample_id"],
        "source_image": {"url": f"images/source/{index:06d}.webp"},
        "source_line_number": row.get("source_line_number"),
        "source_shard_uri": row.get("source_shard_uri"),
        "sparse_layout_count": len(sparse_layout),
        "status": str(row.get("status") or "unknown"),
        "target_box_count": len(edited),
        "target_boxes": edited,
        "target_image": {"url": f"images/target/{index:06d}.webp"},
        "task": str(row.get("task_type") or "unknown"),
    }
    full_record = {
        "audit": {
            "call2_errors": row["call2_audit"]["errors"],
            "call2_valid": row["call2_audit"]["valid"],
            "clean_model": EXPECTED_CLEAN_MODEL,
            "clean_model_provenance": "verified dataset publication manifest; not a per-row field",
            "label_projection_version": LABEL_PROJECTION_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "prompt_version": PROMPT_VERSION,
        },
        "dataset": dataset,
        "instruction": row["instruction"].strip(),
        "original_final_json": original,
        "sample_id": row["sample_id"],
        "source_s3_uri": source_uri,
        "source_shard_uri": row.get("source_shard_uri"),
        "sparse_training_label": sparse,
        "status": row.get("status"),
        "target_s3_uri": target_uri,
        "task_type": row.get("task_type"),
    }
    job = {
        "index": index,
        "source_uri": source_uri,
        "target_uri": target_uri,
        "record_name": record_name,
        "full_record": full_record,
    }
    return entry, job


def existing_image_metadata(output_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    gallery_path = output_dir / "data" / "gallery.json"
    if not gallery_path.is_file():
        return {}
    try:
        rows = json.loads(gallery_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("index"), int):
            continue
        for kind in ("source", "target"):
            value = row.get(f"{kind}_image")
            if isinstance(value, dict) and all(
                isinstance(value.get(field), int)
                for field in ("bytes", "height", "source_height", "source_width", "width")
            ):
                result[(row["index"], kind)] = value
    return result


def build_thumbnail(
    uri: str,
    destination: Path,
    *,
    max_edge: int,
    quality: int,
    cached_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if cached_metadata is not None and destination.is_file():
        try:
            with Image.open(destination) as image:
                if (image.width, image.height) == (
                    cached_metadata["width"],
                    cached_metadata["height"],
                ):
                    return {**cached_metadata, "reused": True}
        except (OSError, ValueError):
            pass

    image = decode_rgb_image(uri)
    source_width, source_height = image.size
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    image.load()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    image.save(temporary, format="WEBP", quality=quality, method=4)
    temporary.replace(destination)
    return {
        "bytes": destination.stat().st_size,
        "height": image.height,
        "reused": False,
        "source_height": source_height,
        "source_width": source_width,
        "width": image.width,
    }


def main() -> int:
    args = parse_args()
    if args.expected_total < 1 or args.workers < 1 or args.thumbnail_max_edge < 64:
        raise ValueError("expected-total/workers/thumbnail-max-edge values are invalid")
    if not (1 <= args.webp_quality <= 100):
        raise ValueError("webp-quality must be in [1, 100]")
    if not args.template.is_file():
        raise FileNotFoundError(args.template)

    os.environ["AWS_PROFILE"] = args.aws_profile
    os.environ["AWS_REGION"] = args.aws_region
    os.environ.setdefault("AWS_DEFAULT_REGION", args.aws_region)
    os.environ.setdefault("T2J_S3_MAX_POOL_CONNECTIONS", str(max(32, args.workers * 2)))
    os.environ.setdefault("T2J_PARQUET_ROW_GROUP_CACHE_BYTES", str(2 * 1024**3))

    input_specs = [parse_input_spec(value) for value in args.input]
    if len({name for name, _ in input_specs}) != len(input_specs):
        raise ValueError("input dataset names must be unique")

    source_rows: list[tuple[str, dict[str, Any]]] = []
    input_summary: list[dict[str, Any]] = []
    for name, path in input_specs:
        rows, trailing_invalid_bytes = load_complete_range_rows(path)
        source_rows.extend((name, row) for row in rows)
        input_summary.append(
            {
                "complete_rows": len(rows),
                "file": path.name,
                "sha256": sha256_file(path),
                "trailing_invalid_bytes_excluded": trailing_invalid_bytes,
            }
        )
        print(f"[input] {name}: {len(rows)} complete rows from {path}", flush=True)
    if len(source_rows) != args.expected_total:
        raise RuntimeError(f"expected {args.expected_total} complete rows, got {len(source_rows)}")

    output_dir = args.output_dir.resolve()
    record_dir = output_dir / "records"
    source_dir = output_dir / "images" / "source"
    target_dir = output_dir / "images" / "target"
    for directory in (record_dir, source_dir, target_dir, output_dir / "data"):
        directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for index, (dataset, row) in enumerate(source_rows):
        entry, job = prepare_row(row, dataset=dataset, index=index)
        entries.append(entry)
        jobs.append(job)
        atomic_write_text(
            record_dir / job["record_name"],
            canonical_json(job["full_record"], pretty=True),
        )

    if any(entry["original_layout_count"] != entry["sparse_layout_count"] + entry["keep_box_count"] for entry in entries):
        raise RuntimeError("layout conservation check failed")

    has_getty = any(
        "s3://design-generation-core-oss.canva.com/datasets/getty-images/" in job["source_uri"]
        for job in jobs
    )
    grant_targets = preflight_known_s3_access_grants() if has_getty else []
    if grant_targets:
        print(f"[access-grants] acquired READ scope for {grant_targets[0]}", flush=True)

    cached = existing_image_metadata(output_dir)
    futures: dict[Any, tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for job in jobs:
            index = job["index"]
            for kind in ("source", "target"):
                destination = (source_dir if kind == "source" else target_dir) / f"{index:06d}.webp"
                futures[
                    executor.submit(
                        build_thumbnail,
                        job[f"{kind}_uri"],
                        destination,
                        max_edge=args.thumbnail_max_edge,
                        quality=args.webp_quality,
                        cached_metadata=cached.get((index, kind)),
                    )
                ] = (index, kind)
        completed = 0
        for future in as_completed(futures):
            index, kind = futures[future]
            try:
                metadata = future.result()
            except Exception as exc:
                raise RuntimeError(f"thumbnail failed for row={index} kind={kind}") from exc
            entries[index][f"{kind}_image"].update(metadata)
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"[images] {completed}/{len(futures)}", flush=True)

    dataset_counts = Counter(entry["dataset"] for entry in entries)
    task_counts = Counter(entry["task"] for entry in entries)
    status_counts = Counter(entry["status"] for entry in entries)
    unique_ids = {entry["sample_id"] for entry in entries}
    summary = {
        "audit": {
            "all_call2_valid": True,
            "all_schema_canonicalized": True,
            "clean_model": EXPECTED_CLEAN_MODEL,
            "clean_model_provenance": "verified dataset publication manifests; model name is not duplicated per row",
            "image_failures": 0,
        },
        "cases_with_empty_sparse_layout": sum(not entry["sparse_layout_count"] for entry in entries),
        "cases_with_parent_rewrite": sum(bool(entry["parent_rewrite_count"]) for entry in entries),
        "cases_with_remove": sum(bool(entry["remove_box_count"]) for entry in entries),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "inputs": input_summary,
        "keep_regions_removed": sum(entry["keep_box_count"] for entry in entries),
        "label_projection_version": LABEL_PROJECTION_VERSION,
        "original_layout_regions": sum(entry["original_layout_count"] for entry in entries),
        "parent_regions_rewritten_to_root": sum(entry["parent_rewrite_count"] for entry in entries),
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "prompt_version": PROMPT_VERSION,
        "reference_boxes": sum(entry["reference_box_count"] for entry in entries),
        "remove_boxes": sum(entry["remove_box_count"] for entry in entries),
        "rows": len(entries),
        "sparse_edited_regions": sum(entry["sparse_layout_count"] for entry in entries),
        "status": "complete",
        "status_counts": dict(sorted(status_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "thumbnail": {
            "max_edge": args.thumbnail_max_edge,
            "source_bytes": sum(entry["source_image"]["bytes"] for entry in entries),
            "target_bytes": sum(entry["target_image"]["bytes"] for entry in entries),
            "webp_quality": args.webp_quality,
        },
        "unique_sample_ids": len(unique_ids),
    }

    atomic_write_text(output_dir / "data" / "gallery.json", canonical_json(entries))
    atomic_write_text(output_dir / "data" / "bundle_summary.json", canonical_json(summary, pretty=True))
    shutil.copyfile(args.template, output_dir / "index.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[done] {output_dir / 'index.html'}", flush=True)
    print(f"[parquet-cache] {json.dumps(parquet_media_cache_stats(), sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
