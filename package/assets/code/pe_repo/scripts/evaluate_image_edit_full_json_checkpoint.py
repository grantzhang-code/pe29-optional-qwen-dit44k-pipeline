#!/usr/bin/env python3
"""Evaluate one converted full-target-JSON checkpoint end to end on 8 GPUs.

The input is a prepared-dataset paths manifest such as the shared 992-row
``hard1000_manifest.json``.  The runner verifies and flattens every declared
shard, launches the existing independent-GPU inference workers, then calls
``scripts/evaluate_image_edit_full_json.py`` on the merged predictions.  The
complete predictions, detailed matches, summary metrics, and a compact Markdown
report are uploaded beneath one immutable S3 result prefix.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_full_json import (  # noqa: E402
    InlineTargetValidationError,
    validate_inline_target,
)
from scripts import evaluate_image_edit_full_json as evaluator  # noqa: E402
from scripts import run_image_edit_full_json_8gpu as inference_runner  # noqa: E402
from scripts.infer_image_edit_full_json import normalize_input_row  # noqa: E402


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 object URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _validate_s3_root(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 root: {uri}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(client: Any, uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        client.download_file(bucket, key, str(destination))
        return
    source = Path(uri).expanduser().resolve(strict=True)
    shutil.copyfile(source, destination)


def _manifest_shards(document: Mapping[str, Any], *, source: str) -> list[dict[str, Any]]:
    raw_shards = document.get("shards")
    if isinstance(raw_shards, list) and raw_shards:
        shards: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_shards):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
                raise ValueError(f"{source}: shards[{index}] must contain a path")
            shards.append(dict(raw))
        return shards

    raw_paths = document.get("paths") or document.get("jsonl_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"{source}: dataset manifest has no shards/paths")
    if not all(isinstance(path, str) and path for path in raw_paths):
        raise ValueError(f"{source}: every paths entry must be a non-empty string")
    return [{"path": path} for path in raw_paths]


def materialize_dataset(
    *,
    client: Any,
    manifest_uri: str,
    expected_manifest_sha256: str,
    expected_count: int,
    work_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Verify a paths manifest and flatten its prepared JSONL shards."""

    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("dataset manifest SHA-256 must be lowercase 64-hex")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "dataset_manifest.json"
    _download(client, manifest_uri, manifest_path)
    actual_manifest_sha256 = _file_sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "dataset manifest SHA-256 mismatch: "
            f"actual={actual_manifest_sha256} expected={expected_manifest_sha256}"
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{manifest_uri}: dataset manifest must be an object")
    shards = _manifest_shards(document, source=manifest_uri)

    flattened_path = work_dir / "evaluation.jsonl"
    shard_dir = work_dir / "shards"
    sample_ids: set[str] = set()
    total_rows = 0
    shard_summaries: list[dict[str, Any]] = []
    with flattened_path.open("wb") as output:
        for shard_index, shard in enumerate(shards):
            uri = str(shard["path"])
            basename = Path(urlparse(uri).path).name or f"part-{shard_index:05d}.jsonl"
            local_path = shard_dir / f"{shard_index:05d}-{basename}"
            _download(client, uri, local_path)
            actual_sha256 = _file_sha256(local_path)
            declared_sha256 = shard.get("sha256")
            if declared_sha256 is not None and actual_sha256 != declared_sha256:
                raise ValueError(
                    f"{uri}: shard SHA-256 mismatch: actual={actual_sha256} "
                    f"expected={declared_sha256}"
                )

            shard_rows = 0
            with local_path.open("rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"{uri}:{line_number}: invalid JSON") from exc
                    if not isinstance(row, Mapping):
                        raise ValueError(f"{uri}:{line_number}: row must be an object")
                    record = normalize_input_row(row, total_rows)
                    if record.sample_id in sample_ids:
                        raise ValueError(f"duplicate evaluation sample_id: {record.sample_id}")
                    sample_ids.add(record.sample_id)
                    if not record.reference_image.startswith(("s3://", "gs://")) or not (
                        record.target_image.startswith(("s3://", "gs://"))
                    ):
                        raise ValueError(
                            f"{uri}:{line_number}: reference and Target must be S3/GCS images"
                        )
                    try:
                        validate_inline_target(row.get("target_json"))
                    except InlineTargetValidationError as exc:
                        raise ValueError(
                            f"{uri}:{line_number}: invalid full-target-JSON ground truth: {exc}"
                        ) from exc
                    output.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    shard_rows += 1
                    total_rows += 1

            declared_rows = shard.get("rows")
            if declared_rows is not None and shard_rows != declared_rows:
                raise ValueError(
                    f"{uri}: shard row mismatch: actual={shard_rows} expected={declared_rows}"
                )
            shard_summaries.append(
                {"uri": uri, "rows": shard_rows, "sha256": actual_sha256}
            )

    declared_total = document.get("total_rows")
    if declared_total is not None and total_rows != declared_total:
        raise ValueError(
            f"dataset total mismatch: actual={total_rows} manifest={declared_total}"
        )
    if total_rows != expected_count:
        raise ValueError(f"dataset count mismatch: actual={total_rows} expected={expected_count}")
    return flattened_path, {
        "manifest_uri": manifest_uri,
        "manifest_sha256": actual_manifest_sha256,
        "format_version": document.get("format_version"),
        "split": document.get("split"),
        "rows": total_rows,
        "unique_sample_ids": len(sample_ids),
        "physical_shards": len(shards),
        "flattened_sha256": _file_sha256(flattened_path),
        "shards": shard_summaries,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(temporary, path)


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def markdown_report(summary: Mapping[str, Any], *, checkpoint_step: int) -> str:
    regions = summary["regions"]
    hierarchy = summary["hierarchy"]
    operations = summary["operations"]
    validity = summary["validity"]["predictions"]
    remove = summary["remove"]
    lines = [
        f"# Full-target-JSON checkpoint-{checkpoint_step} evaluation",
        "",
        f"- Samples: {summary['sample_alignment']['common_rows']}",
        f"- Strict JSON valid: {_percentage(validity['strict_json_valid_rate'])}",
        f"- Schema valid: {_percentage(validity['schema_valid_rate'])}",
        "",
        "| Region split | Precision | Recall | F1 | IoU@0.5 | Parent match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("edited", "Edited"), ("non_edited", "Non-edited")):
        region = regions[key]
        lines.append(
            f"| {label} | {_percentage(region['precision'])} | "
            f"{_percentage(region['recall'])} | {_percentage(region['f1'])} | "
            f"{_percentage(region['bbox']['iou_at_0_5_rate'])} | "
            f"{_percentage(hierarchy[key]['parent_match_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- Edited operation accuracy: "
            f"{_percentage(operations['accuracy_on_spatially_matched_regions'])}",
            f"- Edited operation macro-F1: {_percentage(operations['macro_f1'])}",
            f"- Region and operation both correct: "
            f"{_percentage(operations['correct_operation_and_region_rate'])}",
            f"- Remove count: predicted={remove['predicted']}, "
            f"ground_truth={remove['ground_truth']}, error={remove['signed_count_error']}",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = inference_runner.s3_client()
    status_path = args.output_dir / "evaluation_status.json"
    status = {
        "status": "materializing_dataset",
        "started_at": utc_now(),
        "checkpoint_step": args.checkpoint_step,
        "output_s3": args.output_s3,
    }
    _write_json(status_path, status)
    inference_runner.upload_file(
        client, status_path, inference_runner.join_s3(args.output_s3, status_path.name)
    )

    dataset_path, dataset = materialize_dataset(
        client=client,
        manifest_uri=args.dataset_manifest_uri,
        expected_manifest_sha256=args.dataset_manifest_sha256,
        expected_count=args.expected_count,
        work_dir=args.output_dir / "dataset",
    )
    remote_flattened = inference_runner.join_s3(
        args.output_s3, "input", "evaluation.prepared.jsonl"
    )
    inference_runner.upload_file(client, dataset_path, remote_flattened)

    runner_args = argparse.Namespace(
        model=args.model,
        conversion_provenance=args.conversion_provenance,
        source_dataset_s3=args.source_dataset_s3,
        input_manifest=remote_flattened,
        input_manifest_sha256=dataset["flattened_sha256"],
        output_dir=args.output_dir,
        output_s3=args.output_s3,
        checkpoint_step=args.checkpoint_step,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        expected_count=args.expected_count,
        world_size=args.world_size,
        gpus=args.gpus,
        batch_size=args.batch_size,
        download_workers=args.download_workers,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        progress_every=args.progress_every,
        enforce_eager=args.enforce_eager,
    )
    inference_summary = inference_runner.run(runner_args)

    predictions_path = args.output_dir / "predictions.jsonl"
    predictions = evaluator.load_predictions(predictions_path)
    ground_truth = evaluator.load_ground_truth(dataset_path)
    if set(predictions) != set(ground_truth):
        raise ValueError(
            "prediction/ground-truth sample IDs differ after inference: "
            f"predictions={len(predictions)} ground_truth={len(ground_truth)}"
        )
    evaluation, details = evaluator.evaluate(
        predictions,
        ground_truth,
        iou_threshold=args.iou_threshold,
    )
    metrics_path = args.output_dir / "evaluation.json"
    details_path = args.output_dir / "evaluation.details.jsonl"
    report_path = args.output_dir / "evaluation_report.md"
    combined_path = args.output_dir / "checkpoint_evaluation_summary.json"
    _write_json(metrics_path, evaluation)
    _write_jsonl(details_path, details)
    report_path.write_text(
        markdown_report(evaluation, checkpoint_step=args.checkpoint_step), encoding="utf-8"
    )
    combined = {
        "status": "complete",
        "started_at": status["started_at"],
        "completed_at": utc_now(),
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "dataset": dataset,
        "inference": inference_summary,
        "evaluation": evaluation,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "metrics": metrics_path.name,
            "details": details_path.name,
            "report": report_path.name,
        },
    }
    _write_json(combined_path, combined)
    _write_json(
        status_path,
        {"status": "complete", "completed_at": combined["completed_at"]},
    )
    for path in (metrics_path, details_path, report_path, combined_path, status_path):
        inference_runner.upload_file(
            client, path, inference_runner.join_s3(args.output_s3, path.name)
        )
    return combined


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conversion-provenance", type=Path, required=True)
    parser.add_argument("--source-dataset-s3", required=True)
    parser.add_argument("--dataset-manifest-uri", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1536 * 1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)
    _validate_s3_root(args.source_dataset_s3)
    _validate_s3_root(args.output_s3)
    for name in (
        "expected_count",
        "checkpoint_step",
        "world_size",
        "batch_size",
        "download_workers",
        "max_model_len",
        "max_new_tokens",
        "max_pixels",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if SHA256_RE.fullmatch(args.dataset_manifest_sha256) is None:
        parser.error("--dataset-manifest-sha256 must be lowercase 64-hex")
    if SHA256_RE.fullmatch(args.checkpoint_manifest_sha256) is None:
        parser.error("--checkpoint-manifest-sha256 must be lowercase 64-hex")
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        status_path = args.output_dir / "evaluation_status.json"
        _write_json(status_path, status)
        try:
            inference_runner.upload_file(
                inference_runner.s3_client(),
                status_path,
                inference_runner.join_s3(args.output_s3, status_path.name),
            )
        except BaseException as upload_error:
            print(f"failed to upload terminal evaluation status: {upload_error}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
