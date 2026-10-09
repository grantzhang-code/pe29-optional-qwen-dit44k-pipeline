#!/usr/bin/env python3
"""Materialize the frozen CRAFT hard1000 cohort with native image-edit-v5 labels.

The benchmark cohort supplies the immutable sample order, images, dimensions,
and dataset instruction.  Native V5 labels are selected by exact ``sample_id``
from the immutable consolidated annotation shards.  The resulting inference
manifest never exposes labels to the model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_COHORT = (
    "s3://core-cn-oss.canva.com/usr/gavinruan/design-layout-megatron-swift/"
    "eval/craft-hard1000-v1/cohort.jsonl"
)
DEFAULT_COHORT_VERSION_ID = "xBG7GxKfxF.bDEc4g9Lxr9DA2P1hF2rI"
DEFAULT_COHORT_SHA256 = "f60461d89bb28dcbdbcad3c36dfd9956085f28b1efb688914f31d4dcc2fab707"
DEFAULT_ORDERED_IDS_SHA256 = "94a0daaeca8908b30a284df3cea8ba471ea53d9ad04a173c78cbb5c8afdb3bab"
DEFAULT_V5_MANIFEST = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v5/"
    "full16-single-stage-v3-20260807-v1-consolidated-retry1b/dataset_manifest.json"
)
DEFAULT_V5_MANIFEST_SHA256 = "039eac1c69b12bfa5a7b434280c71550307ec4986347290687e31d693d462977"
SAMPLE_ID_RE = re.compile(rb'"sample_id"\s*:\s*"([^"\\]+)"')


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _get_bytes(client: Any, uri: str, *, version_id: str | None = None) -> bytes:
    if uri.startswith("s3://"):
        bucket, key = _parse_s3(uri)
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if version_id:
            kwargs["VersionId"] = version_id
        return client.get_object(**kwargs)["Body"].read()
    return Path(uri).expanduser().resolve(strict=True).read_bytes()


def _jsonl_rows(raw: bytes, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _manifest_outputs(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = value.get("output_files")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("V5 dataset manifest has no output_files")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(outputs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"output_files[{index}] must be an object")
        item = dict(raw)
        if not isinstance(item.get("s3_uri"), str):
            raise ValueError(f"output_files[{index}] has no s3_uri")
        if not isinstance(item.get("sha256"), str):
            raise ValueError(f"output_files[{index}] has no sha256")
        result.append(item)
    return result


def _download_and_select(
    output: Mapping[str, Any],
    *,
    wanted_ids: frozenset[str],
    work_dir: Path,
    keep_shards: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import boto3

    uri = str(output["s3_uri"])
    bucket, key = _parse_s3(uri)
    destination = work_dir / Path(key).name
    client = boto3.client("s3", region_name="us-east-1")
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    version_id = output.get("version_id")
    if isinstance(version_id, str) and version_id:
        kwargs["VersionId"] = version_id
    response = client.get_object(**kwargs)
    digest = hashlib.sha256()
    byte_count = 0
    with destination.open("wb") as handle:
        while chunk := response["Body"].read(16 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            handle.write(chunk)
    actual_sha256 = digest.hexdigest()
    expected_sha256 = str(output["sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{uri}: SHA-256 mismatch actual={actual_sha256} expected={expected_sha256}"
        )
    expected_bytes = output.get("bytes")
    if isinstance(expected_bytes, int) and byte_count != expected_bytes:
        raise ValueError(f"{uri}: byte count {byte_count} != {expected_bytes}")

    selected: dict[str, dict[str, Any]] = {}
    rows = 0
    try:
        with destination.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rows += 1
                match = SAMPLE_ID_RE.search(line)
                if match is None:
                    raise ValueError(f"{uri}:{line_number}: missing sample_id")
                sample_id = match.group(1).decode("utf-8")
                if sample_id not in wanted_ids:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{uri}:{line_number}: row must be an object")
                if sample_id in selected:
                    raise ValueError(f"{uri}: duplicate selected sample_id {sample_id}")
                selected[sample_id] = value
    finally:
        if not keep_shards:
            destination.unlink(missing_ok=True)
    expected_rows = output.get("sample_count")
    if isinstance(expected_rows, int) and rows != expected_rows:
        raise ValueError(f"{uri}: row count {rows} != {expected_rows}")
    return selected, {
        "uri": uri,
        "version_id": version_id,
        "sha256": actual_sha256,
        "bytes": byte_count,
        "rows": rows,
        "selected_rows": len(selected),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            payload = _canonical_json_bytes(row) + b"\n"
            handle.write(payload)
            digest.update(payload)
    temporary.replace(path)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return _sha256(payload)


def _upload_file(client: Any, path: Path, root: str) -> str:
    bucket, prefix = _parse_s3(root.rstrip("/") + "/placeholder")
    prefix = prefix.rsplit("/", 1)[0]
    key = f"{prefix}/{path.name}" if prefix else path.name
    client.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    import boto3

    client = boto3.client("s3", region_name="us-east-1")
    cohort_raw = _get_bytes(client, args.cohort, version_id=args.cohort_version_id)
    cohort_sha256 = _sha256(cohort_raw)
    if cohort_sha256 != args.cohort_sha256:
        raise ValueError(
            f"cohort SHA-256 mismatch actual={cohort_sha256} expected={args.cohort_sha256}"
        )
    cohort_rows = _jsonl_rows(cohort_raw, source=args.cohort)
    if len(cohort_rows) != args.expected_count:
        raise ValueError(f"cohort rows={len(cohort_rows)} expected={args.expected_count}")
    sample_ids = [str(row.get("sample_id", "")).strip() for row in cohort_rows]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("cohort sample IDs must be non-empty and unique")
    ordered_ids_sha256 = _sha256(_canonical_json_bytes(sample_ids))
    if ordered_ids_sha256 != args.ordered_ids_sha256:
        raise ValueError(
            "ordered sample ID SHA-256 mismatch: "
            f"actual={ordered_ids_sha256} expected={args.ordered_ids_sha256}"
        )

    manifest_raw = _get_bytes(client, args.v5_manifest)
    manifest_sha256 = _sha256(manifest_raw)
    if manifest_sha256 != args.v5_manifest_sha256:
        raise ValueError(
            f"V5 manifest SHA-256 mismatch actual={manifest_sha256} "
            f"expected={args.v5_manifest_sha256}"
        )
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, Mapping):
        raise ValueError("V5 dataset manifest must be an object")
    outputs = _manifest_outputs(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_work = Path(tempfile.mkdtemp(prefix="hard1000-v5-shards.", dir=args.work_dir))
    selected: dict[str, dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(outputs))) as executor:
            futures = {
                executor.submit(
                    _download_and_select,
                    output,
                    wanted_ids=frozenset(sample_ids),
                    work_dir=shard_work,
                    keep_shards=args.keep_shards,
                ): index
                for index, output in enumerate(outputs)
            }
            for future in as_completed(futures):
                index = futures[future]
                rows, shard_summary = future.result()
                duplicates = set(selected) & set(rows)
                if duplicates:
                    raise ValueError(f"duplicate selected IDs across shards: {sorted(duplicates)[:10]}")
                selected.update(rows)
                shard_summaries.append({"shard_index": index, **shard_summary})
    finally:
        if not args.keep_shards:
            shutil.rmtree(shard_work, ignore_errors=True)

    missing = [sample_id for sample_id in sample_ids if sample_id not in selected]
    unexpected = sorted(set(selected) - set(sample_ids))
    if unexpected or (missing and not args.allow_missing_labels):
        raise ValueError(
            f"V5 selection mismatch missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_examples={missing[:10]}"
        )

    inference_rows: list[dict[str, Any]] = []
    ground_truth_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for input_index, cohort in enumerate(cohort_rows):
        sample_id = sample_ids[input_index]
        inference = {
            "input_index": input_index,
            "sample_id": sample_id,
            "instruction": cohort["instruction"],
            "reference_image": cohort["image_uri"],
            "target_image": cohort["target_uri"],
            "reference_image_identity": cohort["image_uri"],
            "target_image_identity": cohort["target_uri"],
            "dataset_source": {
                "benchmark_id": "craft-hard1000-v1",
                "source_split": cohort.get("source_split"),
                "difficulty": cohort.get("difficulty"),
            },
        }
        inference_rows.append(inference)
        if sample_id not in selected:
            continue
        annotation_row = selected[sample_id]
        status = str(annotation_row.get("status", "")).strip()
        annotation = annotation_row.get("v5_annotation")
        if status == "failed" or not isinstance(annotation, Mapping):
            raise ValueError(f"{sample_id}: unusable V5 label status={status!r}")
        status_counts[status] = status_counts.get(status, 0) + 1
        ground_truth_rows.append(
            {
                **inference,
                "source_width": cohort.get("source_width"),
                "source_height": cohort.get("source_height"),
                "target_width": cohort.get("target_width"),
                "target_height": cohort.get("target_height"),
                "annotation_status": status,
                "annotation_sha256": annotation_row.get("v5_annotation_sha256"),
                "target_json": annotation,
            }
        )

    input_path = args.output_dir / "input_manifest.jsonl"
    ground_truth_path = args.output_dir / "ground_truth.v5.jsonl"
    input_sha256 = _write_jsonl(input_path, inference_rows)
    ground_truth_sha256 = _write_jsonl(ground_truth_path, ground_truth_rows)
    summary: dict[str, Any] = {
        "format_version": "craft_hard1000_image_edit_v5_eval_materialization_v1",
        "benchmark_id": "craft-hard1000-v1",
        "cohort_rows": len(sample_ids),
        "v5_label_rows": len(ground_truth_rows),
        "v5_label_coverage": len(ground_truth_rows) / len(sample_ids),
        "missing_v5_label_ids": missing,
        "cohort": {
            "uri": args.cohort,
            "version_id": args.cohort_version_id,
            "sha256": cohort_sha256,
            "ordered_ids_sha256": ordered_ids_sha256,
        },
        "v5_annotations": {
            "manifest_uri": args.v5_manifest,
            "manifest_sha256": manifest_sha256,
            "dataset_format": manifest.get("dataset_format"),
            "status_counts": dict(sorted(status_counts.items())),
            "shards": sorted(shard_summaries, key=lambda item: item["shard_index"]),
        },
        "outputs": {
            "input_manifest": {"path": str(input_path), "rows": len(inference_rows), "sha256": input_sha256},
            "ground_truth": {
                "path": str(ground_truth_path),
                "rows": len(ground_truth_rows),
                "sha256": ground_truth_sha256,
            },
        },
    }
    summary_path = args.output_dir / "materialization_summary.json"
    summary_sha256 = _write_json(summary_path, summary)
    summary["outputs"]["summary"] = {"path": str(summary_path), "sha256": summary_sha256}

    if args.upload_s3:
        uploaded = {
            path.name: _upload_file(client, path, args.upload_s3)
            for path in (input_path, ground_truth_path, summary_path)
        }
        summary["uploaded"] = uploaded
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--cohort-version-id", default=DEFAULT_COHORT_VERSION_ID)
    parser.add_argument("--cohort-sha256", default=DEFAULT_COHORT_SHA256)
    parser.add_argument("--ordered-ids-sha256", default=DEFAULT_ORDERED_IDS_SHA256)
    parser.add_argument("--v5-manifest", default=DEFAULT_V5_MANIFEST)
    parser.add_argument("--v5-manifest-sha256", default=DEFAULT_V5_MANIFEST_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--upload-s3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Keep the 1,000-row inference manifest while omitting cohort IDs absent from native V5 labels.",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.expected_count <= 0:
        parser.error("--workers and --expected-count must be positive")
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = prepare(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
