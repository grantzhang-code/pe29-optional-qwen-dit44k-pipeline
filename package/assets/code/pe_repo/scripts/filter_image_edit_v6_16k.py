#!/usr/bin/env python3
"""Filter prepared image-edit-v6 shards to a configured Qwen3.5 token budget."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    ANNOTATION_CONTRACT_SHA256,
    PROMPT_CONTRACT_SHA256,
    build_user_prompt,
)
from scripts.audit_image_edit_v6_sequence_lengths import (  # noqa: E402
    ASSISTANT_PREFIX,
    CHAT_PREFIX,
    CHAT_SUFFIX,
    VISION_MARKER,
    _audit_batch,
)
from scripts.finalize_image_edit_v6_full64_manifest import (  # noqa: E402
    DEFAULT_IMAGE_MAX_TOKEN_NUM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256,
    DEFAULT_MODEL_TYPE,
    DEFAULT_RUNTIME_IMAGE_DIGEST,
    DEFAULT_TEMPLATE_MODE,
    DEFAULT_TRUNCATION_STRATEGY,
    SEQUENCE_AUDIT_FORMAT,
    dataset_content_sha256_from_summaries,
)
from scripts.prepare_image_edit_v6_full_json_partition import (  # noqa: E402
    SHARD_ASSIGNMENT,
    canonical_bytes,
    json_document_bytes,
    write_atomic,
)


PARTITION_RE = re.compile(r"partition-(\d{3})")
PART_RE = re.compile(r"part-(\d{3})-of-(\d{3})\.jsonl")

def length_rejection_reason(max_length: int) -> str:
    return f"qwen35_conservative_max_length_{max_length}_upper_bound"


@dataclass(frozen=True)
class FilterWorkerOptions:
    paths: tuple[Path, ...]
    input_root: Path
    output_root: Path
    model: Path
    max_length: int
    image_max_token_num: int
    batch_size: int


def _render_fast(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("prepared row is missing sample_id")
    allowed = ("s3://", "parquet+s3://")
    if not isinstance(row.get("source_s3_uri"), str) or not row[
        "source_s3_uri"
    ].startswith(allowed):
        raise ValueError(f"{sample_id}: unsupported Source media URI")
    if not isinstance(row.get("target_s3_uri"), str) or not row[
        "target_s3_uri"
    ].startswith(allowed):
        raise ValueError(f"{sample_id}: unsupported Target media URI")
    instruction = row.get("instruction")
    if not isinstance(instruction, str):
        raise ValueError(f"{sample_id}: instruction is missing")
    user = build_user_prompt(instruction)
    if user.count("<image>") != 2:
        raise ValueError(f"{sample_id}: expected exactly two image slots")
    target = json.dumps(
        row.get("final_json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    rendered = (
        f"{CHAT_PREFIX}{user.replace('<image>', VISION_MARKER)}"
        f"{ASSISTANT_PREFIX}{target}{CHAT_SUFFIX}"
    )
    return sample_id, rendered


def _parse_shard_identity(path: Path) -> tuple[int, int, int]:
    partition_match = PARTITION_RE.fullmatch(path.parents[1].name)
    part_match = PART_RE.fullmatch(path.name)
    if partition_match is None or part_match is None:
        raise ValueError(f"unexpected prepared shard path: {path}")
    return (
        int(partition_match.group(1)),
        int(part_match.group(1)),
        int(part_match.group(2)),
    )


def filter_paths(options: FilterWorkerOptions) -> list[dict[str, Any]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(options.model), local_files_only=True, use_fast=True
    )
    results: list[dict[str, Any]] = []
    for input_path in options.paths:
        partition, part_index, part_count = _parse_shard_identity(input_path)
        output_path = options.output_root / input_path.relative_to(options.input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite filtered shard: {output_path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        output_stream = os.fdopen(descriptor, "wb")
        output_hasher = hashlib.sha256()
        output_bytes = 0
        output_rows = 0
        input_rows = 0
        length_excluded_rows = 0
        output_status_counts: Counter[str] = Counter()
        length_excluded_status_counts: Counter[str] = Counter()
        length_excluded_examples: list[str] = []
        maximum_kept_length = 0
        maximum_kept_sample_id = ""
        batch_samples: list[tuple[str, str]] = []
        batch_rows: list[tuple[bytes, str]] = []

        def consume() -> None:
            nonlocal output_bytes, output_rows, length_excluded_rows
            nonlocal maximum_kept_length, maximum_kept_sample_id
            if not batch_samples:
                return
            lengths = _audit_batch(
                tokenizer,
                batch_samples,
                image_max_token_num=options.image_max_token_num,
            )
            for (sample_id, _text_length, upper_bound), (raw_line, status) in zip(
                lengths, batch_rows, strict=True
            ):
                if upper_bound > options.max_length:
                    length_excluded_rows += 1
                    length_excluded_status_counts[status] += 1
                    if len(length_excluded_examples) < 20:
                        length_excluded_examples.append(sample_id)
                    continue
                output_stream.write(raw_line)
                output_hasher.update(raw_line)
                output_bytes += len(raw_line)
                output_rows += 1
                output_status_counts[status] += 1
                if upper_bound > maximum_kept_length:
                    maximum_kept_length = upper_bound
                    maximum_kept_sample_id = sample_id
            batch_samples.clear()
            batch_rows.clear()

        try:
            with input_path.open("rb") as input_stream:
                for line_number, raw_line in enumerate(input_stream, 1):
                    if not raw_line.strip() or not raw_line.endswith(b"\n"):
                        raise ValueError(f"{input_path}:{line_number}: invalid JSONL framing")
                    value = json.loads(raw_line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{input_path}:{line_number}: expected an object")
                    status = value.get("status")
                    if not isinstance(status, str) or not status:
                        raise ValueError(f"{input_path}:{line_number}: status is missing")
                    input_rows += 1
                    batch_samples.append(_render_fast(value))
                    batch_rows.append((raw_line, status))
                    if len(batch_samples) >= options.batch_size:
                        consume()
            consume()
            output_stream.flush()
            os.fsync(output_stream.fileno())
            output_stream.close()
            if input_rows != output_rows + length_excluded_rows:
                raise AssertionError("sequence-length filter row conservation failed")
            if output_rows < 1:
                raise ValueError(
                    f"sequence-length filter produced an empty shard: {input_path}"
                )
            os.replace(temporary_name, output_path)
        except Exception:
            if not output_stream.closed:
                output_stream.close()
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
            raise
        results.append(
            {
                "input_path": str(input_path),
                "local_path": str(output_path),
                "relative_path": (
                    f"partition-{partition:03d}/{output_path.name}"
                ),
                "source_partition": partition,
                "part_index": part_index,
                "part_count": part_count,
                "assignment": SHARD_ASSIGNMENT,
                "input_rows": input_rows,
                "rows": output_rows,
                "bytes": output_bytes,
                "sha256": output_hasher.hexdigest(),
                "status_counts": dict(sorted(output_status_counts.items())),
                "length_excluded_rows": length_excluded_rows,
                "length_excluded_status_counts": dict(
                    sorted(length_excluded_status_counts.items())
                ),
                "length_excluded_examples": length_excluded_examples,
                "max_observed_length": maximum_kept_length,
                "max_observed_sample_id": maximum_kept_sample_id,
                "publication": {
                    "required_content_type": "application/x-ndjson",
                    "required_metadata": {"sha256": output_hasher.hexdigest()},
                    "version_id": "PENDING_S3_UPLOAD",
                },
            }
        )
    return results


def _chunks(paths: Sequence[Path], workers: int) -> list[tuple[Path, ...]]:
    chunks = [[] for _ in range(min(workers, len(paths)))]
    for index, path in enumerate(paths):
        chunks[index % len(chunks)].append(path)
    return [tuple(chunk) for chunk in chunks if chunk]


def _counter(value: Any) -> Counter[str]:
    return Counter({str(key): int(count) for key, count in dict(value or {}).items()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--image-max-token-num", type=int, default=DEFAULT_IMAGE_MAX_TOKEN_NUM
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = args.input_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"filtered output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(
        input_root.glob("partition-*/prepared/partition-*.summary.json")
    )
    if len(summary_paths) != 64:
        raise ValueError(f"expected 64 summaries, got {len(summary_paths)}")
    summaries = [json.loads(path.read_bytes()) for path in summary_paths]
    shard_paths = [
        Path(shard["local_path"])
        for summary in summaries
        for shard in sorted(summary["shards"], key=lambda item: int(item["part_index"]))
    ]
    if len(shard_paths) != 256:
        raise ValueError(f"expected 256 shards, got {len(shard_paths)}")

    records: list[dict[str, Any]] = []
    rejection_reason = length_rejection_reason(args.max_length)
    chunks = _chunks(shard_paths, args.workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                filter_paths,
                FilterWorkerOptions(
                    paths=chunk,
                    input_root=input_root,
                    output_root=output_root,
                    model=args.model.resolve(strict=True),
                    max_length=args.max_length,
                    image_max_token_num=args.image_max_token_num,
                    batch_size=args.batch_size,
                ),
            )
            for chunk in chunks
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            records.extend(result)
            print(
                f"[v6-length-filter] max_length={args.max_length} "
                f"workers={index}/{len(futures)} "
                f"kept={sum(int(item['rows']) for item in result):,} "
                f"excluded={sum(int(item['length_excluded_rows']) for item in result):,}",
                flush=True,
            )

    records.sort(key=lambda item: (int(item["source_partition"]), int(item["part_index"])))
    records_by_partition = {
        partition: [item for item in records if int(item["source_partition"]) == partition]
        for partition in range(64)
    }
    filtered_summaries: list[dict[str, Any]] = []
    for source_summary in summaries:
        partition = int(source_summary["partition_index"])
        shards = records_by_partition[partition]
        if len(shards) != 4:
            raise ValueError(f"partition {partition}: expected four filtered shards")
        old_statuses = _counter(source_summary.get("status_counts"))
        length_statuses = sum(
            (_counter(item.get("length_excluded_status_counts")) for item in shards),
            Counter(),
        )
        new_statuses = old_statuses - length_statuses
        length_excluded = sum(int(item["length_excluded_rows"]) for item in shards)
        old_output_rows = int(source_summary["output_rows"])
        output_rows = sum(int(item["rows"]) for item in shards)
        if old_output_rows != output_rows + length_excluded:
            raise AssertionError(f"partition {partition}: length-filter conservation failed")
        rejected_statuses = _counter(source_summary.get("rejected_status_counts"))
        rejected_statuses.update(length_statuses)
        rejection_counts = _counter(source_summary.get("rejection_counts"))
        rejection_counts[rejection_reason] += length_excluded
        examples = list(source_summary.get("rejection_examples") or [])
        for sample_id in [
            value for item in shards for value in item["length_excluded_examples"]
        ]:
            if len(examples) >= 40:
                break
            examples.append(
                {"sample_id": sample_id, "reason": rejection_reason}
            )
        summary = dict(source_summary)
        summary.update(
            {
                "rejected_rows": int(source_summary.get("rejected_rows") or 0)
                + length_excluded,
                "output_rows": output_rows,
                "rejected_status_counts": dict(sorted(rejected_statuses.items())),
                "status_counts": dict(sorted(new_statuses.items())),
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "rejection_examples": examples,
                "shards": [
                    {
                        key: item[key]
                        for key in (
                            "local_path",
                            "relative_path",
                            "rows",
                            "bytes",
                            "sha256",
                            "source_partition",
                            "part_index",
                            "part_count",
                            "assignment",
                            "publication",
                        )
                    }
                    for item in shards
                ],
                "length_excluded_rows": length_excluded,
                "length_excluded_status_counts": dict(sorted(length_statuses.items())),
                "length_filter": {
                    "policy": rejection_reason,
                    "max_length": args.max_length,
                    "image_slots": 2,
                    "image_max_token_num_per_slot": args.image_max_token_num,
                    "conservative_upper_bound": True,
                    "semantic_values_modified": False,
                },
            }
        )
        if int(summary["input_rows"]) != (
            int(summary["excluded_rows"])
            + int(summary["rejected_rows"])
            + int(summary["output_rows"])
        ):
            raise AssertionError(f"partition {partition}: final row conservation failed")
        summary_path = (
            output_root
            / f"partition-{partition:03d}"
            / "prepared"
            / f"partition-{partition:03d}.summary.json"
        )
        write_atomic(summary_path, json_document_bytes(summary), overwrite=False)
        filtered_summaries.append(summary)

    total_rows = sum(int(item["output_rows"]) for item in filtered_summaries)
    total_length_excluded = sum(
        int(item["length_excluded_rows"]) for item in filtered_summaries
    )
    maximum_record = max(records, key=lambda item: int(item["max_observed_length"]))
    audit = {
        "format_version": SEQUENCE_AUDIT_FORMAT,
        "complete": True,
        "method": "qwen35_exact_text_plus_two_max_image_token_upper_bound_v1",
        "conservative_upper_bound": True,
        "filter_policy": rejection_reason,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "image_max_token_num": args.image_max_token_num,
        "image_slots_per_row": 2,
        "model_type": DEFAULT_MODEL_TYPE,
        "template_mode": DEFAULT_TEMPLATE_MODE,
        "truncation_strategy": DEFAULT_TRUNCATION_STRATEGY,
        "add_non_thinking_prefix": False,
        "runtime_image_digest": DEFAULT_RUNTIME_IMAGE_DIGEST,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "model_artifact_manifest_sha256": DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "dataset_content_sha256": dataset_content_sha256_from_summaries(
            filtered_summaries
        ),
        "checked_rows": total_rows,
        "overlength_rows": 0,
        "assistant_truncation_rows": 0,
        "encode_error_rows": 0,
        "zero_supervised_assistant_rows": 0,
        "json_eos_incomplete_rows": 0,
        "max_observed_length": int(maximum_record["max_observed_length"]),
        "max_observed_sample_id": str(maximum_record["max_observed_sample_id"]),
        "pre_filter_rows": total_rows + total_length_excluded,
        "length_excluded_rows": total_length_excluded,
    }
    write_atomic(
        output_root / "sequence-length-audit.json",
        json_document_bytes(audit),
        overwrite=False,
    )
    report = {
        "complete": True,
        "physical_shards": len(records),
        "total_rows": total_rows,
        "length_excluded_rows": total_length_excluded,
        "max_observed_length": audit["max_observed_length"],
        "sequence_length_audit": str(output_root / "sequence-length-audit.json"),
    }
    write_atomic(
        output_root / "filter-report.json",
        json_document_bytes(report),
        overwrite=False,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
