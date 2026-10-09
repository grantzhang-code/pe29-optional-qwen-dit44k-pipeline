#!/usr/bin/env python3
"""Audit every prepared image-edit-v6 row against the Qwen3.5 sequence budget.

Text is tokenized with the exact Qwen3.5 chat template.  Each of the two image
slots is then charged the configured maximum image-token budget, making the
result a conservative per-row upper bound that does not require downloading
millions of training images.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    ANNOTATION_CONTRACT_SHA256,
    PROMPT_CONTRACT_SHA256,
    SYSTEM_PROMPT,
    build_user_prompt,
    canonical_v6_target,
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
    json_document_bytes,
    write_atomic,
)


VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
CHAT_PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX = "<|im_end|>\n"


@dataclass(frozen=True)
class AuditWorkerOptions:
    paths: tuple[Path, ...]
    model: Path
    max_length: int
    image_max_token_num: int
    batch_size: int


def _render(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("prepared row is missing sample_id")
    source_uri = row.get("source_s3_uri")
    target_uri = row.get("target_s3_uri")
    allowed_media_prefixes = ("s3://", "parquet+s3://")
    if not isinstance(source_uri, str) or not source_uri.startswith(allowed_media_prefixes):
        raise ValueError(f"{sample_id}: source_s3_uri is not supported")
    if not isinstance(target_uri, str) or not target_uri.startswith(allowed_media_prefixes):
        raise ValueError(f"{sample_id}: target_s3_uri is not supported")
    instruction = row.get("instruction")
    if not isinstance(instruction, str):
        raise ValueError(f"{sample_id}: instruction is missing")
    user = build_user_prompt(instruction)
    if user.count("<image>") != 2:
        raise ValueError(f"{sample_id}: user prompt must contain exactly two images")
    user = user.replace("<image>", VISION_MARKER)
    target = canonical_v6_target(row.get("final_json"))
    rendered = f"{CHAT_PREFIX}{user}{ASSISTANT_PREFIX}{target}{CHAT_SUFFIX}"
    return sample_id, rendered


def _audit_batch(
    tokenizer: Any,
    samples: Sequence[tuple[str, str]],
    *,
    image_max_token_num: int,
) -> list[tuple[str, int, int]]:
    encoded = tokenizer(
        [item[1] for item in samples],
        add_special_tokens=False,
        return_length=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # The rendered template contains one image-pad token per slot.  At runtime
    # each is expanded to at most image_max_token_num vision tokens.
    image_expansion = 2 * (image_max_token_num - 1)
    return [
        (sample_id, int(text_length), int(text_length) + image_expansion)
        for (sample_id, _rendered), text_length in zip(
            samples, encoded["length"], strict=True
        )
    ]


def audit_paths(options: AuditWorkerOptions) -> dict[str, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(options.model), local_files_only=True, use_fast=True
    )
    checked_rows = 0
    overlength_rows = 0
    maximum = 0
    maximum_text = 0
    maximum_sample_id = ""
    batch: list[tuple[str, str]] = []

    def consume() -> None:
        nonlocal checked_rows, overlength_rows, maximum, maximum_text, maximum_sample_id
        if not batch:
            return
        for sample_id, text_length, upper_bound in _audit_batch(
            tokenizer,
            batch,
            image_max_token_num=options.image_max_token_num,
        ):
            checked_rows += 1
            if upper_bound > options.max_length:
                overlength_rows += 1
            if upper_bound > maximum:
                maximum = upper_bound
                maximum_text = text_length
                maximum_sample_id = sample_id
        batch.clear()

    for path in options.paths:
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{line_number}: blank line")
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                batch.append(_render(value))
                if len(batch) >= options.batch_size:
                    consume()
        consume()
    return {
        "checked_rows": checked_rows,
        "overlength_rows": overlength_rows,
        "max_observed_length": maximum,
        "max_observed_text_template_length": maximum_text,
        "max_observed_sample_id": maximum_sample_id,
    }


def _chunks(paths: Sequence[Path], workers: int) -> list[tuple[Path, ...]]:
    result = [[] for _ in range(min(workers, len(paths)))]
    for index, path in enumerate(paths):
        result[index % len(result)].append(path)
    return [tuple(chunk) for chunk in result if chunk]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--summary-glob", default="partition-*/prepared/partition-*.summary.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--image-max-token-num", type=int, default=DEFAULT_IMAGE_MAX_TOKEN_NUM
    )
    parser.add_argument("--runtime-image-digest", default=DEFAULT_RUNTIME_IMAGE_DIGEST)
    parser.add_argument(
        "--model-artifact-manifest-sha256",
        default=DEFAULT_MODEL_ARTIFACT_MANIFEST_SHA256,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")
    summary_paths = sorted(args.summary_dir.glob(args.summary_glob))
    if len(summary_paths) != 64:
        raise ValueError(f"expected 64 partition summaries, got {len(summary_paths)}")
    summaries = [json.loads(path.read_bytes()) for path in summary_paths]
    shard_paths = [
        Path(shard["local_path"])
        for summary in summaries
        for shard in sorted(summary["shards"], key=lambda item: int(item["part_index"]))
    ]
    if len(shard_paths) != 256 or not all(path.is_file() for path in shard_paths):
        raise ValueError("expected 256 local prepared shards")
    model = args.model.resolve(strict=True)
    chunks = _chunks(shard_paths, args.workers)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                audit_paths,
                AuditWorkerOptions(
                    paths=chunk,
                    model=model,
                    max_length=args.max_length,
                    image_max_token_num=args.image_max_token_num,
                    batch_size=args.batch_size,
                ),
            )
            for chunk in chunks
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[v6-sequence-audit] workers={index}/{len(futures)} "
                f"rows={result['checked_rows']:,} max={result['max_observed_length']:,} "
                f"overlength={result['overlength_rows']:,}",
                flush=True,
            )

    total_rows = sum(int(item["checked_rows"]) for item in results)
    overlength_rows = sum(int(item["overlength_rows"]) for item in results)
    maximum_result = max(results, key=lambda item: int(item["max_observed_length"]))
    audit = {
        "format_version": SEQUENCE_AUDIT_FORMAT,
        "complete": True,
        "method": "qwen35_exact_text_plus_two_max_image_token_upper_bound_v1",
        "conservative_upper_bound": True,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "image_max_token_num": args.image_max_token_num,
        "image_slots_per_row": 2,
        "model_type": DEFAULT_MODEL_TYPE,
        "template_mode": DEFAULT_TEMPLATE_MODE,
        "truncation_strategy": DEFAULT_TRUNCATION_STRATEGY,
        "add_non_thinking_prefix": False,
        "runtime_image_digest": args.runtime_image_digest,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "model_artifact_manifest_sha256": args.model_artifact_manifest_sha256,
        "annotation_contract_sha256": ANNOTATION_CONTRACT_SHA256,
        "dataset_content_sha256": dataset_content_sha256_from_summaries(summaries),
        "checked_rows": total_rows,
        "overlength_rows": overlength_rows,
        "assistant_truncation_rows": 0 if overlength_rows == 0 else overlength_rows,
        "encode_error_rows": 0,
        "zero_supervised_assistant_rows": 0,
        "json_eos_incomplete_rows": 0,
        "max_observed_length": int(maximum_result["max_observed_length"]),
        "max_observed_text_template_length": int(
            maximum_result["max_observed_text_template_length"]
        ),
        "max_observed_sample_id": str(maximum_result["max_observed_sample_id"]),
    }
    write_atomic(args.output, json_document_bytes(audit), overwrite=args.overwrite)
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0 if overlength_rows == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
