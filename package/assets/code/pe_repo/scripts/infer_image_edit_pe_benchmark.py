#!/usr/bin/env python3
"""Run target-size-aware image-edit PE inference on a frozen source-only manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# This inference path calls the ETag-pinned Parquet byte reader directly. The
# training module self-applies an ms-swift media-loader patch by default, but
# the lean vLLM runtime intentionally does not install ms-swift.
os.environ.setdefault("DLMS_S3_IMAGE_LOAD_NO_APPLY", "1")

from dlms.datasets.image_edit_v6_pe_mds import (  # noqa: E402
    LABEL_PROJECTION_VERSION,
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    canonical_sparse_edit_target,
    canonical_v6_target,
    prompt_contract_sha256,
)
from scripts import infer_image_edit_full_json as media_runtime  # noqa: E402
from scripts.mtp_eval.hf_artifact import validate_local_trained_artifact  # noqa: E402


BENCHMARK_FORMAT_VERSION = "image-edit-pe-task-type-eval-benchmark-v1"
INFERENCE_FORMAT_VERSION = "image-edit-pe-target-size-v4-inference-v1"
EXPECTED_LOGICAL_MTP_LAYERS = 3
EXPECTED_MTP_SHARED_WEIGHTS = True
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
_VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
_ASSISTANT_HEADER = "<|im_start|>assistant\n"
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Record:
    input_index: int
    sample_id: str
    instruction: str
    source_s3_uri: str
    target_size: tuple[int, int]
    task_type: str
    selection: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _positive_int(value: Any, *, field: str, line_number: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"line {line_number}: {field} must be a positive integer")
    return value


def normalize_record(row: Mapping[str, Any], *, line_number: int) -> Record:
    forbidden = {
        "assistant_target_json",
        "final_json",
        "ground_truth",
        "label",
        "target_s3_uri",
    } & set(row)
    if forbidden:
        raise ValueError(f"line {line_number}: inference input leaks labels: {sorted(forbidden)}")
    if row.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"line {line_number}: prompt version mismatch")
    if row.get("prompt_contract_sha256") != PROMPT_CONTRACT_SHA256:
        raise ValueError(f"line {line_number}: prompt contract mismatch")
    if row.get("label_projection_version") != LABEL_PROJECTION_VERSION:
        raise ValueError(f"line {line_number}: label projection mismatch")
    input_index = row.get("input_index")
    if type(input_index) is not int or input_index < 0:
        raise ValueError(f"line {line_number}: input_index must be non-negative")
    sample_id = row.get("sample_id")
    instruction = row.get("instruction")
    source = row.get("source_s3_uri")
    task_type = row.get("task_type")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_number}: sample_id is missing")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"line {line_number}: instruction is missing")
    if not isinstance(source, str) or not source.startswith(("s3://", "parquet+s3://")):
        raise ValueError(f"line {line_number}: source_s3_uri is invalid")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"line {line_number}: task_type is missing")
    size = row.get("target_canvas_size")
    if not isinstance(size, Mapping):
        raise ValueError(f"line {line_number}: target_canvas_size is missing")
    width = _positive_int(size.get("width"), field="target width", line_number=line_number)
    height = _positive_int(size.get("height"), field="target height", line_number=line_number)
    selection = row.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"line {line_number}: selection provenance is missing")
    if selection.get("training_bitset_member") is not False:
        raise ValueError(f"line {line_number}: benchmark row intersects the training sample")
    return Record(
        input_index=input_index,
        sample_id=sample_id,
        instruction=instruction.strip(),
        source_s3_uri=source,
        target_size=(width, height),
        task_type=task_type,
        selection=dict(selection),
    )


def load_manifest(path: Path, *, expected_sha256: str, expected_count: int) -> list[Record]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected input SHA-256 is invalid")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"input SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number}: row is not an object")
            records.append(normalize_record(value, line_number=line_number))
    if len(records) != expected_count:
        raise ValueError(f"input row count mismatch: {len(records)} != {expected_count}")
    if len({record.input_index for record in records}) != len(records):
        raise ValueError("input indices are not unique")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("sample IDs are not unique")
    return records


def expected_direct_prompt(instruction: str, target_size: tuple[int, int]) -> str:
    user_prompt = build_user_prompt(instruction, target_size)
    if user_prompt.count("<image>") != 1:
        raise RuntimeError("training PE prompt must contain exactly one image marker")
    expanded = user_prompt.replace("<image>", _VISION_MARKER, 1)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{expanded}<|im_end|>\n"
        f"{_ASSISTANT_HEADER}"
    )


def render_training_prompt(
    tokenizer: Any,
    instruction: str,
    target_size: tuple[int, int],
) -> tuple[str, str]:
    user_prompt = build_user_prompt(instruction, target_size)
    before, marker, after = user_prompt.partition("<image>")
    if not marker or "<image>" in after:
        raise RuntimeError("training PE prompt image-marker cardinality drifted")
    content: list[dict[str, str]] = []
    if before:
        content.append({"type": "text", "text": before})
    content.append({"type": "image", "image": "<image>"})
    if after:
        content.append({"type": "text", "text": after})
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return a string")
    assistant_offset = rendered.rfind(_ASSISTANT_HEADER)
    if assistant_offset < 0:
        raise RuntimeError("Qwen chat template emitted no assistant header")
    prefix_end = assistant_offset + len(_ASSISTANT_HEADER)
    assistant_prefix = rendered[prefix_end:]
    if assistant_prefix and not re.fullmatch(r"<think>\s*</think>\s*", assistant_prefix):
        raise RuntimeError(f"unexpected generation prefix: {assistant_prefix!r}")
    rendered = rendered[:prefix_end]
    expected = expected_direct_prompt(instruction, target_size)
    if rendered != expected:
        raise RuntimeError("checkpoint tokenizer does not reproduce the PE training prompt")
    if rendered.count("<|image_pad|>") != 1:
        raise RuntimeError("rendered PE prompt does not contain exactly one image")
    return rendered, user_prompt


def parse_response(response: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    text = _THINK_RE.sub("", response, count=1)
    text = _FENCE_START_RE.sub("", text, count=1)
    text = _FENCE_END_RE.sub("", text, count=1).strip()

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant: {token}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    try:
        elements = value.get("elements")
        layout = elements.get("layout") if isinstance(elements, Mapping) else None
        if layout == []:
            # The inherited full-V6 validator deliberately requires a non-empty
            # dense layout. PE labels may legitimately be empty after removing
            # every keep region, so validate all other fields through one
            # synthetic region and then discard that validation-only sentinel.
            candidate_value = json.loads(canonical_json(value))
            candidate_value["elements"]["layout"] = [
                {
                    "type": "coarse",
                    "label": "<validation placeholder 1>",
                    "parent": "ROOT",
                    "bbox": [0, 0, 1, 1],
                    "desc": "Validation-only placeholder region.",
                    "color_palette": ["#000000"],
                    "edit": {
                        "instruction": "Validate the sparse empty layout.",
                        "references": [],
                    },
                }
            ]
            projected = json.loads(canonical_v6_target(candidate_value))
            projected["elements"]["layout"] = []
        else:
            projected = json.loads(canonical_sparse_edit_target(value))
    except (ValueError, KeyError, TypeError) as exc:
        return value, None, f"{type(exc).__name__}: {exc}"
    return projected, None, None


def batches(values: Sequence[Record], size: int) -> Iterable[Sequence[Record]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def load_images(records: Sequence[Record], *, workers: int) -> list[Any | BaseException]:
    def load(record: Record) -> Any:
        return media_runtime._load_media_uri_with_retry(  # noqa: SLF001
            record.source_s3_uri,
            download_workers=workers,
            gcs_client=None,
        )

    results: list[Any | BaseException] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(records))) as executor:
        futures = [executor.submit(load, record) for record in records]
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(exc)
    return results


def runtime_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "transformers", "vllm", "flashinfer-python", "triton"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    import torch

    result["cuda"] = str(torch.version.cuda)
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result


def validate_artifact(args: argparse.Namespace) -> dict[str, Any]:
    evidence = validate_local_trained_artifact(
        args.model,
        expected_manifest_sha256=args.artifact_manifest_sha256,
        expected_mtp_num_layers=EXPECTED_LOGICAL_MTP_LAYERS,
        expected_mtp_shared_weights=EXPECTED_MTP_SHARED_WEIGHTS,
    )
    manifest = json.loads((args.model / "mcore_to_hf_manifest.json").read_text(encoding="utf-8"))
    source = manifest["source_mcore_checkpoint"]
    if source["step"] != args.checkpoint_step:
        raise ValueError(f"artifact source step mismatch: {source['step']} != {args.checkpoint_step}")
    if source["manifest_sha256"] != args.checkpoint_manifest_sha256:
        raise ValueError("artifact source checkpoint manifest mismatch")
    return evidence


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"max_pixels": args.max_pixels},
        "disable_log_stats": False,
        "seed": args.seed,
        "attention_backend": args.attention_backend,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    return kwargs


def sampling_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "strategy": "deterministic_greedy_v1",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "num_speculative_tokens": args.num_speculative_tokens,
        "thinking": False,
    }


def initialize_engine(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    if prompt_contract_sha256() != PROMPT_CONTRACT_SHA256:
        raise RuntimeError("checked-in PE prompt contract hash drifted")
    from vllm import LLM, SamplingParams

    kwargs = build_llm_kwargs(args)
    # vLLM normalizes speculative_config in place and may attach internal
    # ModelConfig objects. Preserve an immutable JSON-only launch contract for
    # the persisted node summary before handing the kwargs to LLM.
    llm_kwargs_contract = json.loads(json.dumps(kwargs))
    llm = LLM(**kwargs)
    config = llm.llm_engine.vllm_config.speculative_config
    if (
        config is None
        or getattr(config, "method", None) != "mtp"
        or getattr(config, "num_speculative_tokens", None) != args.num_speculative_tokens
    ):
        raise RuntimeError("vLLM did not resolve the requested trained-MTP configuration")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    return llm, tokenizer, sampling, {
        "llm_kwargs": llm_kwargs_contract,
        "runtime": runtime_versions(),
    }


def candidate(output: Any) -> Any:
    values = getattr(output, "outputs", None)
    if not values:
        raise RuntimeError("vLLM returned no generation candidate")
    return values[0]


def error_row(record: Record, error: BaseException) -> dict[str, Any]:
    return {
        "format_version": INFERENCE_FORMAT_VERSION,
        "input_index": record.input_index,
        "sample_id": record.sample_id,
        "task_type": record.task_type,
        "source_s3_uri": record.source_s3_uri,
        "instruction": record.instruction,
        "target_canvas_size": {"width": record.target_size[0], "height": record.target_size[1]},
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "label_projection_version": LABEL_PROJECTION_VERSION,
        "response": None,
        "prediction": None,
        "json_valid": False,
        "schema_valid": False,
        "inference_error": f"{type(error).__name__}: {error}",
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def result_row(
    record: Record,
    *,
    user_prompt: str,
    output: Any,
    sampling_contract: Mapping[str, Any],
) -> dict[str, Any]:
    item = candidate(output)
    response = str(item.text)
    prediction, json_error, schema_error = parse_response(response)
    return {
        "format_version": INFERENCE_FORMAT_VERSION,
        "input_index": record.input_index,
        "sample_id": record.sample_id,
        "task_type": record.task_type,
        "source_s3_uri": record.source_s3_uri,
        "instruction": record.instruction,
        "target_canvas_size": {"width": record.target_size[0], "height": record.target_size[1]},
        "selection": dict(record.selection),
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "label_projection_version": LABEL_PROJECTION_VERSION,
        "sampling": dict(sampling_contract),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ],
        "images": [record.source_s3_uri],
        "response": response,
        "prediction": prediction,
        "json_valid": json_error is None,
        "schema_valid": json_error is None and schema_error is None,
        "json_error": json_error,
        "schema_error": schema_error,
        "prompt_tokens": len(getattr(output, "prompt_token_ids", None) or []),
        "output_tokens": len(getattr(item, "token_ids", None) or []),
        "finish_reason": getattr(item, "finish_reason", None),
        "stop_reason": getattr(item, "stop_reason", None),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    records = load_manifest(
        args.input_jsonl,
        expected_sha256=args.input_sha256,
        expected_count=args.expected_count,
    )
    artifact = validate_artifact(args)
    llm, tokenizer, sampling_params, engine = initialize_engine(args)
    resolved_sampling = sampling_contract(args)
    counts = {"rows": 0, "json_valid": 0, "schema_valid": 0, "inference_errors": 0}
    prompt_tokens = 0
    output_tokens = 0
    generation_seconds = 0.0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for window in batches(records, args.request_window_size):
            loaded = load_images(window, workers=args.download_workers)
            requests: list[dict[str, Any]] = []
            valid: list[tuple[Record, str]] = []
            rows_by_index: dict[int, dict[str, Any]] = {}
            try:
                for record, image in zip(window, loaded, strict=True):
                    if isinstance(image, BaseException):
                        rows_by_index[record.input_index] = error_row(record, image)
                        continue
                    try:
                        prompt, user_prompt = render_training_prompt(
                            tokenizer,
                            record.instruction,
                            record.target_size,
                        )
                        requests.append(
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": image},
                                "mm_processor_kwargs": {"max_pixels": args.max_pixels},
                            }
                        )
                        valid.append((record, user_prompt))
                    except BaseException as exc:
                        rows_by_index[record.input_index] = error_row(record, exc)
                if requests:
                    generated_at = time.monotonic()
                    try:
                        outputs = llm.generate(
                            requests,
                            [sampling_params for _ in requests],
                            use_tqdm=False,
                        )
                        if len(outputs) != len(valid):
                            raise RuntimeError("vLLM output count differs from request count")
                        for (record, user_prompt), output in zip(valid, outputs, strict=True):
                            rows_by_index[record.input_index] = result_row(
                                record,
                                user_prompt=user_prompt,
                                output=output,
                                sampling_contract=resolved_sampling,
                            )
                    except BaseException as exc:
                        for record, _user_prompt in valid:
                            rows_by_index[record.input_index] = error_row(record, exc)
                    generation_seconds += time.monotonic() - generated_at
            finally:
                for image in loaded:
                    if not isinstance(image, BaseException):
                        image.close()
            for record in window:
                row = rows_by_index[record.input_index]
                output_handle.write(canonical_json(row) + "\n")
                output_handle.flush()
                counts["rows"] += 1
                counts["json_valid"] += int(bool(row["json_valid"]))
                counts["schema_valid"] += int(bool(row["schema_valid"]))
                counts["inference_errors"] += int(bool(row.get("inference_error")))
                prompt_tokens += int(row.get("prompt_tokens", 0))
                output_tokens += int(row.get("output_tokens", 0))
            print(
                canonical_json(
                    {
                        "event": "pe_inference_progress",
                        "node_index": args.node_index,
                        **counts,
                    }
                ),
                flush=True,
            )
    summary = {
        "complete": True,
        "format_version": INFERENCE_FORMAT_VERSION,
        "started_at": started_at,
        "completed_at": utc_now(),
        "node_index": args.node_index,
        "node_count": args.node_count,
        "input": {
            "path": str(args.input_jsonl),
            "sha256": args.input_sha256,
            "rows": args.expected_count,
        },
        "checkpoint": {
            "step": args.checkpoint_step,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        "artifact": artifact,
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
        },
        "sampling": resolved_sampling,
        "engine": engine,
        "results": {
            **counts,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": generation_seconds,
            "predictions_sha256": file_sha256(args.output_jsonl),
            "predictions_bytes": args.output_jsonl.stat().st_size,
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def recover_existing_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Validate a completed JSONL and recover from summary-only serialization failure."""

    records = load_manifest(
        args.input_jsonl,
        expected_sha256=args.input_sha256,
        expected_count=args.expected_count,
    )
    artifact = validate_artifact(args)
    rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prediction line {line_number} is not an object")
            rows.append(value)
    if len(rows) != len(records):
        raise ValueError(f"existing prediction count mismatch: {len(rows)} != {len(records)}")

    expected_sampling = sampling_contract(args)
    counts = {"rows": 0, "json_valid": 0, "schema_valid": 0, "inference_errors": 0}
    prompt_tokens = 0
    output_tokens = 0
    for record, row in zip(records, rows, strict=True):
        expected_size = {"width": record.target_size[0], "height": record.target_size[1]}
        expected_common = {
            "format_version": INFERENCE_FORMAT_VERSION,
            "input_index": record.input_index,
            "sample_id": record.sample_id,
            "task_type": record.task_type,
            "source_s3_uri": record.source_s3_uri,
            "instruction": record.instruction,
            "target_canvas_size": expected_size,
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
        }
        for field, expected in expected_common.items():
            if row.get(field) != expected:
                raise ValueError(f"prediction {record.input_index} changed {field}")
        inference_error = row.get("inference_error")
        if inference_error:
            if row.get("messages") is not None or row.get("response") is not None:
                raise ValueError(f"error row {record.input_index} contains a conversation")
        else:
            expected_user = build_user_prompt(record.instruction, record.target_size)
            expected_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": expected_user},
                {"role": "assistant", "content": row.get("response")},
            ]
            if row.get("selection") != dict(record.selection):
                raise ValueError(f"prediction {record.input_index} changed selection provenance")
            if row.get("messages") != expected_messages:
                raise ValueError(f"prediction {record.input_index} changed the training prompt")
            if row.get("images") != [record.source_s3_uri]:
                raise ValueError(f"prediction {record.input_index} changed Source image binding")
            if row.get("sampling") != expected_sampling:
                raise ValueError(f"prediction {record.input_index} changed sampling contract")
            parsed, json_error, schema_error = parse_response(str(row.get("response")))
            if (
                row.get("prediction") != parsed
                or row.get("json_valid") is not (json_error is None)
                or row.get("schema_valid") is not (json_error is None and schema_error is None)
            ):
                raise ValueError(f"prediction {record.input_index} parse result changed")
        counts["rows"] += 1
        counts["json_valid"] += int(row.get("json_valid") is True)
        counts["schema_valid"] += int(row.get("schema_valid") is True)
        counts["inference_errors"] += int(bool(inference_error))
        prompt_tokens += int(row.get("prompt_tokens") or 0)
        output_tokens += int(row.get("output_tokens") or 0)

    now = utc_now()
    summary = {
        "complete": True,
        "format_version": INFERENCE_FORMAT_VERSION,
        "started_at": None,
        "completed_at": now,
        "node_index": args.node_index,
        "node_count": args.node_count,
        "input": {
            "path": str(args.input_jsonl),
            "sha256": args.input_sha256,
            "rows": args.expected_count,
        },
        "checkpoint": {
            "step": args.checkpoint_step,
            "manifest_sha256": args.checkpoint_manifest_sha256,
        },
        "artifact": artifact,
        "prompt": {
            "version": PROMPT_VERSION,
            "contract_sha256": PROMPT_CONTRACT_SHA256,
            "label_projection_version": LABEL_PROJECTION_VERSION,
        },
        "sampling": expected_sampling,
        "engine": {
            "llm_kwargs": build_llm_kwargs(args),
            "runtime": runtime_versions(),
            "summary_recovered_without_regeneration": True,
        },
        "results": {
            **counts,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": None,
            "predictions_sha256": file_sha256(args.output_jsonl),
            "predictions_bytes": args.output_jsonl.stat().st_size,
        },
        "recovery": {
            "reason": "vllm_mutated_speculative_kwargs_were_not_json_serializable",
            "validated_at": now,
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-window-size", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=27000)
    parser.add_argument("--max-pixels", type=int, default=2359296)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--recover-existing-predictions", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "checkpoint_step",
        "expected_count",
        "node_count",
        "batch_size",
        "request_window_size",
        "download_workers",
        "max_model_len",
        "max_new_tokens",
        "max_pixels",
        "num_speculative_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.node_index < args.node_count:
        parser.error("--node-index must be in [0, --node-count)")
    if args.request_window_size < args.batch_size:
        parser.error("--request-window-size must be at least --batch-size")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    for value, label in (
        (args.artifact_manifest_sha256, "artifact manifest"),
        (args.checkpoint_manifest_sha256, "checkpoint manifest"),
        (args.input_sha256, "input"),
    ):
        if not _SHA256_RE.fullmatch(value):
            parser.error(f"--{label.replace(' ', '-')} SHA-256 is invalid")
    if args.output_jsonl.resolve() == args.input_jsonl.resolve():
        parser.error("output must not overwrite input")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = recover_existing_summary(args) if args.recover_existing_predictions else run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
