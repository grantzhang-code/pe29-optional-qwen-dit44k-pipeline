#!/usr/bin/env python3
"""Smoke one image-edit JSONL row through the real Qwen3.5 Swift template.

The script deliberately checks both layers of the multimodal input path:

1. ``ms-swift`` loads both ordered S3 images and expands them into Qwen vision
   tensors/tokens.
2. With ``--forward``, the pinned HF Qwen checkpoint consumes the encoded
   prompt and returns finite next-token logits.

It writes the exact decoded inference prompt (including expanded image-pad
tokens), the human-readable system/user messages, the selected source row, and
a machine-readable report under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable


def _read_row(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("--row-index must be non-negative")
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"row {index} is not a JSON object")
                return row
    raise IndexError(f"row {index} does not exist in {path}")


def _message(row: dict[str, Any], role: str) -> str:
    matches = [item for item in row.get("messages", []) if item.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0].get("content"), str):
        raise ValueError(f"expected exactly one string {role!r} message")
    return matches[0]["content"]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_ints(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(part) for part in shape] if shape is not None else None


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _image_metadata(uri: str, read_s3_bytes: Any) -> dict[str, Any]:
    from PIL import Image

    payload = read_s3_bytes(uri)
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return {
            "uri": uri,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }


def _non_mask_spans(labels: list[int]) -> list[list[int]]:
    spans: list[list[int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        elif label == -100 and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(labels)])
    return spans


def _encode(
    *,
    processor: Any,
    row: dict[str, Any],
    mode: str,
    max_length: int,
    loss_scale: str,
    add_non_thinking_prefix: bool,
) -> tuple[Any, dict[str, Any]]:
    from swift.template import get_template

    template = get_template(
        processor,
        template_type="qwen3_5",
        loss_scale=loss_scale,
        max_length=max_length,
        truncation_strategy="raise",
        add_non_thinking_prefix=add_non_thinking_prefix,
        response_prefix=None if add_non_thinking_prefix else "",
    )
    template.set_mode(mode)
    return template, template.encode(row)


def _forward(
    *,
    model_dir: Path,
    encoded: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    before_allocated = int(torch.cuda.memory_allocated(device)) if device.startswith("cuda") else 0
    load_start = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": encoded["pixel_values"].to(device=device, dtype=model.dtype),
        "image_grid_thw": encoded["image_grid_thw"].to(device=device),
        "use_cache": False,
        "logits_to_keep": 1,
    }
    mm_token_type_ids = encoded.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        inputs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0).to(device=device)

    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    logits = output.logits.detach().float().cpu()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Qwen forward returned non-finite logits")

    top = torch.topk(logits[0, -1], k=5)
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
    return {
        "passed": True,
        "device": device,
        "model_class": type(model).__name__,
        "dtype": str(model.dtype),
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "top5_token_ids": [int(value) for value in top.indices.tolist()],
        "top5_logits": [round(float(value), 6) for value in top.values.tolist()],
        "cuda_allocated_delta_bytes": max(peak_allocated - before_allocated, 0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.aws_profile:
        os.environ["AWS_PROFILE"] = args.aws_profile
    os.environ.setdefault("AWS_REGION", args.aws_region)
    os.environ.setdefault("AWS_DEFAULT_REGION", args.aws_region)

    # Import after the profile/region are installed. Importing this module
    # patches ms-swift's media loader for s3:// URIs.
    from dlms.patches import s3_image_load
    from swift.model import get_model_processor

    row = _read_row(args.dataset, args.row_index)
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError("the smoke row must contain exactly [reference, target] images")
    if any(not isinstance(uri, str) or not uri.startswith("s3://") for uri in images):
        raise ValueError("both image paths must be s3:// URIs")

    system_prompt = _message(row, "system")
    user_prompt = _message(row, "user")
    assistant_target = _message(row, "assistant")
    if user_prompt.count("<image>") != 2:
        raise ValueError("the user prompt must contain exactly two <image> placeholders")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_row.json"
    selected_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (args.output_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
    (args.output_dir / "assistant_target.json").write_text(
        json.dumps(json.loads(assistant_target), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    image_report = [
        _image_metadata(uri, s3_image_load._read_s3_bytes)  # noqa: SLF001 - diagnostic contract
        for uri in images
    ]

    _, processor = get_model_processor(
        str(args.model_dir),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
    )

    training_template, training_encoded = _encode(
        processor=processor,
        row=row,
        mode="train",
        max_length=args.max_length,
        loss_scale=args.loss_scale,
        add_non_thinking_prefix=args.add_non_thinking_prefix,
    )
    inference_row = dict(row)
    inference_row["messages"] = [
        message for message in row["messages"] if message.get("role") != "assistant"
    ]
    inference_template, inference_encoded = _encode(
        processor=processor,
        row=inference_row,
        mode="transformers",
        max_length=args.max_length,
        loss_scale=args.loss_scale,
        add_non_thinking_prefix=args.add_non_thinking_prefix,
    )

    inference_ids = [int(value) for value in inference_encoded["input_ids"]]
    training_ids = [int(value) for value in training_encoded["input_ids"]]
    labels = [int(value) for value in training_encoded["labels"]]
    decoded_instruction = processor.tokenizer.decode(
        inference_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_training = processor.tokenizer.decode(
        training_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assistant_header = "<|im_start|>assistant\n"
    if assistant_header not in decoded_instruction or assistant_header not in decoded_training:
        raise RuntimeError("Qwen rendering is missing the assistant header")
    inference_assistant_prefix = decoded_instruction.rsplit(assistant_header, 1)[1]
    training_assistant_payload = decoded_training.rsplit(assistant_header, 1)[1]
    if not args.add_non_thinking_prefix:
        if inference_assistant_prefix:
            raise RuntimeError(
                "direct-JSON mode unexpectedly inserted text after the inference assistant header"
            )
        if not training_assistant_payload.startswith("{"):
            raise RuntimeError("direct-JSON training payload does not start with '{'")
    (args.output_dir / "qwen_inference_instruction.expanded.txt").write_text(
        decoded_instruction,
        encoding="utf-8",
    )
    readable_instruction = re.sub(
        r"(?:<\|image_pad\|>)+",
        lambda match: (
            f'<|image_pad|>[x{match.group(0).count("<|image_pad|>")}]'
        ),
        decoded_instruction,
    )
    (args.output_dir / "qwen_inference_instruction.readable.txt").write_text(
        readable_instruction,
        encoding="utf-8",
    )
    (args.output_dir / "qwen_training_sequence.expanded.txt").write_text(
        decoded_training,
        encoding="utf-8",
    )

    image_token_id = int(inference_template.image_token_id)
    inference_grid = _as_list(inference_encoded["image_grid_thw"])
    training_grid = _as_list(training_encoded["image_grid_thw"])
    if len(inference_grid) != 2 or inference_grid != training_grid:
        raise RuntimeError("Qwen did not preserve exactly two ordered image grids")
    image_token_count = sum(token == image_token_id for token in inference_ids)
    expected_image_tokens = sum(math.prod(grid) // 4 for grid in inference_grid)
    if image_token_count != expected_image_tokens:
        raise RuntimeError(
            f"image token expansion mismatch: actual={image_token_count}, "
            f"expected={expected_image_tokens}"
        )
    if _shape(inference_encoded.get("pixel_values")) is None:
        raise RuntimeError("Qwen processor did not return pixel_values")

    report: dict[str, Any] = {
        "passed": True,
        "sample_id": row.get("meta", {}).get("sample_id"),
        "dataset": str(args.dataset.resolve()),
        "row_index": args.row_index,
        "row_status": row.get("meta", {}).get("status"),
        "model_dir": str(args.model_dir.resolve()),
        "runtime": {
            "ms_swift": __import__("swift").__version__,
            "transformers": __import__("transformers").__version__,
            "torch": __import__("torch").__version__,
            "template": "qwen3_5",
            "processor_class": type(processor).__name__,
            "add_non_thinking_prefix": args.add_non_thinking_prefix,
            "response_prefix": None if args.add_non_thinking_prefix else "",
            "loss_scale": args.loss_scale,
            "inference_assistant_prefix": inference_assistant_prefix,
            "training_assistant_starts_with_json": training_assistant_payload.startswith("{"),
        },
        "messages": {
            "system_chars": len(system_prompt),
            "user_chars": len(user_prompt),
            "assistant_target_chars": len(assistant_target),
        },
        "images": image_report,
        "processor": {
            "passed": True,
            "ordered_image_grid_thw": inference_grid,
            "pixel_values_shape": _shape(inference_encoded["pixel_values"]),
            "pixel_values_dtype": str(inference_encoded["pixel_values"].dtype),
            "image_token_id": image_token_id,
            "image_token_count": image_token_count,
            "expected_image_token_count": expected_image_tokens,
            "inference_input_tokens": len(inference_ids),
            "inference_input_ids_sha256_int64le": _sha256_ints(inference_ids),
            "training_input_tokens": len(training_ids),
            "training_input_ids_sha256_int64le": _sha256_ints(training_ids),
            "training_supervised_spans": _non_mask_spans(labels),
            "training_supervised_tokens": sum(label != -100 for label in labels),
        },
        "artifacts": {
            "selected_row": str(selected_path.resolve()),
            "system_prompt": str((args.output_dir / "system_prompt.txt").resolve()),
            "user_prompt": str((args.output_dir / "user_prompt.txt").resolve()),
            "assistant_target": str((args.output_dir / "assistant_target.json").resolve()),
            "expanded_inference_instruction": str(
                (args.output_dir / "qwen_inference_instruction.expanded.txt").resolve()
            ),
            "readable_inference_instruction": str(
                (args.output_dir / "qwen_inference_instruction.readable.txt").resolve()
            ),
            "expanded_training_sequence": str(
                (args.output_dir / "qwen_training_sequence.expanded.txt").resolve()
            ),
        },
        "model_forward": {"requested": bool(args.forward), "passed": None},
    }
    if args.forward:
        report["model_forward"] = {
            "requested": True,
            **_forward(model_dir=args.model_dir, encoded=inference_encoded, device=args.device),
        }

    report_path = args.output_dir / "report.json"
    report["artifacts"]["report"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--loss-scale", default="default")
    parser.add_argument(
        "--add-non-thinking-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default false: make the assistant response start directly with JSON.",
    )
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
