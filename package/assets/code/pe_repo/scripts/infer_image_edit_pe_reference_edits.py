#!/usr/bin/env python3
"""Run TextImage2Json inference with the reference-edits-v7 training contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DLMS_IMAGE_EDIT_V6_PE_MDS_LIBRARY_ONLY", "true")

from dlms.datasets.image_edit_v6_pe_mds import (
    REFERENCE_EDITS_LABEL_PROJECTION_VERSION,
    REFERENCE_EDITS_PROMPT_CONTRACT_SHA256,
    REFERENCE_EDITS_PROMPT_VERSION,
    REFERENCE_EDITS_SYSTEM_PROMPT,
    build_reference_edits_user_prompt,
    canonical_reference_edits_target,
    reference_edits_prompt_contract_sha256,
)
from scripts import infer_image_edit_pe_benchmark as runner


INFERENCE_FORMAT_VERSION = "image-edit-pe-reference-edits-target-size-v7-inference-v1"
ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"


def expected_direct_prompt(instruction: str, target_size: tuple[int, int]) -> str:
    """Reproduce the checkpoint's exact non-thinking chat-template prompt."""

    user_prompt = build_reference_edits_user_prompt(instruction, target_size)
    if user_prompt.count("<image>") != 1:
        raise RuntimeError("training PE prompt must contain exactly one image marker")
    expanded = user_prompt.replace("<image>", runner._VISION_MARKER, 1)  # noqa: SLF001
    return (
        f"<|im_start|>system\n{REFERENCE_EDITS_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{expanded}<|im_end|>\n"
        f"{runner._ASSISTANT_HEADER}{ASSISTANT_PREFILL}"  # noqa: SLF001
    )


def render_training_prompt(
    tokenizer: Any,
    instruction: str,
    target_size: tuple[int, int],
) -> tuple[str, str]:
    """Render without stripping the assistant prefix used during training."""

    user_prompt = build_reference_edits_user_prompt(instruction, target_size)
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
            {"role": "system", "content": REFERENCE_EDITS_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return a string")
    assistant_offset = rendered.rfind(runner._ASSISTANT_HEADER)  # noqa: SLF001
    if assistant_offset < 0:
        raise RuntimeError("Qwen chat template emitted no assistant header")
    prefix_end = assistant_offset + len(runner._ASSISTANT_HEADER)  # noqa: SLF001
    assistant_prefix = rendered[prefix_end:]
    if assistant_prefix != ASSISTANT_PREFILL:
        raise RuntimeError(
            "checkpoint tokenizer did not emit the training non-thinking prefix: "
            f"{assistant_prefix!r}"
        )
    if rendered != expected_direct_prompt(instruction, target_size):
        raise RuntimeError("checkpoint tokenizer does not reproduce the PE training prompt")
    if rendered.count("<|image_pad|>") != 1:
        raise RuntimeError("rendered PE prompt does not contain exactly one image")
    return rendered, user_prompt


def parse_response(response: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse strict JSON and validate the exact reference_edits output schema."""

    text = runner._THINK_RE.sub("", response, count=1)  # noqa: SLF001
    text = runner._FENCE_START_RE.sub("", text, count=1)  # noqa: SLF001
    text = runner._FENCE_END_RE.sub("", text, count=1).strip()  # noqa: SLF001

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant: {token}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    try:
        projected = json.loads(canonical_reference_edits_target(value))
    except (ValueError, KeyError, TypeError) as exc:
        return value, None, f"{type(exc).__name__}: {exc}"
    return projected, None, None


def configure_runner() -> None:
    """Bind the reusable PE runner to the checkpoint's exact training contract."""

    runner.PROMPT_VERSION = REFERENCE_EDITS_PROMPT_VERSION
    runner.PROMPT_CONTRACT_SHA256 = REFERENCE_EDITS_PROMPT_CONTRACT_SHA256
    runner.LABEL_PROJECTION_VERSION = REFERENCE_EDITS_LABEL_PROJECTION_VERSION
    runner.SYSTEM_PROMPT = REFERENCE_EDITS_SYSTEM_PROMPT
    runner.build_user_prompt = build_reference_edits_user_prompt
    runner.prompt_contract_sha256 = reference_edits_prompt_contract_sha256
    runner.expected_direct_prompt = expected_direct_prompt
    runner.render_training_prompt = render_training_prompt
    runner.parse_response = parse_response
    runner.INFERENCE_FORMAT_VERSION = INFERENCE_FORMAT_VERSION
    runner.EXPECTED_LOGICAL_MTP_LAYERS = 1
    runner.EXPECTED_MTP_SHARED_WEIGHTS = False


def main() -> int:
    configure_runner()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
