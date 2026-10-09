"""Inference-only V9 prompt with one strongly scoped HLD constraint.

The source 128-H200 run was trained with the V7 prompt. This module preserves
that system prompt and every existing user-prompt byte, then inserts only the
HLD-local constraint below. This keeps the inference delta explicit and
machine-checkable while prohibiting labels absent from ``elements.layout``.
"""

from __future__ import annotations

import hashlib
import json

from dlms.datasets.image_edit_v6_full_json import (
    SYSTEM_PROMPT,
    build_user_prompt as build_training_user_prompt,
)


PROMPT_VERSION = "image_edit_v6_direct_root_xyxy_v7_qwen35_v9"
PROMPT_CONTRACT_SHA256 = "67b27761a29ce7b64afa4dfb79176eeba26c5c4e93cd7da4e1672f380d4ea676"
HLD_LOCAL_CONSTRAINT = (
    "For high_level_description only, every label reference MUST be copied verbatim "
    "from an existing elements.layout[*].label. NEVER invent or hallucinate a label, "
    "and never rename, renumber, or mention any label absent from elements.layout."
)
_INSERTION_ANCHOR = "normalized_edit_instruction. Include style_description.color_palette."


def build_user_prompt(instruction: str) -> str:
    """Add exactly one HLD-only block to the frozen V7 training prompt."""

    prompt = build_training_user_prompt(instruction)
    if prompt.count(_INSERTION_ANCHOR) != 1:
        raise RuntimeError("V7 prompt insertion anchor changed")
    replacement = (
        "normalized_edit_instruction. "
        f"{HLD_LOCAL_CONSTRAINT} "
        "Include style_description.color_palette."
    )
    return prompt.replace(_INSERTION_ANCHOR, replacement, 1)


def prompt_contract_sha256() -> str:
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_user_prompt("__PROMPT_CONTRACT_INSTRUCTION__"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
