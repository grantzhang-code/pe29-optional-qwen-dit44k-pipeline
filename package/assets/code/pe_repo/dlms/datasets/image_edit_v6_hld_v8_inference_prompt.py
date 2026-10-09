"""Inference-only V8 prompt with one local HLD constraint.

The source 128-H200 run was trained with the V7 prompt.  This module preserves
that system prompt and user prompt byte-for-byte except for one sentence scoped
to ``high_level_description`` so an A/B run does not silently change the rest
of the image-edit-v6 contract.
"""

from __future__ import annotations

import hashlib
import json

from dlms.datasets.image_edit_v6_full_json import (
    SYSTEM_PROMPT,
    build_user_prompt as build_training_user_prompt,
)


PROMPT_VERSION = "image_edit_v6_direct_root_xyxy_v7_qwen35_v8"
PROMPT_CONTRACT_SHA256 = "74f5445b236bb4ad01b0396a312bca888113febd83c2ac1ef49e1cdeae8c9783"
HLD_LOCAL_CONSTRAINT = (
    "For high_level_description only, mention only exact labels present in "
    "elements.layout and avoid redundant or consecutive repetition of words, phrases, "
    "relations, or label chains; write concise grammatical prose."
)
_INSERTION_ANCHOR = "normalized_edit_instruction. Include style_description.color_palette."


def build_user_prompt(instruction: str) -> str:
    """Add exactly one HLD-only sentence to the frozen V7 training prompt."""

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
