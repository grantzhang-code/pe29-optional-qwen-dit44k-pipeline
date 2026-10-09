"""Select the exact image-edit training contract used by inference.

The large-scale inference stack is shared by the inline V4, instruction-based
V5, and direct-root V6 tasks.  Selection happens before any worker module
imports prompt constants, so one process can never mix one version's prompt
with another version's validator.  Production jobs must set
``IMAGE_EDIT_TARGET_SCHEMA`` and also pass the matching ``--target-schema`` CLI
value; the latter is checked by each process in the launch chain.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


TARGET_SCHEMA_ENV = "IMAGE_EDIT_TARGET_SCHEMA"
V6_PROMPT_VARIANT_ENV = "IMAGE_EDIT_V6_PROMPT_VARIANT"
V4_TARGET_SCHEMA = "image_edit_v4_inline"
V5_TARGET_SCHEMA = "image_edit_v5_instruction"
V6_TARGET_SCHEMA = "image_edit_v6_direct_root_xyxy"
V6_TRAINING_PROMPT_VARIANT = "training_v7"
V6_HLD_V8_PROMPT_VARIANT = "hld_v8"
V6_HLD_V9_PROMPT_VARIANT = "hld_v9"
SUPPORTED_TARGET_SCHEMAS = (V4_TARGET_SCHEMA, V5_TARGET_SCHEMA, V6_TARGET_SCHEMA)

_ALIASES = {
    "v4": V4_TARGET_SCHEMA,
    "inline_v4": V4_TARGET_SCHEMA,
    V4_TARGET_SCHEMA: V4_TARGET_SCHEMA,
    "v5": V5_TARGET_SCHEMA,
    "instruction_v5": V5_TARGET_SCHEMA,
    V5_TARGET_SCHEMA: V5_TARGET_SCHEMA,
    "v6": V6_TARGET_SCHEMA,
    "direct_root_v6": V6_TARGET_SCHEMA,
    V6_TARGET_SCHEMA: V6_TARGET_SCHEMA,
}


def _selected_schema() -> str:
    raw = os.environ.get(TARGET_SCHEMA_ENV, V4_TARGET_SCHEMA).strip().lower()
    try:
        return _ALIASES[raw]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported {TARGET_SCHEMA_ENV}={raw!r}; expected one of "
            f"{SUPPORTED_TARGET_SCHEMAS!r}"
        ) from exc


TARGET_SCHEMA = _selected_schema()
PROMPT_VARIANT = os.environ.get(
    V6_PROMPT_VARIANT_ENV,
    V6_TRAINING_PROMPT_VARIANT,
).strip().lower()

if TARGET_SCHEMA == V6_TARGET_SCHEMA:
    from dlms.datasets.image_edit_v6_full_json import (
        V6TargetValidationError as InlineTargetValidationError,
        validate_v6_target as validate_inline_target,
    )

    if PROMPT_VARIANT == V6_TRAINING_PROMPT_VARIANT:
        from dlms.datasets.image_edit_v6_full_json import (
            PROMPT_CONTRACT_SHA256,
            PROMPT_VERSION,
            SYSTEM_PROMPT,
            build_user_prompt,
            prompt_contract_sha256,
        )
    elif PROMPT_VARIANT == V6_HLD_V8_PROMPT_VARIANT:
        from dlms.datasets.image_edit_v6_hld_v8_inference_prompt import (
            PROMPT_CONTRACT_SHA256,
            PROMPT_VERSION,
            SYSTEM_PROMPT,
            build_user_prompt,
            prompt_contract_sha256,
        )
    elif PROMPT_VARIANT == V6_HLD_V9_PROMPT_VARIANT:
        from dlms.datasets.image_edit_v6_hld_v9_inference_prompt import (
            PROMPT_CONTRACT_SHA256,
            PROMPT_VERSION,
            SYSTEM_PROMPT,
            build_user_prompt,
            prompt_contract_sha256,
        )
    else:
        raise RuntimeError(
            f"unsupported {V6_PROMPT_VARIANT_ENV}={PROMPT_VARIANT!r}; expected "
            f"one of {V6_TRAINING_PROMPT_VARIANT!r}, "
            f"{V6_HLD_V8_PROMPT_VARIANT!r}, or {V6_HLD_V9_PROMPT_VARIANT!r}"
        )
elif TARGET_SCHEMA == V5_TARGET_SCHEMA:
    from dlms.datasets.image_edit_v5_full_json import (
        PROMPT_CONTRACT_SHA256,
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        V5TargetValidationError as InlineTargetValidationError,
        build_user_prompt,
        prompt_contract_sha256,
        validate_v5_target as validate_inline_target,
    )
else:
    from dlms.datasets.image_edit_full_json import (
        PROMPT_CONTRACT_SHA256,
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        InlineTargetValidationError,
        build_user_prompt,
        prompt_contract_sha256,
        validate_inline_target,
    )


def require_target_schema(value: str) -> str:
    """Reject a CLI/environment mismatch before model or data initialization."""

    requested = _ALIASES.get(str(value).strip().lower())
    if requested is None:
        raise ValueError(
            f"unsupported target schema {value!r}; expected one of "
            f"{SUPPORTED_TARGET_SCHEMAS!r}"
        )
    if requested != TARGET_SCHEMA:
        raise ValueError(
            f"target schema mismatch: CLI requested {requested!r}, but "
            f"{TARGET_SCHEMA_ENV} selected {TARGET_SCHEMA!r} at process import"
        )
    return requested


def contract_metadata() -> Mapping[str, Any]:
    return {
        "target_schema": TARGET_SCHEMA,
        "prompt_variant": PROMPT_VARIANT,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
    }
