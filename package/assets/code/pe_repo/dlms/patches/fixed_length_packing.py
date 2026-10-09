"""Pad every packed sequence to a FIXED length (= packing_length) so the attention
shape is constant across steps.

WHY: FlashAttention-4 (flash-attn-4 CuTe-DSL) JIT-compiles per kernel *shape*. ms-swift's
default `get_padding_to` only pads the packed row to a multiple of TP*CP (=8 here), so every
step's packed length varies (observed 6572..8192) -> FA4 re-traces/recompiles every step
(~48 min/step on 27B). Padding every packed row up to the fixed `packing_length` makes the
shape constant -> FA4 compiles once and reuses (combine with
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1). This is the "pack then pad to fixed length" fix
(works with TransformerEngine 2.16). Waste is small because packing already fills rows near
packing_length.

Inject via `--custom_register_path .../dlms/patches/fixed_length_packing.py` (self-applies on
import). Only meaningful when --packing true; harmless otherwise. Needs packing_length
divisible by TP*CP (8192 % 8 == 0).
"""
from __future__ import annotations

import importlib

_APPLIED = False


def apply() -> bool:
    global _APPLIED
    import swift.megatron.utils.utils as u

    orig = getattr(u, "_dlms_orig_get_padding_to", None) or u.get_padding_to

    def patched_get_padding_to(args):
        base = orig(args)  # TP*CP*fp8 multiple (also the SP divisibility requirement)
        pl = getattr(args, "packing_length", None)
        if getattr(args, "packing", False) and pl:
            # round packing_length up to a multiple of `base` so SP/TP divisibility holds,
            # then use it as the FIXED pad length for every packed row.
            b = base or 1
            fixed = ((pl + b - 1) // b) * b
            return fixed
        return base

    u._dlms_orig_get_padding_to = orig
    u.get_padding_to = patched_get_padding_to
    # propagate to the package re-export and any trainer module that already imported it
    for modname in ("swift.megatron.utils", "swift.megatron.trainers.base",
                    "swift.megatron.trainers.gkd_trainer"):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "get_padding_to"):
                m.get_padding_to = patched_get_padding_to
        except Exception:
            pass
    _APPLIED = True
    print("[dlms] fixed-length packing patch applied: pad packed rows to packing_length "
          "(constant attention shape -> FA4 compiles once)", flush=True)
    return True


apply()
