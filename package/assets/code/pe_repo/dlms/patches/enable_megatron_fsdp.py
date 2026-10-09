"""Enable Megatron-FSDP (ZeRO-3 across DP) in ms-swift's megatron backend.

ms-swift doesn't expose `use_megatron_fsdp` as a CLI arg, and `--megatron_extra_kwargs`
feeds ModelConfig (raises "unexpected keyword argument 'use_megatron_fsdp'"). But
`MegatronSftArguments.initialize_megatron(self)` passes `self` (the args namespace) to
megatron, and megatron reads `args.use_megatron_fsdp`. So we set the FSDP attrs on the args
right before initialize_megatron runs.

Use with TP=1 PP=1 (DP = world_size). Activate via env DLMS_FSDP=1 +
`--custom_register_path .../dlms/patches/enable_megatron_fsdp.py`.
"""
from __future__ import annotations

import importlib
import os


def apply() -> bool:
    if os.environ.get("DLMS_FSDP", "0") != "1":
        return False
    attrs = {
        "use_megatron_fsdp": True,
        "use_custom_fsdp": True,
        "data_parallel_sharding_strategy": os.environ.get("DLMS_FSDP_SHARD", "optim_grads_params"),
    }
    patched = []
    for modname in ("swift.megatron.utils", "swift.megatron.arguments.megatron_args"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        orig = getattr(m, "initialize_megatron", None)
        if orig is None or getattr(orig, "_dlms_fsdp", False):
            continue

        def make(orig):
            def wrapped(args, *a, **k):
                for kk, vv in attrs.items():
                    setattr(args, kk, vv)
                return orig(args, *a, **k)
            wrapped._dlms_fsdp = True
            return wrapped

        setattr(m, "initialize_megatron", make(orig))
        patched.append(modname)
    print(f"[dlms] Megatron-FSDP enable patch applied ({attrs['data_parallel_sharding_strategy']}) -> {patched}", flush=True)
    return True


apply()
