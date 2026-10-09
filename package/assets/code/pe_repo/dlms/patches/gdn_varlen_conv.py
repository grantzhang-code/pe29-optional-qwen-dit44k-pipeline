"""Correct the Qwen3.5 gated-delta-net (GDN) depthwise causal conv under packing/varlen on B300.

WHY (carried over from design-layout-swift on B300 / sm_103):
    Under packing, the Megatron backend uses the *native* mcore GDN
    (``mcore_bridge.model.modules.gated_delta_net.GatedDeltaNet.forward``, and its
    base ``megatron.core.ssm.gated_delta_net``). Its ``chunk_gated_delta_rule`` call
    already threads ``cu_seqlens`` correctly (with fla-core==0.5.1, which fixed the
    0.5.0 nondeterminism). BUT its depthwise causal conv is routed through fla's own
    wrapper with the default ``backend='triton'`` — the exact kernel that MISCOMPILES
    on sm_103 (illegal access / silent wrong values), and whose ``backend='cuda'`` path
    mis-converts multi-segment ``cu_seqlens``.

FIX (identical mechanism to design-layout-swift/swift/model/models/qwen.py:1286-1307):
    When ``cu_seqlens is not None``, route the conv through Dao's
    ``causal_conv1d_fn(x=[B, conv_dim, T], seq_idx=..., cu int64)`` which restarts the
    causal window at every ``seq_idx`` change — correct per-segment behaviour, no
    cross-sample leak. Dao expects channel-last ([B, conv_dim, T] view over
    [B, T, conv_dim] memory) so we transpose WITHOUT ``.contiguous()``.

Apply by importing this module (idempotent) or via ``--external_plugins``:
    from dlms.patches import gdn_varlen_conv; gdn_varlen_conv.apply()
It self-applies on import as well.
"""
from __future__ import annotations

import torch

_APPLIED = False


def _make_patched(orig_causal_conv1d):
    from causal_conv1d import causal_conv1d_fn as _dao_causal_conv1d_fn

    def _patched_causal_conv1d(x, weight, bias=None, activation=None, cu_seqlens=None, **kwargs):
        # Non-varlen (single sequence / no packing): keep the original fla path.
        if cu_seqlens is None:
            return orig_causal_conv1d(
                x, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens, **kwargs
            )
        # Varlen/packing: Dao's kernel with per-segment seq_idx.
        cu = cu_seqlens.to(torch.int64)
        seg_lens = (cu[1:] - cu[:-1]).to(torch.int64)
        seq_idx = torch.repeat_interleave(
            torch.arange(seg_lens.numel(), device=cu.device, dtype=torch.int32), seg_lens
        ).unsqueeze(0)
        # x is [B, T, conv_dim] (B==1 under thd packing). Dao wants [B, conv_dim, T];
        # NOTE: no .contiguous() — the seq_idx path requires the channel-last view.
        out = _dao_causal_conv1d_fn(
            x=x.transpose(1, 2),
            weight=weight,
            bias=bias,
            activation=activation,
            seq_idx=seq_idx,
        ).transpose(1, 2)
        # fla's causal_conv1d returns a tuple; the call site indexes [0].
        return out, None

    _patched_causal_conv1d._dlms_dao_varlen_conv = True  # marker for verification
    return _patched_causal_conv1d


def apply() -> list[str]:
    """Patch the GDN conv symbol in every module that binds it. Idempotent.

    Returns the list of fully-qualified module symbols that were patched.
    """
    global _APPLIED
    patched: list[str] = []
    # Both the mcore-bridge subclass module and the base mcore ssm module import
    # ``causal_conv1d`` as a module-level name and call it as a free function.
    targets = [
        "mcore_bridge.model.modules.gated_delta_net",
        "megatron.core.ssm.gated_delta_net",
    ]
    for mod_name in targets:
        try:
            mod = __import__(mod_name, fromlist=["causal_conv1d"])
        except Exception:
            continue
        cur = getattr(mod, "causal_conv1d", None)
        if cur is None:
            continue
        if getattr(cur, "_dlms_dao_varlen_conv", False):
            patched.append(f"{mod_name}.causal_conv1d (already)")
            continue
        mod.causal_conv1d = _make_patched(cur)
        patched.append(f"{mod_name}.causal_conv1d")
    _APPLIED = True
    if patched:
        print(f"[dlms] GDN varlen conv patched (Dao causal_conv1d_fn + seq_idx): {patched}", flush=True)
    else:
        print("[dlms] GDN varlen conv patch: no target modules found (import order?)", flush=True)
    return patched


# Self-apply on import so `--external_plugins .../gdn_varlen_conv.py` works.
apply()
