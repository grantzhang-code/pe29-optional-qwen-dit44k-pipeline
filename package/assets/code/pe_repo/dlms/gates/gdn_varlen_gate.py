#!/usr/bin/env python3
"""Operator-correctness gates for the Qwen3.5 GDN varlen/packing path on B300.

These are the megatron-swift analogue of design-layout-swift's packing-acceptance
gates. They validate, at the operator level, that the packing (multi-segment
cu_seqlens) path produces the SAME result as computing each segment alone — i.e.
no cross-sample leakage — for the two GDN operators that the fla-triton default
gets wrong on sm_103:

  Gate-A  depthwise causal conv  (Dao causal_conv1d_fn + seq_idx, the dlms fix)
  Gate-B  chunk_gated_delta_rule (fla-core 0.5.1, cu_seqlens)
  Gate-C  determinism (same input, double run -> bitwise identical)

PASS criteria mirror the sibling project: per-segment max|Δ| below tolerance for
the leak tests, and bitwise-0 for the determinism test.

Run:  python -m dlms.gates.gdn_varlen_gate
Exit code 0 = all gates pass.
"""
from __future__ import annotations

import sys
import torch

# Ensure the dlms Dao-conv patch is active (patches fla causal_conv1d symbol).
from dlms.patches import gdn_varlen_conv  # noqa: F401  (self-applies on import)

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)


def _seq_idx_and_cu(seg_lens: list[int]):
    cu = torch.tensor([0, *torch.tensor(seg_lens).cumsum(0).tolist()], device=DEV, dtype=torch.int64)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(seg_lens), device=DEV, dtype=torch.int32),
        torch.tensor(seg_lens, device=DEV),
    ).unsqueeze(0)
    return cu, seq_idx


def gate_a_conv_no_leak(seg_lens=(37, 51, 40, 19), conv_dim=256, width=4, tol=2e-2):
    """Dao varlen conv: each packed segment must equal that segment convolved alone."""
    from causal_conv1d import causal_conv1d_fn

    T = sum(seg_lens)
    x_btd = torch.randn(1, T, conv_dim, device=DEV, dtype=DT)          # [B,T,D] memory
    weight = torch.randn(conv_dim, width, device=DEV, dtype=DT)
    _, seq_idx = _seq_idx_and_cu(list(seg_lens))

    packed = causal_conv1d_fn(
        x=x_btd.transpose(1, 2), weight=weight, bias=None, activation="silu", seq_idx=seq_idx
    ).transpose(1, 2)  # [B,T,D]

    # each segment alone
    alone = torch.empty_like(packed)
    off = 0
    for L in seg_lens:
        seg = x_btd[:, off:off + L, :]
        si = torch.zeros(1, L, device=DEV, dtype=torch.int32)
        out = causal_conv1d_fn(
            x=seg.transpose(1, 2), weight=weight, bias=None, activation="silu", seq_idx=si
        ).transpose(1, 2)
        alone[:, off:off + L, :] = out
        off += L

    d = (packed.float() - alone.float()).abs().max().item()
    ok = d < tol
    print(f"[Gate-A conv-no-leak] max|Δ(packed,alone)| = {d:.3e}  tol={tol}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_b_chunk_no_leak(seg_lens=(37, 51, 40, 19), h=4, dk=128, dv=128, tol=5e-2):
    """chunk_gated_delta_rule varlen: packed segments must equal per-segment scans."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    T = sum(seg_lens)
    cu, _ = _seq_idx_and_cu(list(seg_lens))

    def rand(*s):
        return torch.randn(*s, device=DEV, dtype=DT)

    q = rand(1, T, h, dk); k = rand(1, T, h, dk); v = rand(1, T, h, dv)
    g = torch.nn.functional.logsigmoid(rand(1, T, h).float())
    beta = rand(1, T, h).sigmoid()

    packed, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
    )
    alone = torch.empty_like(packed)
    off = 0
    for L in seg_lens:
        cu_seg = torch.tensor([0, L], device=DEV, dtype=torch.int64)
        o, _ = chunk_gated_delta_rule(
            q[:, off:off + L], k[:, off:off + L], v[:, off:off + L],
            g=g[:, off:off + L], beta=beta[:, off:off + L],
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seg,
        )
        alone[:, off:off + L] = o
        off += L

    d = (packed.float() - alone.float()).abs().max().item()
    ok = d < tol
    print(f"[Gate-B chunk-no-leak] max|Δ(packed,alone)| = {d:.3e}  tol={tol}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_c_determinism(seg_lens=(37, 51, 40, 19), conv_dim=256, width=4):
    """Same input twice -> bitwise identical (fla 0.5.1 determinism on sm_103)."""
    from causal_conv1d import causal_conv1d_fn

    T = sum(seg_lens)
    x = torch.randn(1, T, conv_dim, device=DEV, dtype=DT)
    w = torch.randn(conv_dim, width, device=DEV, dtype=DT)
    _, si = _seq_idx_and_cu(list(seg_lens))
    a = causal_conv1d_fn(x=x.transpose(1, 2), weight=w, bias=None, activation="silu", seq_idx=si)
    b = causal_conv1d_fn(x=x.transpose(1, 2), weight=w, bias=None, activation="silu", seq_idx=si)
    d = (a - b).abs().max().item()
    ok = d == 0.0
    print(f"[Gate-C determinism] max|Δ(run1,run2)| = {d:.3e}  -> {'PASS(bitwise-0)' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device"); return 2
    print(f"device: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
    results = [gate_a_conv_no_leak(), gate_b_chunk_no_leak(), gate_c_determinism()]
    ok = all(results)
    print(f"\n=== GDN varlen gates: {'ALL PASS' if ok else 'FAILURE'} ({sum(results)}/{len(results)}) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
