"""Make TransformerEngine use FlashAttention-2 for the Qwen3.5 hd256 varlen path.

WHY: On B300, TE 2.16's DotProductAttention for Qwen3.5 (head_dim=256, GQA, thd/varlen)
finds NO usable backend: cuDNN FusedAttention rejects hd256, and although flash-attn 2.8.3
is installed and fully functional (flash_attn_2_cuda loads), TE reports it "not installed"
because the wheel is `2.8.3.post1` and TE's `FlashAttentionUtils.max_version == 2.8.3`
(PEP440: 2.8.3.post1 > 2.8.3), so TE disables FA2 -> it falls back to the pure-PyTorch
UnfusedDotProductAttention (very slow). FA4 is the only other option and JIT-recompiles per
shape under packing (~48 min/step).

FA2 IS the right B300 attention (design-layout-swift uses flash_attention_2), it is fast and
does NOT recompile per shape. `2.8.3.post1` is code-identical to `2.8.3`, so we simply tell TE
to treat it as 2.8.3 and enable it.

Inject via `--custom_register_path .../dlms/patches/te_fa2_enable.py` (self-applies on import).
Requires flash-attn (FA2) installed and FA4 NOT installed (same `flash_attn` namespace).
"""
from __future__ import annotations


def apply() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception as e:
        print(f"[dlms] TE-FA2 enable: flash_attn not importable ({e}); skipping")
        return False

    # This patch is FA2-only. On B300 the installed flash_attn is FA4 (CuTe-DSL): it exposes
    # `flash_attn.cute` and has no top-level `__version__`/`flash_attn_func`. FA4 is a different
    # code path (routed via --attention_backend flash), needs no TE version fixup, and touching
    # `flash_attn.__version__` here would crash. Detect FA4 and skip cleanly.
    fa_version = getattr(flash_attn, "__version__", None)
    is_fa4 = fa_version is None or hasattr(flash_attn, "cute")
    if is_fa4:
        print(f"[dlms] TE-FA2 enable: detected FA4 (cute={hasattr(flash_attn, 'cute')}, "
              f"__version__={fa_version!r}); this patch is FA2-only, skipping", flush=True)
        return False

    from transformer_engine.pytorch.attention.dot_product_attention import utils as te_u

    FA = te_u.FlashAttentionUtils
    # Raise the ceiling so post-releases within 2.8.x are accepted, pin the effective
    # version to 2.8.3, and (re)compute the v2_x_plus capability flags + is_installed.
    FA.max_version = te_u.PkgVersion("2.8.99")
    FA.version = te_u.PkgVersion("2.8.3")
    FA.set_flash_attention_version()  # sets is_installed=True + all v2_*_plus flags
    print(f"[dlms] TE-FA2 enabled: FlashAttentionUtils.is_installed={FA.is_installed} "
          f"version={FA.version} (real flash_attn {fa_version})", flush=True)
    return True


apply()
