"""FLUX-style reference-only attention and request-local inference KV state.

Only VAE reference rows are static. Text (including VLM image features) and
target rows remain dynamic. Training uses the same mask without a cache and
keeps gradients through both reference K/V and reference self-attention.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import torch
from src.models.layers import attention_op

from src.models.layers.attention_op import (
    CrossVarlenLayout, attention_cross_varlen, prepare_cross_varlen_attention,
)


@dataclass(frozen=True)
class ReferenceAttentionLayout:
    reference_indices: torch.Tensor
    dynamic_indices: torch.Tensor
    dynamic_text_indices: torch.Tensor
    dynamic_image_indices: torch.Tensor
    restore_kv_indices: torch.Tensor
    reference: CrossVarlenLayout
    dynamic: CrossVarlenLayout
    valid_tokens: int
    masked_q_ranges: torch.Tensor
    masked_k_ranges: torch.Tensor

    @classmethod
    def build(cls, text_lengths, reference_lengths, target_lengths, device):
        prepare_cross_varlen_attention()
        refs, dynamic, txt_dst, img_dst = [], [], [], []
        start, dst = 0, 0
        masked_q, masked_k = [], []
        for text, ref, target in zip(text_lengths, reference_lengths, target_lengths):
            end = start + text + ref + target
            # Disjoint query ranges, each with exactly one softmax key range.
            # Preserve original packed storage: no Q/K/V gather or duplication.
            for qs, qe, ks, ke in (
                (start, start + text, start, end),
                (start + text, start + text + ref, start + text, start + text + ref),
                (start + text + ref, end, start, end),
            ):
                if qe > qs:
                    masked_q.append((qs, qe))
                    masked_k.append((ks, ke))
            refs.extend(range(start + text, start + text + ref))
            dynamic.extend(range(start, start + text))
            dynamic.extend(range(start + text + ref, start + text + ref + target))
            txt_dst.extend(range(dst, dst + text))
            img_dst.extend(range(dst + text, dst + text + target))
            start += text + ref + target
            dst += text + target
        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=device)
        merged = refs + dynamic
        inverse = [0] * len(merged)
        for i, original in enumerate(merged):
            inverse[original] = i
        r_lengths = [n for n in reference_lengths if n]
        return cls(
            indices(refs), indices(dynamic), indices(txt_dst), indices(img_dst),
            indices(inverse),
            CrossVarlenLayout.from_lengths(r_lengths, r_lengths, device),
            CrossVarlenLayout.from_lengths(
                [t + n for t, n in zip(text_lengths, target_lengths)],
                [t + r + n for t, r, n in zip(text_lengths, reference_lengths, target_lengths)],
                device,
            ),
            start,
            torch.tensor(masked_q, dtype=torch.int32).reshape(-1, 2),
            torch.tensor(masked_k, dtype=torch.int32).reshape(-1, 2),
        )


@dataclass
class ReferenceKVLayer:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


def reference_attention(q, k, v, layout, cache: ReferenceKVLayer | None = None):
    if cache is None and attention_op.ACTIVE_BACKEND.name == "fa4":
        n = layout.valid_tokens
        out = attention_op._fa4_gqa_ranges_fwd(
            q[:n], k[:n], v[:n], layout.masked_q_ranges, layout.masked_k_ranges,
        )
        return torch.cat((out, q.new_zeros((q.shape[0] - n, *q.shape[1:]))), dim=0)
    if cache is not None and cache.key is not None:
        if cache.key.device != k.device or cache.key.dtype != k.dtype:
            raise ValueError("reference KV device/dtype changed; start a new cache")
        keys = torch.cat((cache.key, k), dim=0).index_select(0, layout.restore_kv_indices)
        values = torch.cat((cache.value, v), dim=0).index_select(0, layout.restore_kv_indices)
        return attention_cross_varlen(q, keys, values, layout.dynamic)

    r, d = layout.reference_indices, layout.dynamic_indices
    rk, rv = k.index_select(0, r), v.index_select(0, r)
    if cache is not None:
        # Own storage: never pin the complete prefill QKV allocation via a view.
        cache.key, cache.value = rk.clone(), rv.clone()
    ref_out = attention_cross_varlen(q.index_select(0, r), rk, rv, layout.reference)
    dyn_out = attention_cross_varlen(
        q.index_select(0, d), k[:layout.valid_tokens], v[:layout.valid_tokens], layout.dynamic
    )
    return torch.zeros_like(q).index_copy(0, r, ref_out).index_copy(0, d, dyn_out)


def _tensor_stamp(t):
    try:
        version = t._version
    except RuntimeError:  # inference-mode tensors have no version counter
        version = None
    return (id(t), tuple(t.shape), t.device, t.dtype, version)


def _layout_stamp(value):
    # Frozen dataclasses prevent field replacement, but their index tensors can
    # still be modified in place. Guard both nested Q/K schedules and gathers.
    if isinstance(value, torch.Tensor):
        return _tensor_stamp(value)
    if is_dataclass(value):
        return (id(value), tuple(_layout_stamp(getattr(value, field.name))
                                 for field in fields(value)))
    return value


class ReferenceKVCache:
    """One model + one immutable packed layout + one inference request.

    A fresh instance prefills from ALL image rows. A ready instance accepts
    TARGET image rows only; it never accepts replacement reference pixels.
    Use separate instances for CFG branches. Changing conditions, positions,
    model weights, or correspondence settings requires a new instance.
    This is intentionally not a global or cross-request cache.
    """

    def __init__(self):
        self.layers: list[ReferenceKVLayer] = []
        self.ready = False
        self._binding = None
        self._model = None
        self._metadata = None

    def bind(self, model, metadata, corr_code_gate_scale):
        if model.training or torch.is_grad_enabled():
            raise ValueError("reference KV cache requires eval() and no_grad/inference_mode")
        if getattr(model, "conditioning_policy", "full") != "reference_only":
            raise ValueError("reference KV cache requires conditioning_policy=reference_only")
        if metadata.get("reference_attention") is None:
            raise ValueError("reference KV cache requires reference-only packed metadata")
        # Check at the forward boundary, never once per transformer layer.
        binding = (
            id(model), id(metadata), float(corr_code_gate_scale),
            _layout_stamp(metadata["reference_attention"]),
            tuple(_tensor_stamp(p) for p in model.parameters()),
            tuple((name, _tensor_stamp(b)) for name, b in model.named_buffers()),
            tuple((name, _tensor_stamp(value)) for name, value in metadata.items()
                  if isinstance(value, torch.Tensor)),
        )
        if self.ready:
            if binding != self._binding:
                raise ValueError("reference KV cache binding changed; start a new cache")
        else:
            self._model, self._metadata = model, metadata  # keep identities alive
            self._binding = binding
            self.layers = [ReferenceKVLayer() for _ in model.blocks]

    def complete(self):
        if not self.layers or any(layer.key is None for layer in self.layers):
            raise RuntimeError("reference KV prefill did not populate every layer")
        self.ready = True
