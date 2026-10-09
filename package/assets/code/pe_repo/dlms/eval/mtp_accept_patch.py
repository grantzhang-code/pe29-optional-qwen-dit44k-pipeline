"""Measure the MTP (multi-token-prediction) top-1 acceptance rate — offline proxy.

The MTP head predicts the token at position t+2. The accept-rate proxy = fraction of loss
(assistant) positions where `argmax(mtp_logits) == mtp_labels` (the rolled next-next token).

HOOK POINT (important): Qwen3.5 on this stack runs mcore-bridge's `MultimodalGPTModel`, whose
MTP loss is computed INLINE in `mcore_bridge/model/gpt_model.py::_postprocess` — NOT via
megatron's `process_mtp_loss`. That inline code ALREADY contains an acceptance-rate hook, gated on
mcore >= 0.19:

    if hasattr(MTPLossLoggingHelper, 'save_metrics_to_tracker'):
        from megatron.core.transformer.multi_token_prediction import _compute_mtp_acceptance_counts
        correct, total = _compute_mtp_acceptance_counts(mtp_logits, mtp_labels, loss_mask_, ...)
        MTPLossLoggingHelper.save_metrics_to_tracker(loss, correct, total, ...)

megatron-core 0.18.2 ships neither symbol, so the branch is skipped. We simply SUPPLY both:
`_compute_mtp_acceptance_counts` (module fn) + `MTPLossLoggingHelper.save_metrics_to_tracker`
(staticmethod). Then mcore-bridge's own code counts acceptance — with the tensor orientation its
authors intended — and we accumulate the counts. This runs on the `self.training` forward, so the
eval driver does a few TRAIN steps (frozen params -> weights unchanged). Requires TP=1 so
`mtp_logits` is full-vocab (argmax correct).

Orientation (from compute_language_model_loss): labels/loss_mask are [b, s]; logits are [s, b, V];
labels/loss_mask are ALREADY rolled by _postprocess before the acceptance call.

Inject via `--custom_register_path .../dlms/eval/mtp_accept_patch.py`. At process exit writes
{correct,total,mtp_top1_accept_rate} to $DLMS_MTP_ACCEPT_OUT (default /root/dlms_out/mtp_accept.json).
"""
from __future__ import annotations

import atexit
import json
import os

_tracker = {"correct": 0.0, "total": 0.0}


def _dump():
    total = _tracker["total"]
    rate = (_tracker["correct"] / total) if total else float("nan")
    out = {"correct": _tracker["correct"], "total": total, "mtp_top1_accept_rate": rate}
    path = os.environ.get("DLMS_MTP_ACCEPT_OUT", "/root/dlms_out/mtp_accept.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass
    print(f"[dlms] MTP top-1 accept rate = {rate:.4f} ({int(_tracker['correct'])}/{int(total)})", flush=True)


def apply() -> bool:
    import torch
    import megatron.core.transformer.multi_token_prediction as mtp

    if getattr(mtp, "_dlms_mtp_accept_supplied", False):
        return True

    def _compute_mtp_acceptance_counts(mtp_logits, mtp_labels, loss_mask,
                                       output_layer=None, runtime_gather_output=True):
        # mtp_logits [s, b, V] (full vocab @ TP=1); mtp_labels/loss_mask [b, s], already rolled.
        with torch.no_grad():
            pred = mtp_logits.argmax(dim=-1)              # [s, b]
            pred = pred.transpose(0, 1).contiguous()      # [b, s] to align with labels
            m = loss_mask.to(torch.bool)
            correct = ((pred == mtp_labels) & m).sum()
            total = m.sum()
        return correct.detach(), total.detach()

    _orig_save_loss = mtp.MTPLossLoggingHelper.save_loss_to_tracker

    @staticmethod
    def save_metrics_to_tracker(loss, correct, total, layer_number, num_layers,
                                reduce_group=None, avg_group=None):
        # Accumulate acceptance counts (our measurement) ...
        try:
            _tracker["correct"] += float(correct.item() if hasattr(correct, "item") else correct)
            _tracker["total"] += float(total.item() if hasattr(total, "item") else total)
        except Exception as e:
            print(f"[dlms] MTP accept: count accumulate failed ({e})", flush=True)
        # ... and keep the normal per-layer loss logging working.
        _orig_save_loss(loss, layer_number, num_layers, reduce_group=reduce_group, avg_group=avg_group)

    mtp._compute_mtp_acceptance_counts = _compute_mtp_acceptance_counts
    mtp.MTPLossLoggingHelper.save_metrics_to_tracker = save_metrics_to_tracker
    mtp._dlms_mtp_accept_supplied = True
    atexit.register(_dump)
    print("[dlms] MTP accept patch: supplied _compute_mtp_acceptance_counts + "
          "MTPLossLoggingHelper.save_metrics_to_tracker (mcore-bridge inline MTP path; TP=1). "
          "Measures argmax(mtp_logits)==mtp_labels on training forward.", flush=True)
    return True


apply()
