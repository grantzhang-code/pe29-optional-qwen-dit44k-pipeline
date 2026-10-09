"""No-op the checkpoint save for eval-only runs.

The MTP accept-rate eval reuses the `megatron sft` training entrypoint with `train_iters=1`, so
ms-swift's default_flow forces an end-of-training checkpoint save. On an eval-only pass we don't
want a checkpoint at all, and with `freeze_parameters_ratio=1.0` (all params frozen, so the model
fits on ONE GPU without a 27B optimizer) the distributed optimizer's inner torch optimizer is
None -> `save_checkpoint()` crashes in `optimizer.state_dict()`. Eval runs BEFORE save in the
step loop, so the accept rate is already accumulated by then; we just skip the pointless save.

Inject via `--custom_register_path .../dlms/eval/skip_save.py` (eval only). Harmless if imported
on a real training run only in that it would disable saving — so NEVER add it to a training run.
"""
from __future__ import annotations


def apply() -> bool:
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer as _T
    except Exception:
        try:
            from swift.megatron.trainers.base import MegatronTrainer as _T  # fallback name
        except Exception as e:
            print(f"[dlms] skip_save: trainer class not found ({e}); not patched")
            return False

    def _noop_save_checkpoint(self, *args, **kwargs):
        print("[dlms] skip_save: checkpoint save skipped (eval-only run)", flush=True)
        return None

    _T.save_checkpoint = _noop_save_checkpoint
    print("[dlms] skip_save applied (save_checkpoint -> no-op for eval)", flush=True)
    return True


apply()
