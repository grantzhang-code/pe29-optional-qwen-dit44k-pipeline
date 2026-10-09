"""Fix checkpoint save crash: `_validate_global_plan` must return a list, not True.

On torch 2.13 + ms-swift 4.4.2, saving an mcore dist-checkpoint crashes with
`TypeError: can only join an iterable` at
torch/distributed/checkpoint/default_planner.py:
    validation_errors = _validate_global_plan(global_plan, metadata)
    if validation_errors:
        error_summary = "; ".join(validation_errors)   # <- join(True) -> TypeError

ms-swift ships a patch (swift/megatron/init.py) that overrides `_validate_global_plan`
to `return True`, but the call site joins it as an iterable, so `True` crashes. The
correct sentinel for "no validation errors" is an empty list `[]` (this is the
agent-sft runbook fix). We re-patch it to return `[]` (must run AFTER swift.megatron
import, which custom_register_path does).

Inject via `--custom_register_path .../dlms/patches/fix_dist_ckpt_validate.py`.
"""
from __future__ import annotations


def apply() -> bool:
    try:
        from torch.distributed.checkpoint import default_planner
    except Exception as e:
        print(f"[dlms] dist-ckpt validate fix: torch dcp not importable ({e})")
        return False

    def _validate_global_plan(*args, **kwargs):
        return []  # no validation errors; must be an (empty) iterable, NOT True

    default_planner._validate_global_plan = _validate_global_plan
    print("[dlms] dist-ckpt validate fix applied (_validate_global_plan -> [] so checkpoint save doesn't crash)", flush=True)
    return True


apply()
