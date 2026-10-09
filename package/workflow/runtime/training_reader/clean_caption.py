"""Deterministic training caption transforms for native editing inference.

Installed as src.data.dataset.craft_v1_edit_clean_caption by the runtime binder.
Run before tokenization/ROI alignment so every span refers to the final string.
"""
from dataclasses import replace
from functools import lru_cache


CAPTION_PREPARATION_VERSION = "dit44k-pe-order-clean-jfd-v1"


def require_caption_preparation_version(row: dict) -> None:
    if row.get("caption_preparation_version") != CAPTION_PREPARATION_VERSION:
        raise ValueError(
            "Stale native input: rebuild admission and native batches in a new "
            "work directory, then re-encode and regenerate; old bundles/results "
            "cannot be reused after caption normalization changes"
        )


@lru_cache(maxsize=1)
def _clean_processor():
    from src.data.dataset.craft_v1_image_edit_post_drop import craft_v1_edit_post_drop_processor

    # Keep the production recipe's structural palette scope and capitalization.
    # The independent palette policy can still drop colors when no_field_drop
    # fires, so explicitly disable it as well as every stochastic axis.
    return replace(
        craft_v1_edit_post_drop_processor(),
        no_field_drop_prob=1.0,
        high_level_drop_prob=0.0, style_drop_prob=0.0,
        palette_drop_prob=0.0, element_color_palette_drop_prob=0.0,
        element_bbox_drop_prob=0.0, bbox_element_drop_rules=(),
        element_max_depth=None, color_palette_item_policy=None,
        # Geometry is extracted by the native builder after normalization.
        strip_element_bbox=False,
    )


def clean_inference_caption(caption: str) -> str:
    from mariana.data.processors.base import Keys, ProcessorCtx

    sample = _clean_processor()(
        {Keys.CAPTION: caption}, ProcessorCtx(seed=0, epoch=0, sample_id=0),
    )
    if sample is None or sample.get(Keys.POST_DROP_REMOVED_SPANS):
        raise ValueError("Clean inference unexpectedly planned text dropout")
    return str(sample[Keys.CAPTION])
