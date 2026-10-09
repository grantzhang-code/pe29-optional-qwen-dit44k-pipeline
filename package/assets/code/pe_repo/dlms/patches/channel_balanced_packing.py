"""Require every streaming packed row to contain the baseline4 task channels.

ms-swift 4.4 keeps the source ``channel`` on encoded examples, but its public
packing arguments do not include the reference fork's
``packing_required_channels`` option.  This module patches only the grouping
function used by ``IterablePackingDataset``.  The ordinary implementation is
left in place for non-channel data and for unavoidable final overflow.

The patch is applied by :mod:`dlms.datasets.craft_mixed4`; it can also be
applied explicitly from another custom-register plugin.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any


REQUIRED_CHANNELS = (
    "global_desc_bg",
    "full_json_no_region_desc",
    "full_json",
    "layout_only",
    "region_layout",
)

PackedItem = tuple[dict[str, Any], int]
Fallback = Callable[[list[PackedItem], int, bool], tuple[list[list[PackedItem]], list[PackedItem]]]


def _validate_required_channels(required_channels: Sequence[str]) -> tuple[str, ...]:
    channels = tuple(required_channels)
    if not channels or any(not channel for channel in channels):
        raise ValueError("required_channels must contain non-empty names")
    if len(set(channels)) != len(channels):
        raise ValueError(f"required_channels must be unique, got {channels!r}")
    return channels


def _pack_required_channels(
    sequences: list[PackedItem],
    packing_length: int,
    required_channels: Sequence[str],
) -> tuple[list[list[PackedItem]], list[PackedItem]]:
    """Best-fit-decreasing packing, seeded once from every required channel."""
    required_channels = _validate_required_channels(required_channels)
    channel_counts = Counter(sample.get("channel") for sample, _ in sequences)
    max_mixed_bins = min((channel_counts[channel] for channel in required_channels), default=0)
    if max_mixed_bins == 0:
        return [], list(sequences)

    total_length = sum(length for _, length in sequences)
    target_bins = min(max(math.ceil(total_length / packing_length), 1), max_mixed_bins)

    def try_pack(bin_count: int) -> tuple[list[list[PackedItem]], list[PackedItem]]:
        bins: list[list[PackedItem]] = [[] for _ in range(bin_count)]
        bin_lengths = [0] * bin_count
        used_indices: set[int] = set()

        # The shortest examples are the safest seeds.  Distribute the longest
        # selected seed to the lightest bin to keep the mandatory load even.
        for channel in required_channels:
            channel_items = sorted(
                ((index, item) for index, item in enumerate(sequences) if item[0].get("channel") == channel),
                key=lambda value: value[1][1],
            )[:bin_count]
            lightest_bins = sorted(range(bin_count), key=lambda value: bin_lengths[value])
            for (index, item), bin_index in zip(reversed(channel_items), lightest_bins):
                bins[bin_index].append(item)
                bin_lengths[bin_index] += item[1]
                used_indices.add(index)

        if any(length > packing_length for length in bin_lengths):
            return [], list(sequences)

        remaining = sorted(
            (item for index, item in enumerate(sequences) if index not in used_indices),
            key=lambda item: item[1],
            reverse=True,
        )
        overflow: list[PackedItem] = []
        for item in remaining:
            candidates = [
                index for index, length in enumerate(bin_lengths) if length + item[1] <= packing_length
            ]
            if not candidates:
                overflow.append(item)
                continue
            bin_index = max(candidates, key=lambda value: bin_lengths[value])
            bins[bin_index].append(item)
            bin_lengths[bin_index] += item[1]
        return bins, overflow

    while True:
        bins, overflow = try_pack(target_bins)
        if not overflow or target_bins >= max_mixed_bins:
            return bins, overflow
        target_bins += 1


def calculate_channel_balanced_groups(
    sequences: list[PackedItem],
    packing_length: int,
    *,
    is_finished: bool = True,
    required_channels: Sequence[str] = REQUIRED_CHANNELS,
    fallback: Fallback | None = None,
) -> tuple[list[list[PackedItem]], list[PackedItem]]:
    """Group encoded samples while preserving one sample from every channel.

    While the stream is open, the least-filled valid bin and all overflow are
    retained for the next packing interval.  At end-of-stream only, samples
    that cannot meet the constraint are delegated to ms-swift's ordinary
    packer so finite validation streams are not silently dropped.
    """
    if not sequences:
        return [], []
    required_channels = _validate_required_channels(required_channels)
    mixed_bins, overflow = _pack_required_channels(sequences, packing_length, required_channels)

    if not mixed_bins:
        if is_finished and fallback is not None:
            warnings.warn(
                "Final samples cannot satisfy all required packing channels; using ordinary packing.",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback(sequences, packing_length, True)
        return [], list(sequences)

    if is_finished:
        if overflow and fallback is not None:
            warnings.warn(
                "Final packing overflow cannot satisfy all required channels; using ordinary packing for it.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback_bins, _ = fallback(overflow, packing_length, True)
            mixed_bins.extend(fallback_bins)
            overflow = []
        return mixed_bins, overflow

    mixed_bins.sort(key=lambda row: sum(item[1] for item in row), reverse=True)
    retained = list(overflow)
    retained.extend(mixed_bins.pop())
    return mixed_bins, retained


def apply(required_channels: Sequence[str] = REQUIRED_CHANNELS) -> bool:
    """Patch ms-swift 4.4's grouping function; safe to call repeatedly."""
    required_channels = _validate_required_channels(required_channels)
    from swift.dataset import packing as swift_packing

    current = swift_packing.calculate_matched_group
    original = getattr(current, "_dlms_channel_balanced_original", current)

    def patched_calculate_matched_group(
        sequences,
        packing_length: int,
        is_finished: bool = True,
        *args,
        **kwargs,
    ):
        # Newer 4.4.x releases add grouping controls such as ``strategy``.
        # Channel-balanced grouping owns the mixed-row policy, while every
        # stock/fallback path must preserve all extra positional/keyword args.
        # Map-style packing passes integer dataset indexes rather than encoded
        # dictionaries.  It must keep the stock behavior.
        channel_aware = bool(sequences) and all(
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], dict)
            and item[0].get("channel") is not None
            for item in sequences
        )
        if not channel_aware:
            return original(
                sequences,
                packing_length,
                *args,
                is_finished=is_finished,
                **kwargs,
            )

        def fallback(items, length, finished):
            return original(items, length, *args, is_finished=finished, **kwargs)

        return calculate_channel_balanced_groups(
            list(sequences),
            packing_length,
            is_finished=is_finished,
            required_channels=required_channels,
            fallback=fallback,
        )

    patched_calculate_matched_group._dlms_channel_balanced_original = original
    patched_calculate_matched_group._dlms_required_channels = required_channels
    swift_packing.calculate_matched_group = patched_calculate_matched_group

    # Keep the package re-export coherent for diagnostics.  The packing
    # classes themselves resolve the function from swift.dataset.packing.
    import swift.dataset

    if hasattr(swift.dataset, "calculate_matched_group"):
        swift.dataset.calculate_matched_group = patched_calculate_matched_group
    print(f"[dlms] channel-balanced packing enabled: {','.join(required_channels)}", flush=True)
    return True
