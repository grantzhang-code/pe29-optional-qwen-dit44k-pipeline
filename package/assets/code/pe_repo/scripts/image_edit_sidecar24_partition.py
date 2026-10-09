#!/usr/bin/env python3
"""Deterministically split a frozen sidecar24 unit catalog into data parts.

The immutable planner catalog remains the source of truth.  A part owns whole
planner units by ``global_unit_index % part_count``.  Within one invocation,
part-local unit ordinal assigns units to GPU ranks.  This second ordinal is
important: assigning ranks from ``global_unit_index`` would send every unit of
part 0-of-16 to rank 0 in an eight-rank job.

``case_limit`` is a global (not per-rank) canary limit.  It is water-filled
across ranks and each rank takes deterministic prefixes of its own units.  No
planner unit is ever shared by two ranks, so even a clipped canary has no
duplicate source row or output shard identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


PARTITION_FORMAT = "image_edit_sidecar24_partition_v1"
PARTITION_ASSIGNMENT = "global_unit_index_modulo_partition_count"
RANK_ASSIGNMENT = "partition_unit_ordinal_modulo_global_world_size"
LIMIT_ASSIGNMENT = "waterfilled_rank_prefix_without_unit_sharing"


@dataclass(frozen=True)
class PartitionPlan:
    """Fully resolved selection and rank assignment for one invocation."""

    contract: dict[str, Any]
    units_by_rank: tuple[tuple[dict[str, Any], ...], ...]

    @property
    def units(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(unit) for values in self.units_by_rank for unit in values),
            key=lambda unit: int(unit["global_unit_index"]),
        )


def _unit_rows(unit: Mapping[str, Any]) -> int:
    start = unit.get("global_row_start")
    end = unit.get("global_row_end_exclusive")
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError("partition input unit has an invalid global row interval")
    dataset_start = unit.get("dataset_row_start")
    dataset_end = unit.get("dataset_row_end_exclusive")
    if (
        type(dataset_start) is not int
        or type(dataset_end) is not int
        or dataset_end - dataset_start != end - start
    ):
        raise ValueError("partition input unit dataset/global row intervals differ")
    return end - start


def _prefix_ranges(ranges: Any, rows: int) -> list[list[int]]:
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("partition input unit has no local row ranges")
    remaining = rows
    selected: list[list[int]] = []
    for raw_range in ranges:
        if (
            not isinstance(raw_range, list)
            or len(raw_range) != 2
            or not all(type(value) is int for value in raw_range)
        ):
            raise ValueError("partition input unit has an invalid local row range")
        start, end = raw_range
        if start < 0 or end <= start:
            raise ValueError("partition input unit has an empty local row range")
        if remaining <= 0:
            break
        take = min(remaining, end - start)
        selected.append([start, start + take])
        remaining -= take
    if remaining:
        raise ValueError("partition prefix exceeds unit local row ranges")
    return selected


def clip_unit_prefix(unit: Mapping[str, Any], rows: int) -> dict[str, Any]:
    """Return the first *rows* selected rows while preserving source identity."""

    total = _unit_rows(unit)
    if type(rows) is not int or rows <= 0 or rows > total:
        raise ValueError(f"unit prefix rows must be in [1, {total}], got {rows!r}")
    value = dict(unit)
    if rows == total:
        return value
    value["local_row_ranges"] = _prefix_ranges(unit.get("local_row_ranges"), rows)
    value["dataset_row_end_exclusive"] = int(value["dataset_row_start"]) + rows
    value["global_row_end_exclusive"] = int(value["global_row_start"]) + rows
    return value


def _waterfill_targets(capacities: Sequence[int], total: int) -> list[int]:
    if type(total) is not int or total < 0 or total > sum(capacities):
        raise ValueError("global case limit is outside partition capacity")
    if not capacities:
        if total:
            raise ValueError("non-zero case limit has no ranks")
        return []
    if any(type(capacity) is not int or capacity < 0 for capacity in capacities):
        raise ValueError("rank capacities must be non-negative integers")

    low, high = 0, max(capacities, default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(capacity, middle) for capacity in capacities) <= total:
            low = middle
        else:
            high = middle - 1
    targets = [min(capacity, low) for capacity in capacities]
    remaining = total - sum(targets)
    for rank, capacity in enumerate(capacities):
        if remaining == 0:
            break
        if targets[rank] < capacity:
            targets[rank] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("water-fill target allocation did not reach requested total")
    return targets


def _take_lane_prefix(units: Sequence[Mapping[str, Any]], rows: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = rows
    for unit in units:
        if remaining == 0:
            break
        count = min(_unit_rows(unit), remaining)
        selected.append(clip_unit_prefix(unit, count))
        remaining -= count
    if remaining:
        raise AssertionError("rank lane prefix did not reach its allocated row target")
    return selected


def _selection_digest(units_by_rank: Sequence[Sequence[Mapping[str, Any]]]) -> str:
    payload = [
        {
            "rank": rank,
            "units": [
                {
                    "global_unit_index": unit["global_unit_index"],
                    "dataset_unit_index": unit["dataset_unit_index"],
                    "dataset_row_start": unit["dataset_row_start"],
                    "dataset_row_end_exclusive": unit["dataset_row_end_exclusive"],
                    "global_row_start": unit["global_row_start"],
                    "global_row_end_exclusive": unit["global_row_end_exclusive"],
                    "local_row_ranges": unit["local_row_ranges"],
                }
                for unit in units
            ],
        }
        for rank, units in enumerate(units_by_rank)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def basic_contract(
    *,
    partition_count: int,
    partition_index: int,
    global_world_size: int,
    case_limit: int | None,
) -> dict[str, Any]:
    if type(partition_count) is not int or partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if type(partition_index) is not int or not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    if type(global_world_size) is not int or global_world_size <= 0:
        raise ValueError("global_world_size must be positive")
    if case_limit is not None and (type(case_limit) is not int or case_limit <= 0):
        raise ValueError("case_limit must be positive when set")
    return {
        "format_version": PARTITION_FORMAT,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "partition_assignment": PARTITION_ASSIGNMENT,
        "global_world_size": global_world_size,
        "rank_assignment": RANK_ASSIGNMENT,
        "case_limit": case_limit,
        "case_limit_assignment": LIMIT_ASSIGNMENT if case_limit is not None else None,
    }


def build_partition_plan(
    units: Sequence[Mapping[str, Any]],
    *,
    partition_count: int,
    partition_index: int,
    global_world_size: int,
    case_limit: int | None = None,
) -> PartitionPlan:
    """Resolve one mutually exclusive part and its no-overlap rank lanes."""

    base = basic_contract(
        partition_count=partition_count,
        partition_index=partition_index,
        global_world_size=global_world_size,
        case_limit=case_limit,
    )
    previous_index: int | None = None
    partition_units: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    all_partition_unit_counts = [0] * partition_count
    all_partition_row_counts = [0] * partition_count
    for raw_unit in units:
        index = raw_unit.get("global_unit_index")
        if type(index) is not int or index < 0:
            raise ValueError("partition input has an invalid global_unit_index")
        if previous_index is not None and index <= previous_index:
            raise ValueError("partition input units must be strictly ordered by global_unit_index")
        previous_index = index
        rows = _unit_rows(raw_unit)
        owner = index % partition_count
        all_partition_unit_counts[owner] += 1
        all_partition_row_counts[owner] += rows
        if owner == partition_index:
            if index in seen_indices:
                raise ValueError("partition input contains duplicate global_unit_index")
            seen_indices.add(index)
            partition_units.append(dict(raw_unit))

    lanes: list[list[dict[str, Any]]] = [
        partition_units[rank::global_world_size] for rank in range(global_world_size)
    ]
    base_rank_rows = [sum(_unit_rows(unit) for unit in lane) for lane in lanes]
    base_rows = sum(base_rank_rows)
    if not partition_units or base_rows <= 0:
        raise ValueError("selected partition contains no rows")
    if case_limit is not None:
        if case_limit > base_rows:
            raise ValueError(
                f"case_limit={case_limit} exceeds selected partition rows={base_rows}"
            )
        targets = _waterfill_targets(base_rank_rows, case_limit)
        lanes = [_take_lane_prefix(lane, target) for lane, target in zip(lanes, targets, strict=True)]

    selected_rows_by_rank = [sum(_unit_rows(unit) for unit in lane) for lane in lanes]
    selected_units_by_rank = [len(lane) for lane in lanes]
    selected_rows = sum(selected_rows_by_rank)
    expected_rows = case_limit if case_limit is not None else base_rows
    if selected_rows != expected_rows:
        raise AssertionError(
            f"partition selection rows={selected_rows}, expected={expected_rows}"
        )
    selected_indices = [
        int(unit["global_unit_index"]) for lane in lanes for unit in lane
    ]
    if len(selected_indices) != len(set(selected_indices)):
        raise AssertionError("one planner unit was assigned to multiple ranks")

    contract = {
        **base,
        "partition_units": len(partition_units),
        "partition_rows": base_rows,
        "catalog_units": sum(all_partition_unit_counts),
        "catalog_rows": sum(all_partition_row_counts),
        "all_partition_unit_counts": all_partition_unit_counts,
        "all_partition_row_counts": all_partition_row_counts,
        "selected_units": sum(selected_units_by_rank),
        "selected_rows": selected_rows,
        "rank_unit_counts": selected_units_by_rank,
        "rank_row_counts": selected_rows_by_rank,
        "selection_sha256": _selection_digest(lanes),
    }
    return PartitionPlan(
        contract=contract,
        units_by_rank=tuple(tuple(dict(unit) for unit in lane) for lane in lanes),
    )


def selection_output_root(output_s3: str, contract: Mapping[str, Any]) -> str:
    """Namespace parts and canaries so success markers can never collide."""

    root = output_s3.rstrip("/")
    count = int(contract["partition_count"])
    index = int(contract["partition_index"])
    case_limit = contract.get("case_limit")
    if count == 1 and index == 0 and case_limit is None:
        return root
    width = max(2, len(str(count - 1)), len(str(count)))
    part = f"part-{index:0{width}d}-of-{count:0{width}d}"
    mode = "full" if case_limit is None else f"limit-{int(case_limit):012d}"
    return f"{root}/partitions/{part}/{mode}"
