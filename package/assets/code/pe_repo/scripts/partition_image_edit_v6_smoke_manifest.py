#!/usr/bin/env python3
"""Split one frozen V6 smoke manifest into near-balanced immutable node parts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = "image_edit_v6_multinode_smoke_parts_v1"


def file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_identity(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_rows(path: Path, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for physical_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {physical_line} is not an object")
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"line {physical_line} has no sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"input row count mismatch: actual={len(rows)} expected={expected_count}")
    return rows


def partition_manifest(
    input_jsonl: Path,
    output_dir: Path,
    *,
    expected_count: int,
    part_count: int,
    input_uri: str,
) -> dict[str, Any]:
    if expected_count <= 0 or part_count <= 0:
        raise ValueError("expected_count and part_count must be positive")
    input_bytes, input_sha256 = file_identity(input_jsonl)
    rows = load_rows(input_jsonl, expected_count)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(part_count)]
    for global_index, original in enumerate(rows):
        part_index = global_index % part_count
        row = dict(original)
        source = row.get("dataset_source")
        if source is not None and not isinstance(source, Mapping):
            raise ValueError(f"row {global_index}: dataset_source is not an object")
        row["dataset_source"] = {
            **dict(source or {}),
            "smoke_full_manifest_sha256": input_sha256,
            "smoke_global_input_index": global_index,
            "smoke_partition_index": part_index,
            "smoke_partition_count": part_count,
        }
        partitions[part_index].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    part_records: list[dict[str, Any]] = []
    minimum_part_rows, larger_part_count = divmod(expected_count, part_count)
    for part_index, part_rows in enumerate(partitions):
        expected_part_rows = minimum_part_rows + int(part_index < larger_part_count)
        if len(part_rows) != expected_part_rows:
            raise AssertionError("round-robin partitioning produced an imbalanced part")
        filename = f"part-{part_index:03d}-of-{part_count:03d}.jsonl"
        path = output_dir / filename
        size, digest = atomic_write_jsonl(path, part_rows)
        statuses = Counter(
            str(row.get("dataset_source", {}).get("training_status") or "")
            for row in part_rows
        )
        part_records.append(
            {
                "part_index": part_index,
                "part_count": part_count,
                "filename": filename,
                "rows": len(part_rows),
                "bytes": size,
                "sha256": digest,
                "status_counts": dict(sorted(statuses.items())),
                "sample_ids_sha256": hashlib.sha256(
                    ("\n".join(str(row["sample_id"]) for row in part_rows) + "\n").encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    index = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "partition_method": "global_input_index_mod_part_count",
        "input": {
            "uri": input_uri,
            "path": str(input_jsonl),
            "rows": expected_count,
            "bytes": input_bytes,
            "sha256": input_sha256,
        },
        "part_count": part_count,
        "rows_per_part": (
            minimum_part_rows if larger_part_count == 0 else None
        ),
        "minimum_rows_per_part": minimum_part_rows,
        "maximum_rows_per_part": minimum_part_rows + int(larger_part_count > 0),
        "larger_part_count": larger_part_count,
        "parts": part_records,
    }
    atomic_write_json(output_dir / "index.json", index)
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--part-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index = partition_manifest(
        args.input_jsonl,
        args.output_dir,
        expected_count=args.expected_count,
        part_count=args.part_count,
        input_uri=args.input_uri,
    )
    print(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
