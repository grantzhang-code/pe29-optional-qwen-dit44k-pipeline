#!/usr/bin/env python3
"""Use the frozen native MDS processor with PE predictions, never GT captions."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

from tayo_protocol.code import prepare_inputs as native
from tayo_eval.adapt_pe_to_native import adapt


ORIGINAL_DATASET = native.LocalDataset
SOURCE_SHA256 = "dac6e2f3ff92bbccbf5b3038e0922a8c3632046820ac5aa1612bf92a84f4a0d4"
ROW_BINDINGS: dict[tuple[str, int], dict[str, Any]] = {}
ADAPTATIONS: dict[tuple[str, int], dict[str, Any]] = {}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def node_for_ordinal(ordinal: int) -> int:
    return (ordinal // 8) % 3


class PatchedLocalDataset:
    def __init__(self, *, local: str):
        self.source = Path(local).name
        self.backing = ORIGINAL_DATASET(local=local)

    def __getitem__(self, index: int) -> dict:
        key = (self.source, index)
        binding = ROW_BINDINGS[key]
        row = dict(self.backing[index])
        if row.get("sample_id") != binding["sample_id"] or row.get("num_inputs") != 1:
            raise ValueError(f"MDS row identity changed: {key}")
        if row.get("instructions") != binding["instructions"]:
            raise ValueError(f"raw instruction changed: {key}")
        references = native.decode_reference_images(row, max_images=12)
        if len(references) != 1 or digest(bytes(references[0])) != binding["input_images"][0]["sha256"]:
            raise ValueError(f"Source image changed: {key}")
        row["image_edit_v6_annotation"] = ADAPTATIONS[key]
        return row


def initialize(work: str, bindings_path: str, adaptations_path: str) -> None:
    global ROW_BINDINGS, ADAPTATIONS
    native.ROOT = Path(work)
    native.OP = Path(work)
    native.initialize()
    ROW_BINDINGS = {
        (r["source"], r["source_index"]): r
        for r in json.loads(Path(bindings_path).read_text())
    }
    ADAPTATIONS = {
        (r["source"], r["source_index"]): r["annotation"]
        for r in json.loads(Path(adaptations_path).read_text())
    }
    native.LocalDataset = PatchedLocalDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--node-index", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    raw = a.source_manifest.read_bytes()
    if digest(raw) != SOURCE_SHA256:
        raise ValueError("frozen Tayo source manifest changed")
    source = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(source) != 1364 or len({r["request_id"] for r in source}) != 1364:
        raise ValueError("Tayo test1364 manifest changed")
    singles = [r for r in source if r["num_inputs"] == 1 and r["status"] == "ready"]
    if len(singles) != 1164:
        raise ValueError("expected exactly 1164 ready single-reference rows")
    predictions = [json.loads(line) for line in a.predictions.read_bytes().splitlines() if line.strip()]
    by_ordinal = {r["input_index"]: r for r in predictions}
    if len(predictions) != 1164 or len(by_ordinal) != 1164 or set(by_ordinal) != {r["ordinal"] for r in singles}:
        raise ValueError("PE prediction cohort is incomplete or duplicated")
    selected = [r for r in singles if node_for_ordinal(r["ordinal"]) == a.node_index]
    if not selected:
        raise ValueError("empty node selection")
    bindings: list[dict] = []
    adaptations: list[dict] = []
    rejected: list[dict] = []
    jobs: list[tuple[int, str, int]] = []
    for row in selected:
        ordinal = row["ordinal"]
        prediction = by_ordinal[ordinal]
        if (
            prediction.get("sample_id") != row["request_id"]
            or prediction.get("source_s3_uri") != row["input_images"][0]["uri"]
            or prediction.get("instruction") != row["instructions"][0]
            or prediction.get("task_type") != "image_edit"
        ):
            raise ValueError(f"PE input binding changed for ordinal {ordinal}")
        binding = {
            key: row[key]
            for key in ("source", "source_index", "sample_id", "instructions", "input_images")
        }
        bindings.append(binding)
        reason = None
        if prediction.get("schema_valid") is not True or not isinstance(prediction.get("prediction"), dict):
            reason = "invalid_pe_schema"
        else:
            try:
                annotation = adapt(prediction["prediction"])
            except ValueError as exc:
                reason = f"unsupported_pe_annotation: {exc}"
        if reason is not None:
            rejected.append({
                "request_id": row["request_id"], "ordinal": ordinal,
                "source": row["source"], "source_index": row["source_index"],
                "generation_seed": row["generation_seed"], "num_inputs": 1,
                "status": "invalid_pe_caption", "reason": reason,
                "caption_checkpoint_step": a.checkpoint_step,
            })
            continue
        adaptations.append({"source": row["source"], "source_index": row["source_index"], "annotation": annotation})
        jobs.append((ordinal, row["source"], row["source_index"]))
    bindings_path = a.work / "source-bindings.json"
    adaptations_path = a.work / "pe-native-annotations.json"
    bindings_path.write_text(json.dumps(bindings, ensure_ascii=False) + "\n")
    adaptations_path.write_text(json.dumps(adaptations, ensure_ascii=False) + "\n")
    if not (a.work / "mds").is_dir():
        raise ValueError("verified native MDS cache is missing")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=initialize,
        initargs=(str(a.work), str(bindings_path), str(adaptations_path)),
    ) as pool:
        for result in pool.map(native.prepare, jobs):
            results.append(result)
            if len(results) % 50 == 0:
                print("NATIVE_PREPARED", len(results), flush=True)
    all_results = sorted([*results, *rejected], key=lambda r: r["ordinal"])
    if len(all_results) != len(selected) or len({r["request_id"] for r in all_results}) != len(selected):
        raise ValueError("native preparation coverage changed")
    (a.work / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_results))
    summary = {
        "node_index": a.node_index,
        "caption_checkpoint_step": a.checkpoint_step,
        "source_manifest_sha256": SOURCE_SHA256,
        "predictions_sha256": native.sha(a.predictions),
        "samples": len(selected),
        "counts": dict(Counter(r["status"] for r in all_results)),
        "rejections": [r for r in all_results if r["status"] != "ready"],
    }
    native.atomic(a.work / "input-audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rejections"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
