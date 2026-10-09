#!/usr/bin/env python3
"""Verify all 8 native edit archives and publish four score-ready indexes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time

import boto3

from pcb_edit.s3io import ROOT, STEPS, get, maybe_get, put_immutable, sha


def request_id(row: dict) -> str:
    index = row["input_index"]
    return f"{index:04d}-" + sha(("pcbv2:" + row["sample_id"]).encode())[:16]


def gather(client, step: int, edit_step: int, expected: dict[str, dict]) -> dict | None:
    root = ROOT + f"edit/caption-{step}/"
    node_ids = (2 * STEPS.index(step), 2 * STEPS.index(step) + 1)
    markers = []
    for node in node_ids:
        raw = maybe_get(client, root + f"node-{node}/COMPLETE.json")
        if raw is None:
            return None
        marker = json.loads(raw)
        if (marker.get("status"), marker.get("node_index"), marker.get("caption_checkpoint_step"),
                marker.get("edit_checkpoint_step"), marker.get("source_samples")) != (
                "complete", node, step, edit_step, 732):
            raise ValueError(f"edit node completion changed: checkpoint {step} node {node}")
        markers.append(marker)
    samples = []
    score_rows = []
    for marker in markers:
        raw = get(client, marker["score_index_uri"])
        if sha(raw) != marker["score_index_sha256"]:
            raise ValueError(f"score index hash mismatch: checkpoint {step}")
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if len(rows) != marker["outputs"]:
            raise ValueError(f"score index count mismatch: checkpoint {step}")
        score_rows.extend(rows)
        raw = get(client, marker["archive_prefix"] + "manifest.jsonl")
        cohort = [json.loads(line) for line in raw.splitlines() if line]
        if len(cohort) != marker["source_samples"]:
            raise ValueError(f"native manifest count mismatch: checkpoint {step}")
        if any(row["ordinal"] % 2 != marker["node_index"] % 2 for row in cohort):
            raise ValueError(f"native node assignment mismatch: checkpoint {step}")
        if sum(row["status"] == "ready" for row in cohort) != marker["admitted_samples"]:
            raise ValueError(f"native admission count mismatch: checkpoint {step}")
        samples.extend(cohort)
    by_id = {row["request_id"]: row for row in samples}
    if len(samples) != 1464 or len(by_id) != 1464 or set(by_id) != set(expected):
        raise ValueError(f"native sample coverage changed: checkpoint {step}")
    if any(row["ordinal"] != expected[rid]["input_index"] or
           row["original_mds_sample_id"] != expected[rid]["sample_id"]
           for rid, row in by_id.items()):
        raise ValueError(f"native source binding changed: checkpoint {step}")
    admitted = {row["request_id"] for row in samples if row["status"] == "ready"}
    actual = {(row["request_id"], row["cfg"]) for row in score_rows}
    if len(score_rows) != 2 * len(admitted) or len(actual) != len(score_rows) or actual != {
        (rid, cfg) for rid in admitted for cfg in (2, 3)
    }:
        raise ValueError(f"edit output coverage changed: checkpoint {step}")
    for row in score_rows:
        original = expected[row["request_id"]]
        selection = original["selection"]
        if (row["input_index"], row["sample_id"], row["instruction_en"],
                row["original_source_s3_uri"], row["original_gpt_image_2_5_output_sha256"],
                row["original_gpt_image_2_5_output_size"], row["pe_target_canvas_size"],
                row["edit_checkpoint_step"], row["caption_checkpoint_step"], row["weights"]) != (
                original["input_index"], original["sample_id"], original["instruction"],
                original["source_s3_uri"], selection["gpt_output_sha256"],
                [selection["gpt_output_width"], selection["gpt_output_height"]],
                original["target_canvas_size"], edit_step, step, "EMA"):
            raise ValueError(f"score row/source binding changed: checkpoint {step}")
    score_rows.sort(key=lambda row: (row["input_index"], row["cfg"]))
    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                       for row in score_rows)
    index_uri = root + "score-index.jsonl"
    put_immutable(client, index_uri, payload, "application/x-ndjson")
    marker = {
        "version": 1, "status": "complete", "caption_checkpoint_step": step,
        "edit_checkpoint_step": edit_step, "source_samples": 1464,
        "admitted_samples": len(admitted), "outputs": len(score_rows),
        "rejections": dict(Counter(row["status"] for row in samples if row["status"] != "ready")),
        "skipped_gpt_image_2_5_failures": 7,
        "score_index_uri": index_uri, "score_index_sha256": sha(payload),
        "node_markers": [root + f"node-{node}/COMPLETE.json" for node in node_ids],
    }
    put_immutable(client, root + "GLOBAL_COMPLETE.json",
                  (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode())
    print("EDIT_GLOBAL_COMPLETE", step, len(admitted), len(score_rows), flush=True)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    client = boto3.client("s3", region_name="us-east-1")
    edit_lock = json.loads(get(client, ROOT + "edit/edit-checkpoint.lock.json"))
    if edit_lock.get("weights") != "EMA" or edit_lock.get("step") != 54000:
        raise ValueError("edit checkpoint lock changed")
    input_bytes = get(client, ROOT + "inputs/all.jsonl")
    input_lock = json.loads(get(client, ROOT + "inputs/manifest.lock.json"))
    if sha(input_bytes) != input_lock["input_sha256"]:
        raise ValueError("PE input snapshot changed")
    source = [json.loads(line) for line in input_bytes.splitlines() if line]
    expected = {request_id(row): row for row in source}
    if len(source) != 1464 or len(expected) != 1464:
        raise ValueError("PE input cohort changed")
    deadline = time.monotonic() + args.wait_seconds
    completed = {}
    while len(completed) != len(STEPS):
        for step in STEPS:
            if step not in completed:
                result = gather(client, step, edit_lock["step"], expected)
                if result is not None:
                    completed[step] = result
        if len(completed) == len(STEPS):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"edit completion timed out; complete={sorted(completed)}")
        time.sleep(30)
    marker = {
        "version": 1, "status": "complete", "edit_checkpoint_step": edit_lock["step"],
        "weights": "EMA", "caption_steps": list(STEPS), "source_samples_per_caption": 1464,
        "skipped_gpt_image_2_5_failures": 7,
        "total_outputs": sum(completed[step]["outputs"] for step in STEPS),
        "global_markers": [ROOT + f"edit/caption-{step}/GLOBAL_COMPLETE.json" for step in STEPS],
    }
    put_immutable(client, ROOT + "edit/ALL_FOUR_COMPLETE.json",
                  (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode())
    print("EDIT_ALL_FOUR_COMPLETE", marker["total_outputs"], flush=True)


if __name__ == "__main__":
    main()
