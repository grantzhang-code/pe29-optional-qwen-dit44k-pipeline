#!/usr/bin/env python3
"""Verify all 24 PE ranks and freeze one ordered prediction file per ckpt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from tayo_eval.run_caption_node import CHECKPOINTS, ROOT


def parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get(client, uri: str) -> bytes:
    bucket, key = parts(uri)
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def maybe_get(client, uri: str) -> bytes | None:
    try:
        return get(client, uri)
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def put_immutable(client, uri: str, payload: bytes) -> None:
    old = maybe_get(client, uri)
    if old is not None:
        if old != payload:
            raise ValueError(f"existing immutable output differs: {uri}")
        return
    bucket, key = parts(uri)
    client.put_object(Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_step(client, step: int, inputs: dict[int, dict]) -> dict | None:
    base = ROOT + f"captions/checkpoint-{step}/"
    markers = []
    for node in range(3):
        raw = maybe_get(client, base + f"node-{node}/COMPLETE.json")
        if raw is None:
            return None
        marker = json.loads(raw)
        if marker.get("status") != "complete" or marker.get("checkpoint_step") != step or marker.get("checkpoint_manifest_sha256") != CHECKPOINTS[step] or marker.get("node_index") != node:
            raise ValueError(f"node-{node} completion marker changed")
        markers.append(marker)
    ranks = [r for marker in markers for r in marker["rank_results"]]
    if sorted(r["rank"] for r in ranks) != list(range(24)) or sum(r["rows"] for r in ranks) != 1164:
        raise ValueError(f"caption rank coverage changed for checkpoint {step}")
    rows = []
    for rank in ranks:
        raw = get(client, rank["predictions_uri"])
        if sha(raw) != rank["predictions_sha256"]:
            raise ValueError(f"caption rank SHA mismatch: {rank['rank']}")
        shard = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(shard) != rank["rows"]:
            raise ValueError(f"caption rank length mismatch: {rank['rank']}")
        for row in shard:
            ordinal = row["input_index"]
            expected = inputs.get(ordinal)
            if expected is None or ordinal % 24 != rank["rank"]:
                raise ValueError(f"caption rank assignment changed: {ordinal}")
            for field in ("sample_id", "source_s3_uri", "instruction", "target_canvas_size"):
                if row.get(field) != expected[field]:
                    raise ValueError(f"caption input {field} changed: {ordinal}")
        rows.extend(shard)
    if len(rows) != 1164 or len({r["input_index"] for r in rows}) != 1164 or set(r["input_index"] for r in rows) != set(inputs):
        raise ValueError(f"caption global coverage changed: {step}")
    rows.sort(key=lambda r: r["input_index"])
    payload = b"".join((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for r in rows)
    out_uri = base + "predictions.jsonl"
    put_immutable(client, out_uri, payload)
    complete = {
        "version": 1, "status": "complete", "checkpoint_step": step,
        "checkpoint_manifest_sha256": CHECKPOINTS[step],
        "rows": len(rows), "schema_valid": sum(r.get("schema_valid") is True for r in rows),
        "predictions_uri": out_uri, "predictions_sha256": sha(payload),
        "node_markers": [base + f"node-{node}/COMPLETE.json" for node in range(3)],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_uri = base + "GLOBAL_COMPLETE.json"
    old = maybe_get(client, marker_uri)
    if old is not None:
        existing = json.loads(old)
        for field in ("checkpoint_step", "checkpoint_manifest_sha256", "rows", "schema_valid", "predictions_uri", "predictions_sha256", "node_markers"):
            if existing.get(field) != complete[field]:
                raise ValueError(f"global completion changed: {step} {field}")
        return existing
    put_immutable(client, marker_uri, (json.dumps(complete, indent=2, sort_keys=True) + "\n").encode())
    print("CAPTION_GLOBAL_COMPLETE", step, complete["schema_valid"], flush=True)
    return complete


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait-seconds", type=int, default=86400)
    a = p.parse_args()
    client = boto3.client("s3", region_name="us-east-1")
    raw = get(client, ROOT + "inputs/caption-inputs.jsonl")
    lock = json.loads(get(client, ROOT + "inputs/caption-inputs.lock.json"))
    if sha(raw) != lock["caption_inputs_sha256"]:
        raise ValueError("caption input freeze changed")
    inputs = {r["input_index"]: r for line in raw.splitlines() if (r := json.loads(line))}
    if len(inputs) != 1164:
        raise ValueError("caption input cardinality changed")
    deadline = time.monotonic() + a.wait_seconds
    completed = {}
    while len(completed) < len(CHECKPOINTS):
        for step in sorted(CHECKPOINTS, reverse=True):
            if step not in completed:
                marker = collect_step(client, step, inputs)
                if marker is not None:
                    completed[step] = marker
        if len(completed) == len(CHECKPOINTS):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"caption barrier timed out; completed={sorted(completed)}")
        time.sleep(30)
    overall = {
        "version": 1, "status": "complete", "checkpoints": {str(k): completed[k] for k in sorted(completed)},
        "rows_per_checkpoint": 1164, "total_predictions": 1164 * len(CHECKPOINTS),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_uri = ROOT + "captions/ALL_SIX_COMPLETE.json"
    old = maybe_get(client, out_uri)
    if old is None:
        put_immutable(client, out_uri, (json.dumps(overall, indent=2, sort_keys=True) + "\n").encode())
    elif {k: v for k, v in json.loads(old).items() if k != "completed_utc"} != {k: v for k, v in overall.items() if k != "completed_utc"}:
        raise ValueError("existing six-checkpoint barrier changed")
    print("ALL_SIX_COMPLETE", overall["total_predictions"], flush=True)


if __name__ == "__main__":
    main()
