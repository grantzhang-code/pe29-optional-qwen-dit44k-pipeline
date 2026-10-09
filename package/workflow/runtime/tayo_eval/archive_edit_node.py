#!/usr/bin/env python3
"""Archive verified native EMA images and a score-ready manifest immutably."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from tayo_eval.collect_captions import ROOT, get, maybe_get, parts, sha


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def upload(client, source: Path, uri: str) -> dict:
    bucket, key = parts(uri)
    size = source.stat().st_size
    digest = file_sha(source)
    try:
        previous = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) not in ("404", "NoSuchKey", "NotFound"):
            raise
        previous = None
    if previous is None:
        content_type = "image/png" if source.suffix == ".png" else ("application/json" if source.suffix == ".json" else "application/x-ndjson")
        with source.open("rb") as handle:
            try:
                client.put_object(Bucket=bucket, Key=key, Body=handle, ContentLength=size,
                                  ContentType=content_type, Metadata={"sha256": digest}, IfNoneMatch="*")
            except ClientError as exc:
                if str(exc.response.get("Error", {}).get("Code")) != "PreconditionFailed":
                    raise
        previous = client.head_object(Bucket=bucket, Key=key)
    if previous["ContentLength"] != size or previous.get("Metadata", {}).get("sha256") != digest:
        raise ValueError(f"remote archive differs: {uri}")
    return {"uri": uri, "bytes": size, "sha256": digest, "version_id": previous.get("VersionId")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--node-index", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--caption-step", type=int, required=True)
    p.add_argument("--edit-step", type=int, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    a = p.parse_args()
    complete = json.loads((a.work / "COMPLETE.json").read_text())
    plan = json.loads((a.work / "plan.json").read_text())
    if complete.get("status") != "PASS" or plan["step"] != a.edit_step or plan["weights"] != "EMA" or plan["cfg"] != [2, 3]:
        raise ValueError("native EMA output is not verified")
    plan_sha = file_sha(a.work / "plan.json")
    if complete["plan_sha256"] != plan_sha or complete["outputs"] != complete["samples"] * 2:
        raise ValueError("native plan/output count changed")
    manifest = [json.loads(line) for line in (a.work / "manifest.jsonl").read_text().splitlines() if line.strip()]
    ready = {r["request_id"]: r for r in manifest if r["status"] == "ready"}
    if len(manifest) != complete["source_samples"] or len(ready) != complete["samples"]:
        raise ValueError("native admission count changed")
    source = {r["request_id"]: r for line in a.source_manifest.read_bytes().splitlines() if (r := json.loads(line))}
    if len(source) != 1364:
        raise ValueError("frozen source manifest count changed")
    receipts = [json.loads(line) for line in (a.work / "results.jsonl").read_text().splitlines() if line.strip()]
    if len(receipts) != complete["outputs"] or {(r["cfg"], r["request_id"]) for r in receipts} != {(cfg, rid) for cfg in (2, 3) for rid in ready}:
        raise ValueError("native receipt coverage changed")
    client = boto3.client("s3", region_name="us-east-1")
    caption_marker = json.loads(get(client, ROOT + f"captions/checkpoint-{a.caption_step}/GLOBAL_COMPLETE.json"))
    prefix = ROOT + f"edit/caption-{a.caption_step}/node-{a.node_index}/_attempts/{plan_sha[:16]}/"
    files = [
        a.work / name for name in (
            "plan.json", "COMPLETE.json", "manifest.jsonl", "results.jsonl", "input-audit.json",
            "audit.json", "pe-native-annotations.json", "source-bindings.json", "predictions.jsonl",
            "training.json",
        )
    ]
    for r in receipts:
        relative = Path("results") / f"cfg{r['cfg']}" / r["request_id"]
        files.extend((a.work / relative.with_suffix(".png"), a.work / relative.with_suffix(".json")))
    for rid in ready:
        folder = a.work / "inputs" / rid
        files.extend(folder / name for name in ("annotation.json", "model-prompts.json", "input-audit.json"))
    if any(not path.is_file() or not path.resolve().is_relative_to(a.work.resolve()) for path in files):
        raise ValueError("archive input is missing or outside work root")

    def send(path: Path) -> dict:
        relative = path.relative_to(a.work).as_posix()
        return upload(client, path, prefix + relative)

    with ThreadPoolExecutor(max_workers=12) as pool:
        uploaded = list(pool.map(send, files))
    by_uri = {item["uri"]: item for item in uploaded}
    score_rows = []
    for receipt in receipts:
        rid = receipt["request_id"]
        original = source[rid]
        cfg = receipt["cfg"]
        image_uri = prefix + f"results/cfg{cfg}/{rid}.png"
        json_uri = prefix + f"results/cfg{cfg}/{rid}.json"
        if by_uri[image_uri]["sha256"] != receipt["image_sha256"]:
            raise ValueError(f"image SHA differs from native receipt: {rid} cfg{cfg}")
        score_rows.append({
            "request_id": rid, "ordinal": original["ordinal"], "original_mds_sample_id": original["sample_id"],
            "source": original["source"], "source_index": original["source_index"],
            "input_images": original["input_images"], "target_image": original["target_image"],
            "instructions": original["instructions"],
            "caption_checkpoint_step": a.caption_step,
            "caption_predictions_uri": caption_marker["predictions_uri"],
            "caption_predictions_sha256": caption_marker["predictions_sha256"],
            "predicted_native_annotation_uri": prefix + f"inputs/{rid}/annotation.json",
            "model_prompts_uri": prefix + f"inputs/{rid}/model-prompts.json",
            "edit_checkpoint_step": a.edit_step, "edit_checkpoint_uri": plan["checkpoint_uri"],
            "weights": "EMA", "cfg": cfg, "steps": 50, "seed": receipt["seed"],
            "plan_sha256": plan_sha, "prediction_uri": image_uri,
            "prediction_sha256": receipt["image_sha256"], "receipt_uri": json_uri,
        })
    score_rows.sort(key=lambda r: (r["ordinal"], r["cfg"]))
    score_path = a.work / "score-index.jsonl"
    score_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in score_rows))
    index = upload(client, score_path, prefix + "score-index.jsonl")
    marker = {
        "version": 1, "status": "complete", "node_index": a.node_index,
        "caption_checkpoint_step": a.caption_step, "edit_checkpoint_step": a.edit_step,
        "source_samples": len(manifest), "admitted_samples": len(ready),
        "rejections": len(manifest) - len(ready), "outputs": len(receipts),
        "excluded_multi_reference_rows": 200,
        "plan_sha256": plan_sha, "archive_prefix": prefix,
        "score_index_uri": index["uri"], "score_index_sha256": index["sha256"],
        "files_uploaded": len(uploaded) + 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_uri = ROOT + f"edit/caption-{a.caption_step}/node-{a.node_index}/COMPLETE.json"
    payload = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    old = maybe_get(client, marker_uri)
    if old is not None:
        existing = json.loads(old)
        for field in ("node_index", "caption_checkpoint_step", "edit_checkpoint_step", "source_samples", "admitted_samples", "rejections", "outputs", "plan_sha256", "archive_prefix", "score_index_uri", "score_index_sha256"):
            if existing.get(field) != marker[field]:
                raise ValueError(f"existing edit completion changed: {field}")
    else:
        bucket, key = parts(marker_uri)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json", IfNoneMatch="*")
    print("EDIT_ARCHIVE_COMPLETE", a.caption_step, a.node_index, len(receipts), flush=True)


if __name__ == "__main__":
    main()
