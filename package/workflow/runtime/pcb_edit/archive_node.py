#!/usr/bin/env python3
"""Archive native edit images, inputs, and exact caption/source bindings."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from pcb_edit.s3io import ROOT, STEPS, get, maybe_get, parts, sha


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def upload(client, source: Path, uri: str) -> dict:
    bucket, key = parts(uri)
    size, digest = source.stat().st_size, file_sha(source)
    try:
        previous = client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in ("404", "NoSuchKey", "NotFound"):
            raise
        previous = None
    if previous is None:
        content_type = ("image/png" if source.suffix in (".png", ".image") else
                        "application/json" if source.suffix == ".json" else "application/x-ndjson")
        with source.open("rb") as handle:
            try:
                client.put_object(Bucket=bucket, Key=key, Body=handle, ContentLength=size,
                                  ContentType=content_type, Metadata={"sha256": digest}, IfNoneMatch="*")
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) != "PreconditionFailed":
                    raise
        previous = client.head_object(Bucket=bucket, Key=key)
    if previous["ContentLength"] != size or previous.get("Metadata", {}).get("sha256") != digest:
        raise ValueError(f"remote archive differs: {uri}")
    return {"uri": uri, "bytes": size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(8), required=True)
    parser.add_argument("--caption-step", type=int, choices=STEPS, required=True)
    parser.add_argument("--edit-step", type=int, required=True)
    args = parser.parse_args()
    if args.caption_step != STEPS[args.node_index // 2]:
        raise ValueError("caption checkpoint/node assignment changed")
    complete = json.loads((args.work / "COMPLETE.json").read_bytes())
    plan = json.loads((args.work / "plan.json").read_bytes())
    plan_sha = file_sha(args.work / "plan.json")
    if (complete.get("status"), complete.get("plan_sha256"), complete.get("outputs"),
            plan.get("step"), plan.get("weights"), plan.get("cfg")) != (
            "PASS", plan_sha, complete.get("samples", -1) * 2, args.edit_step, "EMA", [2, 3]):
        raise ValueError("native EMA output is not verified")
    manifest = [json.loads(line) for line in (args.work / "manifest.jsonl").read_bytes().splitlines() if line]
    ready = {row["request_id"]: row for row in manifest if row["status"] == "ready"}
    bindings = {row["input_index"]: row for row in json.loads((args.work / "source-bindings.json").read_bytes())}
    if (len(manifest), len(bindings), len(ready), complete["samples"], complete["source_samples"]) != (
            732, 732, len(ready), len(ready), 732):
        raise ValueError("native admission count changed")
    if {row["ordinal"] for row in manifest} != set(bindings):
        raise ValueError("source binding coverage changed")
    receipts = [json.loads(line) for line in (args.work / "results.jsonl").read_bytes().splitlines() if line]
    expected_outputs = {(cfg, rid) for cfg in (2, 3) for rid in ready}
    if len(receipts) != 2 * len(ready) or {(row["cfg"], row["request_id"]) for row in receipts} != expected_outputs:
        raise ValueError("native result coverage changed")
    client = boto3.client("s3", region_name="us-east-1")
    caption_marker = json.loads(get(client, ROOT + f"captions/checkpoint-{args.caption_step}/GLOBAL_COMPLETE.json"))
    if caption_marker.get("rows") != 1464 or file_sha(args.work / "predictions.jsonl") != caption_marker["predictions_sha256"]:
        raise ValueError("PE caption snapshot changed")
    prefix = ROOT + f"edit/caption-{args.caption_step}/node-{args.node_index}/_attempts/{plan_sha[:16]}/"
    files: list[Path] = [args.work / name for name in (
        "plan.json", "COMPLETE.json", "manifest.jsonl", "results.jsonl", "input-audit.json",
        "audit.json", "pe-native-annotations.json", "source-bindings.json", "training.json",
    )]
    for receipt in receipts:
        base = args.work / "results" / f"cfg{receipt['cfg']}" / receipt["request_id"]
        files.extend((base.with_suffix(".png"), base.with_suffix(".json")))
    for row in manifest:
        folder = args.work / "inputs" / row["request_id"]
        if folder.is_dir():
            files.extend(path for path in folder.iterdir() if path.is_file())
    if len(set(files)) != len(files) or any(not path.is_file() or not path.resolve().is_relative_to(args.work.resolve()) for path in files):
        raise ValueError("archive file is missing, duplicated, or outside work root")

    def send(path: Path) -> dict:
        return upload(client, path, prefix + path.relative_to(args.work).as_posix())

    with ThreadPoolExecutor(max_workers=12) as pool:
        uploaded = list(pool.map(send, files))
    by_uri = {item["uri"]: item for item in uploaded}
    score_rows = []
    for receipt in receipts:
        rid, cfg = receipt["request_id"], receipt["cfg"]
        sample = ready[rid]
        binding = bindings[sample["ordinal"]]
        image_uri = prefix + f"results/cfg{cfg}/{rid}.png"
        receipt_uri = prefix + f"results/cfg{cfg}/{rid}.json"
        if by_uri[image_uri]["sha256"] != receipt["image_sha256"]:
            raise ValueError(f"image hash differs from native receipt: {rid} cfg{cfg}")
        reference_uri = prefix + f"inputs/{rid}/reference-00.image"
        placeholder_uri = prefix + f"inputs/{rid}/target.image"
        annotation_uri = prefix + f"inputs/{rid}/annotation.json"
        prompt_uri = prefix + f"inputs/{rid}/model-prompts.json"
        if (by_uri[reference_uri]["sha256"] != sample["reference_sha256"][0]
                or by_uri[placeholder_uri]["sha256"] != sample["target_sha256"]
                or by_uri[annotation_uri]["sha256"] != sample["annotation_sha256"]):
            raise ValueError(f"native input binding changed: {rid}")
        score_rows.append({
            "request_id": rid, "input_index": sample["ordinal"], "sample_id": binding["sample_id"],
            "instruction_en": binding["instruction"], "original_source_s3_uri": binding["source_s3_uri"],
            "original_source_sha256": binding["selection"]["source_sha256"],
            "original_source_size": [binding["selection"]["source_width"], binding["selection"]["source_height"]],
            "original_gpt_image_2_5_output_sha256": binding["selection"]["gpt_output_sha256"],
            "original_gpt_image_2_5_output_size": [binding["selection"]["gpt_output_width"], binding["selection"]["gpt_output_height"]],
            "pe_target_canvas_size": binding["target_canvas_size"],
            "native_resize_policy": "area_le_1048576_aspect_preserved_floor32_lanczos_png_v1",
            "native_reference_uri": reference_uri, "native_reference_sha256": sample["reference_sha256"][0],
            "native_reference_size": sample["source_sizes"][0],
            "native_target_placeholder_uri": placeholder_uri, "native_target_placeholder_sha256": sample["target_sha256"],
            "native_target_size": [sample["output_width"], sample["output_height"]],
            "predicted_native_annotation_uri": annotation_uri, "model_prompts_uri": prompt_uri,
            "caption_checkpoint_step": args.caption_step,
            "caption_predictions_uri": ROOT + f"captions/checkpoint-{args.caption_step}/predictions.jsonl",
            "caption_predictions_sha256": caption_marker["predictions_sha256"],
            "edit_checkpoint_step": args.edit_step, "edit_checkpoint_uri": plan["checkpoint_uri"],
            "weights": "EMA", "cfg": cfg, "steps": 50, "seed": receipt["seed"],
            "plan_sha256": plan_sha, "prediction_uri": image_uri,
            "prediction_sha256": receipt["image_sha256"], "receipt_uri": receipt_uri,
        })
    score_rows.sort(key=lambda row: (row["input_index"], row["cfg"]))
    score_path = args.work / "score-index.jsonl"
    score_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows))
    index = upload(client, score_path, prefix + "score-index.jsonl")
    marker = {
        "version": 1, "status": "complete", "node_index": args.node_index,
        "caption_checkpoint_step": args.caption_step, "edit_checkpoint_step": args.edit_step,
        "source_samples": 732, "admitted_samples": len(ready),
        "rejections": 732 - len(ready), "outputs": len(receipts),
        "plan_sha256": plan_sha, "archive_prefix": prefix,
        "score_index_uri": index["uri"], "score_index_sha256": index["sha256"],
        "files_uploaded": len(uploaded) + 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_uri = ROOT + f"edit/caption-{args.caption_step}/node-{args.node_index}/COMPLETE.json"
    old = maybe_get(client, marker_uri)
    if old is not None:
        previous = json.loads(old)
        if any(previous.get(key) != value for key, value in marker.items() if key != "completed_utc"):
            raise ValueError("existing node completion marker changed")
    else:
        bucket, key = parts(marker_uri)
        client.put_object(Bucket=bucket, Key=key, Body=(json.dumps(marker, indent=2, sort_keys=True) + "\n").encode(),
                          ContentType="application/json", IfNoneMatch="*")
    print("EDIT_ARCHIVE_COMPLETE", args.caption_step, args.node_index, len(receipts), flush=True)


if __name__ == "__main__":
    main()
