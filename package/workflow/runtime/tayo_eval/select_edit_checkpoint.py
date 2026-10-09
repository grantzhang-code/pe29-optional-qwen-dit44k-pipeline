#!/usr/bin/env python3
"""At the six-caption barrier, lock the newest complete Tayo EMA checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import boto3
from torch.distributed.checkpoint import FileSystemReader
import torch

from tayo_eval.collect_captions import ROOT, get, maybe_get, parts, put_immutable, sha


CHECKPOINTS = "s3://core-cn-training.canva.com/pixelcraft/usr/tayo/runs/edit1k-97src-15n-0925-0340/checkpoints/"
STEP_RE = re.compile(r"step=(\d+)\.ckpt/")
SHARD_RE = re.compile(r"__\d+_\d+\.distcp")


def objects(client, bucket: str, prefix: str) -> dict[str, dict]:
    result = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", []):
            name = entry["Key"][len(prefix):]
            if name in ("meta.pt", ".metadata") or SHARD_RE.fullmatch(name):
                result[name] = {"name": name, "key": entry["Key"], "bytes": entry["Size"], "etag": entry["ETag"]}
    return result


def check_candidate(client, uri: str, step: int, frozen_config: dict) -> dict | None:
    bucket, prefix = parts(uri)
    first = objects(client, bucket, prefix)
    if not {"meta.pt", ".metadata"} <= set(first) or not any(SHARD_RE.fullmatch(k) for k in first):
        return None
    if any(obj["bytes"] <= 0 for obj in first.values()):
        return None
    controls = {}
    for name in ("meta.pt", ".metadata"):
        entry = first[name]
        response = client.get_object(Bucket=bucket, Key=entry["key"], IfMatch=entry["etag"])
        raw = response["Body"].read()
        if len(raw) != entry["bytes"]:
            return None
        controls[name] = {"bytes": raw, "sha256": sha(raw), "version_id": response.get("VersionId")}
    control = torch.load(io.BytesIO(controls["meta.pt"]["bytes"]), map_location="cpu", weights_only=False)
    if int(control.get("global_step", -1)) != step:
        return None
    ema = control.get("native_ema_state_dict", {})
    if int(ema.get("previous_step", -1)) != step or int(ema.get("update_count", 0)) <= 0:
        return None
    config = control["mariana_resume_contract"]["effective_config"]
    if config != frozen_config:
        raise ValueError(f"latest checkpoint step {step} changed the frozen training config")
    with tempfile.TemporaryDirectory(prefix="tayo-checkpoint-metadata-") as temp:
        (Path(temp) / ".metadata").write_bytes(controls[".metadata"]["bytes"])
        metadata = FileSystemReader(temp).read_metadata()
    namespace = "state_dict.ema_denoiser."
    keys = [key for key in metadata.state_dict_metadata if key.startswith(namespace)]
    if not keys:
        return None
    required = {v.relative_path for k, v in metadata.storage_data.items() if k.fqn.startswith(namespace)}
    if not required or not required <= set(first):
        return None
    second = objects(client, bucket, prefix)
    if first != second:
        return None
    return {
        "step": step, "uri": uri, "weights": "EMA", "namespace": namespace,
        "ema_keys": len(keys), "shard_count": len(required),
        "metadata_sha256": controls[".metadata"]["sha256"],
        "meta_sha256": controls["meta.pt"]["sha256"],
        "objects": [first[name] for name in sorted(first)],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    client = boto3.client("s3", region_name="us-east-1")
    caption_barrier = get(client, ROOT + "captions/ALL_SIX_COMPLETE.json")
    barrier = json.loads(caption_barrier)
    if barrier.get("status") != "complete" or barrier.get("total_predictions") != 6984:
        raise ValueError("six caption cohorts are not complete")
    config_bytes = a.training_config.read_bytes()
    frozen_config = json.loads(config_bytes)
    bucket, prefix = parts(CHECKPOINTS)
    steps = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            suffix = common["Prefix"][len(prefix):]
            if match := STEP_RE.fullmatch(suffix):
                steps.append(int(match.group(1)))
    chosen = None
    for step in sorted(set(steps), reverse=True):
        candidate = check_candidate(client, CHECKPOINTS + f"step={step}.ckpt/", step, frozen_config)
        if candidate is not None:
            chosen = candidate
            break
    if chosen is None:
        raise ValueError("no complete EMA edit checkpoint with the frozen config")
    chosen.update({
        "version": 1,
        "caption_barrier_sha256": sha(caption_barrier),
        "training_config_sha256": sha(config_bytes),
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    })
    payload = (json.dumps(chosen, indent=2, sort_keys=True) + "\n").encode()
    remote = ROOT + "edit/edit-checkpoint.lock.json"
    old = maybe_get(client, remote)
    if old is not None:
        existing = json.loads(old)
        for field in ("step", "uri", "weights", "namespace", "ema_keys", "shard_count", "metadata_sha256", "meta_sha256", "objects", "caption_barrier_sha256", "training_config_sha256"):
            if existing.get(field) != chosen[field]:
                raise ValueError(f"edit checkpoint lock already exists with different {field}")
        payload = old
    else:
        put_immutable(client, remote, payload)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(payload)
    print("EDIT_CHECKPOINT_SELECTED", chosen["step"], chosen["uri"], chosen["ema_keys"], flush=True)


if __name__ == "__main__":
    main()
