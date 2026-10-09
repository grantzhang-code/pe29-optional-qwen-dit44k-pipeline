#!/usr/bin/env python3
"""The frozen protocol runner with dynamic PCB v2 single-image cardinality.

All Qwen/VAE encode, strict EMA restore, sampling, and output validation still
execute the ZIP's original ``code/handoff/run.py`` functions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from tayo_protocol.code.handoff import run as base


ORIGINAL_IDENTITY = base.script_identity
EXTENSIONS = [
    Path(__file__),
    Path(__file__).with_name("prepare_pcb_native.py"),
    Path(__file__).with_name("canvas_policy.py"),
    Path("/app/tayo_eval/adapt_pe_to_native.py"),
    Path("/app/training_reader/craft_v1_edit_mds.py"),
    Path("/app/training_reader/clean_caption.py"),
    Path(__file__).resolve().parents[1] / "tayo_protocol/code/prepare_inputs.py",
]


def script_identity() -> dict[str, str]:
    identity = ORIGINAL_IDENTITY()
    for path in EXTENSIONS:
        identity[str(path.relative_to(Path("/app")))] = base.sha(path)
    return identity


def prepare(args) -> None:
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from src.data.dataset.craft_v1_edit_clean_caption import require_caption_preparation_version

    base.require(not (base.WORK / "plan.json").exists(), "Work directory already frozen")
    control = torch.load(base.CHECKPOINT / "meta.pt", map_location="cpu", weights_only=False)
    base.require(int(control["global_step"]) == args.step, "Requested checkpoint step mismatch")
    config = control["mariana_resume_contract"]["effective_config"]
    mode = base.policy(config)
    model = config["model"]
    base.require(str(model["conditioner"]["init_args"]["weight_path"]).rstrip("/").endswith("/Qwen3.8-27B"), "Unsupported Qwen asset")
    base.require(str(model["vae"]["init_args"]["weight_path"]).rstrip("/").endswith("/FLUX.2-klein-4B-vae"), "Unsupported VAE asset")
    base.require(model["diffusion_trainer"]["init_args"]["cfg_dropout_mode"] == "multimodal_text_null", "Unsupported CFG-null mode")
    if args.weights == "EMA":
        ema = control.get("native_ema_state_dict", {})
        base.require(int(ema.get("previous_step", -1)) == args.step and int(ema.get("update_count", 0)) > 0, "EMA state is incomplete")
    prefix = "state_dict." + ("ema_denoiser." if args.weights == "EMA" else "denoiser.")
    metadata = FileSystemReader(base.CHECKPOINT).read_metadata()
    keys = [key for key in metadata.state_dict_metadata if key.startswith(prefix)]
    base.require(bool(keys), f"Checkpoint has no {prefix} tensors")
    files = sorted({v.relative_path for k, v in metadata.storage_data.items() if k.fqn.startswith(prefix)})

    def inventory_entry(name: str) -> dict:
        path = base.inside(base.CHECKPOINT, name)
        base.require(path.is_file() and path.stat().st_size > 0, f"Missing DCP shard: {name}")
        return {"name": name, "bytes": path.stat().st_size, "sha256": base.sha(path)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        inventory = list(pool.map(inventory_entry, files))
    base.require((base.CHECKPOINT / "DOWNLOAD.json").exists(), "Checkpoint download incomplete")
    source_rows = [json.loads(line) for line in (base.WORK / "manifest.jsonl").read_text().splitlines()]
    base.require(bool(source_rows) and len({r["request_id"] for r in source_rows}) == len(source_rows), "Non-unique or empty input cohort")
    allowed = {"ready", "invalid_pe_caption", "rejected_by_training_reader", "preparation_error"}
    base.require(all(r["status"] in allowed for r in source_rows), "Unknown preparation status")
    rows = base.all_rows()
    base.require(bool(rows), "No native-admitted PE captions")
    for row in rows:
        require_caption_preparation_version(row)
        base.require("/" not in row["request_id"] and ".." not in row["request_id"], "Invalid request ID")
        base.verified(base.WORK / "native-batches", row["request_id"] + ".pt", row["native_batch_sha256"])
    base.require(config == base.read(base.WORK / "training.json"), "Frozen training config differs from checkpoint")
    p = dict(
        schema=2, step=args.step, weights=args.weights, namespace=prefix,
        checkpoint_uri=args.checkpoint_uri, conditioning_policy=mode,
        use_reference_kv_cache=mode == "reference_only", text_cap=base.TEXT_CAP,
        steps=50, cfg=args.cfg, timeshift=4.6, last_step=0.02, noise_scale=1.0,
        roi_policy="pixel_aspect_fixed", expected_samples=len(rows), source_samples=len(source_rows),
        admission_rejections=[r for r in source_rows if r["status"] != "ready"],
        runtime=base.read(base.HERE / "runtime.json"), scripts=script_identity(),
        manifest_sha256=base.sha(base.WORK / "manifest.jsonl"),
        config_sha256=base.sha(base.WORK / "training.json"),
        control_sha256=base.sha(base.CHECKPOINT / "meta.pt"),
        metadata_sha256=base.sha(base.CHECKPOINT / ".metadata"),
        checkpoint_files=inventory, checkpoint_keys=len(keys),
    )
    base.atomic(base.WORK / "plan.json", p)
    print(json.dumps({"status": "PREPARED", "samples": len(rows), "source_samples": len(source_rows), "policy": mode, "keys": len(keys)}), flush=True)


def verify(args) -> None:
    p = base.plan()
    contract = base.sha(base.WORK / "plan.json")
    rows = base.all_rows()
    receipts = [base.check_result(row, cfg, contract) for cfg in p["cfg"] for row in rows]
    expected = {(cfg, row["request_id"]) for cfg in p["cfg"] for row in rows}
    actual = {(cfg, path.stem) for cfg in p["cfg"] for path in (base.WORK / "results" / f"cfg{cfg}").glob("*.json")}
    base.require(expected == actual, "Unexpected or missing output IDs")
    (base.WORK / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in receipts))
    base.atomic(base.WORK / "COMPLETE.json", dict(
        plan_sha256=contract, samples=p["expected_samples"], source_samples=p["source_samples"],
        admission_rejections=len(p["admission_rejections"]), outputs=len(receipts), status="PASS",
    ))
    print("COMPLETE", len(receipts), flush=True)


if __name__ == "__main__":
    base.script_identity = script_identity
    base.prepare = prepare
    base.verify = verify
    base.main()
