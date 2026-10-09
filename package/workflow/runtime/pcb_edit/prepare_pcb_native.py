#!/usr/bin/env python3
"""Adapt 34cf's source-only PCB v2 captions to the frozen native editor."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from tayo_protocol.code import prepare_inputs as native
from tayo_eval.adapt_pe_to_native import adapt
from pcb_edit.canvas_policy import EDITOR_CANVAS_POLICY, editor_canvas_size


DATASET_SHA256 = "e73cff6cde698d4630f184032670936ef4b58ee772a381e4409fefa85bb87e6b"
PE_INPUT_SHA256 = "dcadb3cde9b3d37d45008cc89aa6b5057201ab051fcdc1804a756181cdab8f03"
RESIZE_POLICY = "area_le_1048576_aspect_preserved_floor32_lanczos_png_v1"
TARGET_POLICY = "verified_gpt_image_2_5_output_png_dimensions"
ROW_BINDINGS: dict[tuple[str, int], dict[str, Any]] = {}
ADAPTATIONS: dict[tuple[str, int], dict[str, Any]] = {}
SOURCE_CACHE = Path("/source-cache")
BLACK_PNG: dict[tuple[int, int], bytes] = {}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def one_k_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    scale = min(1.0, math.sqrt(1_048_576 / (width * height)))
    scaled_w = max(1, round(width * scale))
    scaled_h = max(1, round(height * scale))
    target_w = max(32, scaled_w // 32 * 32)
    target_h = max(32, scaled_h // 32 * 32)
    # Rounding the continuous scale can exceed the cap by one patch. Lower the
    # longer side one patch at a time so even that boundary stays admissible.
    while target_w * target_h > 1_048_576:
        if target_w >= target_h and target_w > 32:
            target_w -= 32
        elif target_h > 32:
            target_h -= 32
        else:
            raise AssertionError("native target exceeds 1K area contract")
    return target_w, target_h


def png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PcbDataset:
    def __init__(self, *, local: str):
        if Path(local).name != "pcbv2":
            raise ValueError(f"unexpected virtual source: {local}")

    def __getitem__(self, index: int) -> dict:
        binding = ROW_BINDINGS[("pcbv2", index)]
        uri = binding["source_s3_uri"]
        selection = binding["selection"]
        filename = Path(urlparse(uri).path).name
        if Path(filename).stem != selection["source_sha256"]:
            raise ValueError(f"source cache key differs from source hash: {index}")
        raw = (SOURCE_CACHE / filename).read_bytes()
        if len(raw) != selection["source_bytes"] or digest(raw) != selection["source_sha256"]:
            raise ValueError(f"source image checksum changed: {index}")
        with Image.open(BytesIO(raw)) as image:
            if image.size != (selection["source_width"], selection["source_height"]):
                raise ValueError(f"source dimensions changed: {index}")
            source_size = one_k_size(*image.size)
            source_png = png(image.convert("RGB").resize(source_size, Image.Resampling.LANCZOS))
        raw_target_size = binding["target_canvas_size"]
        expected_size = (selection["gpt_output_width"], selection["gpt_output_height"])
        if raw_target_size != {"width": expected_size[0], "height": expected_size[1]}:
            raise ValueError(f"GPT target dimension binding changed: {index}")
        target_size = editor_canvas_size(binding)
        target_png = BLACK_PNG.get(target_size)
        if target_png is None:
            target_png = png(Image.new("RGB", target_size, (0, 0, 0)))
            BLACK_PNG[target_size] = target_png
        return {
            "sample_id": binding["sample_id"],
            "instructions": json.dumps([binding["instruction"]], ensure_ascii=False),
            "num_inputs": 1,
            "input_images": source_png,
            "output_image": target_png,
            "image_edit_v6_annotation": ADAPTATIONS[("pcbv2", index)],
            "bbox_order": "yxyx_1000",
            "input_s3_paths": [uri],
            "input_widths": [source_size[0]],
            "input_heights": [source_size[1]],
            "output_width": target_size[0],
            "output_height": target_size[1],
            "meta_info": {
                "annotation_bbox_output_order": "yxyx_1000",
                "mds_resize_contract": {
                    "sources": [{"final_width": source_size[0], "final_height": source_size[1]}],
                    "target": {"final_width": target_size[0], "final_height": target_size[1]},
                },
            },
        }


def initialize(work: str, bindings_path: str, adaptations_path: str) -> None:
    global ROW_BINDINGS, ADAPTATIONS
    native.ROOT = Path(work)
    native.OP = Path(work)
    native.initialize()
    ROW_BINDINGS = {("pcbv2", row["input_index"]): row for row in json.loads(Path(bindings_path).read_text())}
    ADAPTATIONS = {("pcbv2", row["input_index"]): row["annotation"] for row in json.loads(Path(adaptations_path).read_text())}
    native.LocalDataset = PcbDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pe-inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=(0,), required=True)
    parser.add_argument("--caption-step", type=int, required=True)
    parser.add_argument("--limit", type=int, help="Restrict a separate smoke work directory to first N assigned rows")
    parser.add_argument("--unit-index", type=int, default=0)
    parser.add_argument("--unit-count", type=int, default=1)
    args = parser.parse_args()
    assert args.unit_count > 0 and 0 <= args.unit_index < args.unit_count
    args.work.mkdir(parents=True, exist_ok=True)
    input_bytes = args.pe_inputs.read_bytes()
    if digest(input_bytes) != PE_INPUT_SHA256:
        raise ValueError("Frozen source-only PCB v2 input changed")
    all_inputs = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    predictions_bytes = args.predictions.read_bytes()
    contract = json.loads(Path("/control/run-contract.json").read_bytes())
    if (digest(predictions_bytes) != contract["predictions_sha256"]
            or args.caption_step != contract["caption_checkpoint_step"]):
        raise ValueError("34cf caption prediction snapshot changed")
    predictions = [json.loads(line) for line in predictions_bytes.splitlines() if line.strip()]
    if len(all_inputs) != 1464 or len(predictions) != 1464:
        raise ValueError("PE cohort count changed")
    by_index = {row["input_index"]: row for row in predictions}
    if len(by_index) != 1464 or set(by_index) != {row["input_index"] for row in all_inputs}:
        raise ValueError("PE prediction index coverage changed")
    selected = all_inputs[args.unit_index::args.unit_count]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("smoke limit must be positive")
        selected = selected[:args.limit]
    bindings: list[dict] = []
    annotations: list[dict] = []
    jobs: list[tuple[int, str, int]] = []
    rejected: list[dict] = []
    for row in selected:
        index = row["input_index"]
        prediction = by_index[index]
        if (row["selection"]["manifest_sha256"] != DATASET_SHA256
                or row["selection"]["target_canvas_policy"] != TARGET_POLICY
                or any(prediction.get(key) != row[key] for key in
                       ("sample_id", "instruction", "source_s3_uri", "target_canvas_size"))
                or prediction.get("selection") != row["selection"]
                or prediction.get("inference_error")):
            raise ValueError(f"PE input/prediction binding changed: {index}")
        bindings.append(row)
        reason = None
        if prediction.get("schema_valid") is not True or not isinstance(prediction.get("prediction"), dict):
            reason = "invalid_pe_schema"
        else:
            try:
                annotation = adapt(prediction["prediction"])
            except ValueError as error:
                reason = f"unsupported_pe_annotation: {error}"
        if reason is not None:
            request_id = f"{index:04d}-" + digest(("pcbv2:" + row["sample_id"]).encode())[:16]
            rejected.append({"request_id": request_id, "ordinal": index,
                             "source": "pcbv2", "source_index": index,
                             "original_mds_sample_id": row["sample_id"],
                             "generation_seed": 42 + index, "num_inputs": 1,
                             "status": "invalid_pe_caption", "reason": reason,
                             "caption_checkpoint_step": args.caption_step})
            continue
        annotations.append({"input_index": index, "annotation": annotation})
        jobs.append((index, "pcbv2", index))
    bindings_path = args.work / "source-bindings.json"
    annotations_path = args.work / "pe-native-annotations.json"
    bindings_path.write_text(json.dumps(bindings, ensure_ascii=False) + "\n")
    annotations_path.write_text(json.dumps(annotations, ensure_ascii=False) + "\n")
    (args.work / "mds/pcbv2").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=8, initializer=initialize,
                             initargs=(str(args.work), str(bindings_path), str(annotations_path))) as pool:
        for result in pool.map(native.prepare, jobs):
            results.append(result)
            if len(results) % 50 == 0:
                print("NATIVE_PREPARED", args.caption_step, args.node_index, len(results), flush=True)
    manifest = sorted([*results, *rejected], key=lambda row: row["ordinal"])
    if len(manifest) != len(selected) or len({row["request_id"] for row in manifest}) != len(selected):
        raise ValueError("native preparation coverage changed")
    (args.work / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    summary = {
        "node_index": args.node_index,
        "caption_checkpoint_step": args.caption_step,
        "dataset_manifest_sha256": DATASET_SHA256,
        "pe_input_sha256": PE_INPUT_SHA256,
        "predictions_sha256": digest(predictions_bytes),
        "target_canvas_policy": TARGET_POLICY,
        "native_resize_policy": RESIZE_POLICY,
        "editor_target_canvas_policy": EDITOR_CANVAS_POLICY,
        "native_target_placeholder": "black_png_zeroed_before_conditioner",
        "samples": len(selected),
        "counts": dict(Counter(row["status"] for row in manifest)),
        "rejections": [row for row in manifest if row["status"] != "ready"],
    }
    native.atomic(args.work / "input-audit.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rejections"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
