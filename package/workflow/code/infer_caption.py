#!/usr/bin/env python3
"""Run the exact 9c4d training prompt against source-only PCB v2 records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from PIL import Image


TASK = Path(__file__).resolve().parent
PE_REPO = TASK.parents[1] / "assets/code/pe_repo"
TRAINING = TASK.parents[1] / "assets/code/prompt_runtime"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(PE_REPO))

from migration.prompt_iev6_ordered import (  # noqa: E402
    PROJECTION_VERSION, PROMPT_SHA256, PROMPT_VERSION, SYSTEM_PROMPT,
    build_user_prompt, ordered_target, prompt_sha256,
)
from scripts import infer_image_edit_pe_benchmark as runner  # noqa: E402


SOURCE_CACHE = TASK.parent / "source-cache"
MANIFEST_SHA256 = "e73cff6cde698d4630f184032670936ef4b58ee772a381e4409fefa85bb87e6b"
POLICY = "verified_gpt_image_2_5_output_png_dimensions"
ORIGINAL_BUILD_LLM_KWARGS = runner.build_llm_kwargs
ORIGINAL_INITIALIZE_ENGINE = runner.initialize_engine
ORIGINAL_RENDER_TRAINING_PROMPT = runner.render_training_prompt
ORIGINAL_RESULT_ROW = runner.result_row
ORIGINAL_SAMPLING_CONTRACT = runner.sampling_contract


def normalize_record(row: dict, *, line_number: int) -> runner.Record:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: expected object")
    forbidden = {"target_s3_uri", "target_image", "assistant_target_json", "final_json", "ground_truth", "label"} & set(row)
    if forbidden:
        raise ValueError(f"line {line_number}: target data leaked: {sorted(forbidden)}")
    index = row.get("input_index")
    sample_id = row.get("sample_id")
    instruction = row.get("instruction")
    uri = row.get("source_s3_uri")
    canvas = row.get("target_canvas_size")
    selection = row.get("selection")
    if type(index) is not int or not 0 <= index < 1471:
        raise ValueError(f"line {line_number}: invalid index")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_number}: missing sample ID")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"line {line_number}: missing instruction.en")
    if not isinstance(uri, str) or not uri.startswith("s3://core-cn-oss.canva.com/usr/lunalai/pixelcraftbench-v2/release/"):
        raise ValueError(f"line {line_number}: wrong source URI")
    if not isinstance(canvas, dict) or any(type(canvas.get(k)) is not int or canvas[k] <= 0 for k in ("width", "height")):
        raise ValueError(f"line {line_number}: invalid target canvas")
    if (not isinstance(selection, dict) or selection.get("dataset") != "PixelCraftBench v2.0"
            or selection.get("manifest_sha256") != MANIFEST_SHA256
            or selection.get("target_canvas_policy") != POLICY
            or Path(urlparse(uri).path).stem != selection.get("source_sha256")
            or row.get("task_type") != "image_edit"):
        raise ValueError(f"line {line_number}: source-only binding changed")
    if canvas != {"width": selection["gpt_output_width"], "height": selection["gpt_output_height"]}:
        raise ValueError(f"GPT target dimensions changed at {index}")
    from input_policy import validate_instruction
    config = json.loads((TASK.parent / "campaign.json").read_text())
    if config.get('run_configured') is not True:
        raise ValueError('Run configure_pipeline.py before inference')
    validate_instruction(row, config)
    return runner.Record(index, sample_id, instruction, uri, (canvas["width"], canvas["height"]), "image_edit", selection)


def load_images(records: list[runner.Record], *, workers: int):
    def load(record: runner.Record):
        source = record.selection
        path = SOURCE_CACHE / Path(urlparse(record.source_s3_uri).path).name
        payload = path.read_bytes()
        if len(payload) != source["source_bytes"] or hashlib.sha256(payload).hexdigest() != source["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {record.sample_id}")
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.size != (source["source_width"], source["source_height"]):
                raise ValueError(f"Source dimensions mismatch: {record.sample_id}")
            return image.convert("RGB")
    with ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
        futures = [pool.submit(load, record) for record in records]
        result = []
        for future in futures:
            try:
                result.append(future.result())
            except BaseException as exc:
                result.append(exc)
        return result


def parse_response(response: str):
    text = runner._THINK_RE.sub("", response, count=1)  # noqa: SLF001
    text = runner._FENCE_START_RE.sub("", text, count=1)  # noqa: SLF001
    text = runner._FENCE_END_RE.sub("", text, count=1).strip()  # noqa: SLF001
    try:
        value = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Non-JSON constant: {token}")))
        if not isinstance(value, dict):
            raise ValueError("Caption JSON must be an object")
    except (ValueError, RecursionError) as error:
        return None, f"{type(error).__name__}: {error}", None
    try:
        if set(value) != {"normalized_edit_instruction", "high_level_description", "style_description", "elements"}:
            raise ValueError("Caption root fields differ from training schema")
        elements = value["elements"]
        if not isinstance(elements, dict) or set(elements) != {"background", "layout", "reference_edits"}:
            raise ValueError("Caption elements differ from training schema")
        projected = ordered_target({
            "normalized_edit_instruction": value["normalized_edit_instruction"],
            "high_level_description": value["high_level_description"],
            "style_description": value["style_description"],
            "compositional_deconstruction": {
                "background": elements["background"],
                "elements": elements["layout"],
                "reference_edits": elements["reference_edits"],
            },
        })
    except (ValueError, KeyError, TypeError) as error:
        return value, None, f"{type(error).__name__}: {error}"
    return projected, None, None


def render_training_prompt(tokenizer, instruction, target_size):
    # The 9c4d run trained with ADD_NON_THINKING_PREFIX=true.  The generic
    # benchmark runner removes this template suffix for older direct-JSON
    # checkpoints, so restore the exact training-time assistant prefix here.
    direct_prompt, user_prompt = ORIGINAL_RENDER_TRAINING_PROMPT(
        tokenizer, instruction, target_size,
    )
    before, marker, after = user_prompt.partition("<image>")
    if not marker or "<image>" in after:
        raise ValueError("Training prompt must contain one source image")
    content = []
    if before:
        content.append({"type": "text", "text": before})
    content.append({"type": "image", "image": "<image>"})
    if after:
        content.append({"type": "text", "text": after})
    prefix = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assistant_prefix = prefix[len(direct_prompt):]
    if assistant_prefix != "<think>\n\n</think>\n\n":
        raise ValueError(f"Unexpected Qwen non-thinking assistant prefix: {assistant_prefix!r}")
    return direct_prompt + assistant_prefix, user_prompt


def build_llm_kwargs(args):
    kwargs = ORIGINAL_BUILD_LLM_KWARGS(args)
    kwargs["max_num_batched_tokens"] = 32768
    if os.environ.get("PE_DISABLE_MTP") == "1":
        kwargs.pop("speculative_config", None)
    return kwargs


def initialize_engine(args):
    if os.environ.get("PE_DISABLE_MTP") != "1":
        llm, tokenizer, sampling, engine = ORIGINAL_INITIALIZE_ENGINE(args)
        if os.environ.get("PE_RETRY_TEMPERATURE") is None:
            return llm, tokenizer, sampling, engine
        from vllm import SamplingParams
        sampling = SamplingParams(
            temperature=float(os.environ["PE_RETRY_TEMPERATURE"]),
            top_p=float(os.environ["PE_RETRY_TOP_P"]),
            top_k=-1,
            repetition_penalty=float(os.environ["PE_RETRY_REPETITION_PENALTY"]),
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        return llm, tokenizer, sampling, engine
    from vllm import LLM, SamplingParams
    kwargs = build_llm_kwargs(args)
    llm = LLM(**kwargs)
    sampling = SamplingParams(temperature=runner.TEMPERATURE, top_p=runner.TOP_P,
                              top_k=runner.TOP_K, max_tokens=args.max_new_tokens, seed=args.seed)
    return llm, llm.get_tokenizer(), sampling, {"llm_kwargs": kwargs, "runtime": runner.runtime_versions()}


def sampling_contract(args):
    contract = ORIGINAL_SAMPLING_CONTRACT(args)
    if os.environ.get("PE_RETRY_TEMPERATURE") is not None:
        contract.update(
            strategy=os.environ.get("PE_RETRY_STRATEGY", "stochastic_retry_after_invalid_json_v1"),
            temperature=float(os.environ["PE_RETRY_TEMPERATURE"]),
            top_p=float(os.environ["PE_RETRY_TOP_P"]),
            repetition_penalty=float(os.environ["PE_RETRY_REPETITION_PENALTY"]),
        )
    return contract


def result_row(record, *, user_prompt, output, sampling_contract):
    row = ORIGINAL_RESULT_ROW(record, user_prompt=user_prompt, output=output,
                              sampling_contract=sampling_contract)
    row["original_instruction"] = record.selection["original_instruction_en"]
    row["instruction_policy"] = record.selection["instruction_policy"]
    if os.environ.get("PE_DIAG_TOKEN_IDS") == "1":
        tokens = list(getattr(runner.candidate(output), "token_ids", []) or [])
        row["generated_token_ids_head"] = tokens[:64]
        row["generated_token_count"] = len(tokens)
    return row


def validate_artifact(args):
    manifest_path = args.model / "local_export_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != args.artifact_manifest_sha256:
        raise ValueError("Local 9c4d export manifest changed")
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source_mcore_checkpoint"]
    if (manifest["kind"] != "local-mcore-hf-trained-mtp-export-v1"
            or source["step"] != args.checkpoint_step
            or source["manifest_sha256"] != args.checkpoint_manifest_sha256
            or manifest["tensor_count"] != 738
            or manifest["mtp_tensor_count"] != 15
            or not manifest["mtp_changed_tensors"]):
        raise ValueError("Local 9c4d export lineage or MTP contract changed")
    for file in manifest["files"]:
        path = args.model / file["name"]
        if not path.is_file() or path.stat().st_size != file["bytes"]:
            raise ValueError(f"Exported model file changed: {path}")
    return {"kind": manifest["kind"], "source_step": source["step"],
            "manifest_sha256": args.artifact_manifest_sha256,
            "tensor_count": manifest["tensor_count"], "mtp_tensor_count": manifest["mtp_tensor_count"]}


def main() -> int:
    if prompt_sha256() != PROMPT_SHA256:
        raise ValueError("Training prompt contract changed")
    runner.PROMPT_VERSION = PROMPT_VERSION
    runner.PROMPT_CONTRACT_SHA256 = PROMPT_SHA256
    runner.LABEL_PROJECTION_VERSION = PROJECTION_VERSION
    runner.SYSTEM_PROMPT = SYSTEM_PROMPT
    runner.build_user_prompt = build_user_prompt
    runner.prompt_contract_sha256 = prompt_sha256
    runner.EXPECTED_LOGICAL_MTP_LAYERS = 1
    runner.EXPECTED_MTP_SHARED_WEIGHTS = False
    runner.INFERENCE_FORMAT_VERSION = "pcb-v2-9c4d-ordered-refedits-v5-gpt25-canvas-v1"
    runner.normalize_record = normalize_record
    runner.load_images = load_images
    runner.parse_response = parse_response
    runner.render_training_prompt = render_training_prompt
    runner.build_llm_kwargs = build_llm_kwargs
    runner.initialize_engine = initialize_engine
    runner.sampling_contract = sampling_contract
    runner.result_row = result_row
    runner.validate_artifact = validate_artifact
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
