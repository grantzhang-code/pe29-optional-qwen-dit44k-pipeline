"""LIVE-only and native MDS conditioning adapters for the authorized SP2 run.

This module does not initialize a process group or change dependencies. Only
rank zero calls the encoding/decoding loaders; every SP rank restores one LIVE DiT.
No target pixels are used as conditioning. Singleton seeds belong to the caller.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
import zlib

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LIVE_PREFIX = "state_dict.denoiser."


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return _sha(value.detach().contiguous().cpu().numpy().tobytes())


def _config(value) -> dict:
    return copy.deepcopy(value) if isinstance(value, dict) else yaml.safe_load(Path(value).read_text())


def _class(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


@contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _strict_load_live_cpu(model, checkpoint: Path, *, expected_step: int,
                          expected_identity: dict | None = None) -> dict:
    """Independently read only LIVE tensors on CPU, even inside an NCCL group.

    Using no_dist=True is deliberate: both SP ranks require an entire replica,
    not a DCP data-parallel shard or CPU-tensor collective on the NCCL group.
    """
    from torch.distributed.checkpoint import FileSystemReader, load_state_dict

    started = time.monotonic()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Expected canonical DCP directory: {checkpoint}")
    metadata_sha = _sha((checkpoint / ".metadata").read_bytes())
    control_sha = _sha((checkpoint / "meta.pt").read_bytes())
    if expected_identity is not None:
        if expected_identity["weights"] != "LIVE" or expected_identity["namespace"] != LIVE_PREFIX:
            raise ValueError("Expected checkpoint identity is not the LIVE namespace")
        if metadata_sha != expected_identity["metadata_sha256"] or control_sha != expected_identity["control_sha256"]:
            raise ValueError("Pinned LIVE checkpoint metadata/control hashes changed")
    control = torch.load(checkpoint / "meta.pt", map_location="cpu", weights_only=False)
    step = int(control.get("global_step", -1))
    if step != expected_step:
        raise ValueError(f"Checkpoint step {step} != required {expected_step}")
    del control
    reader = FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    entries = {key[len(LIVE_PREFIX):]: value for key, value in metadata.state_dict_metadata.items()
               if key.startswith(LIVE_PREFIX)}
    state = model.state_dict()
    missing = sorted(set(state) - set(entries))
    unexpected = sorted(set(entries) - set(state))
    mismatches = {key: {"model": list(state[key].shape), "checkpoint": list(getattr(entries[key], "size", ()))}
                  for key in set(state) & set(entries)
                  if not hasattr(entries[key], "size") or tuple(state[key].shape) != tuple(entries[key].size)}
    if missing or unexpected or mismatches:
        raise ValueError(f"Strict LIVE audit failed: missing={missing}, unexpected={unexpected}, shapes={mismatches}")
    if not entries or any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("LIVE template must be a nonempty CPU-only state dict")
    if any(value.properties.dtype != torch.float32 for value in entries.values()):
        raise ValueError("LIVE checkpoint tensor storage differs from audited FP32")
    if any(value.dtype != torch.bfloat16 for value in state.values()):
        raise ValueError("LIVE inference template must use BF16 storage")
    namespace = {"denoiser." + key: value for key, value in state.items()}
    load_state_dict({"state_dict": namespace}, storage_reader=reader, no_dist=True)
    restored = {key.removeprefix("denoiser."): value for key, value in namespace.items()}
    result = model.load_state_dict(restored, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict model restore returned incompatible keys")
    proof = {
        "checkpoint": str(checkpoint), "global_step": step, "weights": "LIVE", "namespace": LIVE_PREFIX,
        "key_count": len(entries), "live_key_count": len(entries), "strict_missing": [], "strict_unexpected": [],
        "strict_shape_mismatches": {}, "state_numel": sum(value.numel() for value in state.values()),
        "parameter_numel": sum(p.numel() for p in model.parameters()),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "dcp_load": "CPU LIVE namespace only; no_dist=True; no optimizer/EMA tensors",
        "checkpoint_storage_dtype": "float32", "inference_parameter_dtype": "bfloat16",
        "ema_namespace_loaded": False, "optimizer_loaded": False, "dcp_no_dist": True,
        "metadata_sha256": metadata_sha, "control_sha256": control_sha,
        "cpu_restore_seconds": time.monotonic() - started,
    }
    del state, namespace, restored, entries, metadata, reader
    return proof


def load_denoiser(arm, checkpoint, device, pack_target=10240, *, training_config=None,
                  expected_step=None):
    started = time.monotonic()
    identities = json.loads((ROOT / "config/checkpoint_identities.json").read_text())
    identity = identities[str(arm)]
    expected_step = int(identity["step"] if expected_step is None else expected_step)
    if expected_step != int(identity["step"]) or expected_step != int(identity["checkpoint_step"]):
        raise ValueError("Requested checkpoint step disagrees with this arm's pinned identity")
    raw = _config(training_config or ROOT / "config" / f"{arm}.training.yaml")
    specification = raw["model"]["denoiser"]
    arguments = copy.deepcopy(specification["init_args"])
    arguments.update(pack_target_T=int(pack_target), strict_pack_target=True)
    if arguments.get("use_mup"):
        raise ValueError("This adapter is scoped to the current non-muP LIVE contract")
    # CPU construction keeps ordinary nonpersistent buffers correctly initialized.
    # Meta/to_empty would lose such buffers. Never instantiate a second live DiT.
    with torch.device("cpu"), _default_dtype(torch.bfloat16):
        model = _class(specification["class_path"])(**arguments)
    model.requires_grad_(False).eval()
    proof = _strict_load_live_cpu(model, Path(checkpoint), expected_step=int(expected_step),
                                  expected_identity=identity)
    if proof["key_count"] != identity["key_count"] or proof["state_numel"] != identity["state_numel"]:
        raise ValueError("Restored LIVE architecture differs from this arm's frozen inventory")
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("LIVE parameter storage is not uniformly BF16")
    gc.collect()
    model.to(device=torch.device(device))
    proof.update(arm=str(arm), device=str(device), pack_target_T=int(pack_target),
                 live_denoiser_constructed=True, ema_denoiser_constructed=False,
                 redundant_denoiser_constructed=False, denoiser_instances_created=1,
                 total_load_seconds=time.monotonic() - started)
    model._inference_restore_evidence = proof
    return model


@dataclass
class EncodingContext:
    config: dict
    device: torch.device
    conditioner: Any = None
    vae: Any = None
    fp8_enabled: bool = False
    evidence: dict = field(default_factory=dict)


def load_vae_only(config, modelsroot, device) -> EncodingContext:
    raw = _config(config)
    specification = raw["model"]["vae"]
    arguments = copy.deepcopy(specification["init_args"])
    arguments["weight_path"] = str(Path(modelsroot) / "FLUX.2-klein-4B-vae")
    if arguments.get("dtype") not in ("bf16", "bfloat16"):
        raise ValueError("Expected the frozen BF16 VAE policy")
    vae = _class(specification["class_path"])(**arguments)
    vae.requires_grad_(False).eval().to(device=torch.device(device))
    return EncodingContext(raw, torch.device(device), vae=vae,
                           evidence={"vae_weight_path": arguments["weight_path"], "vae_dtype": "bfloat16"})


def load_conditioner(config, modelsroot, device, *, text_cap=9216) -> EncodingContext:
    from mariana.integrations.transformer_engine import (
        apply_te_fp8, plan_te_fp8_replacement, resolve_te_integration_config,
        validate_te_replacement,
    )
    started = time.monotonic()
    raw = _config(config)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Actual Qwen FP8 encoding requires the selected CUDA runtime")
    torch.cuda.set_device(device)
    from transformer_engine.common.recipe import Float8CurrentScaling
    from transformer_engine.pytorch.quantization import check_recipe_support
    recipe = Float8CurrentScaling()
    check_recipe_support(recipe)
    # Explicitly retain the pinned RTX current-scaling policy on H200. Current
    # scaling has no delayed-amax history/global reduction, and the frozen
    # conditioner forward always runs under no_grad. Do not accidentally make
    # rank zero wait for a rank-one conditioner that does not exist.
    if type(recipe).__name__ != "Float8CurrentScaling":
        raise ValueError(f"FP8 recipe changed from the verified runtime: {recipe!r}")
    specification = raw["model"]["conditioner"]
    arguments = copy.deepcopy(specification["init_args"])
    arguments.update(weight_path=str(Path(modelsroot) / "Qwen3.8-27B"),
                     max_length=int(text_cap), text_cap=int(text_cap), batch_pad_multiple=1)
    if not arguments.get("interleaved_reference_regions") or not arguments.get("vl_mode"):
        raise ValueError("Expected V7 interleaved visual conditioning")
    if raw["model"]["diffusion_trainer"]["init_args"].get("cfg_dropout_mode") != "multimodal_text_null":
        raise ValueError("Expected actual multimodal text-null weak branch")
    conditioner = _class(specification["class_path"])(**arguments)
    conditioner.requires_grad_(False).eval()
    te_config = raw["model"]["te_fp8_config"]
    resolved = resolve_te_integration_config(te_config)
    if not resolved.enabled or not resolved.replace_conditioner or not resolved.conditioner_fp8_weights:
        raise ValueError("Training FP8 conditioner weight-storage policy must be enabled")
    plan = plan_te_fp8_replacement(conditioner.model, frozen=True,
                                   policy=resolved.conditioner_policy, config=te_config,
                                   include_availability=True)
    report = apply_te_fp8(conditioner.model, frozen=True, rank=0, amax_sync_group=None,
                          policy=resolved.conditioner_policy, config=te_config,
                          return_report=True, fp8_weight_storage=True)
    validation = validate_te_replacement(plan, report)
    validation.raise_for_error()
    if report.replaced_count <= 0:
        raise ValueError("No conditioner layers received the training FP8 replacement")
    # Device-only move: preserve FP32 rotary/GDN buffers and FP8 weight storage.
    conditioner.to(device=device)
    if conditioner.use_token_packing:
        raise ValueError("Interleaved ROI conditioning unexpectedly enabled token packing")
    context = load_vae_only(raw, modelsroot, device)
    context.conditioner = conditioner
    context.fp8_enabled = True
    context.evidence.update(qwen_weight_path=arguments["weight_path"], text_cap=int(text_cap),
                            fp8_report=report.to_json(), fp8_validation=validation.to_json(),
                            fp8_recipe_class=type(recipe).__name__, fp8_recipe_repr=repr(recipe),
                            fp8_recipe_selection="explicit-pinned-current-scaling",
                            gpu_name=torch.cuda.get_device_name(device),
                            gpu_capability=list(torch.cuda.get_device_capability(device)),
                            fp8_rank_local=True, fp8_gradient_enabled=False,
                            fp8_collective_policy="current-scaling; no delayed-amax history; no_grad; no tensor/sequence parallel TE linear",
                            actual_token_packing=False, load_seconds=time.monotonic() - started)
    gc.collect()
    return context


def release_conditioner(context: EncodingContext):
    """Drop Qwen while retaining the independent VAE decode handle."""
    context.conditioner = None
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()


def _read_verified(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if _sha(payload) != expected:
        raise ValueError(f"Frozen input hash changed: {path}")
    return payload


def prepare_case(context, row, inputsroot, pack_target=10240):
    """CPU native preparation; also usable with a tokenizer-only test context."""
    from src.data.dataset.craft_v1_edit_mds import CraftV1EditMdsProcessor
    from src.data.dataset.craft_v1_edit_mds_annotation import adapt_native_edit_annotation
    from src.data.dataset.craft_v1_image_edit_packing import build_craft_v1_edit_inference_batch
    from src.inference.craft_v1_edit import _segment_layout

    if context.conditioner is None:
        raise ValueError("Conditioner/tokenizer has been released")
    inputsroot = Path(inputsroot)
    source_path = inputsroot / row["source_path"]
    source = _read_verified(source_path, row["source_sha256"])
    caption_bytes = _read_verified(inputsroot / row["annotation_path"], row["annotation_sha256"])
    native_annotation = json.loads(caption_bytes)
    if native_annotation != row["annotation"]:
        raise ValueError("Frozen manifest annotation differs from native MDS caption")
    annotation = adapt_native_edit_annotation(native_annotation)
    audit = json.loads((inputsroot / row["native_audit_path"]).read_text())
    if audit.get("status") != "passed":
        raise ValueError("Native audit is not passed")
    record_path = source_path.parent / "record.json"
    record = json.loads(_read_verified(record_path, audit["record_json_sha256"]))
    sample_id = str(record["sample_id"])
    if sample_id != str(row["original_mds_sample_id"]) or zlib.crc32(sample_id.encode()) % 200:
        raise ValueError("Native sample identity/heldout membership mismatch")
    if _sha(source) != record["input_images"]["sha256"]:
        raise ValueError("Source differs from the native record")
    geometry = SimpleNamespace(edit=SimpleNamespace(pixel_patch_size=32, image_max_pixels=1048576))
    resize = CraftV1EditMdsProcessor._stored_resize_contract(geometry, record)
    if resize is None:
        raise ValueError("MDS producer geometry is required")
    source, sw, sh = CraftV1EditMdsProcessor._stored_image(geometry, source, name="source", geometry=resize["source"])
    tw, th = int(record["output_width"]), int(record["output_height"])
    if (sw, sh) != (row["source_width"], row["source_height"]) or (tw, th) != (row["expected_output_width"], row["expected_output_height"]):
        raise ValueError("Frozen source/output geometry drift")
    if int(record["num_inputs"]) != 1:
        raise ValueError("This adapter requires exactly one native source")
    batch = build_craft_v1_edit_inference_batch(
        conditioner=context.conditioner, annotation=annotation, reference_images=[source],
        output_height=th, output_width=tw, variant="v7+v4-refid", pack_target_T=int(pack_target),
        pixel_patch_size=32, image_max_pixels=max(1048576, sw * sh, tw * th),
        reference_region_allocation_max_pixels=262144,
        max_reference_images=1, reference_frame_codes=(4097,),
    )
    if batch.n_samples != 1 or batch.img_grids.tolist() != [[sh // 32, sw // 32], [th // 32, tw // 32]]:
        raise ValueError("Native singleton changed image grids")
    target_rows, image_lens = _segment_layout(batch)
    if torch.count_nonzero(batch.img_patches[target_rows]).item() != 0:
        raise ValueError("Target conditioning placeholder is not neutral")
    meta = batch.metadata[0]
    if "post_encoder_text_keep_mask" in meta:
        raise ValueError("Post-encoder text dropping is forbidden in this evaluation")
    positive = int(batch.text_lens[0])
    weak = int(meta["cfg_null_qwen_text_len"])
    cap = int(context.conditioner.text_cap)
    if min(positive, weak) <= 0 or max(positive, weak) >= cap:
        raise ValueError("Full conditioning does not strictly fit the untruncated text cap")
    if image_lens[0] + max(positive, weak) > int(pack_target):
        raise ValueError("Native positive/weak case exceeds strict pack budget")
    actual = {
        "positive_text_tokens": positive, "weak_text_tokens": weak,
        "positive_prompt_sha256": _sha(batch.captions[0].encode()),
        "weak_prompt_sha256": _sha(meta["cfg_null_caption"].encode()),
        "positive_input_ids_sha256": _tensor_sha(meta["qwen_input_ids"]),
        "weak_input_ids_sha256": _tensor_sha(meta["cfg_null_qwen_input_ids"]),
        "positive_text_pos_meta_sha256": _tensor_sha(meta["text_pos_meta"]),
        "weak_text_pos_meta_sha256": _tensor_sha(meta["cfg_null_text_pos_meta"]),
        "strict_pack_cost": image_lens[0] + max(positive, weak),
    }
    for key, value in actual.items():
        if key in audit and audit[key] != value:
            raise ValueError(f"Native prepared input differs from frozen CPU audit: {key}: {value} != {audit[key]}")
    if actual["positive_prompt_sha256"] != row["positive_prompt_sha256"] or actual["weak_prompt_sha256"] != row["weak_prompt_sha256"]:
        raise ValueError("Prepared prompt differs from frozen inference manifest")
    case = dict(actual, request_id=row["request_id"], generation_seed=int(row["generation_seed"]),
                source_sha256=row["source_sha256"], annotation_sha256=row["annotation_sha256"],
                record_json_sha256=audit["record_json_sha256"], sample_id=sample_id,
                source_width=sw, source_height=sh, output_width=tw, output_height=th,
                target_pixels_used=False, truncated=False, max_samples_per_pack=1,
                pack_target_T=int(pack_target), text_cap=cap,
                roi_count=len(meta.get("reference_region_token_counts", [])),
                roi_token_counts=[int(x) for x in meta.get("reference_region_token_counts", [])])
    return batch, case


@torch.no_grad()
def build_case(context, row, inputsroot, pack_target=10240):
    from mariana.integrations.transformer_engine import fp8_autocast_ctx
    from src.native_task import PixelCraftNativeTask
    from src.diffusion.flow_matching.training_self_flow_varlen import SelfFlowVarlenTrainer
    from src.inference.craft_v1_edit import _segment_layout

    batch, case = prepare_case(context, row, inputsroot, pack_target)
    device = context.device
    target_rows, image_lens = _segment_layout(batch)
    metadata = PixelCraftNativeTask._stack_packed_metadata(batch.metadata) or {}
    positive_meta = dict(metadata)
    weak_meta = dict(metadata)
    for dest, source in (("qwen_input_ids", "cfg_null_qwen_input_ids"),
                         ("qwen_attention_mask", "cfg_null_qwen_attention_mask"),
                         ("text_pos_meta", "cfg_null_text_pos_meta")):
        weak_meta[dest] = metadata[source]
    for key in ("post_encoder_text_keep_mask", "qwen_full_text_len", "text_corr_mem_k"):
        weak_meta.pop(key, None)
    branch_features = []
    branch_positions = []
    for captions, meta, expected in (
        (batch.captions, positive_meta, case["positive_text_tokens"]),
        ([str(value) for value in metadata["cfg_null_caption"]], weak_meta, case["weak_text_tokens"]),
    ):
        with fp8_autocast_ctx(enabled=context.fp8_enabled):
            dense = context.conditioner._impl_condition(captions, meta)
        mask = meta["text_embedding_mask"].to(device=dense.device, dtype=torch.bool)
        realized = mask.sum(dim=1).to(dtype=torch.int32, device="cpu").tolist()
        if realized != [expected] or dense.ndim != 4 or tuple(dense.shape[2:]) != (16, 5120):
            raise ValueError(f"Actual Qwen tap contract/length drift: {tuple(dense.shape)}, {realized}, expected={expected}")
        flat = PixelCraftNativeTask._flatten_text_tokens(dense, mask).contiguous()
        del dense
        if flat.dtype != torch.bfloat16 or not torch.isfinite(flat).all().item():
            raise ValueError("Qwen taps must be finite BF16")
        branch_features.append(flat)
        pos = SelfFlowVarlenTrainer._gather_text_pos_meta_valid(meta.get("text_pos_meta"), [expected])
        branch_positions.append(pos)
    encoded = context.vae.encode_packed(batch.img_patches.to(device=device), batch.img_grids)
    if not torch.isfinite(encoded).all().item():
        raise ValueError("VAE produced nonfinite latents")
    positive, weak = branch_features
    bundle = dict(
        img_patches=encoded, is_target_rows=target_rows.to(device=device),
        img_grids=batch.img_grids, img_seg_sample_id=batch.img_seg_sample_id,
        img_seg_frame=batch.img_seg_frame, img_seg_is_target=batch.img_seg_is_target,
        text_tokens=positive, text_lens=[case["positive_text_tokens"]],
        max_seqlen=int(batch.max_seqlen),
        null_text_tokens=weak, null_text_lens=[case["weak_text_tokens"]],
        null_max_seqlen=image_lens[0] + case["weak_text_tokens"],
        text_pos_meta=branch_positions[0], null_text_pos_meta=branch_positions[1],
        text_corr_mem_k=SelfFlowVarlenTrainer._gather_text_corr_mem_k_valid(
            positive_meta.get("text_corr_mem_k"), [case["positive_text_tokens"]]),
        image_corr_mem_q=getattr(batch, "image_corr_mem_q", None),
        image_corr_mem_k=getattr(batch, "image_corr_mem_k", None),
        corr_code_num_groups=getattr(batch, "corr_code_num_groups", None),
        corr_code_row_offset=0, corr_code_gate_scale=1.0,
    )
    case.update(positive_feature_shape=list(positive.shape), weak_feature_shape=list(weak.shape),
                feature_dtype=str(positive.dtype), encoded_shape=list(encoded.shape),
                encoded_dtype=str(encoded.dtype), qwen_fp8=context.fp8_enabled,
                weak_branch="actual_multimodal_text_null")
    return bundle, case


@torch.no_grad()
def decode(context, output_latents, case_metadata):
    from PIL import Image
    from src.models.autoencoder.base import fp2uint8

    if isinstance(output_latents, (list, tuple)):
        if len(output_latents) != 1:
            raise ValueError("Decode requires exactly one target latent image")
        latent = output_latents[0]
    else:
        latent = output_latents
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError(f"Unexpected target latent shape {tuple(latent.shape)}")
    output = context.vae.decode(latent.to(device=context.device))[0]
    if not torch.isfinite(output).all().item():
        raise ValueError("VAE decode produced nonfinite pixels")
    pixels = fp2uint8(output).permute(1, 2, 0).contiguous().cpu().numpy()
    if tuple(pixels.shape) != (int(case_metadata["output_height"]), int(case_metadata["output_width"]), 3):
        raise ValueError(f"Decoded image geometry differs from manifest: {pixels.shape}")
    return Image.fromarray(pixels, mode="RGB")
