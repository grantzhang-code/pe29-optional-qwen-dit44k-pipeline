#!/usr/bin/env python3
"""Run the frozen checkpoint-8000 Image Edit V6 inference contract end to end.

This is the canonical single-entry inference path for checkpoint-8000:

1. verify the restored trained-MTP Hugging Face artifact;
2. build the exact training_v7 Source(frame_0) -> Target prompt;
3. run deterministic vLLM generation;
4. parse and validate the direct-root V6 XYXY response;
5. apply the deterministic small-bbox/HLD-label postprocess and revalidate;
6. convert layout, edit-reference, and remove-reference bboxes to YXYX/1000;
7. write only fully packaged JSONL rows.

The underlying generator deliberately remains shared with large-scale jobs.
This entrypoint freezes the semantic/runtime contract and adds a fail-closed,
streaming packaging boundary, so callers cannot accidentally publish raw XYXY
predictions or omit the HLD postprocess.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v6_bbox_packaging import (  # noqa: E402
    PACKAGED_BBOX_COORDINATE_ORDER,
    PACKAGING_VERSION,
    SOURCE_BBOX_COORDINATE_ORDER,
    V6BBoxPackagingError,
    package_v6_inference_row_yxyx,
)
from dlms.datasets.image_edit_v6_hld_postprocess import (  # noqa: E402
    POSTPROCESS_VERSION,
)


TARGET_SCHEMA = "image_edit_v6_direct_root_xyxy"
PROMPT_VARIANT = "training_v7"
PROMPT_VERSION = "image_edit_v6_direct_root_xyxy_v7_qwen35_v7"
PROMPT_CONTRACT_SHA256 = (
    "7984173abd05fe72ea188ff4065db1305c49df1c41c01f18fc3a284046299f6f"
)
HLD_POSTPROCESS_VERSION = (
    "image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7"
)
BBOX_PACKAGING_VERSION = "image_edit_v6_prediction_xyxy_to_yxyx_v1"
CHECKPOINT_STEP = 8000
CHECKPOINT_MANIFEST_SHA256 = (
    "ffc1c9c5410986d33710d5315674c7ceccd13828e315fe5a62c08c0d4a3bce6e"
)
HF_ARTIFACT_MANIFEST_SHA256 = (
    "14f7260235f775669527bbf211ec9dcd3ca05a4d9175b190eef03b2b4a8c9719"
)
HF_ARTIFACT_VARIANT = "trained_mtp"

MAX_MODEL_LEN = 32768
MAX_NEW_TOKENS = 27000
MAX_PIXELS = 2359296
MAX_NUM_BATCHED_TOKENS = 32768
SEED = 42
NUM_SPECULATIVE_TOKENS = 1

EXPECTED_RUNTIME = {
    "torch": "2.11.0",
    "transformers": "5.8.0",
    "vllm": "0.26.0",
    "flashinfer": "0.6.14",
    "triton": "3.6.0",
    "cuda": "13.0",
}


class CompleteInferenceContractError(RuntimeError):
    """Raised before output when checkpoint-8000's frozen contract drifts."""


@dataclass(frozen=True)
class CompleteInferenceExecution:
    """Generator counters plus final packaged-output evidence."""

    counts: dict[str, int]
    single_prediction: dict[str, Any] | None
    telemetry: dict[str, Any]
    packaging: dict[str, Any]


def contract_manifest() -> dict[str, Any]:
    """Return the exact immutable contract enforced by this entrypoint."""

    return {
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "hf_artifact_manifest_sha256": HF_ARTIFACT_MANIFEST_SHA256,
        "hf_artifact_variant": HF_ARTIFACT_VARIANT,
        "target_schema": TARGET_SCHEMA,
        "prompt_variant": PROMPT_VARIANT,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "image_order": ["reference:frame_0", "target"],
        "hld_postprocess_version": HLD_POSTPROCESS_VERSION,
        "bbox_packaging_version": BBOX_PACKAGING_VERSION,
        "model_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
        "output_bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
        "decode": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": SEED,
            "max_model_len": MAX_MODEL_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_pixels": MAX_PIXELS,
            "mtp": {
                "enabled": True,
                "method": "mtp",
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            },
        },
        "runtime": dict(EXPECTED_RUNTIME),
    }


def _freeze_contract_environment() -> None:
    expected = {
        "IMAGE_EDIT_TARGET_SCHEMA": TARGET_SCHEMA,
        "IMAGE_EDIT_V6_PROMPT_VARIANT": PROMPT_VARIANT,
    }
    for name, value in expected.items():
        current = os.environ.get(name)
        if current is not None and current != value:
            raise CompleteInferenceContractError(
                f"{name}={current!r} conflicts with the frozen value {value!r}"
            )
        os.environ[name] = value


def _load_inference_module() -> Any:
    """Import the shared generator only after selecting the frozen contract."""

    _freeze_contract_environment()
    from dlms.datasets import image_edit_inference_contract as prompt_contract
    from scripts import infer_image_edit_full_json as inference

    observed = {
        "target_schema": prompt_contract.TARGET_SCHEMA,
        "prompt_version": prompt_contract.PROMPT_VERSION,
        "prompt_contract_sha256": prompt_contract.PROMPT_CONTRACT_SHA256,
        "computed_prompt_contract_sha256": prompt_contract.prompt_contract_sha256(),
        "hld_postprocess_version": POSTPROCESS_VERSION,
        "bbox_packaging_version": PACKAGING_VERSION,
    }
    expected = {
        "target_schema": TARGET_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "computed_prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "hld_postprocess_version": contract_manifest()["hld_postprocess_version"],
        "bbox_packaging_version": contract_manifest()["bbox_packaging_version"],
    }
    if observed != expected:
        raise CompleteInferenceContractError(
            f"inference contract drifted: observed={observed!r} expected={expected!r}"
        )
    return inference


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_artifact(
    model_path: Path,
    *,
    expected_manifest_sha256: str = HF_ARTIFACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify that ``model_path`` is the pinned checkpoint-8000 HF artifact."""

    try:
        root = model_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CompleteInferenceContractError(
            f"model path does not exist: {model_path}"
        ) from exc
    if not root.is_dir():
        raise CompleteInferenceContractError(f"model path is not a directory: {root}")
    marker_path = root / "_SUCCESS"
    manifest_path = root / "mcore_to_hf_manifest.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise CompleteInferenceContractError(
            f"model artifact metadata is missing or invalid under {root}: {exc}"
        ) from exc

    actual_manifest_sha256 = _sha256_file(manifest_path)
    marker_expected = {
        "kind": "mcore-model-to-hf-commit",
        "status": "complete",
        "source_step": CHECKPOINT_STEP,
        "variant": HF_ARTIFACT_VARIANT,
        "manifest": manifest_path.name,
        "manifest_sha256": expected_manifest_sha256,
    }
    marker_observed = {key: marker.get(key) for key in marker_expected}
    if marker_observed != marker_expected:
        raise CompleteInferenceContractError(
            f"checkpoint-8000 _SUCCESS marker drifted: {marker_observed!r}"
        )
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise CompleteInferenceContractError(
            "checkpoint-8000 HF manifest SHA-256 drifted: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )

    source = manifest.get("source_mcore_checkpoint")
    expected_source = {
        "step": CHECKPOINT_STEP,
        "manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
    }
    observed_source = (
        {key: source.get(key) for key in expected_source}
        if isinstance(source, Mapping)
        else None
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("variant") != HF_ARTIFACT_VARIANT
        or observed_source != expected_source
    ):
        raise CompleteInferenceContractError(
            "checkpoint-8000 conversion manifest source/variant drifted"
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CompleteInferenceContractError(
            "checkpoint-8000 conversion manifest has no file inventory"
        )
    declared_total_bytes = 0
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise CompleteInferenceContractError(
                f"checkpoint-8000 manifest files[{index}] is not an object"
            )
        relative = item.get("path")
        declared_size = item.get("size")
        declared_sha256 = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(declared_size) is not int
            or declared_size < 0
            or not isinstance(declared_sha256, str)
            or len(declared_sha256) != 64
            or any(character not in "0123456789abcdef" for character in declared_sha256)
        ):
            raise CompleteInferenceContractError(
                f"checkpoint-8000 manifest files[{index}] is invalid"
            )
        try:
            candidate = (root / relative).resolve(strict=True)
        except FileNotFoundError as exc:
            raise CompleteInferenceContractError(
                f"checkpoint-8000 model file is missing: {relative!r}"
            ) from exc
        if root not in candidate.parents or not candidate.is_file():
            raise CompleteInferenceContractError(
                f"checkpoint-8000 model file escapes the artifact root: {relative!r}"
            )
        actual_size = candidate.stat().st_size
        if actual_size != declared_size:
            raise CompleteInferenceContractError(
                f"checkpoint-8000 model file size drifted for {relative!r}: "
                f"{actual_size} != {declared_size}"
            )
        declared_total_bytes += declared_size
    if (
        manifest.get("file_count") != len(files)
        or manifest.get("total_bytes") != declared_total_bytes
    ):
        raise CompleteInferenceContractError(
            "checkpoint-8000 model file count/byte total drifted"
        )
    return {
        "model_path": str(root),
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "hf_artifact_manifest_sha256": actual_manifest_sha256,
        "variant": HF_ARTIFACT_VARIANT,
        "file_count": len(files),
        "total_bytes": declared_total_bytes,
    }


class PackagedJSONLWriter:
    """Convert complete inference windows to final YXYX rows before writing."""

    def __init__(self, output_handle: TextIO) -> None:
        self.output_handle = output_handle
        self.rows = 0
        self.packaged_rows = 0
        self.skipped_invalid_rows = 0
        self.hld_changed_rows = 0
        self.layout_changed_rows = 0
        self.invalid_hld_labels_removed = 0
        self.bboxes_converted = 0
        self.single_prediction: dict[str, Any] | None = None

    def write(self, payload: str) -> int:
        if not isinstance(payload, str):
            raise TypeError("inference output payload must be text")
        if not payload:
            return 0
        if not payload.endswith("\n"):
            raise CompleteInferenceContractError(
                "inference writer received a non-terminated JSONL window"
            )

        packaged_rows: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                raise CompleteInferenceContractError(
                    f"inference writer received blank JSONL line {line_number}"
                )
            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CompleteInferenceContractError(
                    f"inference writer received invalid JSONL line {line_number}: {exc}"
                ) from exc
            try:
                packaged_row, audit = package_v6_inference_row_yxyx(
                    raw_row,
                    expected_hld_postprocess_version=HLD_POSTPROCESS_VERSION,
                )
            except V6BBoxPackagingError as exc:
                raise CompleteInferenceContractError(
                    f"inference row {line_number} failed final packaging: {exc}"
                ) from exc
            if packaged_row.get("bbox_coordinate_order") != PACKAGED_BBOX_COORDINATE_ORDER:
                raise CompleteInferenceContractError(
                    f"packaged line {line_number} has the wrong bbox coordinate order"
                )
            if audit.get("status") not in {
                "packaged",
                "skipped_invalid_prediction",
            }:
                raise CompleteInferenceContractError(
                    f"unexpected bbox packaging status: {audit.get('status')!r}"
                )
            if audit.get("version") != BBOX_PACKAGING_VERSION:
                raise CompleteInferenceContractError(
                    f"bbox packaging version drifted: {audit.get('version')!r}"
                )
            packaged_rows.append(packaged_row)
            audits.append(audit)

        serialized = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in packaged_rows
        )
        written = self.output_handle.write(serialized)
        if written is not None and written != len(serialized):
            raise OSError(
                f"short packaged JSONL write: wrote {written} of {len(serialized)} characters"
            )

        for row, audit in zip(packaged_rows, audits, strict=True):
            self.rows += 1
            if audit.get("status") == "packaged":
                self.packaged_rows += 1
            else:
                self.skipped_invalid_rows += 1
            cleanup = row.get("hld_label_cleanup")
            if isinstance(cleanup, Mapping):
                self.hld_changed_rows += int(cleanup.get("hld_changed") is True)
                self.layout_changed_rows += int(cleanup.get("layout_changed") is True)
                removed = cleanup.get("invalid_label_occurrences_removed", 0)
                if type(removed) is int:
                    self.invalid_hld_labels_removed += removed
            converted = audit.get("total_bboxes_converted", 0)
            if type(converted) is int:
                self.bboxes_converted += converted
            if (
                row.get("input_index") == 0
                and row.get("schema_valid") is True
                and isinstance(row.get("prediction"), dict)
            ):
                self.single_prediction = row["prediction"]
        return len(payload)

    def flush(self) -> None:
        self.output_handle.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "version": BBOX_PACKAGING_VERSION,
            "hld_postprocess_version": HLD_POSTPROCESS_VERSION,
            "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
            "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
            "rows": self.rows,
            "packaged_rows": self.packaged_rows,
            "skipped_invalid_rows": self.skipped_invalid_rows,
            "hld_changed_rows": self.hld_changed_rows,
            "layout_changed_rows": self.layout_changed_rows,
            "invalid_hld_labels_removed": self.invalid_hld_labels_removed,
            "bboxes_converted": self.bboxes_converted,
        }


def infer_complete_records(
    args: argparse.Namespace,
    runtime: Any,
    records: Iterable[Any],
    output_handle: TextIO,
    *,
    inference_module: Any | None = None,
) -> CompleteInferenceExecution:
    """Run shared inference and stream only final packaged rows to ``output``."""

    inference = inference_module or _load_inference_module()
    _require_frozen_args(args)
    packaged_output = PackagedJSONLWriter(output_handle)
    execution = inference.infer_records(args, runtime, records, packaged_output)
    packaged_output.flush()
    if packaged_output.rows != execution.counts.get("rows"):
        raise CompleteInferenceContractError(
            "generator and packaging row counts differ: "
            f"{execution.counts.get('rows')} != {packaged_output.rows}"
        )
    return CompleteInferenceExecution(
        counts=dict(execution.counts),
        single_prediction=packaged_output.single_prediction,
        telemetry=dict(execution.telemetry),
        packaging=packaged_output.summary(),
    )


_FROZEN_DEFAULT_ARGV = (
    "--target-schema", TARGET_SCHEMA,
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-new-tokens", str(MAX_NEW_TOKENS),
    "--max-pixels", str(MAX_PIXELS),
    "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
    "--mm-processor-cache-gb", "0",
    "--attention-backend", "FLASHINFER",
    "--gdn-prefill-backend", "triton",
    "--mm-encoder-attn-backend", "FLASH_ATTN",
    "--cudagraph-mode", "PIECEWISE",
    "--cudagraph-specialize-lora",
    "--enable-tower-connector-lora",
    "--no-enable-prefix-caching",
    "--speculative-attention-backend", "auto",
    "--use-mtp",
    "--num-speculative-tokens", str(NUM_SPECULATIVE_TOKENS),
    "--seed", str(SEED),
    "--expected-torch-version", EXPECTED_RUNTIME["torch"],
    "--expected-transformers-version", EXPECTED_RUNTIME["transformers"],
    "--expected-vllm-version", EXPECTED_RUNTIME["vllm"],
    "--expected-flashinfer-version", EXPECTED_RUNTIME["flashinfer"],
    "--expected-triton-version", EXPECTED_RUNTIME["triton"],
    "--expected-cuda-version", EXPECTED_RUNTIME["cuda"],
)


def _require_frozen_args(args: argparse.Namespace) -> None:
    expected = {
        "target_schema": TARGET_SCHEMA,
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_NEW_TOKENS,
        "max_pixels": MAX_PIXELS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "mm_processor_cache_gb": 0.0,
        "attention_backend": "FLASHINFER",
        "gdn_prefill_backend": "triton",
        "mm_encoder_attn_backend": "FLASH_ATTN",
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_specialize_lora": True,
        "enable_tower_connector_lora": True,
        "enable_prefix_caching": False,
        "speculative_attention_backend": "auto",
        "use_mtp": True,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "seed": SEED,
        "expected_torch_version": EXPECTED_RUNTIME["torch"],
        "expected_transformers_version": EXPECTED_RUNTIME["transformers"],
        "expected_vllm_version": EXPECTED_RUNTIME["vllm"],
        "expected_flashinfer_version": EXPECTED_RUNTIME["flashinfer"],
        "expected_triton_version": EXPECTED_RUNTIME["triton"],
        "expected_cuda_version": EXPECTED_RUNTIME["cuda"],
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        differences = {
            name: {"observed": observed[name], "expected": value}
            for name, value in expected.items()
            if observed[name] != value
        }
        raise CompleteInferenceContractError(
            f"checkpoint-8000 inference arguments drifted: {differences!r}"
        )
    if args.tokenizer is not None and (
        Path(args.tokenizer).resolve() != Path(args.model).resolve()
    ):
        raise CompleteInferenceContractError(
            "checkpoint-8000 must use the tokenizer bundled in --model"
        )


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    inference_module: Any | None = None,
) -> argparse.Namespace:
    inference = inference_module or _load_inference_module()
    user_argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in user_argv or "--help" in user_argv:
        parser = inference.build_parser()
        parser.description = __doc__
        parser.parse_args([*_FROZEN_DEFAULT_ARGV, *user_argv])
        raise AssertionError("argparse --help must terminate the process")
    args = inference.parse_args([*_FROZEN_DEFAULT_ARGV, *user_argv])
    _require_frozen_args(args)
    if args.max_cudagraph_capture_size is None:
        args.max_cudagraph_capture_size = args.batch_size
    return args


def _records(args: argparse.Namespace, inference: Any) -> Iterable[Any]:
    if args.input_jsonl:
        return inference.iter_input_jsonl(
            args.input_jsonl,
            max_samples=args.max_samples,
            rank=args.rank,
            world_size=args.world_size,
        )
    return [
        inference.InferenceRecord(
            input_index=0,
            sample_id=args.sample_id,
            instruction=args.instruction.strip(),
            reference_image=args.reference_image.strip(),
            target_image=args.target_image.strip(),
        )
    ]


def run(
    args: argparse.Namespace,
    *,
    inference_module: Any | None = None,
) -> dict[str, Any]:
    inference = inference_module or _load_inference_module()
    _require_frozen_args(args)
    model_lock = validate_checkpoint_artifact(Path(args.model))
    inference.configure_aws_environment(args)
    runtime = inference.initialize_inference_runtime(args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        execution = infer_complete_records(
            args,
            runtime,
            _records(args, inference),
            output_handle,
            inference_module=inference,
        )

    if args.prediction_output:
        if execution.single_prediction is None:
            raise RuntimeError(
                "cannot write --prediction-output because no packaged schema-valid "
                "input_index=0 prediction was produced"
            )
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        args.prediction_output.write_text(
            json.dumps(
                execution.single_prediction,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "output_jsonl": str(args.output_jsonl),
        "prediction_output": (
            str(args.prediction_output) if args.prediction_output else None
        ),
        "contract": contract_manifest(),
        "model": model_lock,
        "counts": execution.counts,
        "packaging": execution.packaging,
        "telemetry": execution.telemetry,
    }


def main(argv: Sequence[str] | None = None) -> int:
    inference = _load_inference_module()
    args = parse_args(argv, inference_module=inference)
    summary = run(args, inference_module=inference)
    print(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
