#!/usr/bin/env python3
"""Run selected training prompts on TEST 1070 with optional vLLM MTP.

The default launcher starts eight independent tensor-parallel-size-1 workers,
one per GPU.  Each worker downloads its manifest shard once, then reuses those
node-local images for all four prompts.  Result JSONL files retain the original
manifest URI in ``images[0].path`` so they can be consumed directly by the
hierarchical-json TEST 1070 importer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_MANIFEST = "s3://core-cn-oss.canva.com/usr/dexiang/pipeline_1k_test_sample/caption_test_1k.jsonl"
EXPECTED_IMAGE_COUNT = 1070
MAX_PIXELS = 1536 * 1536
MAX_MODEL_LEN = 20480
DEFAULT_MAX_NEW_TOKENS = 18000
DEFAULT_EOS_RESERVE_TOKENS = 1
COMPARISON_CONDITIONS = ("trained_mtp", "original_mtp", "no_mtp")
CONDITIONS = (*COMPARISON_CONDITIONS, "initial_mtp")
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_SAFE_SUFFIXES = frozenset({".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TaskSpec:
    """One task contract used during training."""

    name: str
    output_name: str
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class TrainingProfile:
    """Training-mixture provenance attached to every evaluation result."""

    name: str
    training_config: str
    source_probabilities: Mapping[str, float]
    conditional_primary_task_probabilities: Mapping[str, float]
    effective_overall_task_probabilities: Mapping[str, float]


TASKS = (
    TaskSpec(
        name="global_desc_bg",
        output_name="global_desc_bg",
        system_prompt=(
            "You extract global design context, background, and layout geometry from an image. "
            "Return only valid JSON in the requested compact schema."
        ),
        user_prompt=(
            "<image>\nGenerate JSON containing high_level_description, elements.background, and "
            "elements.layout. Every layout item must contain only type, label, parent, and bbox in "
            "xyxy_1000 coordinates. Do not output style_description, desc, or color_palette."
        ),
    ),
    TaskSpec(
        name="full_json_no_region_desc",
        output_name="full_json_no_region_desc",
        system_prompt=(
            "You extract global design context, style, background, region color palettes, and "
            "layout geometry from an image. Return only valid JSON in the requested schema, "
            "without region-level captions."
        ),
        user_prompt=(
            "<image>\nGenerate JSON containing high_level_description, style_description, "
            "elements.background, and elements.layout. Every layout item must contain only type, "
            "label, parent, bbox in xyxy_1000 coordinates, and color_palette. Do not output a desc "
            "key or any region-level caption. Do not output prose or Markdown."
        ),
    ),
    TaskSpec(
        name="full_json",
        output_name="full_json",
        system_prompt=(
            "You produce a complete structured design description from an image. "
            "Return only valid JSON in the requested full schema."
        ),
        user_prompt=(
            "<image>\nGenerate the complete JSON containing high_level_description, "
            "style_description, elements.background, and elements.layout. Every layout item must "
            "contain type, label, parent, bbox in xyxy_1000 coordinates, desc, and color_palette. "
            'When exact visible text is available for an item of type "text", include it in the '
            "text field."
        ),
    ),
    TaskSpec(
        name="layout_only",
        output_name="layout_only",
        system_prompt=(
            "You extract only layout hierarchy and geometry from an image. "
            "Return only valid JSON and no prose."
        ),
        user_prompt=(
            "<image>\nGenerate only elements.layout. Every layout item must contain only type, "
            "label, parent, and bbox in xyxy_1000 coordinates. Do not output "
            "high_level_description, style_description, elements.background, desc, or "
            "color_palette."
        ),
    ),
)
TASK_BY_NAME = {task.name: task for task in TASKS}

# The cleaned Jimmy 1M+5M run deliberately removed ``color_palette`` from
# both targets and prompts. Keep the actual training task names while choosing
# their prompt contract through the training profile.
IMAGE2JSON_JIMMY_TASKS = (
    TaskSpec(
        name="full_json",
        output_name="full_json",
        system_prompt=(
            "You produce a complete structured design description from an image. "
            "Return only valid JSON in the requested full schema."
        ),
        user_prompt=(
            "<image>\nGenerate the complete JSON containing high_level_description, "
            "style_description, elements.background, and elements.layout. Every layout item must "
            "contain type, label, parent, bbox in xyxy_1000 coordinates, and desc. When exact visible "
            'text is available for an item of type "text", include it in the text field.'
        ),
    ),
    TaskSpec(
        name="layout_only",
        output_name="layout_only",
        system_prompt=(
            "You extract only layout hierarchy and geometry from an image. "
            "Return only valid JSON and no prose."
        ),
        user_prompt=(
            "<image>\nGenerate only elements.layout. Every layout item must contain only type, "
            "label, parent, and bbox in xyxy_1000 coordinates. Do not output high_level_description, "
            "style_description, elements.background, or desc."
        ),
    ),
)
TASKS_BY_TRAINING_PROFILE = {
    profile_name: {task.name: task for task in IMAGE2JSON_JIMMY_TASKS}
    for profile_name in (
        "image2json-jimmy-two-task",
        "image2json-jimmy-chart10-two-task",
    )
}
DEFAULT_EVALUATION_TASKS = ("full_json", "layout_only")


def resolve_tasks(
    value: str | Sequence[str],
    training_profile: str = "c4b-two-task",
) -> tuple[TaskSpec, ...]:
    """Resolve a non-empty, duplicate-free task subset in requested order."""

    task_by_name = TASKS_BY_TRAINING_PROFILE.get(training_profile, TASK_BY_NAME)
    raw_names = value.split(",") if isinstance(value, str) else list(value)
    names = [str(name).strip() for name in raw_names if str(name).strip()]
    if not names:
        raise ValueError("at least one evaluation task is required")
    if len(set(names)) != len(names):
        raise ValueError(f"evaluation tasks contain duplicates: {names!r}")
    unknown = [name for name in names if name not in task_by_name]
    if unknown:
        raise ValueError(
            f"unknown evaluation tasks {unknown!r}; expected one or more of {sorted(task_by_name)!r}"
        )
    return tuple(task_by_name[name] for name in names)

DEFAULT_TRAINING_PROFILE = "c4b-two-task"
TRAINING_PROFILES = {
    "c4b-two-task": TrainingProfile(
        name="c4b-two-task",
        training_config="dlms/datasets/craft_short_two_task.py",
        source_probabilities={"primary_craft": 0.44, "reve2_region_layout": 0.56},
        conditional_primary_task_probabilities={
            "full_json": 15 / 22,
            "layout_only": 7 / 22,
        },
        effective_overall_task_probabilities={
            "full_json": 0.30,
            "layout_only": 0.70,
        },
    ),
    "baseline4": TrainingProfile(
        name="baseline4",
        training_config="job_config/craft27b_full_mtp_4node.yaml",
        source_probabilities={"primary_craft": 0.3, "reve2_region_layout": 0.7},
        conditional_primary_task_probabilities={
            "global_desc_bg": 0.17777777777777778,
            "full_json_no_region_desc": 0.35555555555555557,
            "full_json": 0.26666666666666666,
            "layout_only": 0.2,
        },
        effective_overall_task_probabilities={
            "global_desc_bg": 0.05333333333333334,
            "full_json_no_region_desc": 0.10666666666666667,
            "full_json": 0.08,
            "layout_only": 0.76,
        },
    ),
    "fulljson-tail6000": TrainingProfile(
        name="fulljson-tail6000",
        training_config="job_config/craft27b_mtp_fulljson30_tail_from12000_4node.yaml",
        source_probabilities={"primary_craft": 0.5, "reve2_region_layout": 0.5},
        conditional_primary_task_probabilities={
            "global_desc_bg": 0.12,
            "full_json_no_region_desc": 0.24,
            "full_json": 0.60,
            "layout_only": 0.04,
        },
        effective_overall_task_probabilities={
            "global_desc_bg": 0.06,
            "full_json_no_region_desc": 0.12,
            "full_json": 0.30,
            "layout_only": 0.52,
        },
    ),
    "image2json-jimmy-two-task": TrainingProfile(
        name="image2json-jimmy-two-task",
        training_config="dlms/datasets/image2json_two_task.py",
        source_probabilities={"jimmy_vote_gt50_1m5m": 1.0},
        conditional_primary_task_probabilities={
            "full_json": 0.30,
            "layout_only": 0.70,
        },
        effective_overall_task_probabilities={
            "full_json": 0.30,
            "layout_only": 0.70,
        },
    ),
    "image2json-jimmy-chart10-two-task": TrainingProfile(
        name="image2json-jimmy-chart10-two-task",
        training_config="dlms/datasets/image2json_two_task.py",
        source_probabilities={
            "jimmy_vote_gt50_1m5m": 0.90,
            "chartgalaxy_craft_rev8_layout_only": 0.10,
        },
        conditional_primary_task_probabilities={
            "full_json": 1 / 3,
            "layout_only": 2 / 3,
        },
        effective_overall_task_probabilities={
            "full_json": 0.30,
            "layout_only": 0.70,
        },
    ),
}


def resolve_training_profile(value: str | TrainingProfile) -> TrainingProfile:
    if isinstance(value, TrainingProfile):
        return value
    try:
        return TRAINING_PROFILES[value]
    except KeyError as exc:
        raise ValueError(f"unknown training profile: {value!r}") from exc


def task_probability_provenance(value: str | TrainingProfile) -> dict[str, Any]:
    profile = resolve_training_profile(value)
    if profile.name == "image2json-jimmy-two-task":
        scope = {
            "conditional_primary_task_probabilities": "P(task | Jimmy 1M+5M source)",
            "effective_overall_task_probabilities": "P(task across all training rows)",
        }
    elif profile.name == "image2json-jimmy-chart10-two-task":
        scope = {
            "conditional_primary_task_probabilities": "P(task | Jimmy 1M+5M source)",
            "effective_overall_task_probabilities": (
                "P(task across Jimmy 1M+5M plus the 10% ChartGalaxy layout-only source)"
            ),
        }
    else:
        scope = {
            "conditional_primary_task_probabilities": "P(task | primary CRAFT source)",
            "effective_overall_task_probabilities": (
                "P(task across all training rows); layout_only includes Reve2 region-layout rows"
            ),
        }
    return {
        "profile": profile.name,
        "training_config": profile.training_config,
        "source_probabilities": dict(profile.source_probabilities),
        "conditional_primary_task_probabilities": dict(
            profile.conditional_primary_task_probabilities
        ),
        "effective_overall_task_probabilities": dict(
            profile.effective_overall_task_probabilities
        ),
        "scope": scope,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_initial_mtp_artifact_identity(
    model_dir: Path,
    *,
    expected_source_step: int | None,
    expected_source_checkpoint_manifest_sha256: str | None,
    expected_trained_artifact_manifest_sha256: str | None,
    expected_initial_mtp_artifact_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Verify the sealed legacy FSDP-to-HF bootstrap without rehashing 110 GB per worker."""

    from scripts.checkpoint.hf_artifact_s3 import (
        MANIFEST_NAME,
        PROTOCOL_VERSION,
        SUCCESS_NAME,
        validate_conversion_manifest,
    )

    if expected_trained_artifact_manifest_sha256 is not None:
        raise ValueError(
            "initial_mtp has no trained HF artifact; pin it with "
            "--expected-initial-mtp-artifact-manifest-sha256"
        )

    manifest_path = model_dir / MANIFEST_NAME
    marker_path = model_dir / SUCCESS_NAME
    manifest = _read_json_object(manifest_path)
    marker = _read_json_object(marker_path)
    entries = validate_conversion_manifest(manifest, source=str(manifest_path))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_marker = {
        "version": PROTOCOL_VERSION,
        "kind": "fsdp-model-only-to-hf-commit",
        "status": "complete",
        "source_step": manifest["source_step"],
        "manifest": MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"initial_mtp artifact marker {key} mismatch: "
                f"expected={expected!r} actual={marker.get(key)!r}"
            )

    source_checkpoint_uri = manifest.get("source_checkpoint")
    initial_model_uri = manifest.get("initial_model")
    artifact_uri = manifest.get("artifact_s3")
    for label, uri in (
        ("source_checkpoint", source_checkpoint_uri),
        ("initial_model", initial_model_uri),
        ("artifact_s3", artifact_uri),
    ):
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise ValueError(f"initial_mtp manifest has invalid {label}: {uri!r}")

    coverage = manifest["key_coverage"]
    copied_mtp_files = coverage.get("copied_mtp_files")
    if (
        not isinstance(copied_mtp_files, list)
        or len(copied_mtp_files) != 1
        or copied_mtp_files[0] not in entries
    ):
        raise ValueError(
            "initial_mtp manifest must identify exactly one committed MTP weight file"
        )
    source_checkpoint_sha256 = manifest["source_checkpoint_manifest_sha256"]
    initial_model_sha256 = manifest["initial_model_manifest_sha256"]
    pins = {
        "source checkpoint manifest": (
            expected_source_checkpoint_manifest_sha256,
            source_checkpoint_sha256,
        ),
        "initial MTP artifact manifest": (
            expected_initial_mtp_artifact_manifest_sha256,
            manifest_sha256,
        ),
    }
    if expected_source_step is not None and manifest["source_step"] != expected_source_step:
        raise ValueError(
            f"source step mismatch: expected={expected_source_step} "
            f"actual={manifest['source_step']}"
        )
    for label, (expected, actual) in pins.items():
        if expected is not None and expected != actual:
            raise ValueError(f"{label} mismatch: expected={expected} actual={actual}")

    return {
        "condition": "initial_mtp",
        "variant": "initial_mtp",
        "artifact_protocol": "fsdp-model-only-to-hf",
        "local_manifest_sha256": manifest_sha256,
        "trained_artifact_manifest_sha256": None,
        "source_checkpoint": {
            "kind": "fsdp",
            "step": manifest["source_step"],
            "manifest_sha256": source_checkpoint_sha256,
            "uri": source_checkpoint_uri,
        },
        "initial_mtp_artifact": {
            "uri": artifact_uri,
            "manifest_sha256": manifest_sha256,
            "mtp_file": copied_mtp_files[0],
        },
        "mtp_source_model": {
            "uri": initial_model_uri,
            "manifest_sha256": initial_model_sha256,
        },
        "tensor_coverage": {
            "non_mtp": coverage["merged_non_mtp"],
            "mtp": coverage["mtp"],
            "mtp_files": list(copied_mtp_files),
            "changed_mtp_keys": [],
        },
        "lineage": {
            "backbone": "source_checkpoint",
            "mtp_head": "mtp_source_model",
        },
    }


def verify_model_artifact_identity(
    model_dir: Path,
    condition: str,
    *,
    expected_source_step: int | None = None,
    expected_source_checkpoint_manifest_sha256: str | None = None,
    expected_trained_artifact_manifest_sha256: str | None = None,
    expected_initial_mtp_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind inference provenance to the sealed local HF artifact.

    Restore and original-head derivation already verify every object and tensor.
    This lightweight check intentionally revalidates the immutable manifests,
    condition semantics, and caller-provided lineage pins before an engine is
    loaded, without hashing the 56 GB model once per worker.
    """

    model_dir = model_dir.resolve()
    if condition == "initial_mtp":
        return _verify_initial_mtp_artifact_identity(
            model_dir,
            expected_source_step=expected_source_step,
            expected_source_checkpoint_manifest_sha256=(
                expected_source_checkpoint_manifest_sha256
            ),
            expected_trained_artifact_manifest_sha256=(
                expected_trained_artifact_manifest_sha256
            ),
            expected_initial_mtp_artifact_manifest_sha256=(
                expected_initial_mtp_artifact_manifest_sha256
            ),
        )

    from scripts.mtp_eval.hf_artifact import (
        COMMIT_KIND,
        EXPECTED_MTP_TENSOR_COUNT,
        EXPECTED_NON_MTP_TENSOR_COUNT,
        MANIFEST_NAME,
        PROTOCOL_VERSION,
        SUCCESS_NAME,
        validate_manifest,
    )

    manifest_path = model_dir / MANIFEST_NAME
    marker_path = model_dir / SUCCESS_NAME
    manifest = _read_json_object(manifest_path)
    marker = _read_json_object(marker_path)
    validate_manifest(manifest, source=str(manifest_path))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_variant = "original_mtp" if condition == "original_mtp" else "trained_mtp"
    expected_marker = {
        "version": PROTOCOL_VERSION,
        "kind": COMMIT_KIND,
        "variant": expected_variant,
        "status": "complete",
        "source_step": manifest["source_mcore_checkpoint"]["step"],
        "manifest": MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"artifact marker {key} mismatch for {condition}: "
                f"expected={expected!r} actual={marker.get(key)!r}"
            )
    if manifest.get("variant") != expected_variant:
        raise ValueError(
            f"condition {condition} requires {expected_variant}, got {manifest.get('variant')!r}"
        )

    source_checkpoint = manifest["source_mcore_checkpoint"]
    initial_artifact = manifest["initial_mtp_artifact"]
    coverage = manifest["tensor_coverage"]
    lineage = manifest.get("lineage") if expected_variant == "original_mtp" else None
    if expected_variant == "original_mtp":
        if not isinstance(lineage, dict):
            raise ValueError("original_mtp artifact is missing local derivation lineage")
        trained_artifact_manifest_sha256 = lineage.get("trained_local_manifest_sha256")
        if lineage.get("initial_manifest_sha256") != initial_artifact["manifest_sha256"]:
            raise ValueError("original_mtp initial-head lineage does not match its manifest")
        if lineage.get("initial_artifact_s3") != initial_artifact["uri"]:
            raise ValueError("original_mtp initial-head URI does not match its manifest")
    else:
        trained_artifact_manifest_sha256 = manifest_sha256

    pins = {
        "source checkpoint manifest": (
            expected_source_checkpoint_manifest_sha256,
            source_checkpoint["manifest_sha256"],
        ),
        "trained HF artifact manifest": (
            expected_trained_artifact_manifest_sha256,
            trained_artifact_manifest_sha256,
        ),
        "initial MTP artifact manifest": (
            expected_initial_mtp_artifact_manifest_sha256,
            initial_artifact["manifest_sha256"],
        ),
    }
    if expected_source_step is not None and source_checkpoint["step"] != expected_source_step:
        raise ValueError(
            f"source step mismatch: expected={expected_source_step} actual={source_checkpoint['step']}"
        )
    for label, (expected, actual) in pins.items():
        if expected is not None and expected != actual:
            raise ValueError(f"{label} mismatch: expected={expected} actual={actual}")

    return {
        "condition": condition,
        "variant": expected_variant,
        "local_manifest_sha256": manifest_sha256,
        "trained_artifact_manifest_sha256": trained_artifact_manifest_sha256,
        "source_mcore_checkpoint": dict(source_checkpoint),
        "initial_mtp_artifact": dict(initial_artifact),
        "tensor_coverage": {
            "non_mtp": EXPECTED_NON_MTP_TENSOR_COUNT,
            "mtp": EXPECTED_MTP_TENSOR_COUNT,
            "mtp_files": list(coverage["mtp_files"]),
            "mtp_tensor_sha256": dict(coverage["mtp_tensor_sha256"]),
            "changed_mtp_keys": list(coverage["changed_mtp_keys"]),
        },
        "lineage": dict(lineage) if isinstance(lineage, dict) else None,
    }


def artifact_comparison_lineage(identity: Mapping[str, Any]) -> dict[str, Any]:
    source_checkpoint = (
        identity.get("source_mcore_checkpoint") or identity.get("source_checkpoint") or {}
    )
    initial_artifact = identity.get("initial_mtp_artifact") or {}
    return {
        "trained_artifact_manifest_sha256": identity.get(
            "trained_artifact_manifest_sha256"
        ),
        "source_step": source_checkpoint.get("step"),
        "source_checkpoint_manifest_sha256": source_checkpoint.get("manifest_sha256"),
        "initial_mtp_artifact_manifest_sha256": initial_artifact.get("manifest_sha256"),
    }


def _json_default(value: Any) -> str:
    return str(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def _read_text_uri(uri: str, *, s3_client: Any | None = None) -> str:
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        client = s3_client or _s3_client()
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8")
    if uri.startswith(("http://", "https://")):
        with urlopen(uri) as response:  # noqa: S310 - explicit manifest URI
            return response.read().decode("utf-8")
    if uri.startswith("file://"):
        uri = urlparse(uri).path
    return Path(uri).read_text(encoding="utf-8")


def load_manifest(
    uri: str,
    *,
    expected_count: int = EXPECTED_IMAGE_COUNT,
    s3_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Load and strictly validate the standard TEST 1070 manifest."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_text_uri(uri, s3_client=s3_client).splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not a JSON object")
        image_uri = value.get("s3_image_path")
        if not isinstance(image_uri, str) or not image_uri:
            raise ValueError(f"manifest line {line_number} has no s3_image_path")
        expected_sha256 = value.get("sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError(f"manifest line {line_number} has an invalid image sha256")
        expected_size = value.get("content_length")
        if expected_size is not None and (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ValueError(f"manifest line {line_number} has an invalid content_length")
        rows.append(value)
    if expected_count >= 0 and len(rows) != expected_count:
        raise ValueError(f"manifest row count is {len(rows)}, expected {expected_count}")
    image_uris = [str(row["s3_image_path"]) for row in rows]
    if len(set(image_uris)) != len(image_uris):
        raise ValueError("manifest contains duplicate s3_image_path values")
    return rows


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    value = "\n".join(str(row["s3_image_path"]) for row in rows) + "\n"
    return sha256_text(value)


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    return list(range(rank, total, world_size))


def _cache_destination(cache_dir: Path, index: int, image_uri: str) -> Path:
    suffix = Path(urlparse(image_uri).path).suffix.lower()
    if suffix not in _SAFE_SUFFIXES:
        suffix = ".image"
    digest = sha256_text(image_uri)[:16]
    return cache_dir / f"{index:06d}-{digest}{suffix}"


def _download_image(
    index: int,
    image_uri: str,
    cache_dir: Path,
    s3_client: Any | None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[int, Path]:
    def validate_integrity(path: Path) -> None:
        if expected_size is not None and path.stat().st_size != expected_size:
            raise IOError(
                f"image size mismatch for {image_uri}: "
                f"expected={expected_size} actual={path.stat().st_size}"
            )
        if expected_sha256 is not None:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != expected_sha256:
                raise IOError(
                    f"image SHA-256 mismatch for {image_uri}: "
                    f"expected={expected_sha256} actual={digest}"
                )

    destination = _cache_destination(cache_dir, index, image_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        validate_integrity(destination)
        return index, destination
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        if image_uri.startswith("s3://"):
            bucket, key = _parse_s3_uri(image_uri)
            client = s3_client or _s3_client()
            client.download_file(bucket, key, str(temporary))
        elif image_uri.startswith(("http://", "https://")):
            with urlopen(image_uri) as response, temporary.open("wb") as output:  # noqa: S310
                shutil.copyfileobj(response, output)
        else:
            source = Path(urlparse(image_uri).path if image_uri.startswith("file://") else image_uri)
            shutil.copyfile(source, temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise IOError(f"downloaded image is empty: {image_uri}")
        validate_integrity(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return index, destination


def cache_manifest_shard(
    manifest: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    cache_dir: Path,
    *,
    download_workers: int = 16,
    s3_client: Any | None = None,
) -> dict[int, Path]:
    """Download a worker's complete image shard before any generation starts."""

    if download_workers <= 0:
        raise ValueError("download_workers must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    requests = [
        (
            index,
            str(manifest[index]["s3_image_path"]),
            manifest[index].get("sha256"),
            manifest[index].get("content_length"),
        )
        for index in indices
    ]
    if s3_client is None and any(uri.startswith("s3://") for _, uri, _, _ in requests):
        # botocore clients are thread-safe after construction.  Reuse one client
        # instead of rebuilding the credential/provider chain for every image.
        s3_client = _s3_client()
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        results = executor.map(
            lambda item: _download_image(
                item[0], item[1], cache_dir, s3_client, item[2], item[3]
            ),
            requests,
        )
    return dict(results)


def template_messages(task: TaskSpec) -> list[dict[str, Any]]:
    """Build Qwen3.5 template messages equivalent to the training example."""

    if not task.user_prompt.startswith("<image>"):
        raise ValueError(f"task {task.name} user prompt does not start with <image>")
    return [
        {"role": "system", "content": task.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<image>"},
                # Preserve the newline following the training <image> marker.
                {"type": "text", "text": task.user_prompt.removeprefix("<image>")},
            ],
        },
    ]


def render_training_prompt(tokenizer: Any, task: TaskSpec) -> str:
    rendered = tokenizer.apply_chat_template(
        template_messages(task),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return a string")
    expected_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not rendered.endswith(expected_suffix):
        raise RuntimeError(
            f"Qwen3.5 template does not match the training inference suffix; expected {expected_suffix!r}"
        )
    if task.system_prompt not in rendered or task.user_prompt.removeprefix("<image>") not in rendered:
        raise RuntimeError(f"Qwen3.5 template dropped content for task {task.name}")
    return rendered


def parse_json_response(response: Any) -> dict[str, Any]:
    text = _THINK_RE.sub("", str(response or ""), count=1)
    text = _FENCE_START_RE.sub("", text, count=1)
    text = _FENCE_END_RE.sub("", text, count=1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer or args.model),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"max_pixels": args.max_pixels},
        "disable_log_stats": False,
        "seed": args.seed,
    }
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.condition != "no_mtp":
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
        }
    return kwargs


def engine_comparison_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "attention_backend": args.attention_backend,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "vllm_runtime_env": {
            name: os.environ.get(name)
            for name in (
                "VLLM_USE_DEEP_GEMM",
                "VLLM_MOE_USE_DEEP_GEMM",
                "VLLM_USE_FLASHINFER_SAMPLER",
            )
        },
    }


def collect_runtime_provenance(artifact_identity: Mapping[str, Any]) -> dict[str, Any]:
    image_uri = os.environ.get("MTP_EVAL_RUNTIME_IMAGE_URI", "")
    image_match = re.search(r"(sha256:[0-9a-f]{64})$", image_uri)
    overlay_sha256 = os.environ.get("MTP_EVAL_CODE_OVERLAY_SHA256")
    if overlay_sha256 is not None and _SHA256_RE.fullmatch(overlay_sha256) is None:
        overlay_sha256 = None

    gpu_hardware_and_clock = None
    gpu_hardware_and_clock_error = None
    try:
        import torch

        properties = torch.cuda.get_device_properties(0)
        gpu_hardware_and_clock = {
            "name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "max_sm_clock_mhz": None,
            "max_memory_clock_mhz": None,
        }
    except Exception as exc:  # noqa: BLE001 - optional provenance must not abort inference
        gpu_hardware_and_clock_error = (
            f"torch CUDA property query failed: {type(exc).__name__}: {exc}"
        )
    else:
        gpu_token = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.max.sm,clocks.max.memory",
                    "--format=csv,noheader,nounits",
                    "-i",
                    gpu_token,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            clock_values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
            if len(clock_values) != 2 or any(
                value in {"", "N/A", "[N/A]"} for value in clock_values
            ):
                raise RuntimeError(
                    f"nvidia-smi returned unusable max clocks: {result.stdout!r}"
                )
            gpu_hardware_and_clock["max_sm_clock_mhz"] = float(clock_values[0])
            gpu_hardware_and_clock["max_memory_clock_mhz"] = float(clock_values[1])
        except Exception as exc:  # noqa: BLE001 - clock telemetry is best-effort
            gpu_hardware_and_clock_error = (
                f"nvidia-smi clock query failed: {type(exc).__name__}: {exc}"
            )

    trained_manifest = artifact_identity.get("trained_artifact_manifest_sha256")
    if isinstance(trained_manifest, str):
        target_backbone_fingerprint = (
            f"trained-hf-artifact-manifest-sha256:{trained_manifest}"
        )
    elif artifact_identity.get("variant") == "initial_mtp" and isinstance(
        artifact_identity.get("local_manifest_sha256"), str
    ):
        target_backbone_fingerprint = (
            "initial-fsdp-hf-artifact-manifest-sha256:"
            f"{artifact_identity['local_manifest_sha256']}"
        )
    else:
        target_backbone_fingerprint = None
    try:
        vllm_version = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        vllm_version = None
    return {
        "target_backbone_fingerprint": target_backbone_fingerprint,
        "runtime_image_digest": image_match.group(1) if image_match else None,
        "vllm_version": vllm_version,
        "code_overlay_sha256": overlay_sha256,
        "gpu_hardware_and_clock": gpu_hardware_and_clock,
        "gpu_hardware_and_clock_error": gpu_hardware_and_clock_error,
    }


def validate_resolved_speculative_config(llm: Any, condition: str) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config.speculative_config
    if condition == "no_mtp":
        if config is not None:
            raise RuntimeError("no_mtp unexpectedly resolved a speculative config")
        return {"enabled": False, "method": None, "num_speculative_tokens": 0}
    if config is None:
        raise RuntimeError(f"{condition} did not resolve a speculative config")
    method = getattr(config, "method", None)
    tokens = getattr(config, "num_speculative_tokens", None)
    if method != "mtp" or tokens != 1:
        raise RuntimeError(
            "resolved speculative config differs from method=mtp,num_speculative_tokens=1: "
            f"method={method!r}, tokens={tokens!r}"
        )
    draft_config = getattr(config, "draft_model_config", None)
    hf_config = getattr(draft_config, "hf_config", None)
    architectures = list(getattr(hf_config, "architectures", None) or [])
    if architectures and "Qwen3_5MTP" not in architectures:
        raise RuntimeError(f"unexpected MTP draft architectures: {architectures!r}")
    return {
        "enabled": True,
        "method": method,
        "num_speculative_tokens": tokens,
        "draft_architectures": architectures,
    }


def _metric_samples(metrics: Iterable[Any]) -> Iterable[tuple[str, float]]:
    for family in metrics:
        samples = getattr(family, "samples", None)
        if samples is None:
            name = str(getattr(family, "name", ""))
            value = getattr(family, "value", None)
            if isinstance(value, (int, float)):
                yield name, float(value)
            continue
        for sample in samples:
            value = getattr(sample, "value", None)
            if isinstance(value, (int, float)):
                yield str(getattr(sample, "name", "")), float(value)


def speculative_metric_snapshot(metrics: Iterable[Any]) -> dict[str, float]:
    """Extract stable counters from vLLM 0.24 Prometheus metric families."""

    result = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    for raw_name, value in _metric_samples(metrics):
        name = raw_name.lower().removesuffix("_total")
        if "spec_decode" not in name and "speculative" not in name:
            continue
        if "accepted_tokens_per_pos" in name:
            continue
        if name.endswith("num_draft_tokens"):
            result["draft_tokens"] += value
        elif name.endswith("num_drafts"):
            result["drafts"] += value
        elif name.endswith("num_accepted_tokens"):
            result["accepted_tokens"] += value
    return result


def metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in ("drafts", "draft_tokens", "accepted_tokens"):
        delta = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        if delta < 0.0:
            raise RuntimeError(
                f"speculative metric counter reset for {key}: "
                f"before={before.get(key, 0.0)} after={after.get(key, 0.0)}"
            )
        values[key] = delta
    denominator = values["draft_tokens"] or values["drafts"]
    values["acceptance_rate"] = values["accepted_tokens"] / denominator if denominator else None
    return values


def validate_speculative_metrics(condition: str, values: Mapping[str, Any]) -> None:
    drafts = float(values.get("drafts", 0.0))
    draft_tokens = float(values.get("draft_tokens", 0.0))
    accepted = float(values.get("accepted_tokens", 0.0))
    if condition == "no_mtp":
        if any(value != 0.0 for value in (drafts, draft_tokens, accepted)):
            raise RuntimeError(f"no_mtp unexpectedly reported speculative metrics: {dict(values)}")
        return
    if drafts <= 0.0 or draft_tokens <= 0.0:
        raise RuntimeError(f"{condition} reported no MTP draft metrics: {dict(values)}")
    if draft_tokens > drafts:
        raise RuntimeError(
            f"{condition} K=1 reported more draft tokens than draft events: "
            f"drafts={drafts} draft_tokens={draft_tokens}"
        )
    if accepted < 0.0 or accepted > draft_tokens:
        raise RuntimeError(
            f"{condition} accepted-token counter is invalid: accepted={accepted} "
            f"draft_tokens={draft_tokens}"
        )


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value >= 0.0]
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean) if clean else None,
    }


def prompt_token_count_from_engine_input(engine_input: Mapping[str, Any]) -> int:
    """Count expanded text+visual tokens in a vLLM 0.24 EngineInput."""

    value: Mapping[str, Any] = engine_input
    decoder = value.get("decoder")
    if isinstance(decoder, Mapping):
        value = decoder
    token_ids = value.get("prompt_token_ids")
    if token_ids is not None:
        return len(token_ids)
    prompt_embeds = value.get("prompt_embeds")
    shape = getattr(prompt_embeds, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    raise ValueError("vLLM preprocessed input has no prompt tokens or embeddings")


def _parse_vllm_model_prompt(model_config: Any, request: Mapping[str, Any]) -> Any:
    from vllm.renderers.inputs.preprocess import parse_model_prompt

    return parse_model_prompt(model_config, request)


def preprocess_prompt_token_count(llm: Any, request: Mapping[str, Any]) -> int:
    """Count expanded tokens through vLLM's isolated readonly MM cache."""

    renderer = getattr(llm, "renderer", None)
    model_config = getattr(llm, "model_config", None)
    render_cmpl = getattr(renderer, "render_cmpl", None)
    default_tok_params = getattr(renderer, "default_cmpl_tok_params", None)
    if model_config is None or render_cmpl is None or default_tok_params is None:
        raise RuntimeError("vLLM does not expose the pinned 0.24 renderer API")
    mm_processor_kwargs = request.get("mm_processor_kwargs")
    prompt_extras = None if mm_processor_kwargs is None else {"mm_processor_kwargs": mm_processor_kwargs}
    engine_inputs = render_cmpl(
        [_parse_vllm_model_prompt(model_config, request)],
        default_tok_params.with_kwargs(),
        prompt_extras=prompt_extras,
        skip_mm_cache=True,
    )
    if len(engine_inputs) != 1:
        raise RuntimeError(f"vLLM renderer returned {len(engine_inputs)} inputs, expected one")
    engine_input = engine_inputs[0]
    return prompt_token_count_from_engine_input(engine_input)


def preprocess_prompt_token_counts(
    llm: Any,
    requests: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Count prompts without inserting hash-only entries in the sender cache."""

    return [preprocess_prompt_token_count(llm, request) for request in requests]


def dynamic_max_new_tokens(
    *,
    prompt_tokens: int,
    max_model_len: int,
    cli_max_new_tokens: int,
    eos_reserve_tokens: int,
) -> int:
    """Cap generation by each request's real context usage and EOS reserve."""

    available = max_model_len - prompt_tokens - eos_reserve_tokens
    result = min(cli_max_new_tokens, available)
    if result <= 0:
        raise ValueError(
            "no output context remains: "
            f"max_model_len={max_model_len}, prompt_tokens={prompt_tokens}, "
            f"eos_reserve_tokens={eos_reserve_tokens}"
        )
    return result


def request_latency_metrics(request_output: Any) -> dict[str, float | None]:
    """Read per-request timing when vLLM returns RequestStateStats."""

    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return {
            "end_to_end_seconds": None,
            "time_to_first_token_seconds": None,
            "decode_seconds": None,
            "engine_inference_seconds": None,
        }
    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    scheduled = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    first_token = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_token = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    decode = last_token - first_token if last_token >= first_token > 0.0 else None
    inference = last_token - scheduled if last_token >= scheduled > 0.0 else None
    end_to_end = first_token_latency + (decode or 0.0) if first_token_latency > 0.0 else None
    return {
        "end_to_end_seconds": end_to_end,
        "time_to_first_token_seconds": first_token_latency or None,
        "decode_seconds": decode,
        "engine_inference_seconds": inference,
    }


def response_sequence_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row["input_index"]))
    canonical = [[int(row["input_index"]), str(row["response_sha256"])] for row in ordered]
    return sha256_text(json.dumps(canonical, separators=(",", ":")))


def make_output_row(
    *,
    task: TaskSpec,
    condition: str,
    training_profile: str | TrainingProfile = DEFAULT_TRAINING_PROFILE,
    artifact_identity: Mapping[str, Any] | None = None,
    manifest_row: Mapping[str, Any],
    input_index: int,
    response: str,
    prompt_tokens: int,
    output_tokens: int,
    max_new_tokens_budget: int,
    request_timing: Mapping[str, float | None],
    finish_reason: Any,
    stop_reason: Any,
) -> dict[str, Any]:
    profile = resolve_training_profile(training_profile)
    try:
        parse_json_response(response)
        json_valid = True
        json_error = None
    except Exception as exc:  # noqa: BLE001 - error is recorded in the result row
        json_valid = False
        json_error = f"{type(exc).__name__}: {exc}"
    image_uri = str(manifest_row["s3_image_path"])
    row = {
        "response": response,
        "labels": "{}",
        "logprobs": None,
        "messages": [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt},
            {"role": "assistant", "content": response},
        ],
        "images": [{"bytes": None, "path": image_uri}],
        "condition": condition,
        "task": task.output_name,
        "training_task_name": task.name,
        "training_profile": profile.name,
        "training_config": profile.training_config,
        "conditional_primary_task_probability": profile.conditional_primary_task_probabilities[
            task.name
        ],
        "effective_overall_task_probability": profile.effective_overall_task_probabilities[
            task.name
        ],
        "input_index": input_index,
        "source": manifest_row.get("source"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "max_new_tokens_budget": max_new_tokens_budget,
        "request_timing": dict(request_timing),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "json_valid": json_valid,
        "json_error": json_error,
        "response_sha256": sha256_text(response),
    }
    if artifact_identity is not None:
        source_checkpoint = (
            artifact_identity.get("source_mcore_checkpoint")
            or artifact_identity.get("source_checkpoint")
            or {}
        )
        row["model_artifact_variant"] = artifact_identity.get("variant")
        row["model_artifact_manifest_sha256"] = artifact_identity.get(
            "local_manifest_sha256"
        )
        row["trained_artifact_manifest_sha256"] = artifact_identity.get(
            "trained_artifact_manifest_sha256"
        )
        row["source_checkpoint_manifest_sha256"] = source_checkpoint.get(
            "manifest_sha256"
        )
    return row


def _write_jsonl_line(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default))
    handle.write("\n")
    handle.flush()


def _sum_task_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sum(int(summary.get("rows", 0)) for summary in summaries)
    output_tokens = sum(int(summary.get("output_tokens", 0)) for summary in summaries)
    prompt_tokens = sum(int(summary.get("prompt_tokens", 0)) for summary in summaries)
    json_valid = sum(int(summary.get("json_valid", 0)) for summary in summaries)
    drafts = sum(float(summary.get("drafts", 0.0)) for summary in summaries)
    draft_tokens = sum(float(summary.get("draft_tokens", 0.0)) for summary in summaries)
    accepted = sum(float(summary.get("accepted_tokens", 0.0)) for summary in summaries)
    generation_wall = max(
        (
            float(
                summary.get(
                    "generation_wall_seconds",
                    summary.get("generation_elapsed_seconds", 0.0),
                )
            )
            for summary in summaries
        ),
        default=0.0,
    )
    image_decode = max(
        (float(summary.get("image_decode_seconds", 0.0)) for summary in summaries),
        default=0.0,
    )
    multimodal_preprocess = max(
        (float(summary.get("multimodal_preprocess_seconds", 0.0)) for summary in summaries),
        default=0.0,
    )
    elapsed = max(
        (float(summary.get("elapsed_seconds", 0.0)) for summary in summaries),
        default=0.0,
    )
    latency_samples = [
        float(value) for summary in summaries for value in summary.get("request_latency_samples_seconds", [])
    ]
    minimum_budgets = [
        int(summary["min_max_new_tokens_budget"])
        for summary in summaries
        if summary.get("min_max_new_tokens_budget") is not None
    ]
    maximum_budgets = [
        int(summary["max_max_new_tokens_budget"])
        for summary in summaries
        if summary.get("max_max_new_tokens_budget") is not None
    ]
    denominator = draft_tokens or drafts
    return {
        "rows": rows,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "json_valid": json_valid,
        "json_valid_rate": json_valid / rows if rows else None,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / denominator if denominator else None,
        "generation_wall_seconds": generation_wall,
        "generation_elapsed_seconds": generation_wall,
        "image_decode_seconds": image_decode,
        "multimodal_preprocess_seconds": multimodal_preprocess,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output_tokens / generation_wall if generation_wall else None,
        "tokens_per_second": output_tokens / generation_wall if generation_wall else None,
        "images_per_second": rows / generation_wall if generation_wall else None,
        "request_latency_seconds": latency_summary(latency_samples),
        "request_latency_samples_seconds": latency_samples,
        "min_max_new_tokens_budget": min(minimum_budgets) if minimum_budgets else None,
        "max_max_new_tokens_budget": max(maximum_budgets) if maximum_budgets else None,
        "sum_rank_generation_elapsed_seconds": sum(
            float(
                summary.get(
                    "generation_wall_seconds",
                    summary.get("generation_elapsed_seconds", 0.0),
                )
            )
            for summary in summaries
        ),
        "sum_rank_image_decode_seconds": sum(
            float(summary.get("image_decode_seconds", 0.0)) for summary in summaries
        ),
        "sum_rank_multimodal_preprocess_seconds": sum(
            float(summary.get("multimodal_preprocess_seconds", 0.0)) for summary in summaries
        ),
        "sum_rank_elapsed_seconds": sum(float(summary.get("elapsed_seconds", 0.0)) for summary in summaries),
    }


def _event(rank_dir: Path, event: str, **fields: Any) -> None:
    payload = {"time": utc_now(), "event": event, **fields}
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    rank_dir.mkdir(parents=True, exist_ok=True)
    with (rank_dir / "runner.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _join_s3_uri(root: str, *parts: str) -> str:
    bucket, key = _parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    full_key = "/".join(part for part in (key.rstrip("/"), suffix) if part)
    return f"s3://{bucket}/{full_key}"


def condition_output_s3(root: str, condition: str) -> str:
    """Accept either a shared output root or an already-qualified condition URI."""

    bucket, key = _parse_s3_uri(root)
    if key.rstrip("/").split("/")[-1] == condition:
        return f"s3://{bucket}/{key.rstrip('/')}"
    return _join_s3_uri(root, condition)


def cross_condition_output_s3(root: str, condition: str) -> str:
    """Return the common root even if a JobSet supplied ``.../<condition>``."""

    bucket, key = _parse_s3_uri(root)
    components = key.rstrip("/").split("/")
    if components[-1] == condition:
        parent = "/".join(components[:-1])
        if not parent:
            raise ValueError("condition-qualified --output-s3 must have a parent prefix")
        return f"s3://{bucket}/{parent}"
    return f"s3://{bucket}/{key.rstrip('/')}"


def upload_file(local_path: Path, destination: str, *, s3_client: Any | None = None) -> None:
    bucket, key = _parse_s3_uri(destination)
    (s3_client or _s3_client()).upload_file(str(local_path), bucket, key)


def upload_tree(
    local_root: Path,
    destination_root: str,
    *,
    s3_client: Any | None = None,
) -> int:
    """Recursively upload regular files using boto3 only."""

    client = s3_client or _s3_client()
    count = 0
    if not local_root.exists():
        return count
    for path in sorted(local_root.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        relative = path.relative_to(local_root).as_posix()
        upload_file(path, _join_s3_uri(destination_root, relative), s3_client=client)
        count += 1
    return count


def _upload_rank_best_effort(args: argparse.Namespace, rank_dir: Path) -> None:
    if not args.output_s3:
        return
    destination = _join_s3_uri(
        condition_output_s3(args.output_s3, args.condition),
        "ranks",
        rank_dir.name,
    )
    try:
        count = upload_tree(rank_dir, destination)
        _event(rank_dir, "rank_upload_complete", files=count, destination=destination)
        # Upload the event appended immediately above as well.
        upload_file(rank_dir / "runner.log", _join_s3_uri(destination, "runner.log"))
    except Exception as exc:  # noqa: BLE001 - upload is explicitly best effort
        _event(rank_dir, "rank_upload_failed", error=f"{type(exc).__name__}: {exc}")


def _load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB")


def _output_candidate(request_output: Any) -> Any:
    candidates = getattr(request_output, "outputs", None) or []
    if len(candidates) != 1:
        raise RuntimeError(f"vLLM returned {len(candidates)} candidates, expected one")
    candidate = candidates[0]
    request_id = getattr(request_output, "request_id", None)
    finish_reason = getattr(candidate, "finish_reason", None)
    if str(finish_reason).lower() == "error":
        detail = getattr(candidate, "stop_reason", None) or getattr(request_output, "error", None)
        raise RuntimeError(f"vLLM request {request_id!r} finished with error: {detail!r}")
    text = getattr(candidate, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"vLLM request {request_id!r} returned empty output (finish_reason={finish_reason!r})"
        )
    return candidate


def run_worker(args: argparse.Namespace, rank: int, world_size: int) -> dict[str, Any]:
    tasks = tuple(args.tasks)
    training_profile = resolve_training_profile(args.training_profile)
    probability_provenance = task_probability_provenance(training_profile)
    artifact_identity = verify_model_artifact_identity(
        Path(args.model),
        args.condition,
        expected_source_step=args.expected_source_step,
        expected_source_checkpoint_manifest_sha256=(
            args.expected_source_checkpoint_manifest_sha256
        ),
        expected_trained_artifact_manifest_sha256=(
            args.expected_trained_artifact_manifest_sha256
        ),
        expected_initial_mtp_artifact_manifest_sha256=(
            args.expected_initial_mtp_artifact_manifest_sha256
        ),
    )
    condition_dir = Path(args.output_dir) / args.condition
    rank_dir = condition_dir / "ranks" / f"rank-{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    worker_started = time.monotonic()
    atomic_write_json(
        rank_dir / "status.json",
        {"status": "running", "rank": rank, "world_size": world_size, "started_at": utc_now()},
    )
    _event(
        rank_dir,
        "worker_start",
        rank=rank,
        world_size=world_size,
        condition=args.condition,
        evaluation_tasks=[task.output_name for task in tasks],
        training_profile=training_profile.name,
        artifact_identity=artifact_identity,
    )

    manifest_started = time.monotonic()
    manifest = load_manifest(args.manifest, expected_count=args.expected_count)
    manifest_load_seconds = time.monotonic() - manifest_started
    indices = shard_indices(len(manifest), rank, world_size)
    _event(
        rank_dir,
        "manifest_loaded",
        rows=len(manifest),
        shard_rows=len(indices),
        elapsed_seconds=manifest_load_seconds,
    )
    rank_cache = Path(args.cache_dir) / f"rank-{rank:03d}"
    download_started = time.monotonic()
    cache_paths = cache_manifest_shard(
        manifest,
        indices,
        rank_cache,
        download_workers=args.download_workers,
    )
    image_download_seconds = time.monotonic() - download_started
    _event(
        rank_dir,
        "image_shard_cached",
        images=len(cache_paths),
        cache_dir=str(rank_cache),
        elapsed_seconds=image_download_seconds,
    )

    # Keep CUDA imports below GPU narrowing and image prefetch.
    from vllm import LLM, SamplingParams

    runtime_provenance = collect_runtime_provenance(artifact_identity)
    llm_kwargs = build_llm_kwargs(args)
    _event(rank_dir, "engine_loading", llm_kwargs=llm_kwargs)
    engine_load_started = time.monotonic()
    llm = LLM(**llm_kwargs)
    engine_load_seconds = time.monotonic() - engine_load_started
    speculative = validate_resolved_speculative_config(llm, args.condition)
    tokenizer = llm.get_tokenizer()
    template_render_started = time.monotonic()
    rendered_prompts = {task.name: render_training_prompt(tokenizer, task) for task in tasks}
    template_render_seconds = time.monotonic() - template_render_started
    cold_start_seconds = time.monotonic() - worker_started
    _event(
        rank_dir,
        "engine_loaded",
        speculative=speculative,
        engine_load_seconds=engine_load_seconds,
        template_render_seconds=template_render_seconds,
        cold_start_seconds=cold_start_seconds,
    )

    task_summaries: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_started = time.monotonic()
        generation_wall = 0.0
        image_decode_seconds = 0.0
        multimodal_preprocess_seconds = 0.0
        request_latencies: list[float] = []
        before = speculative_metric_snapshot(llm.get_metrics())
        partial_path = rank_dir / f"{task.output_name}.partial.jsonl"
        final_path = rank_dir / f"{task.output_name}.jsonl"
        rows: list[dict[str, Any]] = []
        with partial_path.open("w", encoding="utf-8") as output_handle:
            for offset in range(0, len(indices), args.batch_size):
                batch_indices = indices[offset : offset + args.batch_size]
                decode_started = time.monotonic()
                images = [_load_rgb_image(cache_paths[index]) for index in batch_indices]
                image_decode_seconds += time.monotonic() - decode_started
                requests = [
                    {
                        "prompt": rendered_prompts[task.name],
                        "multi_modal_data": {"image": image},
                        "mm_processor_kwargs": {"max_pixels": args.max_pixels},
                    }
                    for image in images
                ]
                preprocess_started = time.monotonic()
                preprocessed_prompt_tokens = preprocess_prompt_token_counts(llm, requests)
                multimodal_preprocess_seconds += time.monotonic() - preprocess_started
                max_new_token_budgets = [
                    dynamic_max_new_tokens(
                        prompt_tokens=prompt_tokens,
                        max_model_len=args.max_model_len,
                        cli_max_new_tokens=args.max_tokens,
                        eos_reserve_tokens=args.eos_reserve_tokens,
                    )
                    for prompt_tokens in preprocessed_prompt_tokens
                ]
                batch_sampling_params = [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=max_new_tokens,
                        seed=args.seed,
                    )
                    for max_new_tokens in max_new_token_budgets
                ]
                try:
                    generation_started = time.monotonic()
                    outputs = llm.generate(requests, batch_sampling_params, use_tqdm=False)
                    generation_wall += time.monotonic() - generation_started
                finally:
                    for image in images:
                        image.close()
                if len(outputs) != len(batch_indices):
                    raise RuntimeError(
                        f"vLLM output count {len(outputs)} != request count {len(batch_indices)}"
                    )
                for input_index, request_output, expected_prompt_tokens, max_new_tokens in zip(
                    batch_indices,
                    outputs,
                    preprocessed_prompt_tokens,
                    max_new_token_budgets,
                    strict=True,
                ):
                    candidate = _output_candidate(request_output)
                    actual_prompt_tokens = len(getattr(request_output, "prompt_token_ids", None) or [])
                    if actual_prompt_tokens != expected_prompt_tokens:
                        raise RuntimeError(
                            "vLLM prompt-token count changed between budget preprocessing and "
                            f"generation: {expected_prompt_tokens} != {actual_prompt_tokens}"
                        )
                    request_timing = request_latency_metrics(request_output)
                    if request_timing["end_to_end_seconds"] is not None:
                        request_latencies.append(float(request_timing["end_to_end_seconds"]))
                    row = make_output_row(
                        task=task,
                        condition=args.condition,
                        training_profile=training_profile,
                        artifact_identity=artifact_identity,
                        manifest_row=manifest[input_index],
                        input_index=input_index,
                        response=str(candidate.text),
                        prompt_tokens=actual_prompt_tokens,
                        output_tokens=len(getattr(candidate, "token_ids", None) or []),
                        max_new_tokens_budget=max_new_tokens,
                        request_timing=request_timing,
                        finish_reason=getattr(candidate, "finish_reason", None),
                        stop_reason=getattr(candidate, "stop_reason", None),
                    )
                    rows.append(row)
                    _write_jsonl_line(output_handle, row)
                _event(
                    rank_dir,
                    "generation_progress",
                    task=task.output_name,
                    completed=len(rows),
                    total=len(indices),
                )
        os.replace(partial_path, final_path)
        after = speculative_metric_snapshot(llm.get_metrics())
        spec_metrics = metric_delta(before, after)
        if rows:
            validate_speculative_metrics(args.condition, spec_metrics)
        elapsed = time.monotonic() - task_started
        output_tokens = sum(int(row["output_tokens"]) for row in rows)
        prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
        json_valid = sum(bool(row["json_valid"]) for row in rows)
        task_summary = {
            "task": task.output_name,
            "training_task_name": task.name,
            "training_profile": training_profile.name,
            "conditional_primary_task_probability": (
                training_profile.conditional_primary_task_probabilities[task.name]
            ),
            "effective_overall_task_probability": (
                training_profile.effective_overall_task_probabilities[task.name]
            ),
            "rows": len(rows),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "json_valid": json_valid,
            "json_valid_rate": json_valid / len(rows) if rows else None,
            "generation_wall_seconds": generation_wall,
            "generation_elapsed_seconds": generation_wall,
            "image_decode_seconds": image_decode_seconds,
            "multimodal_preprocess_seconds": multimodal_preprocess_seconds,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": output_tokens / generation_wall if generation_wall else None,
            "tokens_per_second": output_tokens / generation_wall if generation_wall else None,
            "images_per_second": len(rows) / generation_wall if generation_wall else None,
            "request_latency_seconds": latency_summary(request_latencies),
            "request_latency_samples_seconds": request_latencies,
            "min_max_new_tokens_budget": min(
                (int(row["max_new_tokens_budget"]) for row in rows),
                default=None,
            ),
            "max_max_new_tokens_budget": max(
                (int(row["max_new_tokens_budget"]) for row in rows),
                default=None,
            ),
            "response_sequence_sha256": response_sequence_sha256(rows),
            "speculative_metrics_before": dict(before),
            "speculative_metrics_after": dict(after),
            **spec_metrics,
        }
        task_summaries[task.output_name] = task_summary
        atomic_write_json(
            rank_dir / "progress.json",
            {"completed_tasks": task_summaries, "updated_at": utc_now()},
        )
        _event(rank_dir, "task_complete", **task_summary)

    task_values = list(task_summaries.values())
    total = _sum_task_summaries(task_values)
    # On one rank tasks run sequentially, so totals use sums rather than the
    # cross-rank max used by _sum_task_summaries.
    total["generation_wall_seconds"] = sum(float(item["generation_wall_seconds"]) for item in task_values)
    total["generation_elapsed_seconds"] = total["generation_wall_seconds"]
    total["image_decode_seconds"] = sum(float(item["image_decode_seconds"]) for item in task_values)
    total["multimodal_preprocess_seconds"] = sum(
        float(item["multimodal_preprocess_seconds"]) for item in task_values
    )
    total["elapsed_seconds"] = time.monotonic() - worker_started
    total["output_tokens_per_second"] = (
        total["output_tokens"] / total["generation_wall_seconds"]
        if total["generation_wall_seconds"]
        else None
    )
    total["tokens_per_second"] = total["output_tokens_per_second"]
    total["images_per_second"] = (
        total["rows"] / total["generation_wall_seconds"] if total["generation_wall_seconds"] else None
    )
    rank_summary = {
        "status": "complete",
        "condition": args.condition,
        "model": str(args.model),
        "evaluation_tasks": [task.output_name for task in tasks],
        "training_profile": training_profile.name,
        "training_config": training_profile.training_config,
        "task_probability_provenance": probability_provenance,
        "artifact_identity": artifact_identity,
        **runtime_provenance,
        "rank": rank,
        "world_size": world_size,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha256(manifest),
        "shard_indices": indices,
        "speculative": speculative,
        "seed": args.seed,
        "engine": engine_comparison_contract(args),
        "max_pixels": args.max_pixels,
        "max_model_len": args.max_model_len,
        "max_new_tokens_cli_cap": args.max_tokens,
        "eos_reserve_tokens": args.eos_reserve_tokens,
        "timing": {
            "manifest_load_seconds": manifest_load_seconds,
            "image_download_seconds": image_download_seconds,
            "engine_load_seconds": engine_load_seconds,
            "template_render_seconds": template_render_seconds,
            "cold_start_seconds": cold_start_seconds,
        },
        "tasks": task_summaries,
        "total": total,
        "completed_at": utc_now(),
    }
    atomic_write_json(rank_dir / "summary.json", rank_summary)
    atomic_write_json(
        rank_dir / "status.json",
        {"status": "complete", "rank": rank, "world_size": world_size, "completed_at": utc_now()},
    )
    _event(rank_dir, "worker_complete", total=total)
    _upload_rank_best_effort(args, rank_dir)
    if rank == 0:
        condition_summary = aggregate_condition(args, manifest, world_size)
        return condition_summary
    return rank_summary


def _wait_for_rank_summaries(condition_dir: Path, world_size: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        complete = 0
        for rank in range(world_size):
            rank_dir = condition_dir / "ranks" / f"rank-{rank:03d}"
            status_path = rank_dir / "status.json"
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "failed":
                raise RuntimeError(f"rank {rank} failed: {status.get('error')}")
            if status.get("status") == "complete" and (rank_dir / "summary.json").exists():
                complete += 1
        if complete == world_size:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {complete}/{world_size} rank summaries arrived")
        time.sleep(2.0)


def _merge_task_rows(
    condition_dir: Path,
    task: TaskSpec,
    world_size: int,
    expected_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = condition_dir / "ranks" / f"rank-{rank:03d}" / f"{task.output_name}.jsonl"
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: int(row["input_index"]))
    indices = [int(row["input_index"]) for row in rows]
    if indices != list(range(expected_count)):
        raise ValueError(f"merged {task.output_name} indices do not cover 0..{expected_count - 1}")
    image_uris = [str((row.get("images") or [{}])[0].get("path") or "") for row in rows]
    if len(set(image_uris)) != expected_count:
        raise ValueError(f"merged {task.output_name} has duplicate image paths")
    destination = condition_dir / f"{task.output_name}.jsonl"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_jsonl_line(handle, row)
    os.replace(temporary, destination)
    return rows


_GPU_STABLE_IDENTITY_FIELDS = (
    "name",
    "compute_capability",
    "multiprocessor_count",
)
_GPU_PER_RANK_MEASUREMENT_FIELDS = (
    "total_memory_bytes",
    "max_sm_clock_mhz",
    "max_memory_clock_mhz",
)


def _aggregate_gpu_hardware_and_clock(
    rank_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate GPU identity and retain card-specific capacity/clock measurements."""

    available: list[tuple[int, Mapping[str, Any]]] = []
    unavailable_ranks: list[int] = []
    for index, summary in enumerate(rank_summaries):
        rank = int(summary.get("rank", index))
        hardware = summary.get("gpu_hardware_and_clock")
        if hardware is None:
            error = summary.get("gpu_hardware_and_clock_error")
            if not isinstance(error, str) or not error.strip():
                raise ValueError(
                    "GPU hardware provenance is unavailable without a recorded collection "
                    f"error for rank {rank}"
                )
            unavailable_ranks.append(rank)
            continue
        if not isinstance(hardware, Mapping):
            raise ValueError(f"GPU hardware provenance is invalid for rank {rank}")
        available.append((rank, hardware))
    if not available:
        return None

    identity: dict[str, Any] = {}
    for field in _GPU_STABLE_IDENTITY_FIELDS:
        missing_ranks = [
            rank
            for rank, hardware in available
            if hardware.get(field) is None
        ]
        if missing_ranks:
            raise ValueError(
                f"GPU hardware identity field {field!r} is missing for ranks {missing_ranks!r}"
            )
        reference_rank, reference_hardware = available[0]
        reference = reference_hardware[field]
        mismatched_ranks = [
            rank
            for rank, hardware in available
            if hardware[field] != reference
        ]
        if mismatched_ranks:
            raise ValueError(
                "rank summaries disagree on stable GPU hardware identity field "
                f"{field!r}; rank {reference_rank} has {reference!r}, "
                f"mismatched ranks {mismatched_ranks!r}"
            )
        identity[field] = reference

    known_fields = set(_GPU_STABLE_IDENTITY_FIELDS) | set(_GPU_PER_RANK_MEASUREMENT_FIELDS)
    extra_fields = sorted(set().union(*(set(hardware) for _, hardware in available)) - known_fields)
    for field in extra_fields:
        reference = available[0][1].get(field)
        mismatched_ranks = [
            rank
            for rank, hardware in available
            if hardware.get(field) != reference
        ]
        if mismatched_ranks:
            raise ValueError(
                "unclassified GPU hardware field differs across ranks; classify it as a stable "
                f"identity or per-rank measurement before accepting it: {field!r}"
            )
        identity[field] = reference

    rank_measurements: list[dict[str, Any]] = []
    all_ranks = {int(summary.get("rank", index)) for index, summary in enumerate(rank_summaries)}
    measurement_ranges: dict[str, dict[str, float | int] | None] = {}
    measurement_unavailable_ranks: dict[str, list[int]] = {}
    for field in _GPU_PER_RANK_MEASUREMENT_FIELDS:
        measurements: list[float | int] = []
        measurement_ranks: set[int] = set()
        for rank, hardware in available:
            value = hardware.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"GPU per-rank measurement {field!r} is invalid for rank {rank}"
                )
            measurements.append(value)
            measurement_ranks.add(rank)
        measurement_ranges[field] = (
            {"min": min(measurements), "max": max(measurements)} if measurements else None
        )
        measurement_unavailable_ranks[field] = sorted(all_ranks - measurement_ranks)

    for rank, hardware in available:
        rank_measurements.append(
            {
                "rank": rank,
                **{field: hardware.get(field) for field in _GPU_PER_RANK_MEASUREMENT_FIELDS},
            }
        )
    rank_measurements.sort(key=lambda item: int(item["rank"]))
    return {
        **identity,
        "available_ranks": sorted(rank for rank, _ in available),
        "unavailable_ranks": sorted(unavailable_ranks),
        "measurement_ranges": measurement_ranges,
        "measurement_unavailable_ranks": measurement_unavailable_ranks,
        "rank_measurements": rank_measurements,
    }


def aggregate_condition(
    args: argparse.Namespace,
    manifest: Sequence[Mapping[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    tasks = tuple(args.tasks)
    training_profile = resolve_training_profile(args.training_profile)
    probability_provenance = task_probability_provenance(training_profile)
    condition_dir = Path(args.output_dir) / args.condition
    _wait_for_rank_summaries(condition_dir, world_size, args.rank_wait_timeout)
    rank_summaries = [
        json.loads(
            (condition_dir / "ranks" / f"rank-{rank:03d}" / "summary.json").read_text(encoding="utf-8")
        )
        for rank in range(world_size)
    ]
    mismatched_profiles = {
        str(summary.get("training_profile"))
        for summary in rank_summaries
        if summary.get("training_profile") != training_profile.name
    }
    if mismatched_profiles:
        raise ValueError(
            f"rank summaries do not match training profile {training_profile.name!r}: "
            f"{sorted(mismatched_profiles)!r}"
        )
    expected_task_names = [task.output_name for task in tasks]
    mismatched_task_sets = {
        tuple(summary.get("evaluation_tasks") or [])
        for summary in rank_summaries
        if summary.get("evaluation_tasks") != expected_task_names
    }
    if mismatched_task_sets:
        raise ValueError(
            f"rank summaries do not match evaluation tasks {expected_task_names!r}: "
            f"{sorted(mismatched_task_sets)!r}"
        )
    artifact_identities = {
        json.dumps(summary.get("artifact_identity"), sort_keys=True, separators=(",", ":"))
        for summary in rank_summaries
    }
    if len(artifact_identities) != 1:
        raise ValueError("rank summaries do not share one model artifact identity")
    artifact_identity = rank_summaries[0]["artifact_identity"]
    runtime_fields = (
        "target_backbone_fingerprint",
        "runtime_image_digest",
        "vllm_version",
        "code_overlay_sha256",
    )
    runtime_provenance: dict[str, Any] = {}
    for field in runtime_fields:
        values = {
            json.dumps(summary.get(field), sort_keys=True, separators=(",", ":"))
            for summary in rank_summaries
        }
        if len(values) != 1:
            raise ValueError(f"rank summaries disagree on runtime provenance field {field}")
        runtime_provenance[field] = rank_summaries[0].get(field)
    runtime_provenance["gpu_hardware_and_clock"] = _aggregate_gpu_hardware_and_clock(
        rank_summaries
    )
    gpu_provenance_errors = sorted(
        {
            str(summary["gpu_hardware_and_clock_error"])
            for summary in rank_summaries
            if summary.get("gpu_hardware_and_clock_error")
        }
    )
    runtime_provenance["gpu_hardware_and_clock_error"] = (
        gpu_provenance_errors or None
    )
    task_summaries: dict[str, dict[str, Any]] = {}
    task_hashes: dict[str, str] = {}
    for task in tasks:
        rows = _merge_task_rows(condition_dir, task, world_size, len(manifest))
        rank_task_summaries = [summary["tasks"][task.output_name] for summary in rank_summaries]
        aggregate = _sum_task_summaries(rank_task_summaries)
        aggregate["training_profile"] = training_profile.name
        aggregate["conditional_primary_task_probability"] = (
            training_profile.conditional_primary_task_probabilities[task.name]
        )
        aggregate["effective_overall_task_probability"] = (
            training_profile.effective_overall_task_probabilities[task.name]
        )
        aggregate["response_sequence_sha256"] = response_sequence_sha256(rows)
        aggregate["output"] = f"{task.output_name}.jsonl"
        task_summaries[task.output_name] = aggregate
        task_hashes[task.output_name] = aggregate["response_sequence_sha256"]
    total = _sum_task_summaries([summary["total"] for summary in rank_summaries])
    worker_timing_fields = (
        "manifest_load_seconds",
        "image_download_seconds",
        "engine_load_seconds",
        "template_render_seconds",
        "cold_start_seconds",
    )
    timing = {
        f"{field.removesuffix('_seconds')}_wall_seconds": max(
            float(summary.get("timing", {}).get(field, 0.0)) for summary in rank_summaries
        )
        for field in worker_timing_fields
    }
    timing.update(
        {
            f"sum_rank_{field}": sum(
                float(summary.get("timing", {}).get(field, 0.0)) for summary in rank_summaries
            )
            for field in worker_timing_fields
        }
    )
    summary = {
        "status": "complete",
        "condition": args.condition,
        "model": str(args.model),
        "evaluation_tasks": expected_task_names,
        "training_profile": training_profile.name,
        "training_config": training_profile.training_config,
        "task_probability_provenance": probability_provenance,
        "artifact_identity": artifact_identity,
        **runtime_provenance,
        "world_size": world_size,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha256(manifest),
        "images": len(manifest),
        "seed": args.seed,
        "engine": engine_comparison_contract(args),
        "max_pixels": args.max_pixels,
        "max_model_len": args.max_model_len,
        "max_new_tokens_cli_cap": args.max_tokens,
        "eos_reserve_tokens": args.eos_reserve_tokens,
        "timing": timing,
        "tasks": task_summaries,
        "task_hashes": task_hashes,
        "total": total,
        "rank_summaries": [f"ranks/rank-{rank:03d}/summary.json" for rank in range(world_size)],
        "completed_at": utc_now(),
    }
    atomic_write_json(condition_dir / "summary.json", summary)
    cross = update_cross_condition_hashes(Path(args.output_dir))
    if cross is not None:
        summary["cross_condition_hashes"] = str(cross.name)
        atomic_write_json(condition_dir / "summary.json", summary)
    atomic_write_json(
        condition_dir / "status.json",
        {"status": "complete", "condition": args.condition, "completed_at": utc_now()},
    )
    if args.output_s3:
        upload_tree(condition_dir, condition_output_s3(args.output_s3, args.condition))
        if cross is not None:
            reconcile_cross_condition_s3(
                args.output_s3,
                args.condition,
                cross,
            )
    return summary


def _condition_rows(output_dir: Path, condition: str, task_name: str) -> dict[int, str]:
    path = output_dir / condition / f"{task_name}.jsonl"
    if not path.exists():
        return {}
    return {int(row["input_index"]): str(row["response_sha256"]) for row in read_jsonl(path)}


def _summary_task_names(summary: Mapping[str, Any]) -> tuple[str, ...]:
    names = summary.get("evaluation_tasks")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return tuple(names)
    hashes = summary.get("task_hashes")
    if isinstance(hashes, Mapping):
        return tuple(str(name) for name in hashes)
    return tuple(task.output_name for task in TASKS)


def performance_raw(summary: Mapping[str, Any]) -> dict[str, Any]:
    total = summary.get("total", {})
    return {
        "rows": total.get("rows"),
        "prompt_tokens": total.get("prompt_tokens"),
        "output_tokens": total.get("output_tokens"),
        "generation_wall_seconds": total.get(
            "generation_wall_seconds",
            total.get("generation_elapsed_seconds"),
        ),
        "output_tokens_per_second": total.get(
            "output_tokens_per_second",
            total.get("tokens_per_second"),
        ),
        "images_per_second": total.get("images_per_second"),
        "request_latency_seconds": total.get("request_latency_seconds"),
        "image_decode_seconds": total.get("image_decode_seconds"),
        "multimodal_preprocess_seconds": total.get("multimodal_preprocess_seconds"),
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    return float(numerator) / float(denominator) if denominator else None


def speedup_vs_no_mtp(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline_summary = summaries.get("no_mtp")
    if baseline_summary is None:
        return {}
    baseline = performance_raw(baseline_summary)
    comparisons: dict[str, Any] = {}
    for condition in ("trained_mtp", "original_mtp"):
        candidate_summary = summaries.get(condition)
        if candidate_summary is None:
            continue
        candidate = performance_raw(candidate_summary)
        comparisons[condition] = {
            "baseline_condition": "no_mtp",
            "baseline_raw": baseline,
            "candidate_raw": candidate,
            "generation_wall_speedup": _safe_ratio(
                baseline["generation_wall_seconds"],
                candidate["generation_wall_seconds"],
            ),
            "output_token_throughput_speedup": _safe_ratio(
                candidate["output_tokens_per_second"],
                baseline["output_tokens_per_second"],
            ),
            "image_throughput_speedup": _safe_ratio(
                candidate["images_per_second"],
                baseline["images_per_second"],
            ),
        }
    return comparisons


def comparison_provenance(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Check that independently finished conditions share one target and run contract."""

    conditions: dict[str, Any] = {}
    missing_identity: list[str] = []
    missing_contract_fields: dict[str, list[str]] = {}
    contract_fields = (
        "training_profile",
        "training_config",
        "task_probability_provenance",
        "evaluation_tasks",
        "manifest_sha256",
        "world_size",
        "images",
        "max_pixels",
        "max_model_len",
        "max_new_tokens_cli_cap",
        "eos_reserve_tokens",
        "seed",
        "engine",
        "target_backbone_fingerprint",
        "runtime_image_digest",
        "vllm_version",
        "code_overlay_sha256",
        "gpu_hardware_and_clock",
    )
    for condition, summary in summaries.items():
        identity = summary.get("artifact_identity")
        if not isinstance(identity, Mapping):
            missing_identity.append(condition)
            continue
        comparison_contract = {field: summary.get(field) for field in contract_fields}
        missing = [field for field, value in comparison_contract.items() if value is None]
        if missing:
            missing_contract_fields[condition] = missing
        conditions[condition] = {
            "training_profile": summary.get("training_profile"),
            "artifact_variant": identity.get("variant"),
            "local_manifest_sha256": identity.get("local_manifest_sha256"),
            "comparison_lineage": artifact_comparison_lineage(identity),
            "comparison_contract": comparison_contract,
        }
    lineage_values = {
        json.dumps(value["comparison_lineage"], sort_keys=True, separators=(",", ":"))
        for value in conditions.values()
    }
    contract_values = {
        json.dumps(value["comparison_contract"], sort_keys=True, separators=(",", ":"))
        for value in conditions.values()
    }
    compatible = (
        not missing_identity
        and not missing_contract_fields
        and len(conditions) == len(summaries)
        and len(lineage_values) == 1
        and len(contract_values) == 1
    )
    return {
        "compatible": compatible,
        "missing_artifact_identity": sorted(missing_identity),
        "missing_contract_fields": missing_contract_fields,
        "conditions": conditions,
        "reason": (
            "all conditions share one pinned trained target/source lineage and inference contract"
            if compatible
            else "conditions do not share one fully recorded artifact and inference contract"
        ),
    }


def update_cross_condition_hashes(output_dir: Path) -> Path | None:
    """Atomically update response equality diagnostics across completed conditions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "cross_condition_hashes.json"
    lock_path = output_dir / ".cross_condition_hashes.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summaries: dict[str, dict[str, Any]] = {}
        for condition in COMPARISON_CONDITIONS:
            path = output_dir / condition / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") == "complete":
                summaries[condition] = summary
        if len(summaries) < 2:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return None
        provenance = comparison_provenance(summaries)
        observed_speedup = speedup_vs_no_mtp(summaries) if provenance["compatible"] else {}
        payload: dict[str, Any] = {
            "updated_at": utc_now(),
            "comparison_provenance": provenance,
            "conditions": {
                condition: {
                    "model": summary.get("model"),
                    "evaluation_tasks": summary.get("evaluation_tasks"),
                    "training_profile": summary.get("training_profile"),
                    "training_config": summary.get("training_config"),
                    "task_probability_provenance": summary.get(
                        "task_probability_provenance"
                    ),
                    "artifact_identity": summary.get("artifact_identity"),
                    "manifest_sha256": summary.get("manifest_sha256"),
                    "task_hashes": summary.get("task_hashes", {}),
                    "performance_raw": performance_raw(summary),
                }
                for condition, summary in summaries.items()
            },
            "pairs": {},
            "speedup_vs_no_mtp": observed_speedup,
            "speedup_vs_no_mtp_observed": observed_speedup,
            "speedup_semantics": (
                "observed only; use finalize_test1070_speed.py for strict output-comparability gating"
            ),
        }
        names = sorted(summaries)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                task_comparisons: dict[str, Any] = {}
                task_names = sorted(
                    set(_summary_task_names(summaries[left]))
                    | set(_summary_task_names(summaries[right]))
                )
                for task_name in task_names:
                    left_rows = _condition_rows(output_dir, left, task_name)
                    right_rows = _condition_rows(output_dir, right, task_name)
                    shared = sorted(set(left_rows) & set(right_rows))
                    matching = sum(left_rows[index] == right_rows[index] for index in shared)
                    task_comparisons[task_name] = {
                        "rows_left": len(left_rows),
                        "rows_right": len(right_rows),
                        "shared_rows": len(shared),
                        "matching_rows": matching,
                        "match_rate": matching / len(shared) if shared else None,
                        "all_equal": (len(left_rows) == len(right_rows) == len(shared) == matching),
                    }
                payload["pairs"][f"{left}__vs__{right}"] = {
                    "tasks": task_comparisons,
                    "all_tasks_equal": all(value["all_equal"] for value in task_comparisons.values()),
                }
        atomic_write_json(destination, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return destination


def _read_s3_json(uri: str, client: Any) -> dict[str, Any] | None:
    bucket, key = _parse_s3_uri(uri)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - botocore is imported lazily
        error = getattr(exc, "response", {}).get("Error", {})
        if str(error.get("Code", "")) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    value = json.loads(response["Body"].read())
    if not isinstance(value, dict):
        raise ValueError(f"S3 JSON is not an object: {uri}")
    return value


def reconcile_cross_condition_s3(
    output_s3: str,
    condition: str,
    local_destination: Path,
    *,
    s3_client: Any | None = None,
) -> Path | None:
    """Let the last isolated JobSet condition publish the combined hash report."""

    client = s3_client or _s3_client()
    common_root = cross_condition_output_s3(output_s3, condition)
    summaries: dict[str, dict[str, Any]] = {}
    for name in COMPARISON_CONDITIONS:
        summary = _read_s3_json(_join_s3_uri(common_root, name, "summary.json"), client)
        if summary is not None and summary.get("status") == "complete":
            summaries[name] = summary
    # A one-condition report is misleading and can overwrite a report written
    # by a concurrently finishing sibling.  The final condition will always see
    # the summaries uploaded by earlier successful conditions.
    if len(summaries) < 2:
        return None
    provenance = comparison_provenance(summaries)
    observed_speedup = speedup_vs_no_mtp(summaries) if provenance["compatible"] else {}
    payload: dict[str, Any] = {
        "updated_at": utc_now(),
        "source": "s3_condition_summaries",
        "comparison_provenance": provenance,
        "conditions": {
            name: {
                "model": summary.get("model"),
                "evaluation_tasks": summary.get("evaluation_tasks"),
                "training_profile": summary.get("training_profile"),
                "training_config": summary.get("training_config"),
                "task_probability_provenance": summary.get("task_probability_provenance"),
                "artifact_identity": summary.get("artifact_identity"),
                "manifest_sha256": summary.get("manifest_sha256"),
                "task_hashes": summary.get("task_hashes", {}),
                "performance_raw": performance_raw(summary),
            }
            for name, summary in summaries.items()
        },
        "pairs": {},
        "speedup_vs_no_mtp": observed_speedup,
        "speedup_vs_no_mtp_observed": observed_speedup,
        "speedup_semantics": (
            "observed only; use finalize_test1070_speed.py for strict output-comparability gating"
        ),
    }
    names = sorted(summaries)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_summary = summaries[left]
            right_summary = summaries[right]
            manifest_equal = left_summary.get("manifest_sha256") == right_summary.get("manifest_sha256")
            task_names = sorted(
                set(_summary_task_names(left_summary))
                | set(_summary_task_names(right_summary))
            )
            tasks = {
                task_name: {
                    "left_sha256": left_summary.get("task_hashes", {}).get(task_name),
                    "right_sha256": right_summary.get("task_hashes", {}).get(task_name),
                    "hash_equal": left_summary.get("task_hashes", {}).get(task_name)
                    == right_summary.get("task_hashes", {}).get(task_name),
                }
                for task_name in task_names
            }
            payload["pairs"][f"{left}__vs__{right}"] = {
                "manifest_equal": manifest_equal,
                "tasks": tasks,
                "all_tasks_equal": manifest_equal and all(value["hash_equal"] for value in tasks.values()),
            }
    atomic_write_json(local_destination, payload)
    upload_file(
        local_destination,
        _join_s3_uri(common_root, local_destination.name),
        s3_client=client,
    )
    return local_destination


def _failure_payload(
    *, condition: str, error: BaseException, rank: int | None = None, world_size: int | None = None
) -> dict[str, Any]:
    return {
        "status": "failed",
        "condition": condition,
        "rank": rank,
        "world_size": world_size,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "failed_at": utc_now(),
    }


def _write_condition_failure(args: argparse.Namespace, error: BaseException) -> None:
    condition_dir = Path(args.output_dir) / args.condition
    payload = _failure_payload(condition=args.condition, error=error)
    atomic_write_json(condition_dir / "status.json", payload)
    atomic_write_json(condition_dir / "summary.json", payload)
    if args.output_s3:
        try:
            upload_tree(condition_dir, condition_output_s3(args.output_s3, args.condition))
        except Exception as upload_error:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "event": "condition_failure_upload_failed",
                        "error": f"{type(upload_error).__name__}: {upload_error}",
                    }
                ),
                flush=True,
            )


def _resolve_rank(args: argparse.Namespace) -> tuple[int | None, int]:
    if args.rank is not None:
        return args.rank, int(os.environ.get("LOCAL_RANK", "0"))
    if "MTP_EVAL_RANK" in os.environ:
        return int(os.environ["MTP_EVAL_RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ:
        return int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    return None, 0


def _narrow_cuda_device(gpu_tokens: Sequence[str], local_rank: int) -> None:
    visible = [
        value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()
    ]
    if len(visible) == 1:
        return
    if local_rank < 0 or local_rank >= len(gpu_tokens):
        raise ValueError(f"local rank {local_rank} has no GPU in {gpu_tokens!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_tokens[local_rank]


def _sanitize_distributed_environment() -> None:
    # Each evaluator worker intentionally runs an independent TP=1 vLLM engine.
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


def run_worker_safely(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    local_rank: int,
) -> int:
    gpu_tokens = [token.strip() for token in args.gpus.split(",") if token.strip()]
    _narrow_cuda_device(gpu_tokens, local_rank)
    _sanitize_distributed_environment()
    rank_dir = Path(args.output_dir) / args.condition / "ranks" / f"rank-{rank:03d}"
    try:
        run_worker(args, rank, world_size)
        return 0
    except BaseException as exc:  # noqa: BLE001 - persist partial outputs before exiting
        rank_dir.mkdir(parents=True, exist_ok=True)
        payload = _failure_payload(
            condition=args.condition,
            error=exc,
            rank=rank,
            world_size=world_size,
        )
        atomic_write_json(rank_dir / "status.json", payload)
        _event(rank_dir, "worker_failed", error=payload["error"])
        _upload_rank_best_effort(args, rank_dir)
        if rank == 0:
            _write_condition_failure(args, exc)
        traceback.print_exc()
        return 1


def launch_local_workers(args: argparse.Namespace) -> int:
    gpu_tokens = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if len(gpu_tokens) < args.world_size:
        raise ValueError(f"--gpus supplies {len(gpu_tokens)} devices for world size {args.world_size}")
    condition_dir = Path(args.output_dir) / args.condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        condition_dir / "status.json",
        {
            "status": "running",
            "condition": args.condition,
            "evaluation_tasks": [task.output_name for task in args.tasks],
            "training_profile": args.training_profile,
            "world_size": args.world_size,
            "started_at": utc_now(),
        },
    )
    processes: list[subprocess.Popen[Any]] = []
    for rank in range(args.world_size):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_tokens[rank]
        environment["MTP_EVAL_RANK"] = str(rank)
        environment["LOCAL_RANK"] = "0"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            "--rank",
            str(rank),
        ]
        processes.append(subprocess.Popen(command, env=environment))
    try:
        while processes:
            failed: subprocess.Popen[Any] | None = None
            for process in processes:
                result = process.poll()
                if result not in (None, 0):
                    failed = process
                    break
            if failed is not None:
                raise RuntimeError(f"evaluation worker exited with code {failed.returncode}")
            if all(process.poll() == 0 for process in processes):
                return 0
            time.sleep(1.0)
    except BaseException as exc:  # noqa: BLE001
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        _write_condition_failure(args, exc)
        raise
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--training-profile",
        choices=tuple(TRAINING_PROFILES),
        default=os.environ.get("MTP_EVAL_PROFILE", DEFAULT_TRAINING_PROFILE),
        help="Training data/task mixture whose provenance is attached to every result.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_EVALUATION_TASKS),
        help="Comma-separated subset of training task names to evaluate.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--expected-source-step",
        type=int,
        default=os.environ.get("MTP_EVAL_EXPECTED_SOURCE_STEP") or None,
    )
    parser.add_argument(
        "--expected-source-checkpoint-manifest-sha256",
        default=os.environ.get("MTP_EVAL_EXPECTED_SOURCE_CHECKPOINT_MANIFEST_SHA256") or None,
    )
    parser.add_argument(
        "--expected-trained-artifact-manifest-sha256",
        default=os.environ.get("MTP_EVAL_EXPECTED_TRAINED_ARTIFACT_MANIFEST_SHA256") or None,
    )
    parser.add_argument(
        "--expected-initial-mtp-artifact-manifest-sha256",
        default=os.environ.get("MTP_EVAL_EXPECTED_INITIAL_MTP_ARTIFACT_MANIFEST_SHA256") or None,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_IMAGE_COUNT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-s3",
        help="Optional S3 root; boto3 uploads <root>/<condition>/ and cross-condition hashes.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("MTP_EVAL_CACHE_DIR", "/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/mtp-test1070-images")),
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--eos-reserve-tokens",
        type=int,
        default=DEFAULT_EOS_RESERVE_TOKENS,
        help="Context positions kept free after each dynamically capped request.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--rank-wait-timeout", type=float, default=7200.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.tasks = resolve_tasks(args.tasks, training_profile=args.training_profile)
    except ValueError as exc:
        parser.error(str(exc))
    if args.world_size <= 0 or args.batch_size <= 0 or args.max_tokens <= 0:
        parser.error("--world-size, --batch-size, and --max-new-tokens must be positive")
    if args.eos_reserve_tokens < 1:
        parser.error("--eos-reserve-tokens must be at least one")
    if args.max_model_len != MAX_MODEL_LEN:
        parser.error(f"--max-model-len must remain {MAX_MODEL_LEN} for this reproducibility run")
    if args.max_pixels != MAX_PIXELS:
        parser.error(f"--max-pixels must remain {MAX_PIXELS} (1536 squared)")
    if args.expected_source_step is not None and args.expected_source_step <= 0:
        parser.error("--expected-source-step must be positive")
    for option in (
        "expected_source_checkpoint_manifest_sha256",
        "expected_trained_artifact_manifest_sha256",
        "expected_initial_mtp_artifact_manifest_sha256",
    ):
        value = getattr(args, option)
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            parser.error(f"--{option.replace('_', '-')} must be a lowercase 64-hex SHA-256")
    if args.output_s3:
        _parse_s3_uri(args.output_s3)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rank, local_rank = _resolve_rank(args)
    if args.worker or rank is not None:
        if rank is None:
            raise ValueError("worker mode requires a rank")
        return run_worker_safely(args, rank, args.world_size, local_rank)
    try:
        return launch_local_workers(args)
    except BaseException as exc:  # noqa: BLE001
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
