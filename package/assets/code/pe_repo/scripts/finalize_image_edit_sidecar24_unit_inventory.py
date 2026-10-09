#!/usr/bin/env python3
"""Finalize a sidecar24 result root from immutable unit commits.

The regular sidecar24 finalizer consumes node summaries from one frozen
topology.  Recovery and tail repartitioning can legitimately commit the same
frozen unit catalog through several scheduling topologies, so those node
summaries no longer form one truthful tree.  This finalizer instead verifies
the authoritative unit layer directly:

* the result listing contains exactly one output and one marker per catalog
  unit, with no unknown objects below ``datasets/``;
* every marker body is digest-bound, has the exact frozen unit identity and an
  explicitly allowed generation contract;
* every output HEAD matches the marker's byte count, row/count metadata,
  checkpoint, HF artifact, prompt, and catalog fingerprints; and
* successful units contain zero inference errors.

It writes a deterministic manifest and then conditionally creates ``_SUCCESS``
last.  It never reads or rewrites output JSONL bodies and never mutates unit
objects.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


EXPECTED_ENVIRONMENT = {
    "IMAGE_EDIT_TARGET_SCHEMA": "image_edit_v6_direct_root_xyxy",
    "IMAGE_EDIT_V6_PROMPT_VARIANT": "training_v7",
}
for _name, _expected in EXPECTED_ENVIRONMENT.items():
    _actual = os.environ.get(_name)
    if _actual not in {None, _expected}:
        raise RuntimeError(f"{_name}={_actual!r}; expected {_expected!r}")
    os.environ[_name] = _expected

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_inference_contract import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
)
from scripts import build_craftv0_image_edit_v6_repartition1600 as repartition  # noqa: E402
from scripts import image_edit_sidecar24_catalog as source_catalog_module  # noqa: E402
from scripts import infer_image_edit_full_json_sidecar24_rank as rank_worker  # noqa: E402
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module  # noqa: E402


MANIFEST_FORMAT = "image_edit_full_json_sidecar24_unit_inventory_manifest_v1"
SUCCESS_FORMAT = "image_edit_full_json_sidecar24_unit_inventory_commit_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    bucket, key = repartition.parse_s3_uri(uri)
    return bucket, key.rstrip("/")


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def is_not_found(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code")) in NOT_FOUND_CODES


def generation_sha256(generation: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(generation))


def parse_generation_count(value: str) -> tuple[str, int]:
    digest, separator, raw_count = value.partition("=")
    if not separator or SHA256_RE.fullmatch(digest) is None:
        raise argparse.ArgumentTypeError("expected SHA256=COUNT")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("generation count must be an integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("generation count must be positive")
    return digest, count


def catalog_row_count(unit: Mapping[str, Any]) -> int:
    return int(unit["global_row_end_exclusive"]) - int(unit["global_row_start"])


def marker_and_output_keys(
    unit: Mapping[str, Any],
    *,
    result_key_prefix: str,
) -> tuple[str, str]:
    output_key, marker_key = repartition.result_keys(
        unit,
        result_key_prefix=result_key_prefix,
    )
    return marker_key, output_key


def compact_shard(
    marker: Mapping[str, Any],
    *,
    marker_uri: str,
) -> dict[str, Any]:
    unit = marker["binding"]["unit"]
    return {
        "global_unit_index": unit["global_unit_index"],
        "dataset_id": unit["dataset_id"],
        "dataset_unit_index": unit["dataset_unit_index"],
        "source": dict(unit["source"]),
        "local_row_ranges": [list(value) for value in unit["local_row_ranges"]],
        "dataset_row_start": unit["dataset_row_start"],
        "dataset_row_end_exclusive": unit["dataset_row_end_exclusive"],
        "global_row_start": unit["global_row_start"],
        "global_row_end_exclusive": unit["global_row_end_exclusive"],
        "output": dict(marker["output"]),
        "success_marker_uri": marker_uri,
        "disposition": "validated_committed",
    }


def validate_one_unit(
    unit: Mapping[str, Any],
    *,
    client: Any,
    bucket: str,
    result_key_prefix: str,
    source_catalog_sha256: str,
    unit_catalog_sha256: str,
    checkpoint_step: int,
    checkpoint_manifest_sha256: str,
    hf_artifact_manifest_sha256: str,
    allowed_generation_sha256: frozenset[str],
) -> tuple[dict[str, Any], str, str]:
    marker_key, output_key = marker_and_output_keys(
        unit,
        result_key_prefix=result_key_prefix,
    )
    marker_response = client.get_object(Bucket=bucket, Key=marker_key)
    body = marker_response["Body"]
    try:
        marker_payload = body.read()
    finally:
        body.close()
    if int(marker_response["ContentLength"]) != len(marker_payload):
        raise ValueError(f"{marker_key}: marker GET length mismatch")
    if marker_response.get("ContentType") != rank_worker.MARKER_CONTENT_TYPE:
        raise ValueError(f"{marker_key}: marker ContentType mismatch")
    marker_digest = sha256_bytes(marker_payload)
    marker_metadata = {
        str(key).lower(): str(value)
        for key, value in marker_response.get("Metadata", {}).items()
    }
    if marker_metadata.get("sha256") != marker_digest:
        raise ValueError(f"{marker_key}: marker SHA-256 metadata mismatch")
    marker = rank_worker._strict_json_object(  # noqa: SLF001
        marker_payload,
        location=s3_uri(bucket, marker_key),
    )
    binding = marker.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError(f"{marker_key}: marker has no binding object")
    generation = binding.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError(f"{marker_key}: marker has no generation contract")
    generation_digest = generation_sha256(generation)
    if generation_digest not in allowed_generation_sha256:
        raise ValueError(
            f"{marker_key}: generation SHA-256 {generation_digest} is not allowed"
        )
    expected_binding = rank_worker.unit_binding(
        unit,
        source_catalog_sha256=source_catalog_sha256,
        unit_catalog_sha256=unit_catalog_sha256,
        checkpoint_step=checkpoint_step,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        hf_artifact_manifest_sha256=hf_artifact_manifest_sha256,
        generation=generation,
    )
    output_head = client.head_object(Bucket=bucket, Key=output_key)
    expected_output_uri = s3_uri(bucket, output_key)
    rank_worker.validate_resume_marker(
        marker,
        expected_binding=expected_binding,
        expected_output_uri=expected_output_uri,
        output_head=output_head,
    )
    if int(marker["output"]["counts"]["inference_errors"]) != 0:
        raise ValueError(f"{marker_key}: non-zero inference errors")
    shard = compact_shard(marker, marker_uri=s3_uri(bucket, marker_key))
    return shard, generation_digest, str(marker["completed_at"])


def add_counts(target: dict[str, int], shard: Mapping[str, Any]) -> None:
    output = shard["output"]
    counts = output["counts"]
    target["units"] += 1
    target["bytes"] += int(output["bytes"])
    for key in ("rows", "json_valid", "schema_valid", "failed", "inference_errors"):
        target[key] += int(counts[key])


def empty_counts() -> dict[str, int]:
    return {
        "units": 0,
        "rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "failed": 0,
        "inference_errors": 0,
        "bytes": 0,
    }


def batches(values: Sequence[Mapping[str, Any]], size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_manifest(
    path: Path,
    *,
    header: Mapping[str, Any],
    inventory_path: Path,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    written = 0

    def emit(stream: Any, payload: bytes) -> None:
        nonlocal written
        stream.write(payload)
        digest.update(payload)
        written += len(payload)

    with path.open("wb") as destination:
        emit(destination, b"{")
        for index, (key, value) in enumerate(header.items()):
            if index:
                emit(destination, b",")
            emit(destination, canonical_json_bytes(str(key)))
            emit(destination, b":")
            emit(destination, canonical_json_bytes(value))
        if header:
            emit(destination, b",")
        emit(destination, b'"shards":[')
        with inventory_path.open("rb") as inventory:
            first = True
            for line in inventory:
                payload = line.rstrip(b"\n")
                if not payload:
                    continue
                if not first:
                    emit(destination, b",")
                emit(destination, payload)
                first = False
        emit(destination, b"]}")
        destination.flush()
        os.fsync(destination.fileno())
    return digest.hexdigest(), written


def head_or_none(client: Any, *, bucket: str, key: str) -> Mapping[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if is_not_found(error):
            return None
        raise


def put_file_once(
    client: Any,
    *,
    bucket: str,
    key: str,
    path: Path,
    digest: str,
    content_type: str,
) -> None:
    size = path.stat().st_size
    existing = head_or_none(client, bucket=bucket, key=key)
    if existing is not None:
        metadata = {str(k).lower(): str(v) for k, v in existing.get("Metadata", {}).items()}
        if (
            int(existing["ContentLength"]) != size
            or existing.get("ContentType") != content_type
            or metadata.get("sha256") != digest
        ):
            raise FileExistsError(f"s3://{bucket}/{key} exists with different identity")
        return
    try:
        with path.open("rb") as stream:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentLength=size,
                ContentType=content_type,
                Metadata={"sha256": digest},
                IfNoneMatch="*",
            )
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in {
            "PreconditionFailed",
            "412",
        }:
            raise
    verified = head_or_none(client, bucket=bucket, key=key)
    if verified is None:
        raise IOError(f"s3://{bucket}/{key}: upload disappeared")
    metadata = {str(k).lower(): str(v) for k, v in verified.get("Metadata", {}).items()}
    if (
        int(verified["ContentLength"]) != size
        or verified.get("ContentType") != content_type
        or metadata.get("sha256") != digest
    ):
        raise IOError(f"s3://{bucket}/{key}: uploaded identity mismatch")


def put_json_once(
    client: Any,
    *,
    bucket: str,
    key: str,
    value: Mapping[str, Any],
) -> str:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    existing = head_or_none(client, bucket=bucket, key=key)
    if existing is not None:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            existing_payload = body.read()
        finally:
            body.close()
        if existing_payload != payload:
            raise FileExistsError(f"s3://{bucket}/{key} exists with different content")
        return digest
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentLength=len(payload),
            ContentType="application/json",
            Metadata={"sha256": digest},
            IfNoneMatch="*",
        )
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in {
            "PreconditionFailed",
            "412",
        }:
            raise
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            if body.read() != payload:
                raise FileExistsError(f"s3://{bucket}/{key} raced with different content")
        finally:
            body.close()
    return digest


def validate_existing_success(
    client: Any,
    *,
    bucket: str,
    success_key: str,
    expected_contract: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if head_or_none(client, bucket=bucket, key=success_key) is None:
        return None
    response = client.get_object(Bucket=bucket, Key=success_key)
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        body.close()
    value = rank_worker._strict_json_object(  # noqa: SLF001
        payload,
        location=s3_uri(bucket, success_key),
    )
    if value.get("format_version") != SUCCESS_FORMAT or value.get("status") != "complete":
        raise ValueError("existing root _SUCCESS has an unsupported format")
    if value.get("contract") != expected_contract:
        raise ValueError("existing root _SUCCESS contract differs from this invocation")
    manifest_uri = value.get("manifest_uri")
    if not isinstance(manifest_uri, str):
        raise ValueError("existing root _SUCCESS has no manifest URI")
    manifest_bucket, manifest_key = parse_s3_uri(manifest_uri)
    if manifest_bucket != bucket:
        raise ValueError("existing root manifest bucket differs")
    manifest_head = client.head_object(Bucket=manifest_bucket, Key=manifest_key)
    metadata = {
        str(k).lower(): str(v) for k, v in manifest_head.get("Metadata", {}).items()
    }
    if metadata.get("sha256") != value.get("manifest_sha256"):
        raise ValueError("existing root manifest digest metadata differs from _SUCCESS")
    return value


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    source_catalog = source_catalog_module.load_catalog(args.source_catalog)
    source_digest = source_catalog_module.canonical_sha256(source_catalog)
    unit_catalog = unit_catalog_module.load_unit_catalog(
        args.unit_catalog,
        source_catalog=source_catalog,
    )
    unit_digest = unit_catalog_module.canonical_unit_catalog_sha256(
        unit_catalog,
        source_catalog=source_catalog,
    )
    if args.source_catalog_sha256 and source_digest != args.source_catalog_sha256:
        raise ValueError("source catalog SHA-256 mismatch")
    if args.unit_catalog_sha256 and unit_digest != args.unit_catalog_sha256:
        raise ValueError("unit catalog SHA-256 mismatch")
    units = unit_catalog["units"]
    expected_rows = int(unit_catalog["total_selected_rows"])
    if expected_rows != sum(catalog_row_count(unit) for unit in units):
        raise ValueError("unit catalog row total is inconsistent")

    expected_generation_counts = dict(args.expected_generation_count)
    allowed_generation = frozenset(args.allowed_generation_sha256)
    if expected_generation_counts and set(expected_generation_counts) != set(allowed_generation):
        raise ValueError("expected generation counts must cover exactly the allowed hashes")
    if expected_generation_counts and sum(expected_generation_counts.values()) != len(units):
        raise ValueError("expected generation counts do not sum to the unit count")

    session = boto3.Session(profile_name=args.aws_profile) if args.aws_profile else boto3.Session()
    client = session.client(
        "s3",
        region_name=args.aws_region,
        config=Config(
            max_pool_connections=max(args.workers + 16, 64),
            connect_timeout=15,
            read_timeout=300,
            retries={"max_attempts": 20, "mode": "adaptive"},
        ),
    )
    bucket, result_key_prefix = parse_s3_uri(args.output_s3)
    manifest_key = f"{result_key_prefix}/manifest.json"
    success_key = f"{result_key_prefix}/_SUCCESS"
    contract = {
        "source_catalog_uri": args.source_catalog_uri,
        "source_catalog_sha256": source_digest,
        "unit_catalog_uri": args.unit_catalog_uri,
        "unit_catalog_sha256": unit_digest,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "hf_artifact_manifest_sha256": args.hf_artifact_manifest_sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "allowed_generation_sha256": sorted(allowed_generation),
        "expected_units": len(units),
        "expected_rows": expected_rows,
    }
    existing_success = validate_existing_success(
        client,
        bucket=bucket,
        success_key=success_key,
        expected_contract=contract,
    )
    if existing_success is not None:
        return {"status": "already_complete", **dict(existing_success)}

    committed, listing = repartition.list_committed_indices(
        client,
        bucket=bucket,
        result_key_prefix=result_key_prefix,
        units=units,
    )
    expected_indices = {int(unit["global_unit_index"]) for unit in units}
    if committed != expected_indices:
        raise ValueError(
            f"unit coverage mismatch: committed={len(committed)} expected={len(expected_indices)}"
        )
    expected_listing = {
        "recognized_outputs": len(units),
        "recognized_markers": len(units),
        "unknown_result_objects": 0,
        "output_without_marker": 0,
        "marker_without_output": 0,
    }
    for key, expected in expected_listing.items():
        if int(listing[key]) != expected:
            raise ValueError(f"result listing {key}={listing[key]}, expected {expected}")

    temp_parent = args.temp_dir if args.temp_dir else None
    with tempfile.TemporaryDirectory(prefix="sidecar24-finalize-", dir=temp_parent) as raw_temp:
        temp_root = Path(raw_temp)
        inventory_path = temp_root / "shards.jsonl"
        totals = empty_counts()
        dataset_totals: dict[str, dict[str, int]] = defaultdict(empty_counts)
        generation_counts: Counter[str] = Counter()
        completion_timestamps: list[str] = []
        inventory_digest = hashlib.sha256()

        def validate(unit: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
            return validate_one_unit(
                unit,
                client=client,
                bucket=bucket,
                result_key_prefix=result_key_prefix,
                source_catalog_sha256=source_digest,
                unit_catalog_sha256=unit_digest,
                checkpoint_step=args.checkpoint_step,
                checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
                hf_artifact_manifest_sha256=args.hf_artifact_manifest_sha256,
                allowed_generation_sha256=allowed_generation,
            )

        completed = 0
        with inventory_path.open("wb") as inventory, ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="unit-inventory",
        ) as executor:
            for batch in batches(units, args.batch_size):
                for shard, generation_digest, completed_at in executor.map(validate, batch):
                    payload = canonical_json_bytes(shard) + b"\n"
                    inventory.write(payload)
                    inventory_digest.update(payload)
                    add_counts(totals, shard)
                    add_counts(dataset_totals[str(shard["dataset_id"])], shard)
                    generation_counts[generation_digest] += 1
                    completion_timestamps.append(completed_at)
                    completed += 1
                inventory.flush()
                if completed % args.progress_every == 0 or completed == len(units):
                    print(
                        json.dumps(
                            {
                                "event": "validation_progress",
                                "completed_units": completed,
                                "total_units": len(units),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            os.fsync(inventory.fileno())

        if totals["units"] != len(units) or totals["rows"] != expected_rows:
            raise ValueError(f"validated totals mismatch: {totals}")
        if totals["inference_errors"] != 0:
            raise ValueError("validated results contain inference errors")
        if expected_generation_counts and dict(generation_counts) != expected_generation_counts:
            raise ValueError(
                f"generation lineage counts={dict(generation_counts)}, "
                f"expected={expected_generation_counts}"
            )
        dataset_summaries: list[dict[str, Any]] = []
        for dataset in unit_catalog["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            actual = dataset_totals.get(dataset_id, empty_counts())
            if actual["units"] != int(dataset["unit_count"]):
                raise ValueError(f"{dataset_id}: validated unit count mismatch")
            if actual["rows"] != int(dataset["selected_rows"]):
                raise ValueError(f"{dataset_id}: validated row count mismatch")
            dataset_summaries.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_ordinal": int(dataset["dataset_ordinal"]),
                    "family": dataset["family"],
                    "adapter": dataset["adapter"],
                    "main_s3_uri": dataset["main_s3_uri"],
                    "sidecar_s3_uri": dataset.get("sidecar_s3_uri"),
                    "selected_rows": int(dataset["selected_rows"]),
                    "unit_count": int(dataset["unit_count"]),
                    "main_inventory_sha256": dataset["main_inventory_sha256"],
                    "sidecar_inventory_sha256": dataset.get("sidecar_inventory_sha256"),
                    "totals": actual,
                }
            )

        completed_at = max(
            completion_timestamps,
            key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
        )
        validation = {
            "format_version": "image_edit_full_json_sidecar24_unit_inventory_validation_v1",
            "listing": listing,
            "unit_marker": "GET body SHA-256 metadata, strict JSON schema, exact frozen binding",
            "output": "HEAD content length/type and all resumability metadata",
            "output_body_downloaded": False,
            "inference_errors_required": 0,
            "inventory_jsonl_sha256": inventory_digest.hexdigest(),
        }
        lineage = {
            "generation_contract_sha256_counts": {
                key: generation_counts[key] for key in sorted(generation_counts)
            }
        }
        header = {
            "format_version": MANIFEST_FORMAT,
            "status": "complete",
            "completed_at": completed_at,
            "contract": contract,
            "totals": totals,
            "datasets": dataset_summaries,
            "lineage": lineage,
            "validation": validation,
        }
        manifest_path = temp_root / "manifest.json"
        manifest_digest, manifest_bytes = write_manifest(
            manifest_path,
            header=header,
            inventory_path=inventory_path,
        )
        result = {
            "format_version": SUCCESS_FORMAT,
            "status": "complete",
            "completed_at": completed_at,
            "manifest_uri": s3_uri(bucket, manifest_key),
            "manifest_sha256": manifest_digest,
            "manifest_bytes": manifest_bytes,
            "contract": contract,
            "totals": totals,
            "datasets": len(dataset_summaries),
            "lineage": lineage,
            "validation": validation,
        }
        if args.validate_only:
            return {"status": "validated_not_published", **result}
        put_file_once(
            client,
            bucket=bucket,
            key=manifest_key,
            path=manifest_path,
            digest=manifest_digest,
            content_type="application/json",
        )
        put_json_once(client, bucket=bucket, key=success_key, value=result)
        validated = validate_existing_success(
            client,
            bucket=bucket,
            success_key=success_key,
            expected_contract=contract,
        )
        if validated is None:
            raise IOError("root _SUCCESS was not readable after publication")
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--source-catalog-uri", required=True)
    parser.add_argument("--unit-catalog-uri", required=True)
    parser.add_argument("--source-catalog-sha256")
    parser.add_argument("--unit-catalog-sha256")
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--hf-artifact-manifest-sha256", required=True)
    parser.add_argument(
        "--allowed-generation-sha256",
        action="append",
        required=True,
        help="Repeat for each accepted scheduling lineage.",
    )
    parser.add_argument(
        "--expected-generation-count",
        action="append",
        default=[],
        type=parse_generation_count,
        metavar="SHA256=COUNT",
    )
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=2048)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "source_catalog_sha256",
        "unit_catalog_sha256",
        "checkpoint_manifest_sha256",
        "hf_artifact_manifest_sha256",
    ):
        value = getattr(args, name)
        if value is not None and SHA256_RE.fullmatch(value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    for value in args.allowed_generation_sha256:
        if SHA256_RE.fullmatch(value) is None:
            parser.error("--allowed-generation-sha256 must be a lowercase SHA-256")
    if len(args.allowed_generation_sha256) != len(set(args.allowed_generation_sha256)):
        parser.error("allowed generation hashes must be unique")
    if len(args.expected_generation_count) != len(dict(args.expected_generation_count)):
        parser.error("expected generation hashes must be unique")
    if args.workers <= 0 or args.batch_size <= 0 or args.progress_every <= 0:
        parser.error("worker, batch, and progress values must be positive")
    if not args.source_catalog.is_file() or not args.unit_catalog.is_file():
        parser.error("catalog files must exist")
    if args.temp_dir is not None and not args.temp_dir.is_dir():
        parser.error("--temp-dir must be an existing directory")
    parse_s3_uri(args.source_catalog_uri)
    parse_s3_uri(args.unit_catalog_uri)
    parse_s3_uri(args.output_s3)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = finalize(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
