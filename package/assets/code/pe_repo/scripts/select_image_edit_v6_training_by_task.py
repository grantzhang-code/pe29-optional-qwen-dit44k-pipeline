#!/usr/bin/env python3
"""Select an exact, instruction-task-stratified subset of V6 training JSONL.

The source rows do not contain a native task label.  This pipeline applies the
repository's versioned, mutually-exclusive instruction taxonomy, counts every
source row, assigns exact proportional or sqrt-frequency quotas with Hamilton
apportionment, and then selects deterministic hashes within every
(task, source-shard) stratum.  An immutable prior selection can be reconstructed
and excluded per shard, with sample-id digests proving zero overlap.

Eight restartable UTP workers run both passes.  All writes are create-only and
``_SUCCESS`` is published only after the output inventory, counts, schema
samples, and immutable source identities have been verified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_retry_instruction_eval_set import (  # noqa: E402
    TASK_RULES,
    TAXONOMY_VERSION,
    classify_instruction,
)


DEFAULT_PIPELINE_VERSION = "image_edit_v6_training_instruction_stratified_2m_v1"
PIPELINE_VERSION = DEFAULT_PIPELINE_VERSION
APPORTIONMENT_VERSION = "hamilton_largest_remainder_lexical_tiebreak_v1"
SELECTION_VERSION = "sha256_task_shard_rank_v1"
OUTPUT_ORDER_VERSION = "sha256_shard_output_order_v1"
FORMAT_VERSION = f"{PIPELINE_VERSION}_shard_v1"
PROPORTIONAL_POLICY = "proportional"
SQRT_REMAINING_POLICY = "sqrt_remaining"
ALLOCATION_POLICIES = (PROPORTIONAL_POLICY, SQRT_REMAINING_POLICY)
SQRT_WEIGHT_SCALE = 10**9
SQRT_WEIGHT_VERSION = "integer_sqrt_remaining_frequency_scale_1e9_v1"

EXPECTED_INPUT_ROWS = 7_236_807
EXPECTED_INPUT_SHARDS = 256
EXPECTED_INPUT_MANIFEST_FORMAT = "image_edit_v6_training_hld_postprocess_v3_manifest_v1"
EXPECTED_BASE_MANIFEST_FORMAT = "image_edit_v6_training_manifest_v1"
EXPECTED_POSTPROCESS_VERSION = "image_edit_v6_small_bbox_hallucinated_clause_cleanup_v7"

DEFAULT_INPUT_MANIFEST_URI = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
    "full64-v7-call2-final-prepared-32k-256shards-gavin-disjoint-"
    "no-recovery-dedup-v2-hld-postprocess-v7-fixed-v1/"
    "hld_postprocess_manifest.json"
)
DEFAULT_INPUT_MANIFEST_SHA256 = "807185074997b14fb9e0e5433efb11d5efc01485e96b92a0b178b3f2b76ca91a"
DEFAULT_INPUT_MANIFEST_VERSION_ID = "NJumi3.xvgI3UpmTEY78N0N2HTydFmkI"
DEFAULT_BASE_MANIFEST_URI = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
    "full64-v7-call2-final-prepared-32k-256shards-gavin-disjoint-"
    "no-recovery-dedup-v2/image_edit_v6_full_json.manifest.json"
)
DEFAULT_BASE_MANIFEST_SHA256 = "74683be13008294481c18991e26476b4bf582d7d95ef2c711bc78d32114016ea"
DEFAULT_BASE_MANIFEST_VERSION_ID = "xyRt1dwACYF4Y0Ylh9XLvhm410O8RNFi"
DEFAULT_OUTPUT_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v6/training/"
    "full64-v7-call2-final-prepared-32k-256shards-gavin-disjoint-"
    "no-recovery-dedup-v2-hld-postprocess-v7-fixed-v1-"
    "edit-task-stratified-2m-v1/"
)
DEFAULT_TARGET_ROWS = 2_000_000
DEFAULT_WORKER_COUNT = 8
DEFAULT_PROCESS_COUNT = 32
DEFAULT_SEED = "image-edit-v6-training-task-stratified-2m-20260818-v1"
DEFAULT_WAIT_SECONDS = 7_200
DEFAULT_POLL_SECONDS = 20
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
REQUIRED_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "status",
        "instruction",
        "source_s3_uri",
        "target_s3_uri",
        "contract_version",
        "contract_sha256",
        "bbox_coordinate_order",
        "call2_audit",
        "final_json",
        "hld_label_cleanup",
    }
)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


_PROCESS_CLIENT: Any | None = None
_PROCESS_REGION = "us-east-1"


def configure_pipeline_version(value: str) -> None:
    """Set the campaign-specific format namespace before any durable writes."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,127}", value):
        raise ValueError(f"invalid pipeline version: {value!r}")
    global PIPELINE_VERSION, FORMAT_VERSION
    PIPELINE_VERSION = value
    FORMAT_VERSION = f"{PIPELINE_VERSION}_shard_v1"


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def parse_s3_uri(value: str, *, require_key: bool = True) -> S3Location:
    if not value.startswith("s3://"):
        raise ValueError(f"expected an S3 URI, got {value!r}")
    bucket, separator, key = value[5:].partition("/")
    normalized = key.strip("/")
    if not bucket or (require_key and (not separator or not normalized)):
        raise ValueError(f"incomplete S3 URI: {value!r}")
    return S3Location(bucket, normalized)


def make_s3_client(*, profile: str | None, region: str) -> Any:
    session = (
        boto3.Session(profile_name=profile, region_name=region)
        if profile
        else boto3.Session(region_name=region)
    )
    return session.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"max_attempts": 12, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=300,
            max_pool_connections=64,
        ),
    )


def reset_process_client(region: str = "us-east-1") -> None:
    global _PROCESS_CLIENT, _PROCESS_REGION
    _PROCESS_CLIENT = None
    _PROCESS_REGION = region


def process_client() -> Any:
    global _PROCESS_CLIENT
    if _PROCESS_CLIENT is None:
        _PROCESS_CLIENT = make_s3_client(profile=None, region=_PROCESS_REGION)
    return _PROCESS_CLIENT


def is_not_found(exc: ClientError) -> bool:
    return str(exc.response.get("Error", {}).get("Code", "")) in NOT_FOUND_CODES


class OutputStore:
    def __init__(self, root: str, client: Any) -> None:
        location = parse_s3_uri(root)
        self.bucket = location.bucket
        self.prefix = location.key.rstrip("/") + "/"
        self.client = client

    def key(self, relative: str) -> str:
        return self.prefix + relative.lstrip("/")

    def read_json(self, relative: str) -> dict[str, Any] | None:
        key = self.key(relative)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if is_not_found(exc):
                return None
            raise
        body = response["Body"]
        try:
            value = json.loads(body.read())
        finally:
            body.close()
        if not isinstance(value, dict):
            raise TypeError(f"s3://{self.bucket}/{key} must contain a JSON object")
        return value

    def put_json_once(self, relative: str, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = canonical_json_bytes(dict(value), newline=True)
        key = self.key(relative)
        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                Metadata={"sha256": sha256_bytes(payload)},
                IfNoneMatch="*",
            )
            reused = False
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code", "")) not in {
                "412",
                "PreconditionFailed",
            }:
                raise
            existing = self.client.get_object(Bucket=self.bucket, Key=key)
            body = existing["Body"]
            try:
                existing_payload = body.read()
            finally:
                body.close()
            if existing_payload != payload:
                raise ValueError(f"create-only JSON collision at s3://{self.bucket}/{key}") from exc
            response = existing
            reused = True
        return {
            "uri": f"s3://{self.bucket}/{key}",
            "key": key,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "version_id": str(response.get("VersionId") or ""),
            "reused": reused,
        }

    def put_file_once(
        self,
        relative: str,
        path: Path,
        *,
        rows: int,
        metadata: Mapping[str, str],
    ) -> dict[str, Any]:
        size, digest = sha256_file(path)
        key = self.key(relative)
        object_metadata = {
            **{str(k): str(v) for k, v in metadata.items()},
            "rows": str(rows),
            "sha256": digest,
        }
        with path.open("rb") as stream:
            try:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=stream,
                    ContentType="application/x-ndjson",
                    Metadata=object_metadata,
                    IfNoneMatch="*",
                )
                reused = False
            except ClientError as exc:
                if str(exc.response.get("Error", {}).get("Code", "")) not in {
                    "412",
                    "PreconditionFailed",
                }:
                    raise
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                existing_metadata = head.get("Metadata") or {}
                if (
                    int(head.get("ContentLength", -1)) != size
                    or existing_metadata.get("sha256") != digest
                    or int(existing_metadata.get("rows", -1)) != rows
                    or any(existing_metadata.get(k) != v for k, v in object_metadata.items())
                ):
                    raise ValueError(f"create-only JSONL collision at s3://{self.bucket}/{key}") from exc
                response = head
                reused = True
        return {
            "uri": f"s3://{self.bucket}/{key}",
            "key": key,
            "bytes": size,
            "rows": rows,
            "sha256": digest,
            "version_id": str(response.get("VersionId") or ""),
            "reused": reused,
        }


def read_pinned_json(
    client: Any,
    uri: str,
    *,
    version_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    location = parse_s3_uri(uri)
    response = client.get_object(
        Bucket=location.bucket,
        Key=location.key,
        VersionId=version_id,
    )
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        body.close()
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"manifest SHA mismatch for {uri}: {actual} != {expected_sha256}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"{uri} must contain a JSON object")
    return value


def _validate_base_manifest(value: Mapping[str, Any]) -> None:
    sample_audit = value.get("sample_id_audit") or {}
    if (
        value.get("format_version") != EXPECTED_BASE_MANIFEST_FORMAT
        or value.get("complete") is not True
        or int(value.get("total_rows", -1)) != EXPECTED_INPUT_ROWS
        or int(value.get("physical_shard_count", -1)) != EXPECTED_INPUT_SHARDS
        or sample_audit.get("complete") is not True
        or int(sample_audit.get("unique_sample_ids", -1)) != EXPECTED_INPUT_ROWS
        or int(sample_audit.get("duplicate_rows", -1)) != 0
    ):
        raise ValueError("base training manifest violates its immutable uniqueness contract")


def _validate_input_manifest(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = value.get("aggregate") or {}
    totals = aggregate.get("totals") or {}
    raw_objects = value.get("objects")
    if (
        value.get("format_version") != EXPECTED_INPUT_MANIFEST_FORMAT
        or value.get("postprocess_version") != EXPECTED_POSTPROCESS_VERSION
        or value.get("complete") is not True
        or int(value.get("physical_shard_count", -1)) != EXPECTED_INPUT_SHARDS
        or int(aggregate.get("objects", -1)) != EXPECTED_INPUT_SHARDS
        or int(totals.get("rows", -1)) != EXPECTED_INPUT_ROWS
        or not isinstance(raw_objects, list)
        or len(raw_objects) != EXPECTED_INPUT_SHARDS
    ):
        raise ValueError("HLD input manifest violates the expected publication contract")

    result: list[dict[str, Any]] = []
    relative_keys: set[str] = set()
    rows = 0
    for item in raw_objects:
        if not isinstance(item, Mapping):
            raise TypeError("input manifest object entry must be a mapping")
        output = item.get("output")
        relative = item.get("relative_key")
        if not isinstance(output, Mapping) or not isinstance(relative, str):
            raise ValueError("input manifest object entry is incomplete")
        normalized_relative = "shards/" + relative.lstrip("/")
        if normalized_relative in relative_keys:
            raise ValueError(f"duplicate source relative key: {normalized_relative}")
        relative_keys.add(normalized_relative)
        expected_suffix = "/" + normalized_relative
        if not str(output.get("key") or "").endswith(expected_suffix):
            raise ValueError(f"source key does not match relative path: {output}")
        record = {
            "bucket": str(output.get("bucket") or ""),
            "key": str(output.get("key") or ""),
            "relative_key": normalized_relative,
            "bytes": int(output.get("bytes") or 0),
            "rows": int(output.get("rows") or 0),
            "sha256": str(output.get("sha256") or ""),
        }
        if (
            not record["bucket"]
            or not record["key"]
            or record["bytes"] <= 0
            or record["rows"] <= 0
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"invalid input shard identity: {record}")
        rows += record["rows"]
        result.append(record)
    if rows != EXPECTED_INPUT_ROWS:
        raise ValueError(f"input shard row sum {rows} != {EXPECTED_INPUT_ROWS}")
    return sorted(result, key=lambda item: item["relative_key"])


def load_input_contract(args: argparse.Namespace, client: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = read_pinned_json(
        client,
        args.base_manifest_uri,
        version_id=args.base_manifest_version_id,
        expected_sha256=args.base_manifest_sha256,
    )
    _validate_base_manifest(base)
    manifest = read_pinned_json(
        client,
        args.input_manifest_uri,
        version_id=args.input_manifest_version_id,
        expected_sha256=args.input_manifest_sha256,
    )
    shards = _validate_input_manifest(manifest)
    if manifest.get("input_root") != str(args.base_manifest_uri).rsplit("/", 1)[0] + "/":
        raise ValueError("HLD manifest input root differs from the pinned base manifest root")
    return manifest, shards


def _validate_pinned_record(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    record = {
        "uri": str(value.get("uri") or ""),
        "version_id": str(value.get("version_id") or ""),
        "sha256": str(value.get("sha256") or ""),
    }
    parse_s3_uri(record["uri"])
    if (
        not record["version_id"]
        or record["version_id"] == "null"
        or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    ):
        raise ValueError(f"invalid pinned {label} record: {record}")
    return record


def load_exclusion_contract(args: argparse.Namespace, client: Any) -> dict[str, Any] | None:
    """Load and validate an immutable previously selected dataset to exclude."""

    if not args.exclude_manifest_uri:
        return None
    manifest = read_pinned_json(
        client,
        args.exclude_manifest_uri,
        version_id=args.exclude_manifest_version_id,
        expected_sha256=args.exclude_manifest_sha256,
    )
    pipeline_version = str(manifest.get("pipeline_version") or "")
    validation = manifest.get("validation") or {}
    outputs = manifest.get("outputs")
    task_counts = {str(k): int(v) for k, v in (manifest.get("task_counts") or {}).items()}
    if (
        not pipeline_version
        or manifest.get("format_version") != f"{pipeline_version}_manifest_v1"
        or manifest.get("complete") is not True
        or manifest.get("taxonomy_version") != TAXONOMY_VERSION
        or int(manifest.get("rows", -1)) <= 0
        or int(manifest.get("rows", -1)) >= EXPECTED_INPUT_ROWS
        or int(manifest.get("physical_shard_count", -1)) != EXPECTED_INPUT_SHARDS
        or sum(task_counts.values()) != int(manifest.get("rows", -1))
        or validation.get("exact_target_rows") is not True
        or validation.get("source_rows_copied_byte_for_byte") is not True
        or not isinstance(outputs, list)
        or len(outputs) != EXPECTED_INPUT_SHARDS
    ):
        raise ValueError("exclusion manifest violates the immutable selected-dataset contract")

    plan_record = _validate_pinned_record(
        manifest.get("selection_plan") or {},
        label="exclusion selection plan",
    )
    plan = read_pinned_json(
        client,
        plan_record["uri"],
        version_id=plan_record["version_id"],
        expected_sha256=plan_record["sha256"],
    )
    shard_quotas = plan.get("shard_quotas")
    plan_task_quotas = {str(k): int(v) for k, v in (plan.get("task_quotas") or {}).items()}
    if (
        plan.get("format_version") != f"{pipeline_version}_selection_plan_v1"
        or plan.get("complete") is not True
        or plan.get("taxonomy_version") != TAXONOMY_VERSION
        or plan.get("input_manifest_sha256") != args.input_manifest_sha256
        or int(plan.get("source_rows", -1)) != EXPECTED_INPUT_ROWS
        or int(plan.get("target_rows", -1)) != int(manifest["rows"])
        or not isinstance(plan.get("seed"), str)
        or not plan.get("seed")
        or plan_task_quotas != task_counts
        or not isinstance(shard_quotas, Mapping)
        or len(shard_quotas) != EXPECTED_INPUT_SHARDS
    ):
        raise ValueError("exclusion selection plan does not match its pinned manifest")

    verifier_record = _validate_pinned_record(
        manifest.get("verifier_summary") or {},
        label="exclusion verifier",
    )
    verifier = read_pinned_json(
        client,
        verifier_record["uri"],
        version_id=verifier_record["version_id"],
        expected_sha256=verifier_record["sha256"],
    )
    if (
        verifier.get("pass") is not True
        or int(verifier.get("row_count", -1)) != int(manifest["rows"])
        or int(verifier.get("shard_count", -1)) != EXPECTED_INPUT_SHARDS
        or {str(k): int(v) for k, v in (verifier.get("category_counts") or {}).items()} != task_counts
    ):
        raise ValueError("exclusion verifier does not match its pinned manifest")

    outputs_by_relative: dict[str, dict[str, Any]] = {}
    observed_tasks: Counter[str] = Counter()
    observed_rows = 0
    for item in outputs:
        if not isinstance(item, Mapping):
            raise TypeError("exclusion output entry must be a mapping")
        relative = str(item.get("relative_key") or "")
        quotas = {str(k): int(v) for k, v in (item.get("quotas") or {}).items()}
        selected_counts = {str(k): int(v) for k, v in (item.get("selected_task_counts") or {}).items()}
        selected_rows = int(item.get("selected_rows", -1))
        selected_id_digest = str(item.get("selected_sample_ids_sorted_sha256") or "")
        if (
            not relative.startswith("shards/")
            or relative in outputs_by_relative
            or relative not in shard_quotas
            or quotas != {str(k): int(v) for k, v in shard_quotas[relative].items()}
            or selected_counts != {k: v for k, v in quotas.items() if v}
            or selected_rows != sum(quotas.values())
            or int(item.get("selected_unique_sample_ids", -1)) != selected_rows
            or not re.fullmatch(r"[0-9a-f]{64}", selected_id_digest)
        ):
            raise ValueError(f"invalid exclusion output audit for {relative!r}")
        outputs_by_relative[relative] = dict(item)
        observed_tasks.update(selected_counts)
        observed_rows += selected_rows
    if (
        set(outputs_by_relative) != set(shard_quotas)
        or observed_rows != int(manifest["rows"])
        or dict(sorted(observed_tasks.items())) != dict(sorted(task_counts.items()))
    ):
        raise ValueError("exclusion output inventory does not close against its manifest")

    return {
        "manifest_record": {
            "uri": args.exclude_manifest_uri,
            "version_id": args.exclude_manifest_version_id,
            "sha256": args.exclude_manifest_sha256,
            "rows": int(manifest["rows"]),
        },
        "plan_record": plan_record,
        "verifier_record": verifier_record,
        "plan": plan,
        "outputs_by_relative": outputs_by_relative,
    }


def validate_training_row(row: Mapping[str, Any], *, location: str) -> tuple[str, str, str]:
    missing = REQUIRED_ROW_FIELDS - set(row)
    if missing:
        raise ValueError(f"{location}: missing required fields {sorted(missing)}")
    sample_id = row.get("sample_id")
    instruction = row.get("instruction")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{location}: invalid sample_id")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{location}: invalid instruction")
    if row.get("status") not in {"validated", "manual_review_required"}:
        raise ValueError(f"{location}: invalid status {row.get('status')!r}")
    source_uri = row.get("source_s3_uri")
    target_uri = row.get("target_s3_uri")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError(f"{location}: invalid source_s3_uri")
    if not isinstance(target_uri, str) or not target_uri.strip():
        raise ValueError(f"{location}: invalid target_s3_uri")
    if source_uri.strip() == target_uri.strip():
        raise ValueError(f"{location}: source and target media URIs must differ")
    if not isinstance(row.get("final_json"), Mapping):
        raise ValueError(f"{location}: final_json must be a mapping")
    call2 = row.get("call2_audit")
    if not isinstance(call2, Mapping) or call2.get("valid") is not True:
        raise ValueError(f"{location}: call2_audit.valid must be true")
    cleanup = row.get("hld_label_cleanup")
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("version") != EXPECTED_POSTPROCESS_VERSION
        or int(cleanup.get("remaining_invalid_numbered_label_occurrences", -1)) != 0
    ):
        raise ValueError(f"{location}: invalid HLD cleanup contract")
    task, matched_rule = classify_instruction(instruction)
    return sample_id, task, str(matched_rule or "")


def _source_head(source: Mapping[str, Any]) -> dict[str, Any]:
    client = process_client()
    head = client.head_object(Bucket=source["bucket"], Key=source["key"])
    metadata = head.get("Metadata") or {}
    if (
        int(head.get("ContentLength", -1)) != int(source["bytes"])
        or metadata.get("sha256") != source["sha256"]
        or int(metadata.get("rows", -1)) != int(source["rows"])
        or metadata.get("postprocess-version") != EXPECTED_POSTPROCESS_VERSION
    ):
        raise ValueError(f"source HEAD identity mismatch: s3://{source['bucket']}/{source['key']}")
    version_id = str(head.get("VersionId") or "")
    if not version_id or version_id == "null":
        raise ValueError(f"source VersionId missing: s3://{source['bucket']}/{source['key']}")
    return {**dict(source), "version_id": version_id}


def _iter_pinned_lines(source: Mapping[str, Any]) -> Iterable[tuple[int, bytes]]:
    response = process_client().get_object(
        Bucket=source["bucket"],
        Key=source["key"],
        VersionId=source["version_id"],
    )
    body = response["Body"]
    digest = hashlib.sha256()
    bytes_read = 0
    rows = 0
    try:
        for line_number, raw in enumerate(body.iter_lines(chunk_size=1024 * 1024), 1):
            if not raw:
                raise ValueError(f"{source['key']}:{line_number}: blank JSONL line")
            reconstructed = raw + b"\n"
            digest.update(reconstructed)
            bytes_read += len(reconstructed)
            rows += 1
            yield line_number, raw
    finally:
        body.close()
    if (
        rows != int(source["rows"])
        or bytes_read != int(source["bytes"])
        or digest.hexdigest() != source["sha256"]
    ):
        raise ValueError(
            f"pinned source verification failed for {source['key']}: "
            f"rows={rows}/{source['rows']} bytes={bytes_read}/{source['bytes']} "
            f"sha256={digest.hexdigest()}/{source['sha256']}"
        )


def count_source_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    source = _source_head(task["source"])
    task_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    key_signatures: Counter[str] = Counter()
    for line_number, raw in _iter_pinned_lines(source):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source['key']}:{line_number}: invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise TypeError(f"{source['key']}:{line_number}: row must be an object")
        sample_id, category, matched_rule = validate_training_row(
            row, location=f"{source['key']}:{line_number}"
        )
        if sample_id in seen_ids:
            raise ValueError(f"{source['key']}:{line_number}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        task_counts[category] += 1
        status_counts[str(row["status"])] += 1
        rule_counts[sha256_bytes(matched_rule.encode("utf-8")) if matched_rule else "none"] += 1
        key_signatures[sha256_bytes("\0".join(sorted(row)).encode("utf-8"))] += 1
    if sum(task_counts.values()) != source["rows"]:
        raise AssertionError("task counting did not conserve rows")
    return {
        "format_version": f"{PIPELINE_VERSION}_count_shard_v1",
        "complete": True,
        "taxonomy_version": TAXONOMY_VERSION,
        "source": source,
        "rows": source["rows"],
        "task_counts": dict(sorted(task_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "matched_rule_sha256_counts": dict(sorted(rule_counts.items())),
        "top_level_key_signature_counts": dict(sorted(key_signatures.items())),
        "unique_sample_ids_within_shard": len(seen_ids),
    }


def count_audit_relative(relative_key: str) -> str:
    return f"control/counts/{relative_key}.count.json"


def output_audit_relative(relative_key: str) -> str:
    return f"_audit/outputs/{relative_key}.audit.json"


def worker_count_relative(worker_index: int, worker_count: int) -> str:
    return f"control/workers/count-worker-{worker_index:03d}-of-{worker_count:03d}.json"


def worker_select_relative(worker_index: int, worker_count: int) -> str:
    return f"control/workers/select-worker-{worker_index:03d}-of-{worker_count:03d}.json"


def _valid_count_audit(value: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    recorded = value.get("source") or {}
    return bool(
        value.get("format_version") == f"{PIPELINE_VERSION}_count_shard_v1"
        and value.get("complete") is True
        and value.get("taxonomy_version") == TAXONOMY_VERSION
        and recorded.get("bucket") == source["bucket"]
        and recorded.get("key") == source["key"]
        and recorded.get("sha256") == source["sha256"]
        and int(recorded.get("rows", -1)) == int(source["rows"])
        and int(recorded.get("bytes", -1)) == int(source["bytes"])
        and int(value.get("rows", -1)) == int(source["rows"])
        and sum(int(v) for v in (value.get("task_counts") or {}).values()) == int(source["rows"])
    )


def aggregate_count_audits(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    rows = 0
    source_bytes = 0
    for audit in audits:
        rows += int(audit["rows"])
        source_bytes += int(audit["source"]["bytes"])
        tasks.update({k: int(v) for k, v in audit["task_counts"].items()})
        statuses.update({k: int(v) for k, v in audit["status_counts"].items()})
    return {
        "shards": len(audits),
        "rows": rows,
        "source_bytes": source_bytes,
        "task_counts": dict(sorted(tasks.items())),
        "status_counts": dict(sorted(statuses.items())),
    }


def _hamilton_unbounded(weights: Mapping[str, int], target: int) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in weights.items()}
    if target < 0 or any(value < 0 for value in normalized.values()):
        raise ValueError("Hamilton weights and target must be non-negative")
    total = sum(normalized.values())
    if total == 0:
        if target:
            raise ValueError("cannot allocate a nonzero target from zero weight")
        return {key: 0 for key in sorted(normalized)}
    floors = {key: target * value // total for key, value in normalized.items()}
    remainder = target - sum(floors.values())
    order = sorted(
        normalized,
        key=lambda key: (-(target * normalized[key] % total), key),
    )
    for key in order[:remainder]:
        floors[key] += 1
    if sum(floors.values()) != target:
        raise AssertionError("Hamilton allocation failed")
    return dict(sorted(floors.items()))


def hamilton_weighted_quotas(
    weights: Mapping[str, int],
    target: int,
    *,
    capacities: Mapping[str, int],
) -> dict[str, int]:
    """Allocate integer quotas from weights while respecting hard capacities."""

    keys = sorted(set(weights) | set(capacities))
    normalized_weights = {key: int(weights.get(key, 0)) for key in keys}
    normalized_capacities = {key: int(capacities.get(key, 0)) for key in keys}
    if (
        target < 0
        or any(value < 0 for value in normalized_weights.values())
        or any(value < 0 for value in normalized_capacities.values())
    ):
        raise ValueError("Hamilton weights, capacities, and target must be non-negative")
    available = sum(normalized_capacities.values())
    if target > available:
        raise ValueError(f"target {target} exceeds available rows {available}")
    result = {key: 0 for key in keys}
    remaining = target
    active = {
        key: normalized_weights[key]
        for key in keys
        if normalized_weights[key] > 0 and normalized_capacities[key] > 0
    }
    while remaining:
        if not active:
            raise ValueError("positive capacity has zero allocation weight")
        proposal = _hamilton_unbounded(active, remaining)
        saturated = [key for key in active if proposal[key] > normalized_capacities[key] - result[key]]
        if not saturated:
            for key, value in proposal.items():
                result[key] += value
            remaining = 0
            break
        for key in sorted(saturated):
            capacity = normalized_capacities[key] - result[key]
            result[key] += capacity
            remaining -= capacity
            del active[key]
    if sum(result.values()) != target or any(result[key] > normalized_capacities[key] for key in keys):
        raise AssertionError("capacitated Hamilton allocation failed")
    return result


def hamilton_quotas(counts: Mapping[str, int], target: int) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in counts.items()}
    return hamilton_weighted_quotas(normalized, target, capacities=normalized)


def sqrt_frequency_weights(counts: Mapping[str, int]) -> dict[str, int]:
    """Return deterministic fixed-point sqrt-frequency weights (alpha=0.5)."""

    normalized = {str(key): int(value) for key, value in counts.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("sqrt-frequency counts must be non-negative")
    return {
        key: math.isqrt(value * SQRT_WEIGHT_SCALE * SQRT_WEIGHT_SCALE)
        for key, value in sorted(normalized.items())
    }


def build_selection_plan(
    count_audits: Sequence[Mapping[str, Any]],
    *,
    target_rows: int,
    seed: str,
    input_manifest_sha256: str,
    allocation_policy: str = PROPORTIONAL_POLICY,
    exclusion_plan: Mapping[str, Any] | None = None,
    exclusion_manifest_record: Mapping[str, Any] | None = None,
    exclusion_plan_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted(count_audits, key=lambda item: str(item["source"]["relative_key"]))
    source_counts: Counter[str] = Counter()
    per_shard: dict[str, dict[str, int]] = {}
    for audit in ordered:
        relative = str(audit["source"]["relative_key"])
        counts = {str(k): int(v) for k, v in audit["task_counts"].items()}
        if relative in per_shard:
            raise ValueError(f"duplicate count audit for {relative}")
        per_shard[relative] = counts
        source_counts.update(counts)
    if len(per_shard) != EXPECTED_INPUT_SHARDS or sum(source_counts.values()) != EXPECTED_INPUT_ROWS:
        raise ValueError("count audits do not cover the complete input inventory")

    excluded_shard_quotas: dict[str, dict[str, int]] = {relative: {} for relative in per_shard}
    if exclusion_plan is not None:
        raw_excluded = exclusion_plan.get("shard_quotas")
        if not isinstance(raw_excluded, Mapping) or set(raw_excluded) != set(per_shard):
            raise ValueError("exclusion plan does not cover the current source shard inventory")
        for relative, source_task_counts in per_shard.items():
            excluded = {str(k): int(v) for k, v in raw_excluded[relative].items()}
            if any(value < 0 or value > source_task_counts.get(key, 0) for key, value in excluded.items()):
                raise ValueError(f"exclusion quota exceeds source availability for {relative}")
            excluded_shard_quotas[relative] = dict(sorted(excluded.items()))

    excluded_counts: Counter[str] = Counter()
    remaining_per_shard: dict[str, dict[str, int]] = {}
    for relative, counts in per_shard.items():
        excluded = excluded_shard_quotas[relative]
        remaining = {category: count - excluded.get(category, 0) for category, count in counts.items()}
        if any(value < 0 for value in remaining.values()):
            raise ValueError(f"negative remaining availability for {relative}")
        remaining_per_shard[relative] = remaining
        excluded_counts.update(excluded)
    remaining_counts = {
        category: source_counts[category] - excluded_counts.get(category, 0)
        for category in sorted(source_counts)
    }
    if target_rows > sum(remaining_counts.values()):
        raise ValueError("target exceeds rows remaining after exclusion")
    if allocation_policy == PROPORTIONAL_POLICY:
        allocation_weights = dict(remaining_counts)
        allocation_weight_version = "remaining_frequency_v1"
    elif allocation_policy == SQRT_REMAINING_POLICY:
        allocation_weights = sqrt_frequency_weights(remaining_counts)
        allocation_weight_version = SQRT_WEIGHT_VERSION
    else:
        raise ValueError(f"unsupported allocation policy: {allocation_policy!r}")
    task_quotas = hamilton_weighted_quotas(
        allocation_weights,
        target_rows,
        capacities=remaining_counts,
    )
    shard_quotas: dict[str, dict[str, int]] = {relative: {} for relative in per_shard}
    for category, quota in task_quotas.items():
        available = {relative: counts.get(category, 0) for relative, counts in remaining_per_shard.items()}
        allocation = hamilton_quotas(available, quota)
        for relative, value in allocation.items():
            if value:
                shard_quotas[relative][category] = value
    shard_rows = {relative: sum(values.values()) for relative, values in shard_quotas.items()}
    if sum(shard_rows.values()) != target_rows:
        raise AssertionError("per-shard allocations do not sum to target")
    return {
        "format_version": f"{PIPELINE_VERSION}_selection_plan_v1",
        "complete": True,
        "input_manifest_sha256": input_manifest_sha256,
        "taxonomy_version": TAXONOMY_VERSION,
        "apportionment_version": APPORTIONMENT_VERSION,
        "selection_version": SELECTION_VERSION,
        "output_order_version": OUTPUT_ORDER_VERSION,
        "allocation_policy": allocation_policy,
        "allocation_weight_version": allocation_weight_version,
        "seed": seed,
        "source_rows": EXPECTED_INPUT_ROWS,
        "excluded_rows": sum(excluded_counts.values()),
        "remaining_rows": sum(remaining_counts.values()),
        "target_rows": target_rows,
        "source_task_counts": dict(sorted(source_counts.items())),
        "excluded_task_counts": dict(sorted(excluded_counts.items())),
        "remaining_task_counts": dict(sorted(remaining_counts.items())),
        "allocation_weights": dict(sorted(allocation_weights.items())),
        "task_quotas": task_quotas,
        "excluded_shard_quotas": {
            k: dict(sorted(v.items())) for k, v in sorted(excluded_shard_quotas.items())
        },
        "shard_quotas": {k: dict(sorted(v.items())) for k, v in sorted(shard_quotas.items())},
        "shard_target_rows": dict(sorted(shard_rows.items())),
        **(
            {
                "exclusion_manifest": dict(exclusion_manifest_record or {}),
                "exclusion_selection_plan": dict(exclusion_plan_record or {}),
                "exclusion_seed": str(exclusion_plan.get("seed") or ""),
            }
            if exclusion_plan is not None
            else {}
        ),
        "validation": {
            "source_shards": len(per_shard),
            "task_count": len(source_counts),
            "source_row_accounting_closes": (
                sum(source_counts.values()) == sum(excluded_counts.values()) + sum(remaining_counts.values())
            ),
            "task_quotas_sum_to_target": sum(task_quotas.values()) == target_rows,
            "shard_quotas_sum_to_target": sum(shard_rows.values()) == target_rows,
            "all_quotas_within_remaining_counts": all(
                task_quotas[key] <= remaining_counts[key] for key in source_counts
            ),
        },
    }


def selection_digest(seed: str, category: str, sample_id: str) -> str:
    return sha256_bytes(
        b"selection\0"
        + seed.encode("utf-8")
        + b"\0"
        + category.encode("utf-8")
        + b"\0"
        + sample_id.encode("utf-8")
    )


def output_order_digest(seed: str, sample_id: str) -> str:
    return sha256_bytes(b"output-order\0" + seed.encode("utf-8") + b"\0" + sample_id.encode("utf-8"))


def select_raw_rows(
    candidates: Mapping[str, Sequence[tuple[str, str, bytes]]],
    *,
    quotas: Mapping[str, int],
    seed: str,
) -> tuple[list[tuple[str, str, bytes]], dict[str, int]]:
    selected: list[tuple[str, str, bytes]] = []
    selected_counts: dict[str, int] = {}
    for category in sorted(set(candidates) | set(quotas)):
        rows = list(candidates.get(category, ()))
        quota = int(quotas.get(category, 0))
        if quota > len(rows):
            raise ValueError(f"category {category!r} has {len(rows)} rows but quota {quota}")
        rows.sort(key=lambda item: (item[0], item[1]))
        chosen = rows[:quota]
        selected.extend(chosen)
        if quota:
            selected_counts[category] = quota
    selected.sort(key=lambda item: (output_order_digest(seed, item[1]), item[1]))
    return selected, dict(sorted(selected_counts.items()))


def exclude_raw_rows(
    candidates: Mapping[str, Sequence[tuple[str, str, bytes]]],
    *,
    quotas: Mapping[str, int],
    seed: str,
) -> tuple[
    dict[str, list[tuple[str, str, bytes]]],
    dict[str, int],
    set[str],
    str,
]:
    """Reconstruct a prior deterministic selection and return the remaining rows."""

    remaining: dict[str, list[tuple[str, str, bytes]]] = {}
    excluded_counts: dict[str, int] = {}
    excluded_ids: set[str] = set()
    for category in sorted(set(candidates) | set(quotas)):
        rows = list(candidates.get(category, ()))
        quota = int(quotas.get(category, 0))
        if quota < 0 or quota > len(rows):
            raise ValueError(f"excluded category {category!r} has {len(rows)} rows but quota {quota}")
        ranked = sorted(
            rows,
            key=lambda item: (selection_digest(seed, category, item[1]), item[1]),
        )
        category_excluded = ranked[:quota]
        category_excluded_ids = {item[1] for item in category_excluded}
        if excluded_ids & category_excluded_ids:
            raise ValueError("duplicate sample_id across exclusion categories")
        excluded_ids.update(category_excluded_ids)
        remaining[category] = [item for item in rows if item[1] not in category_excluded_ids]
        if quota:
            excluded_counts[category] = quota
    excluded_digest = sha256_bytes("\n".join(sorted(excluded_ids)).encode("utf-8"))
    return remaining, dict(sorted(excluded_counts.items())), excluded_ids, excluded_digest


def select_source_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(task["source"])
    quotas = {str(k): int(v) for k, v in task["quotas"].items()}
    excluded_quotas = {str(k): int(v) for k, v in (task.get("excluded_quotas") or {}).items()}
    exclusion_seed = str(task.get("exclusion_seed") or "")
    candidates: dict[str, list[tuple[str, str, bytes]]] = defaultdict(list)
    source_task_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_number, raw in _iter_pinned_lines(source):
        row = json.loads(raw)
        if not isinstance(row, Mapping):
            raise TypeError(f"{source['key']}:{line_number}: row must be an object")
        sample_id, category, _ = validate_training_row(row, location=f"{source['key']}:{line_number}")
        if sample_id in seen_ids:
            raise ValueError(f"{source['key']}:{line_number}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        source_task_counts[category] += 1
        status_counts[str(row["status"])] += 1
        candidates[category].append(
            (selection_digest(str(task["seed"]), category, sample_id), sample_id, raw)
        )
    expected_counts = {str(k): int(v) for k, v in task["expected_task_counts"].items()}
    if dict(sorted(source_task_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(f"source task counts changed for {source['relative_key']}")
    if any(excluded_quotas.values()) and not exclusion_seed:
        raise ValueError("exclusion seed is required for nonzero excluded quotas")
    remaining_candidates, excluded_counts, excluded_ids, excluded_id_digest = exclude_raw_rows(
        candidates,
        quotas=excluded_quotas,
        seed=exclusion_seed,
    )
    expected_excluded_digest = str(task.get("expected_excluded_sample_ids_sorted_sha256") or "")
    if expected_excluded_digest and excluded_id_digest != expected_excluded_digest:
        raise ValueError(
            f"excluded sample-id digest differs from pinned manifest for {source['relative_key']}"
        )
    selected, selected_counts = select_raw_rows(
        remaining_candidates,
        quotas=quotas,
        seed=str(task["seed"]),
    )
    output_path = Path(str(task["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    output_bytes = 0
    selected_ids: list[str] = []
    with output_path.open("wb") as stream:
        for _, sample_id, raw in selected:
            payload = raw + b"\n"
            stream.write(payload)
            output_digest.update(payload)
            output_bytes += len(payload)
            selected_ids.append(sample_id)
    overlap = excluded_ids.intersection(selected_ids)
    if overlap:
        raise ValueError(f"selection overlaps excluded dataset in {source['relative_key']}")
    sorted_id_digest = sha256_bytes("\n".join(sorted(selected_ids)).encode("utf-8"))
    return {
        "source": source,
        "source_task_counts": dict(sorted(source_task_counts.items())),
        "source_status_counts": dict(sorted(status_counts.items())),
        "excluded_quotas": dict(sorted(excluded_quotas.items())),
        "excluded_task_counts": excluded_counts,
        "excluded_rows": len(excluded_ids),
        "excluded_sample_ids_sorted_sha256": excluded_id_digest,
        "excluded_digest_matches_pinned_manifest": (
            not expected_excluded_digest or excluded_id_digest == expected_excluded_digest
        ),
        "selected_excluded_overlap": len(overlap),
        "quotas": dict(sorted(quotas.items())),
        "selected_task_counts": selected_counts,
        "selected_rows": len(selected),
        "selected_unique_sample_ids": len(set(selected_ids)),
        "selected_sample_ids_sorted_sha256": sorted_id_digest,
        "output_path": str(output_path),
        "output_bytes": output_bytes,
        "output_sha256": output_digest.hexdigest(),
    }


def _valid_output_audit(
    value: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    quotas: Mapping[str, int],
    excluded_quotas: Mapping[str, int],
    expected_excluded_sample_ids_sorted_sha256: str,
    plan_sha256: str,
) -> bool:
    recorded_source = value.get("source") or {}
    return bool(
        value.get("format_version") == FORMAT_VERSION
        and value.get("complete") is True
        and value.get("taxonomy_version") == TAXONOMY_VERSION
        and value.get("plan_sha256") == plan_sha256
        and recorded_source.get("key") == source["key"]
        and recorded_source.get("version_id") == source["version_id"]
        and recorded_source.get("sha256") == source["sha256"]
        and {str(k): int(v) for k, v in (value.get("quotas") or {}).items()}
        == {str(k): int(v) for k, v in quotas.items()}
        and {str(k): int(v) for k, v in (value.get("excluded_quotas") or {}).items()}
        == {str(k): int(v) for k, v in excluded_quotas.items()}
        and int(value.get("excluded_rows", -1)) == sum(int(v) for v in excluded_quotas.values())
        and (
            not expected_excluded_sample_ids_sorted_sha256
            or value.get("excluded_sample_ids_sorted_sha256") == expected_excluded_sample_ids_sorted_sha256
        )
        and value.get("excluded_digest_matches_pinned_manifest") is True
        and int(value.get("selected_excluded_overlap", -1)) == 0
        and int(value.get("selected_rows", -1)) == sum(int(v) for v in quotas.values())
        and int(value.get("selected_unique_sample_ids", -1)) == int(value.get("selected_rows", -1))
    )


def commit_selected_shard(
    result: Mapping[str, Any],
    *,
    store: OutputStore,
    relative_key: str,
    plan_sha256: str,
    exclusion_plan_sha256: str | None,
) -> dict[str, Any]:
    source = result["source"]
    metadata = {
        "pipeline-version": PIPELINE_VERSION,
        "taxonomy-version": TAXONOMY_VERSION,
        "plan-sha256": plan_sha256,
        "source-version-id": str(source["version_id"]),
        "source-sha256": str(source["sha256"]),
    }
    if exclusion_plan_sha256:
        metadata["exclusion-plan-sha256"] = exclusion_plan_sha256
    output = store.put_file_once(
        relative_key,
        Path(str(result["output_path"])),
        rows=int(result["selected_rows"]),
        metadata=metadata,
    )
    if output["sha256"] != result["output_sha256"] or output["bytes"] != result["output_bytes"]:
        raise AssertionError("committed output identity differs from local selection")
    audit = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "pipeline_version": PIPELINE_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "plan_sha256": plan_sha256,
        "source": source,
        "source_task_counts": result["source_task_counts"],
        "source_status_counts": result["source_status_counts"],
        "excluded_quotas": result["excluded_quotas"],
        "excluded_task_counts": result["excluded_task_counts"],
        "excluded_rows": result["excluded_rows"],
        "excluded_sample_ids_sorted_sha256": result["excluded_sample_ids_sorted_sha256"],
        "excluded_digest_matches_pinned_manifest": result["excluded_digest_matches_pinned_manifest"],
        "selected_excluded_overlap": result["selected_excluded_overlap"],
        "quotas": result["quotas"],
        "selected_task_counts": result["selected_task_counts"],
        "selected_rows": result["selected_rows"],
        "selected_unique_sample_ids": result["selected_unique_sample_ids"],
        "selected_sample_ids_sorted_sha256": result["selected_sample_ids_sorted_sha256"],
        "output": output,
        "source_row_bytes_preserved": True,
    }
    store.put_json_once(output_audit_relative(relative_key), audit)
    return audit


def _wait_for_worker_documents(
    store: OutputStore,
    *,
    relative_for_worker: Any,
    worker_count: int,
    wait_seconds: int,
    poll_seconds: int,
    phase: str,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + wait_seconds
    while True:
        documents: list[dict[str, Any]] = []
        missing: list[int] = []
        for worker_index in range(worker_count):
            value = store.read_json(relative_for_worker(worker_index, worker_count))
            if value is None:
                missing.append(worker_index)
            else:
                documents.append(value)
        if not missing:
            return documents
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {phase} workers: {missing}")
        print(f"[{phase}] waiting for {len(missing)}/{worker_count} workers", flush=True)
        time.sleep(poll_seconds)


def _wait_for_json(
    store: OutputStore,
    relative: str,
    *,
    wait_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    while True:
        value = store.read_json(relative)
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {relative}")
        time.sleep(poll_seconds)


def _taxonomy_contract() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "method": "ordered_regex_first_match",
        "label_origin": "heuristic_instruction_taxonomy_not_source_metadata",
        "mutually_exclusive_primary_task": True,
        "rules": [
            {
                "name": rule.name,
                "description": rule.description,
                "patterns": list(rule.patterns),
            }
            for rule in TASK_RULES
        ],
        "fallback": "other",
    }


def ensure_run_contract(args: argparse.Namespace, store: OutputStore) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    classifier_path = Path(sys.modules[classify_instruction.__module__].__file__).resolve()
    _, script_sha = sha256_file(script_path)
    _, classifier_sha = sha256_file(classifier_path)
    if args.expected_script_sha256 and script_sha != args.expected_script_sha256:
        raise ValueError(f"runtime script SHA mismatch: {script_sha} != {args.expected_script_sha256}")
    if args.expected_taxonomy_source_sha256 and classifier_sha != args.expected_taxonomy_source_sha256:
        raise ValueError(
            f"taxonomy source SHA mismatch: {classifier_sha} != {args.expected_taxonomy_source_sha256}"
        )
    contract = {
        "format_version": f"{PIPELINE_VERSION}_run_contract_v1",
        "pipeline_version": PIPELINE_VERSION,
        "input_manifest": {
            "uri": args.input_manifest_uri,
            "version_id": args.input_manifest_version_id,
            "sha256": args.input_manifest_sha256,
            "rows": EXPECTED_INPUT_ROWS,
            "shards": EXPECTED_INPUT_SHARDS,
        },
        "base_manifest": {
            "uri": args.base_manifest_uri,
            "version_id": args.base_manifest_version_id,
            "sha256": args.base_manifest_sha256,
            "unique_sample_ids": EXPECTED_INPUT_ROWS,
        },
        "output_root": args.output_root,
        "target_rows": args.target_rows,
        "seed": args.seed,
        "allocation_policy": args.allocation_policy,
        "allocation_weight_version": (
            SQRT_WEIGHT_VERSION
            if args.allocation_policy == SQRT_REMAINING_POLICY
            else "remaining_frequency_v1"
        ),
        "exclusion_manifest": (
            {
                "uri": args.exclude_manifest_uri,
                "version_id": args.exclude_manifest_version_id,
                "sha256": args.exclude_manifest_sha256,
            }
            if args.exclude_manifest_uri
            else None
        ),
        "worker_count": args.worker_count,
        "process_count": args.process_count,
        "taxonomy": _taxonomy_contract(),
        "apportionment_version": APPORTIONMENT_VERSION,
        "selection_version": SELECTION_VERSION,
        "output_order_version": OUTPUT_ORDER_VERSION,
        "source_row_bytes_preserved": True,
        "runtime": {
            "script_sha256": script_sha,
            "taxonomy_source_sha256": classifier_sha,
        },
    }
    record = store.put_json_once("run-contract.json", contract)
    return {**contract, "record": record}


def _parallel_map(
    function: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    processes: int,
    region: str,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(processes, len(tasks)),
        initializer=reset_process_client,
        initargs=(region,),
    ) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if completed % 4 == 0 or completed == len(futures):
                print(f"[parallel] {function.__name__} completed={completed}/{len(futures)}", flush=True)
    return results


def _publish_count_worker_summary(
    args: argparse.Namespace,
    store: OutputStore,
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "format_version": f"{PIPELINE_VERSION}_count_worker_v1",
        "complete": True,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "taxonomy_version": TAXONOMY_VERSION,
        "aggregate": aggregate_count_audits(audits),
        "count_audits": [
            {
                "relative_key": audit["source"]["relative_key"],
                "source": audit["source"],
                "task_counts": audit["task_counts"],
                "status_counts": audit["status_counts"],
            }
            for audit in sorted(audits, key=lambda item: item["source"]["relative_key"])
        ],
    }
    store.put_json_once(worker_count_relative(args.worker_index, args.worker_count), value)
    return value


def _load_all_count_audits(worker_documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for expected_index, document in enumerate(worker_documents):
        if (
            document.get("format_version") != f"{PIPELINE_VERSION}_count_worker_v1"
            or document.get("complete") is not True
            or int(document.get("worker_index", -1)) != expected_index
            or int(document.get("worker_count", -1)) != len(worker_documents)
            or document.get("taxonomy_version") != TAXONOMY_VERSION
        ):
            raise ValueError(f"invalid count worker summary {expected_index}")
        for item in document.get("count_audits") or []:
            audits.append(
                {
                    "format_version": f"{PIPELINE_VERSION}_count_shard_v1",
                    "complete": True,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "source": item["source"],
                    "rows": item["source"]["rows"],
                    "task_counts": item["task_counts"],
                    "status_counts": item["status_counts"],
                }
            )
    return audits


def _publish_select_worker_summary(
    args: argparse.Namespace,
    store: OutputStore,
    audits: Sequence[Mapping[str, Any]],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    selected_tasks: Counter[str] = Counter()
    rows = output_bytes = 0
    for audit in audits:
        rows += int(audit["selected_rows"])
        output_bytes += int(audit["output"]["bytes"])
        selected_tasks.update({k: int(v) for k, v in audit["selected_task_counts"].items()})
    value = {
        "format_version": f"{PIPELINE_VERSION}_select_worker_v1",
        "complete": True,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "taxonomy_version": TAXONOMY_VERSION,
        "plan_sha256": plan_sha256,
        "aggregate": {
            "shards": len(audits),
            "rows": rows,
            "output_bytes": output_bytes,
            "task_counts": dict(sorted(selected_tasks.items())),
        },
        "outputs": [
            {
                "relative_key": audit["source"]["relative_key"],
                "source": audit["source"],
                "excluded_quotas": audit["excluded_quotas"],
                "excluded_task_counts": audit["excluded_task_counts"],
                "excluded_rows": audit["excluded_rows"],
                "excluded_sample_ids_sorted_sha256": audit["excluded_sample_ids_sorted_sha256"],
                "excluded_digest_matches_pinned_manifest": audit["excluded_digest_matches_pinned_manifest"],
                "selected_excluded_overlap": audit["selected_excluded_overlap"],
                "quotas": audit["quotas"],
                "selected_task_counts": audit["selected_task_counts"],
                "selected_rows": audit["selected_rows"],
                "selected_unique_sample_ids": audit["selected_unique_sample_ids"],
                "selected_sample_ids_sorted_sha256": audit["selected_sample_ids_sorted_sha256"],
                "output": audit["output"],
                "audit_relative_key": output_audit_relative(audit["source"]["relative_key"]),
            }
            for audit in sorted(audits, key=lambda item: item["source"]["relative_key"])
        ],
    }
    store.put_json_once(worker_select_relative(args.worker_index, args.worker_count), value)
    return value


def _sample_output_schema(client: Any, item: Mapping[str, Any]) -> dict[str, Any]:
    output = item["output"]
    response = client.get_object(
        Bucket=parse_s3_uri(output["uri"]).bucket,
        Key=parse_s3_uri(output["uri"]).key,
        VersionId=output["version_id"],
        Range="bytes=0-1048575",
    )
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        body.close()
    newline = payload.find(b"\n")
    if newline < 0:
        raise ValueError(f"could not read a complete sample row from {output['uri']}")
    row = json.loads(payload[:newline])
    if not isinstance(row, Mapping):
        raise TypeError("sample output row is not a mapping")
    sample_id, category, _ = validate_training_row(row, location=output["uri"])
    return {
        "relative_key": item["relative_key"],
        "sample_id_sha256": sha256_bytes(sample_id.encode("utf-8")),
        "category": category,
        "top_level_fields": sorted(row),
    }


def finalize(
    args: argparse.Namespace,
    store: OutputStore,
    client: Any,
    documents: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for expected_index, document in enumerate(documents):
        if (
            document.get("format_version") != f"{PIPELINE_VERSION}_select_worker_v1"
            or document.get("complete") is not True
            or int(document.get("worker_index", -1)) != expected_index
            or int(document.get("worker_count", -1)) != args.worker_count
            or document.get("plan_sha256") != plan_record["sha256"]
        ):
            raise ValueError(f"invalid select worker summary {expected_index}")
        outputs.extend(document.get("outputs") or [])
    outputs.sort(key=lambda item: item["relative_key"])
    if len(outputs) != EXPECTED_INPUT_SHARDS or len({item["relative_key"] for item in outputs}) != len(
        outputs
    ):
        raise ValueError("output inventory is incomplete or duplicated")

    observed_tasks: Counter[str] = Counter()
    observed_excluded_tasks: Counter[str] = Counter()
    rows = output_bytes = excluded_rows = overlap_rows = 0
    exclusion_plan_sha256 = str((plan.get("exclusion_selection_plan") or {}).get("sha256") or "")
    for item in outputs:
        rows += int(item["selected_rows"])
        output_bytes += int(item["output"]["bytes"])
        excluded_rows += int(item.get("excluded_rows", 0))
        overlap_rows += int(item.get("selected_excluded_overlap", -1))
        observed_tasks.update({k: int(v) for k, v in item["selected_task_counts"].items()})
        observed_excluded_tasks.update(
            {k: int(v) for k, v in (item.get("excluded_task_counts") or {}).items()}
        )
        if item.get("excluded_digest_matches_pinned_manifest") is not True:
            raise ValueError(f"excluded sample-id digest mismatch for {item['relative_key']}")
        head = client.head_object(
            Bucket=parse_s3_uri(item["output"]["uri"]).bucket,
            Key=parse_s3_uri(item["output"]["uri"]).key,
            VersionId=item["output"]["version_id"],
        )
        metadata = head.get("Metadata") or {}
        if (
            int(head.get("ContentLength", -1)) != int(item["output"]["bytes"])
            or metadata.get("sha256") != item["output"]["sha256"]
            or int(metadata.get("rows", -1)) != int(item["selected_rows"])
            or metadata.get("plan-sha256") != plan_record["sha256"]
            or (exclusion_plan_sha256 and metadata.get("exclusion-plan-sha256") != exclusion_plan_sha256)
        ):
            raise ValueError(f"output HEAD verification failed for {item['relative_key']}")
    expected_tasks = {str(k): int(v) for k, v in plan["task_quotas"].items()}
    expected_excluded_tasks = {str(k): int(v) for k, v in (plan.get("excluded_task_counts") or {}).items()}
    if (
        rows != args.target_rows
        or dict(sorted(observed_tasks.items())) != dict(sorted(expected_tasks.items()))
        or excluded_rows != int(plan.get("excluded_rows", 0))
        or dict(sorted(observed_excluded_tasks.items()))
        != {k: v for k, v in sorted(expected_excluded_tasks.items()) if v}
        or overlap_rows != 0
    ):
        raise ValueError("final output count or task distribution differs from the selection plan")

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        schema_samples = list(pool.map(lambda item: _sample_output_schema(client, item), outputs))
    field_signatures = Counter(
        sha256_bytes("\0".join(sample["top_level_fields"]).encode("utf-8")) for sample in schema_samples
    )
    verifier = {
        "format_version": f"{PIPELINE_VERSION}_verifier_v1",
        "input_prefix": str(args.input_manifest_uri).rsplit("/", 1)[0] + "/",
        "output_prefix": args.output_root,
        "pass": True,
        "shard_count": len(outputs),
        "row_count": rows,
        "key_column": "sample_id",
        "schema_summary": {
            "required_fields": sorted(REQUIRED_ROW_FIELDS),
            "sampled_shards": len(schema_samples),
            "top_level_field_signature_counts": dict(sorted(field_signatures.items())),
        },
        "sampled_evidence": schema_samples,
        "allocation_policy": args.allocation_policy,
        "category_counts": dict(sorted(observed_tasks.items())),
        "excluded_category_counts": dict(sorted(observed_excluded_tasks.items())),
        "validation": {
            "all_output_objects_head_verified": True,
            "one_schema_sample_per_shard": True,
            "exact_target_rows": True,
            "exact_weighted_hamilton_task_quotas": True,
            "source_rows_copied_byte_for_byte": True,
            "unique_sample_ids_proof": (
                "pinned base manifest proves 7,236,807 globally unique sample_ids; "
                "each selected row is copied once from exactly one disjoint source shard; "
                "every output shard audit proves local uniqueness"
            ),
            **(
                {
                    "zero_overlap_with_exclusion_manifest": True,
                    "excluded_sample_id_digests_match_pinned_manifest": True,
                    "excluded_rows_reconstructed": excluded_rows,
                    "disjoint_sample_ids_proof": (
                        "every source shard reconstructs the pinned exclusion selection with "
                        "the original seed and quotas; its sorted sample-id SHA256 matches the "
                        "pinned exclusion manifest, and new rows are selected only from the "
                        "remaining candidates"
                    ),
                }
                if exclusion_plan_sha256
                else {}
            ),
        },
    }
    verifier_record = store.put_json_once("verifier_summary.json", verifier)
    manifest = {
        "format_version": f"{PIPELINE_VERSION}_manifest_v1",
        "complete": True,
        "pipeline_version": PIPELINE_VERSION,
        "rows": rows,
        "physical_shard_count": len(outputs),
        "output_bytes": output_bytes,
        "task_counts": dict(sorted(observed_tasks.items())),
        "allocation_policy": args.allocation_policy,
        "excluded_rows": excluded_rows,
        "exclusion_manifest": plan.get("exclusion_manifest"),
        "taxonomy_version": TAXONOMY_VERSION,
        "selection_plan": plan_record,
        "run_contract": run_contract["record"],
        "verifier_summary": verifier_record,
        "outputs": outputs,
        "validation": verifier["validation"],
    }
    manifest_record = store.put_json_once("manifest.json", manifest)
    success = {
        "format_version": f"{PIPELINE_VERSION}_success_v1",
        "complete": True,
        "rows": rows,
        "shards": len(outputs),
        "output_bytes": output_bytes,
        "task_counts": dict(sorted(observed_tasks.items())),
        "allocation_policy": args.allocation_policy,
        "excluded_rows": excluded_rows,
        "exclusion_manifest": plan.get("exclusion_manifest"),
        "taxonomy_version": TAXONOMY_VERSION,
        "manifest": manifest_record,
        "verifier_summary": verifier_record,
        "output_root": args.output_root,
    }
    store.put_json_once("_SUCCESS", success)
    return success


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_pipeline_version(args.pipeline_version)
    if args.worker_count != DEFAULT_WORKER_COUNT:
        raise ValueError(f"this campaign is pinned to {DEFAULT_WORKER_COUNT} workers")
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index is outside worker count")
    client = make_s3_client(profile=args.aws_profile, region=args.aws_region)
    exclusion = load_exclusion_contract(args, client)
    _, shards = load_input_contract(args, client)
    store = OutputStore(args.output_root, client)
    run_contract = ensure_run_contract(args, store)
    assigned = [
        source for index, source in enumerate(shards) if index % args.worker_count == args.worker_index
    ]
    if not assigned:
        raise ValueError("worker has no assigned source shards")

    count_audits: list[dict[str, Any]] = []
    pending_count: list[dict[str, Any]] = []
    for source in assigned:
        existing = store.read_json(count_audit_relative(source["relative_key"]))
        if existing is not None and _valid_count_audit(existing, source):
            count_audits.append(existing)
        elif existing is not None:
            raise ValueError(f"invalid existing count audit for {source['relative_key']}")
        else:
            pending_count.append({"source": source})
    print(
        f"[worker {args.worker_index}] count pending={len(pending_count)} reused={len(count_audits)}",
        flush=True,
    )
    for result in _parallel_map(
        count_source_shard,
        pending_count,
        processes=args.process_count,
        region=args.aws_region,
    ):
        store.put_json_once(count_audit_relative(result["source"]["relative_key"]), result)
        count_audits.append(result)
    count_audits.sort(key=lambda item: item["source"]["relative_key"])
    _publish_count_worker_summary(args, store, count_audits)

    if args.worker_index == 0:
        count_documents = _wait_for_worker_documents(
            store,
            relative_for_worker=worker_count_relative,
            worker_count=args.worker_count,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            phase="count",
        )
        all_count_audits = _load_all_count_audits(count_documents)
        plan = build_selection_plan(
            all_count_audits,
            target_rows=args.target_rows,
            seed=args.seed,
            input_manifest_sha256=args.input_manifest_sha256,
            allocation_policy=args.allocation_policy,
            exclusion_plan=exclusion["plan"] if exclusion else None,
            exclusion_manifest_record=exclusion["manifest_record"] if exclusion else None,
            exclusion_plan_record=exclusion["plan_record"] if exclusion else None,
        )
        plan_record = store.put_json_once("selection-plan.json", plan)
        print(
            f"[plan] tasks={len(plan['task_quotas'])} target_rows={plan['target_rows']}",
            flush=True,
        )
    else:
        plan = _wait_for_json(
            store,
            "selection-plan.json",
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
        )
        plan_record = {
            "uri": f"{args.output_root.rstrip('/')}/selection-plan.json",
            "key": store.key("selection-plan.json"),
            "sha256": sha256_bytes(canonical_json_bytes(plan, newline=True)),
        }
    if (
        plan.get("format_version") != f"{PIPELINE_VERSION}_selection_plan_v1"
        or plan.get("complete") is not True
        or int(plan.get("target_rows", -1)) != args.target_rows
        or plan.get("seed") != args.seed
        or plan.get("allocation_policy") != args.allocation_policy
        or (bool(args.exclude_manifest_uri) != bool((plan.get("exclusion_manifest") or {}).get("sha256")))
        or (
            args.exclude_manifest_uri
            and (plan.get("exclusion_manifest") or {}).get("sha256") != args.exclude_manifest_sha256
        )
    ):
        raise ValueError("selection plan contract mismatch")

    count_by_relative = {audit["source"]["relative_key"]: audit for audit in count_audits}
    selection_audits: list[dict[str, Any]] = []
    pending_selection: list[dict[str, Any]] = []
    staging = Path(args.staging_dir).resolve() / f"worker-{args.worker_index:03d}"
    staging.mkdir(parents=True, exist_ok=True)
    for ordinal, source in enumerate(assigned):
        count_audit = count_by_relative[source["relative_key"]]
        pinned_source = count_audit["source"]
        quotas = plan["shard_quotas"][source["relative_key"]]
        excluded_quotas = plan["excluded_shard_quotas"][source["relative_key"]]
        exclusion_output = exclusion["outputs_by_relative"][source["relative_key"]] if exclusion else None
        expected_excluded_digest = str(
            (exclusion_output or {}).get("selected_sample_ids_sorted_sha256") or ""
        )
        existing = store.read_json(output_audit_relative(source["relative_key"]))
        if existing is not None and _valid_output_audit(
            existing,
            source=pinned_source,
            quotas=quotas,
            excluded_quotas=excluded_quotas,
            expected_excluded_sample_ids_sorted_sha256=expected_excluded_digest,
            plan_sha256=plan_record["sha256"],
        ):
            selection_audits.append(existing)
            continue
        if existing is not None:
            raise ValueError(f"invalid existing output audit for {source['relative_key']}")
        pending_selection.append(
            {
                "source": pinned_source,
                "expected_task_counts": count_audit["task_counts"],
                "quotas": quotas,
                "excluded_quotas": excluded_quotas,
                "exclusion_seed": str(plan.get("exclusion_seed") or ""),
                "expected_excluded_sample_ids_sorted_sha256": expected_excluded_digest,
                "seed": args.seed,
                "output_path": str(staging / f"selected-{ordinal:03d}.jsonl"),
            }
        )
    print(
        f"[worker {args.worker_index}] select pending={len(pending_selection)} "
        f"reused={len(selection_audits)}",
        flush=True,
    )
    for result in _parallel_map(
        select_source_shard,
        pending_selection,
        processes=args.process_count,
        region=args.aws_region,
    ):
        audit = commit_selected_shard(
            result,
            store=store,
            relative_key=result["source"]["relative_key"],
            plan_sha256=plan_record["sha256"],
            exclusion_plan_sha256=str((plan.get("exclusion_selection_plan") or {}).get("sha256") or "")
            or None,
        )
        selection_audits.append(audit)
        Path(str(result["output_path"])).unlink(missing_ok=True)
    selection_audits.sort(key=lambda item: item["source"]["relative_key"])
    summary = _publish_select_worker_summary(
        args,
        store,
        selection_audits,
        plan_sha256=plan_record["sha256"],
    )
    shutil.rmtree(staging, ignore_errors=True)
    print(
        f"[worker {args.worker_index}] selected_rows={summary['aggregate']['rows']} "
        f"shards={summary['aggregate']['shards']}",
        flush=True,
    )

    if args.worker_index == 0:
        select_documents = _wait_for_worker_documents(
            store,
            relative_for_worker=worker_select_relative,
            worker_count=args.worker_count,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            phase="select",
        )
        success = finalize(
            args,
            store,
            client,
            select_documents,
            plan=plan,
            plan_record=plan_record,
            run_contract=run_contract,
        )
        print(json.dumps(success, ensure_ascii=False, sort_keys=True), flush=True)
        return success
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest-uri", default=DEFAULT_INPUT_MANIFEST_URI)
    parser.add_argument("--input-manifest-sha256", default=DEFAULT_INPUT_MANIFEST_SHA256)
    parser.add_argument("--input-manifest-version-id", default=DEFAULT_INPUT_MANIFEST_VERSION_ID)
    parser.add_argument("--base-manifest-uri", default=DEFAULT_BASE_MANIFEST_URI)
    parser.add_argument("--base-manifest-sha256", default=DEFAULT_BASE_MANIFEST_SHA256)
    parser.add_argument("--base-manifest-version-id", default=DEFAULT_BASE_MANIFEST_VERSION_ID)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    parser.add_argument(
        "--allocation-policy",
        choices=ALLOCATION_POLICIES,
        default=PROPORTIONAL_POLICY,
    )
    parser.add_argument("--exclude-manifest-uri")
    parser.add_argument("--exclude-manifest-sha256")
    parser.add_argument("--exclude-manifest-version-id")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--process-count", type=int, default=DEFAULT_PROCESS_COUNT)
    parser.add_argument("--staging-dir", default="/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/task-stratified-2m")
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--expected-script-sha256")
    parser.add_argument("--expected-taxonomy-source-sha256")
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    for field in ("target_rows", "worker_count", "process_count", "wait_seconds", "poll_seconds"):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    for field in (
        "input_manifest_sha256",
        "base_manifest_sha256",
        "exclude_manifest_sha256",
        "expected_script_sha256",
        "expected_taxonomy_source_sha256",
    ):
        value = getattr(args, field)
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error(f"--{field.replace('_', '-')} must be a lowercase SHA-256")
    exclusion_fields = (
        args.exclude_manifest_uri,
        args.exclude_manifest_sha256,
        args.exclude_manifest_version_id,
    )
    if any(exclusion_fields) and not all(exclusion_fields):
        parser.error("all --exclude-manifest-* arguments must be provided together")
    if args.allocation_policy == SQRT_REMAINING_POLICY and not args.exclude_manifest_uri:
        parser.error("--allocation-policy sqrt_remaining requires a pinned exclusion manifest")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,127}", args.pipeline_version):
        parser.error("--pipeline-version must contain only lowercase letters, numbers, and underscores")
    input_root = str(args.input_manifest_uri).rsplit("/", 1)[0].rstrip("/") + "/"
    output_root = args.output_root.rstrip("/") + "/"
    if input_root == output_root or input_root.startswith(output_root) or output_root.startswith(input_root):
        parser.error("input and output roots must be disjoint")
    if args.exclude_manifest_uri:
        exclusion_root = str(args.exclude_manifest_uri).rsplit("/", 1)[0].rstrip("/") + "/"
        if (
            exclusion_root == output_root
            or exclusion_root.startswith(output_root)
            or output_root.startswith(exclusion_root)
        ):
            parser.error("exclusion and output roots must be disjoint")
    args.output_root = output_root
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
