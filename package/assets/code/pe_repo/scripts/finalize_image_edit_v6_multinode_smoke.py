#!/usr/bin/env python3
"""Commit one index and deterministically finalize a multi-Pod V6 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from dlms.datasets.image_edit_v6_bbox_packaging import (
    PACKAGED_BBOX_COORDINATE_ORDER,
    PACKAGING_VERSION,
    SOURCE_BBOX_COORDINATE_ORDER,
    package_v6_inference_row_yxyx,
)
from scripts.partition_image_edit_v6_smoke_manifest import FORMAT_VERSION


NODE_FORMAT = "image_edit_v6_multinode_smoke_node_v1"
RESULT_FORMAT = "image_edit_v6_multinode_smoke_results_v1"
SUCCESS_FORMAT = "image_edit_v6_multinode_smoke_commit_v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, key


def join_s3(root: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{'/'.join(value for value in (key, suffix) if value)}"


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
    )


def _missing_s3_error(error: BaseException) -> bool:
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def read_bytes(client: Any, uri: str, *, missing_ok: bool = False) -> bytes | None:
    bucket, key = parse_s3_uri(uri)
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except BaseException as exc:
        if missing_ok and _missing_s3_error(exc):
            return None
        raise


def read_json(client: Any, uri: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
    payload = read_bytes(client, uri, missing_ok=missing_ok)
    if payload is None:
        return None
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return value


def put_identical(client: Any, uri: str, payload: bytes, content_type: str) -> str:
    digest = sha256_bytes(payload)
    existing = read_bytes(client, uri, missing_ok=True)
    if existing is not None:
        if existing != payload:
            raise ValueError(f"refusing to replace non-identical object: {uri}")
        return digest
    bucket, key = parse_s3_uri(uri)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    observed = read_bytes(client, uri)
    if observed != payload:
        raise IOError(f"uploaded object verification failed: {uri}")
    return digest


def load_index(client: Any, uri: str, expected_sha256: str) -> dict[str, Any]:
    payload = read_bytes(client, uri)
    assert payload is not None
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"partition index SHA-256 mismatch: actual={actual_sha256} expected={expected_sha256}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected partition index format")
    if value.get("complete") is not True:
        raise ValueError("partition index is incomplete")
    return value


def node_name(index: int, count: int) -> str:
    return f"node-{index:03d}-of-{count:03d}"


def node_root(output_s3: str, index: int, count: int) -> str:
    return join_s3(output_s3, "nodes", node_name(index, count))


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: actual={actual!r} expected={expected!r}")


def mark_node(
    client: Any,
    *,
    output_s3: str,
    index: Mapping[str, Any],
    node_index: int,
) -> dict[str, Any]:
    node_count = int(index["part_count"])
    if node_index < 0 or node_index >= node_count:
        raise ValueError("node_index is outside the partition range")
    part = index["parts"][node_index]
    root = node_root(output_s3, node_index, node_count)
    summary_uri = join_s3(root, "summary.json")
    predictions_uri = join_s3(root, "predictions.jsonl")
    summary = read_json(client, summary_uri)
    if summary is None:
        raise FileNotFoundError(summary_uri)
    require_equal(summary.get("status"), "complete", "node summary status")
    require_equal(summary.get("output_s3"), root + "/", "node summary output root")
    require_equal(summary.get("input_manifest", {}).get("sha256"), part["sha256"], "part SHA")
    require_equal(summary.get("input_manifest", {}).get("rows"), part["rows"], "part rows")
    require_equal(summary.get("results", {}).get("rows"), part["rows"], "result rows")
    predictions = read_bytes(client, predictions_uri)
    assert predictions is not None
    result_digest = sha256_bytes(predictions)
    require_equal(
        summary.get("results", {}).get("predictions_sha256"),
        result_digest,
        "predictions SHA-256",
    )
    marker = {
        "format_version": NODE_FORMAT,
        "status": "complete",
        "completed_at": summary["completed_at"],
        "node_index": node_index,
        "node_count": node_count,
        "part": dict(part),
        "checkpoint_step": summary["checkpoint_step"],
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
        "target_schema": summary["target_schema"],
        "prompt_version": summary["prompt_version"],
        "prompt_contract_sha256": summary["prompt_contract_sha256"],
        "generation": summary["generation"],
        "results": dict(summary["results"]),
        "summary_uri": summary_uri,
        "predictions_uri": predictions_uri,
        "predictions_bytes": len(predictions),
        "predictions_sha256": result_digest,
    }
    put_identical(
        client,
        join_s3(root, "_SUCCESS"),
        canonical_json_bytes(marker),
        "application/json",
    )
    return marker


def load_node_markers(
    client: Any,
    *,
    output_s3: str,
    index: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    node_count = int(index["part_count"])
    markers: list[dict[str, Any]] = []
    for node_index in range(node_count):
        marker = read_json(
            client,
            join_s3(node_root(output_s3, node_index, node_count), "_SUCCESS"),
            missing_ok=True,
        )
        if marker is None:
            return None
        require_equal(marker.get("format_version"), NODE_FORMAT, "node marker format")
        require_equal(marker.get("node_index"), node_index, "node marker index")
        require_equal(marker.get("node_count"), node_count, "node marker count")
        require_equal(marker.get("part"), index["parts"][node_index], "node marker part")
        markers.append(marker)
    return markers


def finalize(
    client: Any,
    *,
    output_s3: str,
    index_uri: str,
    index_sha256: str,
    index: Mapping[str, Any],
    markers: Sequence[Mapping[str, Any]],
    expected_hld_postprocess_version: str | None = None,
) -> dict[str, Any]:
    contract_fields = (
        "checkpoint_step",
        "checkpoint_manifest_sha256",
        "target_schema",
        "prompt_version",
        "prompt_contract_sha256",
        "generation",
    )
    reference = markers[0]
    for marker in markers[1:]:
        for field in contract_fields:
            require_equal(marker.get(field), reference.get(field), f"node {field}")

    expected_rows = int(index["input"]["rows"])
    rows_by_global_index: dict[int, dict[str, Any]] = {}
    node_summaries: list[dict[str, Any]] = []
    for marker in markers:
        payload = read_bytes(client, str(marker["predictions_uri"]))
        assert payload is not None
        require_equal(sha256_bytes(payload), marker["predictions_sha256"], "node predictions digest")
        rows = 0
        for physical_line, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"node prediction line {physical_line} is not an object")
            source = value.get("dataset_source")
            if not isinstance(source, Mapping):
                raise ValueError("prediction is missing smoke dataset_source provenance")
            global_index = int(source["smoke_global_input_index"])
            require_equal(
                int(source["smoke_partition_index"]),
                int(marker["node_index"]),
                "prediction partition index",
            )
            require_equal(
                str(source["smoke_full_manifest_sha256"]),
                str(index["input"]["sha256"]),
                "prediction full-manifest digest",
            )
            if global_index in rows_by_global_index:
                raise ValueError(f"duplicate global input index: {global_index}")
            value["input_index"] = global_index
            rows_by_global_index[global_index] = value
            rows += 1
        require_equal(rows, marker["part"]["rows"], "node prediction rows")
        node_summaries.append(
            {
                "node_index": marker["node_index"],
                "success_uri": join_s3(
                    node_root(output_s3, int(marker["node_index"]), int(marker["node_count"])),
                    "_SUCCESS",
                ),
                "rows": rows,
                "predictions_sha256": marker["predictions_sha256"],
            }
        )
    require_equal(sorted(rows_by_global_index), list(range(expected_rows)), "global row coverage")
    ordered = [rows_by_global_index[index] for index in range(expected_rows)]
    sample_ids = [str(row["sample_id"]) for row in ordered]
    require_equal(len(set(sample_ids)), expected_rows, "unique sample IDs")
    predictions_payload = b"".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in ordered
    )
    predictions_uri = join_s3(output_s3, "predictions.jsonl")
    predictions_sha256 = put_identical(
        client, predictions_uri, predictions_payload, "application/x-ndjson"
    )

    packaged_result: dict[str, Any] | None = None
    hld_postprocess: dict[str, Any] | None = None
    if expected_hld_postprocess_version is not None:
        packaged_rows: list[dict[str, Any]] = []
        packaging_audits: list[dict[str, Any]] = []
        cleanup_audits: list[Mapping[str, Any]] = []
        for row in ordered:
            packaged_row, packaging_audit = package_v6_inference_row_yxyx(
                row,
                expected_hld_postprocess_version=(
                    expected_hld_postprocess_version
                ),
            )
            packaged_rows.append(packaged_row)
            packaging_audits.append(packaging_audit)
            cleanup = row.get("hld_label_cleanup")
            if isinstance(cleanup, Mapping):
                cleanup_audits.append(cleanup)
        packaged_payload = b"".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for row in packaged_rows
        )
        packaged_uri = join_s3(
            output_s3, "predictions.packaged-yxyx.jsonl"
        )
        packaged_sha256 = put_identical(
            client,
            packaged_uri,
            packaged_payload,
            "application/x-ndjson",
        )
        packaged_result = {
            "version": PACKAGING_VERSION,
            "source_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
            "bbox_coordinate_order": PACKAGED_BBOX_COORDINATE_ORDER,
            "rows": len(packaged_rows),
            "packaged_rows": sum(
                audit.get("status") == "packaged" for audit in packaging_audits
            ),
            "skipped_invalid_rows": sum(
                audit.get("status") == "skipped_invalid_prediction"
                for audit in packaging_audits
            ),
            "bboxes_converted": sum(
                int(audit.get("total_bboxes_converted", 0))
                for audit in packaging_audits
            ),
            "predictions_uri": packaged_uri,
            "predictions_bytes": len(packaged_payload),
            "predictions_sha256": packaged_sha256,
        }
        hld_postprocess = {
            "version": expected_hld_postprocess_version,
            "audited_schema_valid_rows": len(cleanup_audits),
            "changed_rows": sum(
                cleanup.get("changed") is True for cleanup in cleanup_audits
            ),
            "hld_changed_rows": sum(
                cleanup.get("hld_changed") is True for cleanup in cleanup_audits
            ),
            "layout_changed_rows": sum(
                cleanup.get("layout_changed") is True for cleanup in cleanup_audits
            ),
            "invalid_label_occurrences_removed": sum(
                int(cleanup.get("invalid_label_occurrences_removed", 0))
                for cleanup in cleanup_audits
            ),
            "small_bbox_elements_removed": sum(
                int(
                    cleanup.get("bbox_cleanup", {}).get(
                        "small_bbox_elements_removed", 0
                    )
                )
                for cleanup in cleanup_audits
            ),
            "remaining_invalid_numbered_label_occurrences": 0,
        }
    completed_at = max(str(marker["completed_at"]) for marker in markers)
    summary = {
        "format_version": RESULT_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "input_partition_index_uri": index_uri,
        "input_partition_index_sha256": index_sha256,
        "input_manifest": dict(index["input"]),
        "checkpoint_step": reference["checkpoint_step"],
        "checkpoint_manifest_sha256": reference["checkpoint_manifest_sha256"],
        "target_schema": reference["target_schema"],
        "prompt_version": reference["prompt_version"],
        "prompt_contract_sha256": reference["prompt_contract_sha256"],
        "model_bbox_coordinate_order": SOURCE_BBOX_COORDINATE_ORDER,
        "generation": reference["generation"],
        "nodes": node_summaries,
        "results": {
            "rows": expected_rows,
            "json_valid": sum(bool(row.get("json_valid")) for row in ordered),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in ordered),
            "inference_errors": sum(bool(row.get("inference_error")) for row in ordered),
            "unique_sample_ids": len(set(sample_ids)),
            "predictions_uri": predictions_uri,
            "predictions_bytes": len(predictions_payload),
            "predictions_sha256": predictions_sha256,
        },
    }
    if hld_postprocess is not None:
        summary["hld_postprocess"] = hld_postprocess
    if packaged_result is not None:
        summary["packaged_yxyx"] = packaged_result
    summary_payload = canonical_json_bytes(summary)
    summary_uri = join_s3(output_s3, "summary.json")
    summary_sha256 = put_identical(client, summary_uri, summary_payload, "application/json")
    success = {
        "format_version": SUCCESS_FORMAT,
        "status": "complete",
        "completed_at": completed_at,
        "summary_uri": summary_uri,
        "summary_sha256": summary_sha256,
        "predictions_uri": predictions_uri,
        "predictions_sha256": predictions_sha256,
        "rows": expected_rows,
        "checkpoint_step": reference["checkpoint_step"],
        "checkpoint_manifest_sha256": reference["checkpoint_manifest_sha256"],
        "prompt_contract_sha256": reference["prompt_contract_sha256"],
    }
    if hld_postprocess is not None:
        success["hld_postprocess_version"] = hld_postprocess["version"]
    if packaged_result is not None:
        success["packaged_bbox_coordinate_order"] = packaged_result[
            "bbox_coordinate_order"
        ]
        success["packaged_predictions_uri"] = packaged_result[
            "predictions_uri"
        ]
        success["packaged_predictions_sha256"] = packaged_result[
            "predictions_sha256"
        ]
    put_identical(client, join_s3(output_s3, "_SUCCESS"), canonical_json_bytes(success), "application/json")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = s3_client()
    index = load_index(client, args.partition_index_uri, args.partition_index_sha256)
    if int(index["part_count"]) != args.node_count:
        raise ValueError("partition index node count changed")
    marker = mark_node(
        client,
        output_s3=args.output_s3,
        index=index,
        node_index=args.node_index,
    )
    deadline = time.monotonic() + args.wait_seconds
    while True:
        markers = load_node_markers(client, output_s3=args.output_s3, index=index)
        if markers is not None:
            return finalize(
                client,
                output_s3=args.output_s3,
                index_uri=args.partition_index_uri,
                index_sha256=args.partition_index_sha256,
                index=index,
                markers=markers,
                expected_hld_postprocess_version=(
                    args.expected_hld_postprocess_version
                ),
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for all node _SUCCESS markers")
        print(
            json.dumps(
                {
                    "event": "waiting_for_nodes",
                    "node_index": args.node_index,
                    "node_marker": marker["completed_at"],
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(args.poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--partition-index-uri", required=True)
    parser.add_argument("--partition-index-sha256", required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--expected-hld-postprocess-version")
    args = parser.parse_args(argv)
    parse_s3_uri(args.output_s3)
    parse_s3_uri(args.partition_index_uri)
    if len(args.partition_index_sha256) != 64:
        parser.error("--partition-index-sha256 must be a SHA-256 digest")
    if args.node_count <= 0 or args.wait_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("node_count, wait_seconds, and poll_seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
