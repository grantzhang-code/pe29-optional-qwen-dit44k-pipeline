#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
#     "pyyaml",
# ]
# ///
"""Fail-closed identity preflight for an image-edit-v6 training manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path

import yaml
from botocore.exceptions import BotoCoreError, ClientError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["DLMS_IMAGE_EDIT_V6_LIBRARY_ONLY"] = "true"

from dlms.datasets.craft_mixed4 import _s3_client, _split_s3_uri  # noqa: E402
from dlms.datasets.image_edit_v6_full_json import (  # noqa: E402
    _manifest_documents,
    load_runtime_config,
)


def _verify_object_identity(
    record: dict, uri_field: str, require_sha256_metadata: bool
) -> None:
    uri = str(record[uri_field])
    bucket, key = _split_s3_uri(uri)
    version_id = str(record["version_id"])
    head = _s3_client().head_object(Bucket=bucket, Key=key, VersionId=version_id)
    if int(head["ContentLength"]) != int(record["bytes"]):
        raise ValueError(f"{uri}: pinned object byte count changed")
    if str(head.get("VersionId") or "") != version_id:
        raise ValueError(f"{uri}: pinned S3 VersionId changed")
    if require_sha256_metadata and str(
        (head.get("Metadata") or {}).get("sha256") or ""
    ) != str(record["sha256"]):
        raise ValueError(f"{uri}: S3 sha256 metadata does not match the pinned digest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    document = yaml.safe_load(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("training config must be a YAML mapping")
    data_config = document.get("image_edit_v6_full_json_config")
    if not isinstance(data_config, dict):
        raise ValueError("image_edit_v6_full_json_config must be a mapping")
    env_vars = document.get("env_vars")
    if not isinstance(env_vars, dict):
        raise ValueError("env_vars must be a mapping")
    max_length = int(env_vars.get("MAX_LENGTH") or 0)
    packing_length = int(env_vars.get("PACKING_LENGTH") or 0)
    audited_max_length = int(
        data_config.get("sequence_length_audit_max_length") or 0
    )
    if max_length < 1 or packing_length < max_length:
        raise ValueError("MAX_LENGTH/PACKING_LENGTH are invalid")
    external_review_policy = str(data_config.get("external_review_policy") or "")
    if audited_max_length and audited_max_length != max_length:
        raise ValueError(
            "sequence_length_audit_max_length must equal training MAX_LENGTH"
        )
    if not audited_max_length and external_review_policy != "externally_reviewed_no_reaudit_requested":
        raise ValueError(
            "a sequence audit or the explicit externally reviewed policy is required"
        )
    manifest_sha256 = str(data_config.get("manifest_sha256") or "")
    try:
        manifest_sha_bytes = bytes.fromhex(manifest_sha256)
    except ValueError as exc:
        raise ValueError(
            "manifest_sha256 is pending; pin it after Call-2 publication"
        ) from exc
    if len(manifest_sha_bytes) != 32 or manifest_sha256 != manifest_sha256.lower():
        raise ValueError("manifest_sha256 must be a lowercase SHA-256")
    manifest_version_id = str(data_config.get("manifest_version_id") or "")
    if not manifest_version_id or manifest_version_id == "PENDING_AFTER_CALL2_PUBLICATION":
        raise ValueError("manifest_version_id is pending; pin it after Call-2 publication")
    required_total_rows = int(data_config.get("required_total_rows") or 0)
    required_status_counts = data_config.get("required_status_counts")
    if (
        required_total_rows < 1
        or not isinstance(required_status_counts, dict)
        or not required_status_counts
        or sum(int(value) for value in required_status_counts.values())
        != required_total_rows
    ):
        raise ValueError(
            "required_total_rows/status_counts are pending; pin them after Call-2 publication"
        )
    config = load_runtime_config(document, resolve_paths=True)
    manifests = _manifest_documents(data_config)
    object_records: list[tuple[dict, str, bool]] = []
    for _uri, manifest, _digest in manifests:
        contract = manifest.get("dataset_contract") or {}
        source_object = contract.get("source_object")
        if isinstance(source_object, dict):
            object_records.append((dict(source_object), "uri", True))
        source_objects = contract.get("source_objects")
        if isinstance(source_objects, list):
            object_records.extend(
                (dict(record), "uri", True)
                for record in source_objects
                if isinstance(record, dict)
            )
        object_records.extend(
            (dict(record), "uri", True)
            for record in (manifest.get("source_partitions") or [])
        )
        object_records.extend(
            (dict(record), "path", True) for record in (manifest.get("shards") or [])
        )
    if len(config.paths) < config.minimum_physical_shards or not object_records:
        raise ValueError("manifest does not contain enough immutable shard identities")
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(lambda item: _verify_object_identity(*item), object_records))
    print(
        json.dumps(
            {
                "dataset": config.dataset_name,
                "physical_shards": len(config.paths),
                "required_source_partitions": config.required_source_partitions,
                "max_sequence_length": max_length,
                "sequence_review": (
                    "exact_token_audit" if audited_max_length else external_review_policy
                ),
                "versioned_objects_verified": len(object_records),
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BotoCoreError, ClientError, OSError, ValueError) as exc:
        print(f"ERROR: image-edit-v6 manifest preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
