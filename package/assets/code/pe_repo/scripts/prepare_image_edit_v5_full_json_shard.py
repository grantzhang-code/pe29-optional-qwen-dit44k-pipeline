#!/usr/bin/env python3
"""Convert one consolidated image-edit-v5 shard to Qwen prepared JSONL.

The consolidated V5 rows contain labels but intentionally omit the global edit
instruction and media paths.  This converter recovers those fields from the
immutable source Parquet range for the corresponding V4 manifest shard and
joins by exact ``sample_id``.  It also requires ``dataset_row_index`` to match
the source global row, so a shifted or accidental positional join fails.

The output remains prompt-free.  ``dlms/datasets/image_edit_v5_full_json.py``
injects the pinned system/user prompts and ordered image slots at training time.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlms.datasets.image_edit_v5_full_json import (  # noqa: E402
    PROMPT_CONTRACT_SHA256,
    PROMPT_VERSION,
    V5TargetValidationError,
    canonical_training_target,
    contains_operation_key,
    prompt_contract_sha256,
)
from scripts.prepare_image_edit_full_json_shard import (  # noqa: E402
    DEFAULT_SOURCE_DATASET,
    S3Location,
    _aws_session,
    _json_bytes,
    _read_s3_bytes,
    _read_s3_json,
    _shard_summaries,
    load_or_build_inventory,
    output_part_paths,
    parse_s3_uri,
    recover_source_metadata,
    verify_images,
)


DEFAULT_V5_ROOT = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v5/"
    "full16-single-stage-v3-20260807-v1-consolidated-retry1b"
)
DEFAULT_DATASET_MANIFEST = f"{DEFAULT_V5_ROOT}/dataset_manifest.json"
DEFAULT_V4_GLOBAL_SUMMARY = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v4/full/global_summary.json"
)
DEFAULT_SOURCE_INVENTORY = (
    "/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/"
    "craft_v7_full_shard_00_inline_edit_20260804/source_dataset_inventory.json"
)
EXPECTED_DATASET_FORMAT = "image_edit_v5_consolidated_retry_replacement_v1"
FORMAT_VERSION = "image_edit_v5_full_json_qwen_shard_v1"
PREPARED_MANIFEST_FORMAT = "image_edit_v5_full_json_prepared_manifest_v1"


def _progress(message: str) -> None:
    print(f"[prepare-image-edit-v5] {message}", file=sys.stderr, flush=True)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _open_atomic(path: Path, *, overwrite: bool) -> tuple[BinaryIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    return os.fdopen(descriptor, "wb"), Path(temporary_name)


def _read_document(uri: str, *, s3_client: Any) -> tuple[dict[str, Any], bytes]:
    if uri.startswith("s3://"):
        raw = _read_s3_bytes(s3_client, parse_s3_uri(uri))
    else:
        raw = Path(uri).expanduser().resolve(strict=True).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{uri}: expected a JSON object")
    return value, raw


def _manifest_shards(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("dataset_format") != EXPECTED_DATASET_FORMAT:
        raise ValueError(
            f"unsupported V5 dataset format: {document.get('dataset_format')!r}"
        )
    raw = document.get("output_files")
    if not isinstance(raw, list) or not raw:
        raise ValueError("V5 dataset manifest has no output_files")
    shards = [dict(item) for item in raw]
    for index, item in enumerate(shards):
        expected_suffix = f"/final_results/shard_{index:02d}.jsonl"
        if not str(item.get("s3_uri") or "").endswith(expected_suffix):
            raise ValueError(f"V5 manifest shard {index} has unexpected URI")
        if int(item.get("sample_count", 0)) <= 0:
            raise ValueError(f"V5 manifest shard {index} has no rows")
        if len(str(item.get("sha256") or "")) != 64:
            raise ValueError(f"V5 manifest shard {index} has no SHA-256")
    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("V5 dataset manifest has no summary")
    if sum(int(item["sample_count"]) for item in shards) != int(
        summary.get("dataset_sample_count", -1)
    ):
        raise ValueError("V5 shard counts do not match dataset_sample_count")
    return shards


def _source_range(
    shard: int,
    *,
    summary_dir: Path | None,
    global_summary_uri: str,
    s3_client: Any,
) -> tuple[int, int, str]:
    if summary_dir is not None:
        path = summary_dir / f"image_edit_v4.shard-{shard:02d}.prepared.jsonl.summary.json"
        document = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        if int(document.get("selected_shard", -1)) != shard:
            raise ValueError(f"{path}: selected_shard mismatch")
        start = int(document["global_row_start"])
        end = int(document["global_row_end_exclusive"])
        return start, end, str(path.resolve())
    summary = _read_s3_json(s3_client, parse_s3_uri(global_summary_uri))
    shards = _shard_summaries(summary)
    if shard < 0 or shard >= len(shards):
        raise ValueError(f"shard must be between 0 and {len(shards) - 1}")
    start = sum(int(item["expected_rows"]) for item in shards[:shard])
    end = start + int(shards[shard]["expected_rows"])
    return start, end, global_summary_uri


def _load_excluded_ids(paths: Sequence[Path]) -> tuple[set[str], dict[str, Any]]:
    sample_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        payload = resolved.read_bytes()
        ids = [line.strip() for line in payload.decode("utf-8").splitlines() if line.strip()]
        duplicate_count = len(ids) - len(set(ids))
        if duplicate_count:
            raise ValueError(f"{resolved}: contains {duplicate_count} duplicate IDs")
        sample_ids.update(ids)
        source_files.append(
            {
                "path": str(resolved),
                "rows": len(ids),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    ordered = sorted(sample_ids)
    return sample_ids, {
        "policy": "exclude_exact_sample_id_v1" if paths else "none",
        "unique_sample_ids": len(ordered),
        "sample_ids_sha256": hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest(),
        "sources": source_files,
    }


class VerifiedS3Jsonl:
    """Stream an immutable S3 object while preserving its byte-level digest."""

    def __init__(self, client: Any, identity: Mapping[str, Any]) -> None:
        self.client = client
        self.identity = identity
        self.bytes_read = 0
        self.rows_read = 0
        self.sha256 = ""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        location = parse_s3_uri(str(self.identity["s3_uri"]))
        version_id = str(self.identity.get("version_id") or "")
        head_kwargs: dict[str, Any] = {"Bucket": location.bucket, "Key": location.key}
        if version_id:
            head_kwargs["VersionId"] = version_id
        head = self.client.head_object(**head_kwargs)
        if int(head["ContentLength"]) != int(self.identity["bytes"]):
            raise ValueError(f"source V5 object size changed: {location.uri}")
        if version_id and str(head.get("VersionId") or "") != version_id:
            raise ValueError(f"source V5 object version changed: {location.uri}")
        response = self.client.get_object(**head_kwargs)
        body = response["Body"]
        digest = hashlib.sha256()
        pending = b""
        line_number = 0
        try:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                self.bytes_read += len(chunk)
                digest.update(chunk)
                pending += chunk
                pieces = pending.split(b"\n")
                pending = pieces.pop()
                for raw in pieces:
                    line_number += 1
                    if not raw.strip():
                        continue
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError(f"{location.uri}:{line_number}: expected an object")
                    self.rows_read += 1
                    yield value
            if pending.strip():
                line_number += 1
                value = json.loads(pending)
                if not isinstance(value, dict):
                    raise ValueError(f"{location.uri}:{line_number}: expected an object")
                self.rows_read += 1
                yield value
        finally:
            body.close()
        self.sha256 = digest.hexdigest()
        if self.bytes_read != int(self.identity["bytes"]):
            raise ValueError(f"short V5 object read: {self.bytes_read} bytes")
        if self.sha256 != str(self.identity["sha256"]):
            raise ValueError(
                f"V5 object SHA-256 mismatch: actual={self.sha256}, "
                f"expected={self.identity['sha256']}"
            )
        if self.rows_read != int(self.identity["sample_count"]):
            raise ValueError(
                f"V5 object row count mismatch: actual={self.rows_read}, "
                f"expected={self.identity['sample_count']}"
            )


def build_prepared_record(
    metadata: Mapping[str, Any],
    *,
    status: str,
    target_json: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_training_target(target_json, status=status)
    references = metadata.get("reference_images")
    target_image = metadata.get("target_image")
    if (
        not isinstance(references, list)
        or len(references) != 1
        or not isinstance(references[0], Mapping)
        or references[0].get("reference_id") != "frame_0"
        or not isinstance(references[0].get("s3_uri"), str)
        or not isinstance(target_image, Mapping)
        or not isinstance(target_image.get("s3_uri"), str)
    ):
        raise ValueError("invalid ordered media binding; expected frame_0 reference then target")
    reference_uri = references[0]["s3_uri"].strip()
    target_uri = target_image["s3_uri"].strip()
    parse_s3_uri(reference_uri)
    parse_s3_uri(target_uri)
    if reference_uri == target_uri:
        raise ValueError("reference and target image paths must differ")
    return {
        "sample_id": metadata["sample_id"],
        "status": status,
        "instruction": metadata["instruction"],
        "reference_images": [{"reference_id": "frame_0", "s3_uri": reference_uri}],
        "target_image": {"s3_uri": target_uri},
        "target_json": target_json,
    }


def _join_uri(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def _dataset_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_format": "prepared",
        "row_format_version": FORMAT_VERSION,
        "runtime_loader": "dlms/datasets/image_edit_v5_full_json.py",
        "target_schema": "image_edit_v5_instruction",
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "image_order": ["reference frame_0", "target"],
        "selection": summary["selection"],
        "allowed_statuses": list(summary["allowed_statuses"]),
        "target_validation_by_status": summary["target_validation_by_status"],
        "forbidden_target_keys_recursive": ["operation"],
        "source_v5_dataset_manifest": summary["source_v5_dataset_manifest"],
        "source_v5_dataset_manifest_sha256": summary[
            "source_v5_dataset_manifest_sha256"
        ],
        "source_dataset": summary["source_dataset"],
        "expected_annotation_shards": int(summary["annotation_shard_count"]),
        "exclusion": summary["exclusion"],
    }


def update_prepared_manifest(
    manifest_path: Path,
    *,
    summary: Mapping[str, Any],
    part_paths: Sequence[Path],
    part_rows: Sequence[int],
    part_sha256: Sequence[str],
    entry_prefix: str | None,
) -> dict[str, Any]:
    if not (len(part_paths) == len(part_rows) == len(part_sha256)):
        raise ValueError("manifest part metadata lengths differ")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        contract = _dataset_contract(summary)
        if manifest_path.exists():
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            if document.get("format_version") != PREPARED_MANIFEST_FORMAT:
                raise ValueError(f"{manifest_path}: incompatible manifest format")
            if document.get("dataset_contract") != contract:
                raise ValueError(f"{manifest_path}: dataset contract differs")
            shards = [
                dict(item)
                for item in document.get("shards", [])
                if int(item.get("annotation_shard", -1))
                != int(summary["selected_shard"])
            ]
        else:
            shards = []
        for part_index, (path, rows, digest) in enumerate(
            zip(part_paths, part_rows, part_sha256, strict=True)
        ):
            destination = _join_uri(entry_prefix, path.name) if entry_prefix else str(path.resolve())
            shards.append(
                {
                    "path": destination,
                    "rows": int(rows),
                    "sha256": digest,
                    "annotation_shard": int(summary["selected_shard"]),
                    "annotation_shard_complete": int(summary["limit"]) == 0,
                    "part_index": part_index,
                    "part_count": len(part_paths),
                    "source_v5_sha256": summary["source_v5_shard_sha256"],
                }
            )
        shards.sort(key=lambda item: (int(item["annotation_shard"]), int(item["part_index"])))
        annotation_shards = sorted({int(item["annotation_shard"]) for item in shards})
        expected = int(summary["annotation_shard_count"])
        complete = annotation_shards == list(range(expected)) and all(
            bool(item["annotation_shard_complete"]) for item in shards
        )
        result = {
            "format_version": PREPARED_MANIFEST_FORMAT,
            "complete": complete,
            "dataset_contract": contract,
            "paths": [str(item["path"]) for item in shards],
            "shards": shards,
            "physical_shard_count": len(shards),
            "annotation_shards": annotation_shards,
            "expected_annotation_shards": expected,
            "total_rows": sum(int(item["rows"]) for item in shards),
        }
        _write_atomic(manifest_path, _json_bytes(result), overwrite=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return result


def convert(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if prompt_contract_sha256() != PROMPT_CONTRACT_SHA256:
        raise ValueError("V5 prompt contract drifted before dataset publication")
    _, s3_client = _aws_session(args.aws_profile, args.region)
    dataset_manifest, manifest_raw = _read_document(args.dataset_manifest, s3_client=s3_client)
    source_shards = _manifest_shards(dataset_manifest)
    shard = args.shard
    if shard < 0 or shard >= len(source_shards):
        raise ValueError(f"--shard must be between 0 and {len(source_shards) - 1}")
    source_identity = source_shards[shard]
    global_start, global_end, range_source = _source_range(
        shard,
        summary_dir=args.v4_summary_dir,
        global_summary_uri=args.v4_global_summary,
        s3_client=s3_client,
    )
    _progress(
        f"shard={shard:02d} V5_rows={int(source_identity['sample_count']):,} "
        f"source_range=[{global_start:,},{global_end:,})"
    )

    source_dataset = parse_s3_uri(args.source_dataset.rstrip("/"))
    inventory_cache = args.inventory_cache or args.output.with_name(
        "image_edit_v4.source_inventory.json"
    )
    inventory = load_or_build_inventory(
        source_inventory=args.source_inventory,
        inventory_cache=inventory_cache.resolve(),
        source_dataset=source_dataset,
        s3_client=s3_client,
        workers=args.inventory_workers,
        overwrite=args.overwrite,
    )
    metadata_rows = recover_source_metadata(
        inventory,
        s3_client=s3_client,
        source_dataset_uri=source_dataset.uri,
        global_start=global_start,
        global_end=global_end,
        workers=args.parquet_workers,
    )
    metadata_by_id = {str(row["sample_id"]): row for row in metadata_rows}
    excluded_ids, exclusion = _load_excluded_ids(args.exclude_sample_ids)
    allowed_statuses = tuple(dict.fromkeys(args.allowed_status))

    output = args.output.resolve()
    part_paths = output_part_paths(output, args.output_parts)
    opened = [_open_atomic(path, overwrite=args.overwrite) for path in part_paths]
    streams = [item[0] for item in opened]
    temporary_paths = [item[1] for item in opened]
    part_rows = [0] * len(part_paths)
    part_hashers = [hashlib.sha256() for _ in part_paths]
    source_status_counts: Counter[str] = Counter()
    selected_status_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    rejection_examples: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    emitted = 0
    excluded_matches = 0
    stream = VerifiedS3Jsonl(s3_client, source_identity)
    try:
        for input_index, row in enumerate(stream):
            sample_id = row.get("sample_id")
            status = row.get("status")
            dataset_row_index = row.get("dataset_row_index")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"V5 row {input_index}: missing sample_id")
            if sample_id in seen_ids:
                raise ValueError(f"V5 shard contains duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            if not isinstance(status, str) or not status:
                raise ValueError(f"{sample_id}: missing authoritative status")
            source_status_counts[status] += 1
            metadata = metadata_by_id.get(sample_id)
            if metadata is None:
                raise ValueError(
                    f"{sample_id}: not found in immutable source range "
                    f"[{global_start},{global_end})"
                )
            expected_row = int(metadata["dataset_source"]["dataset_global_row"])
            if type(dataset_row_index) is not int or dataset_row_index != expected_row:
                raise ValueError(
                    f"{sample_id}: dataset_row_index={dataset_row_index!r} does not match "
                    f"source global row {expected_row}"
                )
            if status not in allowed_statuses:
                continue
            validation = row.get("validation")
            if status == "validated" and (
                not isinstance(validation, Mapping)
                or validation.get("valid") is not True
                or validation.get("status") != "validated"
            ):
                raise ValueError(f"{sample_id}: validated status disagrees with validation report")
            if sample_id in excluded_ids:
                excluded_matches += 1
                continue
            target = row.get("v5_annotation")
            if not isinstance(target, Mapping):
                reason = "v5_annotation is not an object"
            elif contains_operation_key(target):
                reason = "target_json contains forbidden operation key"
            else:
                declared_digest = row.get("v5_annotation_sha256")
                actual_digest = hashlib.sha256(_canonical_bytes(target)).hexdigest()
                if declared_digest != actual_digest:
                    raise ValueError(f"{sample_id}: v5_annotation_sha256 mismatch")
                try:
                    prepared = build_prepared_record(
                        metadata,
                        status=status,
                        target_json=target,
                    )
                    reason = ""
                except (V5TargetValidationError, ValueError) as exc:
                    reason = str(exc)
            if reason:
                rejection_counts[reason] += 1
                if len(rejection_examples) < args.rejection_examples:
                    rejection_examples.append({"sample_id": sample_id, "reason": reason})
                if args.invalid_policy == "error":
                    raise ValueError(f"{sample_id}: invalid V5 target: {reason}")
                continue
            part_index = emitted % len(streams)
            line = _canonical_bytes(prepared) + b"\n"
            streams[part_index].write(line)
            part_hashers[part_index].update(line)
            part_rows[part_index] += 1
            emitted += 1
            selected_status_counts[status] += 1
            if args.progress_every and emitted % args.progress_every == 0:
                _progress(f"shard={shard:02d} emitted={emitted:,}")
            if args.limit and emitted >= args.limit:
                break
        for stream_handle in streams:
            stream_handle.flush()
            os.fsync(stream_handle.fileno())
            stream_handle.close()
        for temporary, destination in zip(temporary_paths, part_paths, strict=True):
            os.replace(temporary, destination)
    except Exception:
        for stream_handle in streams:
            if not stream_handle.closed:
                stream_handle.close()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise

    if not args.limit:
        expected_counts = {str(key): int(value) for key, value in source_identity["status_counts"].items()}
        if dict(source_status_counts) != expected_counts:
            raise ValueError(
                f"V5 status counts differ from manifest: actual={dict(source_status_counts)}, "
                f"expected={expected_counts}"
            )
    eligible_metadata = [
        row
        for row in metadata_rows
        if row["sample_id"] in seen_ids
        and row["sample_id"] not in excluded_ids
    ]
    image_verification = verify_images(
        eligible_metadata,
        s3_client=s3_client,
        mode=args.verify_images,
        sample_count=args.image_check_samples,
        workers=args.image_workers,
    )
    part_digests = [hasher.hexdigest() for hasher in part_hashers]
    summary = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "source_v5_dataset_manifest": args.dataset_manifest,
        "source_v5_dataset_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_v5_dataset_format": dataset_manifest["dataset_format"],
        "source_v5_shard": str(source_identity["s3_uri"]),
        "source_v5_shard_version_id": source_identity.get("version_id"),
        "source_v5_shard_sha256": source_identity["sha256"],
        "source_v5_shard_bytes": int(source_identity["bytes"]),
        "selected_shard": shard,
        "annotation_shard_count": len(source_shards),
        "global_row_start": global_start,
        "global_row_end_exclusive": global_end,
        "source_range_authority": range_source,
        "source_dataset": source_dataset.uri,
        "source_inventory": args.source_inventory or str(inventory_cache.resolve()),
        "selection": "authoritative_status",
        "allowed_statuses": list(allowed_statuses),
        "target_validation_by_status": {
            "validated": "strict_image_edit_v5_instruction_schema",
            "manual_review_required": "preserve_complete_json_object_without_repair",
        },
        "exclusion": exclusion,
        "excluded_matching_rows": excluded_matches,
        "output_format": "prepared",
        "output": str(output),
        "physical_outputs": [str(path) for path in part_paths],
        "physical_output_rows": part_rows,
        "physical_output_sha256": part_digests,
        "output_rows": emitted,
        "input_rows": stream.rows_read,
        "source_status_counts": dict(sorted(source_status_counts.items())),
        "selected_status_counts": dict(sorted(selected_status_counts.items())),
        "schema_rejection_count": sum(rejection_counts.values()),
        "schema_rejection_counts": dict(sorted(rejection_counts.items())),
        "schema_rejection_examples": rejection_examples,
        "operation_key_rejection_count": rejection_counts.get(
            "target_json contains forbidden operation key", 0
        ),
        "image_verification": image_verification,
        "limit": args.limit,
        "semantic_values_modified": False,
        "media_paths_inferred": False,
        "media_join": "exact sample_id plus dataset_row_index",
        "model_api_calls_added": 0,
        "qwen_contract": {
            "template": "qwen3_5",
            "materialization": "runtime_custom_loader",
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "image_order": ["reference frame_0", "target"],
            "image_placeholders": 2,
            "target_schema": "image_edit_v5_instruction",
            "forbidden_target_keys_recursive": ["operation"],
            "think_prefix": False,
        },
    }
    summary_path = output.with_name(output.name + ".summary.json")
    manifest_path: Path | None = None
    if not args.no_manifest:
        manifest_path = (
            args.manifest.resolve()
            if args.manifest
            else output.parent / "image_edit_v5_full_json.prepared.manifest.json"
        )
        summary["manifest"] = str(manifest_path)
        summary["manifest_entry_prefix"] = args.manifest_entry_prefix
    _write_atomic(summary_path, _json_bytes(summary), overwrite=args.overwrite)
    if manifest_path is not None:
        update_prepared_manifest(
            manifest_path,
            summary=summary,
            part_paths=part_paths,
            part_rows=part_rows,
            part_sha256=part_digests,
            entry_prefix=args.manifest_entry_prefix,
        )
    _progress(
        f"shard={shard:02d} complete emitted={emitted:,} excluded={excluded_matches:,} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-manifest", default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-inventory", default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--inventory-cache", type=Path)
    parser.add_argument("--v4-global-summary", default=DEFAULT_V4_GLOBAL_SUMMARY)
    parser.add_argument(
        "--v4-summary-dir",
        type=Path,
        help="Optional local directory containing prior V4 per-shard summaries",
    )
    parser.add_argument("--allowed-status", action="append", default=None)
    parser.add_argument(
        "--exclude-sample-ids",
        action="append",
        type=Path,
        default=[],
        help="Newline-delimited IDs to exclude; repeat for multiple split files",
    )
    parser.add_argument("--output-parts", type=int, default=8)
    manifest_group = parser.add_mutually_exclusive_group()
    manifest_group.add_argument("--manifest", type=Path)
    manifest_group.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--manifest-entry-prefix")
    parser.add_argument("--aws-profile")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--parquet-workers", type=int, default=16)
    parser.add_argument("--image-workers", type=int, default=32)
    parser.add_argument("--verify-images", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--image-check-samples", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--invalid-policy", choices=("skip", "error"), default="error")
    parser.add_argument("--rejection-examples", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.allowed_status is None:
        args.allowed_status = ["validated"]
    if not args.allowed_status or any(not value for value in args.allowed_status):
        raise ValueError("allowed statuses must be non-empty")
    if args.output_parts <= 0:
        raise ValueError("--output-parts must be positive")
    if min(args.inventory_workers, args.parquet_workers, args.image_workers) <= 0:
        raise ValueError("worker counts must be positive")
    if args.limit < 0 or args.progress_every < 0:
        raise ValueError("--limit and --progress-every must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
