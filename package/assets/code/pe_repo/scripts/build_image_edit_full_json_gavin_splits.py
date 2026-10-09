#!/usr/bin/env python3
"""Build Gavin-disjoint full-target-JSON train and shared evaluation splits.

The current full-target-JSON corpus and Gavin's descriptive CRAFT corpus use
different assistant schemas, but retain the same ``sample_id`` and input image
URIs.  This utility treats Gavin Medium validation/test IDs as the authoritative
split assignment, removes every matching ID from full-target-JSON training, and
emits the shared validation, test, and hard1000 subsets in the original prepared
row format.

Inputs may be local paths or S3 URIs.  ``--*-root`` remaps manifest entries to a
local cache by basename, which avoids downloading the same multi-GB prepared
source again.  Outputs are built locally; ``--published-root`` controls the S3
paths recorded in the generated manifests but does not upload files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse


FORMAT_VERSION = "image_edit_full_json_gavin_disjoint_v1"
SPLIT_POLICY = "gavin_medium_validation_test_sample_id_v1"


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class Reader:
    def __init__(self, aws_profile: str | None = None) -> None:
        self.aws_profile = aws_profile
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            session = boto3.Session(profile_name=self.aws_profile) if self.aws_profile else boto3.Session()
            self._client = session.client("s3")
        return self._client

    def bytes(self, uri: str) -> bytes:
        if not _is_s3(uri):
            return Path(uri).read_bytes()
        bucket, key = _split_s3_uri(uri)
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def lines(self, uri: str) -> Iterator[bytes]:
        if not _is_s3(uri):
            with Path(uri).open("rb") as handle:
                yield from handle
            return
        bucket, key = _split_s3_uri(uri)
        body = self.client.get_object(Bucket=bucket, Key=key)["Body"]
        for line in body.iter_lines():
            yield line + b"\n"


def _json_document(reader: Reader, uri: str) -> Mapping[str, Any]:
    value = json.loads(reader.bytes(uri))
    if not isinstance(value, Mapping):
        raise ValueError(f"{uri}: expected a JSON object")
    return value


def _manifest_entries(document: Mapping[str, Any], uri: str) -> list[str]:
    raw_entries = document.get("paths") or document.get("shards") or document.get("jsonl_paths")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"{uri}: manifest has no paths")
    result: list[str] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            result.append(entry["path"])
        elif isinstance(entry, Mapping) and isinstance(entry.get("s3_path"), str):
            result.append(entry["s3_path"])
        else:
            raise ValueError(f"{uri}: unsupported manifest entry: {entry!r}")
    return list(dict.fromkeys(result))


def _basename(uri: str) -> str:
    return Path(urlparse(uri).path).name if _is_s3(uri) else Path(uri).name


def _resolve_manifest(
    reader: Reader,
    manifest_uri: str,
    *,
    local_root: Path | None,
) -> tuple[Mapping[str, Any], list[str]]:
    document = _json_document(reader, manifest_uri)
    paths = _manifest_entries(document, manifest_uri)
    if local_root is not None:
        resolved = [str(local_root / _basename(path)) for path in paths]
        missing = [path for path in resolved if not Path(path).is_file()]
        if missing:
            raise ValueError(f"{manifest_uri}: {len(missing)} remapped files are missing: {missing[:3]}")
        paths = resolved
    return document, paths


def _parse_row(raw: bytes, *, source: str, line_number: int) -> Mapping[str, Any]:
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
    if not isinstance(row, Mapping):
        raise ValueError(f"{source}:{line_number}: expected a JSON object")
    return row


def _load_authority_rows(
    reader: Reader,
    paths: Sequence[str],
    *,
    name: str,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    order: list[str] = []
    rows: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for line_number, raw in enumerate(reader.lines(path), start=1):
            if not raw.strip():
                continue
            row = _parse_row(raw, source=path, line_number=line_number)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            if sample_id in rows:
                raise ValueError(f"{name}: duplicate sample_id {sample_id}")
            order.append(sample_id)
            rows[sample_id] = row
    return order, rows


def _input_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    references = row.get("reference_images")
    reference_uri = ""
    if isinstance(references, list) and references and isinstance(references[0], Mapping):
        reference_uri = str(references[0].get("s3_uri") or references[0].get("uri") or "")
    target = row.get("target_image")
    target_uri = ""
    if isinstance(target, Mapping):
        target_uri = str(target.get("s3_uri") or target.get("uri") or "")
    elif isinstance(target, str):
        target_uri = target
    return reference_uri, target_uri, str(row.get("instruction") or "")


def _authority_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("image_uri") or row.get("source_uri") or ""),
        str(row.get("target_uri") or ""),
        str(row.get("instruction") or ""),
    )


def _part_for_id(sample_id: str, parts: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[8:16], "big") % parts


@dataclass
class PartMetadata:
    path: Path
    rows: int
    sha256: str


class PartWriter:
    def __init__(self, root: Path, split: str, parts: int) -> None:
        if parts <= 0:
            raise ValueError("parts must be positive")
        self.split = split
        self.parts = parts
        self.root = root / split
        self.root.mkdir(parents=True)
        self.paths = [self.root / f"part-{index:05d}-of-{parts:05d}.jsonl" for index in range(parts)]
        self.handles: list[BinaryIO | None] = [None] * parts
        self.rows = [0] * parts
        self.hashers = [hashlib.sha256() for _ in range(parts)]

    def write(self, part: int, raw: bytes) -> None:
        if not 0 <= part < self.parts:
            raise ValueError(f"invalid {self.split} part {part}")
        if not raw.endswith(b"\n"):
            raw += b"\n"
        handle = self.handles[part]
        if handle is None:
            handle = self.paths[part].open("wb")
            self.handles[part] = handle
        handle.write(raw)
        self.rows[part] += 1
        self.hashers[part].update(raw)

    def close(self) -> list[PartMetadata]:
        for index, handle in enumerate(self.handles):
            if handle is not None:
                handle.close()
            else:
                self.paths[index].touch()
        return [
            PartMetadata(path=path, rows=self.rows[index], sha256=self.hashers[index].hexdigest())
            for index, path in enumerate(self.paths)
        ]


def _join_uri(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def _manifest(
    *,
    split: str,
    parts: Sequence[PartMetadata],
    published_root: str,
    source_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    shards = [
        {
            "part_index": index,
            "path": _join_uri(published_root, split, part.path.name),
            "rows": part.rows,
            "sha256": part.sha256,
        }
        for index, part in enumerate(parts)
    ]
    contract = dict(source_contract)
    contract.update(
        {
            "format_version": FORMAT_VERSION,
            "split": split,
            "split_policy": SPLIT_POLICY,
            "training_excludes": ["validation", "test"],
        }
    )
    return {
        "format_version": FORMAT_VERSION,
        "split": split,
        "total_rows": sum(part.rows for part in parts),
        "physical_shard_count": len(parts),
        "paths": [item["path"] for item in shards],
        "shards": shards,
        "dataset_contract": contract,
        "provenance": dict(provenance),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_ids(path: Path, sample_ids: Iterable[str]) -> dict[str, Any]:
    payload = "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    path.write_bytes(payload)
    return {"path": path.name, "rows": payload.count(b"\n"), "sha256": hashlib.sha256(payload).hexdigest()}


def _assert_expected(name: str, actual: int, expected: int) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f"{name}: expected {expected}, observed {actual}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    reader = Reader(args.aws_profile)
    source_manifest, source_paths = _resolve_manifest(
        reader,
        args.source_manifest,
        local_root=args.source_root,
    )
    validation_manifest, validation_paths = _resolve_manifest(
        reader,
        args.gavin_validation_manifest,
        local_root=args.gavin_validation_root,
    )
    test_manifest, test_paths = _resolve_manifest(
        reader,
        args.gavin_test_manifest,
        local_root=args.gavin_test_root,
    )
    validation_order, validation_rows = _load_authority_rows(
        reader, validation_paths, name="Gavin Medium validation"
    )
    test_order, test_rows = _load_authority_rows(reader, test_paths, name="Gavin Medium test")
    overlap = set(validation_rows).intersection(test_rows)
    if overlap:
        raise ValueError(f"Gavin validation/test overlap: {sorted(overlap)[:3]}")

    hard_order, hard_rows = _load_authority_rows(
        reader, [args.hard1000_cohort], name="Gavin hard1000"
    )
    hard_outside = set(hard_rows).difference(set(validation_rows) | set(test_rows))
    if hard_outside:
        raise ValueError(f"hard1000 contains IDs outside validation/test: {sorted(hard_outside)[:3]}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise ValueError(f"output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    stage = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    train_writer = PartWriter(stage, "train", args.train_parts)
    validation_writer = PartWriter(stage, "validation", args.validation_parts)
    test_writer = PartWriter(stage, "test", args.test_parts)
    hard_lines: dict[str, bytes] = {}
    seen: set[str] = set()
    shared_validation: set[str] = set()
    shared_test: set[str] = set()
    input_mismatches: list[dict[str, Any]] = []
    source_rows = 0

    try:
        for source_index, source_path in enumerate(source_paths):
            for line_number, raw in enumerate(reader.lines(source_path), start=1):
                if not raw.strip():
                    continue
                source_rows += 1
                row = _parse_row(raw, source=source_path, line_number=line_number)
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{source_path}:{line_number}: missing sample_id")
                if sample_id in seen:
                    raise ValueError(f"prepared source contains duplicate sample_id: {sample_id}")
                seen.add(sample_id)

                authority: Mapping[str, Any] | None = None
                if sample_id in validation_rows:
                    authority = validation_rows[sample_id]
                    shared_validation.add(sample_id)
                    validation_writer.write(_part_for_id(sample_id, args.validation_parts), raw)
                elif sample_id in test_rows:
                    authority = test_rows[sample_id]
                    shared_test.add(sample_id)
                    test_writer.write(_part_for_id(sample_id, args.test_parts), raw)
                else:
                    train_part = (
                        source_index
                        if args.train_parts == len(source_paths)
                        else _part_for_id(sample_id, args.train_parts)
                    )
                    train_writer.write(train_part, raw)

                if authority is not None and _input_identity(row) != _authority_identity(authority):
                    input_mismatches.append(
                        {
                            "sample_id": sample_id,
                            "full_json": _input_identity(row),
                            "gavin": _authority_identity(authority),
                        }
                    )
                if sample_id in hard_rows:
                    hard_lines[sample_id] = raw
    except Exception:
        train_writer.close()
        validation_writer.close()
        test_writer.close()
        raise

    train_parts = train_writer.close()
    validation_parts = validation_writer.close()
    test_parts = test_writer.close()
    if input_mismatches:
        raise ValueError(f"shared input identity mismatches: {input_mismatches[:3]}")

    shared_hard_order = [sample_id for sample_id in hard_order if sample_id in hard_lines]
    hard_writer = PartWriter(stage, "hard1000", args.hard1000_parts)
    for index, sample_id in enumerate(shared_hard_order):
        part = min(index * args.hard1000_parts // len(shared_hard_order), args.hard1000_parts - 1)
        hard_writer.write(part, hard_lines[sample_id])
    hard_parts = hard_writer.close()

    train_count = sum(part.rows for part in train_parts)
    validation_count = sum(part.rows for part in validation_parts)
    test_count = sum(part.rows for part in test_parts)
    hard_count = sum(part.rows for part in hard_parts)
    _assert_expected("source rows", source_rows, args.expected_source_rows)
    _assert_expected("train rows", train_count, args.expected_train_rows)
    _assert_expected("shared validation rows", validation_count, args.expected_shared_validation)
    _assert_expected("shared test rows", test_count, args.expected_shared_test)
    _assert_expected("shared hard1000 rows", hard_count, args.expected_shared_hard1000)
    if source_rows != train_count + validation_count + test_count:
        raise ValueError("train/validation/test do not partition the prepared source exactly")

    audit_dir = stage / "audit"
    audit_dir.mkdir()
    audit_files = {
        "shared_validation_ids": _write_ids(
            audit_dir / "shared_validation_ids.txt",
            (sample_id for sample_id in validation_order if sample_id in shared_validation),
        ),
        "shared_test_ids": _write_ids(
            audit_dir / "shared_test_ids.txt",
            (sample_id for sample_id in test_order if sample_id in shared_test),
        ),
        "shared_hard1000_ids": _write_ids(
            audit_dir / "shared_hard1000_ids.txt", shared_hard_order
        ),
        "missing_validation_ids": _write_ids(
            audit_dir / "missing_validation_ids.txt",
            (sample_id for sample_id in validation_order if sample_id not in shared_validation),
        ),
        "missing_test_ids": _write_ids(
            audit_dir / "missing_test_ids.txt",
            (sample_id for sample_id in test_order if sample_id not in shared_test),
        ),
        "missing_hard1000_ids": _write_ids(
            audit_dir / "missing_hard1000_ids.txt",
            (sample_id for sample_id in hard_order if sample_id not in hard_lines),
        ),
    }

    provenance = {
        "source_manifest": args.source_manifest,
        "source_manifest_total_rows": source_manifest.get("total_rows"),
        "gavin_validation_manifest": args.gavin_validation_manifest,
        "gavin_validation_rows": validation_manifest.get("record_count"),
        "gavin_test_manifest": args.gavin_test_manifest,
        "gavin_test_rows": test_manifest.get("record_count"),
        "gavin_hard1000_cohort": args.hard1000_cohort,
        "split_policy": SPLIT_POLICY,
    }
    source_contract = source_manifest.get("dataset_contract")
    if not isinstance(source_contract, Mapping):
        source_contract = {}
    manifests = {
        "train": _manifest(
            split="train",
            parts=train_parts,
            published_root=args.published_root,
            source_contract=source_contract,
            provenance=provenance,
        ),
        "validation": _manifest(
            split="validation",
            parts=validation_parts,
            published_root=args.published_root,
            source_contract=source_contract,
            provenance=provenance,
        ),
        "test": _manifest(
            split="test",
            parts=test_parts,
            published_root=args.published_root,
            source_contract=source_contract,
            provenance=provenance,
        ),
        "hard1000": _manifest(
            split="hard1000",
            parts=hard_parts,
            published_root=args.published_root,
            source_contract=source_contract,
            provenance=provenance,
        ),
    }
    for split, manifest in manifests.items():
        _write_json(stage / f"{split}_manifest.json", manifest)

    combined_shards = [
        {**item, "split": "validation"} for item in manifests["validation"]["shards"]
    ] + [{**item, "split": "train"} for item in manifests["train"]["shards"]]
    combined_contract = dict(source_contract)
    combined_contract.update(
        {
            "format_version": FORMAT_VERSION,
            "split": "validation_then_train",
            "split_policy": SPLIT_POLICY,
            "training_excludes": ["validation", "test"],
        }
    )
    combined_manifest = {
        "format_version": FORMAT_VERSION,
        "split": "validation_then_train",
        "val_path_count": len(validation_parts),
        "total_rows": validation_count + train_count,
        "paths": [item["path"] for item in combined_shards],
        "shards": combined_shards,
        "dataset_contract": combined_contract,
        "provenance": provenance,
    }
    _write_json(stage / "train_with_validation_manifest.json", combined_manifest)

    summary = {
        "format_version": FORMAT_VERSION,
        "split_policy": SPLIT_POLICY,
        "published_root": args.published_root,
        "source_rows": source_rows,
        "source_unique_ids": len(seen),
        "train_rows": train_count,
        "shared_validation_rows": validation_count,
        "shared_test_rows": test_count,
        "shared_hard1000_rows": hard_count,
        "gavin_validation_rows": len(validation_rows),
        "gavin_test_rows": len(test_rows),
        "gavin_hard1000_rows": len(hard_rows),
        "missing_validation_rows": len(validation_rows) - validation_count,
        "missing_test_rows": len(test_rows) - test_count,
        "missing_hard1000_rows": len(hard_rows) - hard_count,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
        "shared_input_identity_mismatches": 0,
        "audit_files": audit_files,
        "manifests": {
            split: f"{split}_manifest.json" for split in ("train", "validation", "test", "hard1000")
        }
        | {"train_with_validation": "train_with_validation_manifest.json"},
    }
    _write_json(stage / "split_summary.json", summary)
    stage.rename(output_dir)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--gavin-validation-manifest", required=True)
    parser.add_argument("--gavin-validation-root", type=Path)
    parser.add_argument("--gavin-test-manifest", required=True)
    parser.add_argument("--gavin-test-root", type=Path)
    parser.add_argument("--hard1000-cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--published-root", required=True)
    parser.add_argument("--aws-profile")
    parser.add_argument("--train-parts", type=int, default=128)
    parser.add_argument("--validation-parts", type=int, default=32)
    parser.add_argument("--test-parts", type=int, default=32)
    parser.add_argument("--hard1000-parts", type=int, default=32)
    parser.add_argument("--expected-source-rows", type=int, default=0)
    parser.add_argument("--expected-train-rows", type=int, default=0)
    parser.add_argument("--expected-shared-validation", type=int, default=0)
    parser.add_argument("--expected-shared-test", type=int, default=0)
    parser.add_argument("--expected-shared-hard1000", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in ("train_parts", "validation_parts", "test_parts", "hard1000_parts"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = build(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
