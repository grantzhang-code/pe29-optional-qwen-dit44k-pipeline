#!/usr/bin/env python3
"""Report committed unit/row progress for a frozen sidecar24 result root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_craftv0_image_edit_v6_repartition1600 as repartition
from scripts import image_edit_sidecar24_catalog as source_catalog_module
from scripts import prepare_image_edit_sidecar24_units as unit_catalog_module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unit_rows(unit: dict) -> int:
    return int(unit["global_row_end_exclusive"]) - int(unit["global_row_start"])


def scan(args: argparse.Namespace) -> dict:
    source = source_catalog_module.load_catalog(args.source_catalog)
    units_document = unit_catalog_module.load_unit_catalog(
        args.unit_catalog,
        source_catalog=source,
    )
    units = units_document["units"]
    bucket, result_key_prefix = repartition.parse_s3_uri(args.output_s3)
    client = boto3.client(
        "s3",
        region_name=args.aws_region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )

    started_at = utc_now()
    committed, listing = repartition.list_committed_indices(
        client,
        bucket=bucket,
        result_key_prefix=result_key_prefix,
        units=units,
    )
    total_rows = sum(unit_rows(unit) for unit in units)
    completed_rows = sum(
        unit_rows(unit)
        for unit in units
        if int(unit["global_unit_index"]) in committed
    )
    try:
        client.head_object(Bucket=bucket, Key=f"{result_key_prefix}/_SUCCESS")
        root_success = True
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            raise
        root_success = False

    return {
        "scan_started_at": started_at,
        "completed_at": utc_now(),
        "result_root": args.output_s3.rstrip("/"),
        "completed_units": len(committed),
        "completed_rows": completed_rows,
        "pending_units": len(units) - len(committed),
        "pending_rows": total_rows - completed_rows,
        "total_units": len(units),
        "total_rows": total_rows,
        "listing": listing,
        "root_success": root_success,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--unit-catalog", type=Path, required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args(argv)
    repartition.parse_s3_uri(args.output_s3)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = scan(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
