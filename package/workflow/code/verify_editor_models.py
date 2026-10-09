#!/usr/bin/env python3
"""Verify frozen Qwen text encoder and VAE assets for the 48k editor."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    files = [row for model in lock["models"] for row in model["files"]]

    def check(row: dict) -> str:
        path = args.models / row["path"]
        assert path.is_file() and path.stat().st_size == row["bytes"], row["path"]
        assert digest(path) == row["sha256"], row["path"]
        return row["path"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        names = list(pool.map(check, files))
    assert len(names) == len(set(names)) == len(files)
    marker = {"status": "PASS", "lock_sha256": digest(args.lock),
              "files": len(files), "bytes": sum(row["bytes"] for row in files)}
    (args.models / "MODELS_VERIFIED.json").write_text(json.dumps(marker, indent=2) + "\n")
    print("EDITOR_MODELS_VERIFIED", len(files), marker["bytes"], flush=True)


if __name__ == "__main__":
    main()
