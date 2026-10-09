#!/usr/bin/env python3
"""Unpack the pinned Docker image layers for native canary use in a held Pod."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


EXPECTED_CONFIG = "baa2ff80885bc029e33cbd11c433da5d14cfd5533ccbb09e9380595f37ef0af4.json"


def safe(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image path: {relative}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"image path leaves root: {relative}")
    return path


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--rootfs", required=True, type=Path)
    parser.add_argument("--start-layer", type=int, default=0)
    args = parser.parse_args()
    archive = args.archive_dir.resolve()
    root = args.rootfs.resolve()
    manifest = json.loads((archive / "manifest.json").read_text())
    assert len(manifest) == 1 and manifest[0]["Config"] == EXPECTED_CONFIG
    layers = manifest[0]["Layers"]
    root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(layers):
        if index < args.start_layer:
            continue
        layer = archive / name
        assert layer.is_file(), layer
        opaque: list[Path] = []
        deletes: list[Path] = []
        markers: list[Path] = []
        replacements: list[Path] = []
        with tarfile.open(layer) as stream:
            for member in stream:
                relative = PurePosixPath(member.name.removeprefix("./"))
                if not relative.parts:
                    continue
                basename = relative.name
                if basename == ".wh..wh..opq":
                    opaque.append(safe(root, relative.parent))
                    markers.append(safe(root, relative))
                elif basename.startswith(".wh."):
                    deletes.append(safe(root, relative.parent / basename.removeprefix(".wh.")))
                    markers.append(safe(root, relative))
                else:
                    path = root.joinpath(*relative.parts)
                    if not path.parent.resolve().is_relative_to(root):
                        # A parent symlink is replaced by an earlier layer member.
                        continue
                    if member.isdir() and (path.is_symlink() or path.is_file()):
                        replacements.append(path)
                    elif not member.isdir() and path.is_dir():
                        replacements.append(path)
        for directory in opaque:
            if directory.is_dir():
                for child in directory.iterdir():
                    remove(child)
        for path in deletes:
            remove(path)
        for path in replacements:
            remove(path)
        subprocess.run(["tar", "-xpf", str(layer), "-C", str(root), "--no-same-owner",
                        "--delay-directory-restore"], check=True)
        for marker in markers:
            if marker.is_file() or marker.is_symlink():
                marker.unlink()
        if index % 5 == 0 or index == len(layers) - 1:
            print("IMAGE_LAYER", index + 1, len(layers), name, flush=True)
    (root / "IMAGE_READY.json").write_text(json.dumps({"config": EXPECTED_CONFIG,
                                                          "layers": len(layers)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
