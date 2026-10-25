#!/usr/bin/env python3
"""Collect manually transferred NAOMi PNGs into one image dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILE = json.loads((ROOT / "profile.json").read_text(encoding="utf-8"))
NAME = re.compile(r"naomi_(\d{4})_seed(\d+)\.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "incoming", help="Folder containing PNGs, including subfolders")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_png", help="Folder for collected PNGs")
    parser.add_argument("--require-complete", action="store_true", help="Require all expected image numbers")
    args = parser.parse_args()
    source_root = args.input.resolve()
    output = args.output.resolve()
    if not source_root.is_dir():
        parser.error(f"Input folder does not exist: {source_root}")
    if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        parser.error("Input and output folders must not contain one another")

    expected_size = (int(PROFILE["image_shape_yx"][1]), int(PROFILE["image_shape_yx"][0]))
    seed_base = int(PROFILE["seed_base"])
    total = int(PROFILE["total_images"])
    found: dict[int, dict] = {}
    for path in sorted(source_root.rglob("*.png")):
        match = NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected PNG filename: {path.name}; expected naomi_0000_seed20263000.png")
        index, seed = map(int, match.groups())
        if not 0 <= index < total or seed != seed_base + index:
            raise ValueError(f"Wrong image number or seed: {path.name}")
        if index in found:
            raise ValueError(f"Duplicate image number {index}: {found[index]['path']} and {path}")
        if png_size(path) != expected_size:
            raise ValueError(f"Wrong PNG size for {path.name}; expected {expected_size[0]}x{expected_size[1]}")
        found[index] = {
            "index": index,
            "seed": seed,
            "filename": path.name,
            "sha256": sha256(path),
            "source": str(path.relative_to(source_root)),
            "path": path,
        }
    if not found:
        raise ValueError(f"No PNGs found in {source_root}")
    if args.require_complete and len(found) != total:
        missing = sorted(set(range(total)) - set(found))
        raise ValueError(f"Only {len(found)}/{total} images; first missing numbers: {missing[:20]}")

    images = output / "images"
    expected_names = {item["filename"] for item in found.values()}
    for existing in images.glob("*.png"):
        if existing.name not in expected_names:
            raise ValueError(f"Old PNG in output is absent from input: {existing}")
    images.mkdir(parents=True, exist_ok=True)
    copied = 0
    for index in sorted(found):
        item = found[index]
        destination = images / item["filename"]
        if destination.exists():
            if sha256(destination) != item["sha256"]:
                raise ValueError(f"Output has different pixels for {item['filename']}: {destination}")
            continue
        temporary = images / (item["filename"] + f".tmp.{os.getpid()}")
        try:
            shutil.copy2(item["path"], temporary)
            if sha256(temporary) != item["sha256"]:
                raise ValueError(f"Copy failed checksum check: {item['path']}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        copied += 1

    manifest = output / "manifest.csv"
    temporary_manifest = manifest.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as stream:
        columns = ["index", "seed", "filename", "sha256", "source"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index in sorted(found):
            writer.writerow({key: found[index][key] for key in columns})
    os.replace(temporary_manifest, manifest)
    print(f"Collected {len(found)}/{total} PNGs in {images}; copied {copied} new files")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
