#!/usr/bin/env python3
"""Download the official BeaverTails 330k train/test exports with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/main/round0/330k/"
FILES = {"train": "train.jsonl.xz", "test": "test.jsonl.xz"}


def download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    names = ("train", "test") if args.include_test else ("train",)
    manifest = {
        name: {
            "url": BASE + FILES[name],
            "path": str(args.output_directory / FILES[name]),
            "sha256": download(BASE + FILES[name], args.output_directory / FILES[name]),
        }
        for name in names
    }
    (args.output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
