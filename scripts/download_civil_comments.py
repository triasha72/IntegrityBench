#!/usr/bin/env python3
"""Download the pinned public Civil Comments parquet files with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/civil_comments/resolve/main/"
FILES = {
    "train": "data/train-00000-of-00002.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    manifest = {
        name: {
            "url": BASE + source,
            "path": str(args.output_directory / f"{name}.parquet"),
            "sha256": download(BASE + source, args.output_directory / f"{name}.parquet"),
        }
        for name, source in FILES.items()
    }
    (args.output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
