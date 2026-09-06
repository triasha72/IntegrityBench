#!/usr/bin/env python3
"""Create two independent, balanced volunteer-review pilot files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_pilot(source: Path, output_dir: Path, per_slice: int = 4) -> None:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    slices = sorted({str(row["slice_name"]) for row in rows})
    selected = []
    for slice_name in slices:
        candidates = [row for row in rows if row["slice_name"] == slice_name]
        candidates.sort(key=lambda row: hashlib.sha256(row["case_id"].encode()).hexdigest())
        selected.extend(candidates[:per_slice])
    selected.sort(key=lambda row: row["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in ("reviewer_a", "reviewer_b"):
        target = output_dir / f"integritybench_pilot_{reviewer}.jsonl"
        target.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "1.0",
        "purpose": "unpaid_volunteer_pilot",
        "cases": len(selected),
        "cases_per_slice": per_slice,
        "reviewers_required": 2,
        "compensation": "none",
        "human_reviews_completed": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/annotation/integritybench_v0_2_blinded.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-slice", type=int, default=4)
    args = parser.parse_args()
    if args.per_slice < 1:
        raise ValueError("per-slice must be positive")
    build_pilot(args.source, args.output_dir, args.per_slice)


if __name__ == "__main__":
    main()
