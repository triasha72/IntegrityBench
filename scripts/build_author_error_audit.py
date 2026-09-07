#!/usr/bin/env python3
"""Build a deterministic 50-case author error-audit sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/annotation/integritybench_v0_2_blinded.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    if args.size < 1 or args.size > len(rows):
        raise ValueError("size must fit within the source pack")
    rows.sort(key=lambda row: hashlib.sha256(row["case_id"].encode()).hexdigest())
    output = []
    for row in rows[: args.size]:
        output.append(
            {
                "case_id": row["case_id"],
                "reviewer_role": "project_author",
                "model_decision": None,
                "reference_decision": None,
                "error_category": None,
                "error_notes": None,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in output) + "\n")


if __name__ == "__main__":
    main()
