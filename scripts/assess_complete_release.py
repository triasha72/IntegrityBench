#!/usr/bin/env python3
"""Apply the complete model, shift, and human-review release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integritybench.monitoring import assess_complete_release


def optional_json(path: Path | None):
    return None if path is None else json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--external-shift", type=Path)
    parser.add_argument("--human-agreement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_complete_release(
        json.loads(args.candidate.read_text()),
        optional_json(args.external_shift),
        optional_json(args.human_agreement),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
