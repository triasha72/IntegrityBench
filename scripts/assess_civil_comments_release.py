#!/usr/bin/env python3
"""Generate the frozen deployment assessment for the real-data baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integritybench.monitoring import assess_civil_comments_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assessment = assess_civil_comments_release(json.loads(args.artifact.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assessment, indent=2) + "\n")
    print(f"decision={assessment['decision']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
