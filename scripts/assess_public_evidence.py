#!/usr/bin/env python3
"""Apply the three-dataset research-evidence policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integritybench.monitoring import assess_public_evidence


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--toxic-chat", type=Path, required=True)
    parser.add_argument("--beavertails", type=Path, required=True)
    parser.add_argument("--author-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_public_evidence(
        load(args.candidate),
        load(args.toxic_chat),
        load(args.beavertails),
        None if args.author_audit is None else load(args.author_audit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
