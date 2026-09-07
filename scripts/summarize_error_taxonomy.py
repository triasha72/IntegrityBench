"""Write a source-free IntegrityBench error taxonomy receipt from JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integritybench.error_taxonomy import summarize_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line]
    result = summarize_errors(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
