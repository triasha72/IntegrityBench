"""Validate two completed annotation files and report agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "annotated_decision",
    "annotated_rule_id",
    "supporting_passage",
    "reason",
    "review_required",
    "annotation_confidence",
}


def load(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        missing = [name for name in REQUIRED if row.get(name) is None]
        if missing:
            raise ValueError(f"{row['case_id']} is missing: {', '.join(sorted(missing))}")
        rows[str(row["case_id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = load(args.annotator_a)
    second = load(args.annotator_b)
    if first.keys() != second.keys():
        raise ValueError("Annotators must label exactly the same case IDs.")
    ids = sorted(first)
    disagreements = [
        case_id
        for case_id in ids
        if first[case_id]["annotated_decision"] != second[case_id]["annotated_decision"]
        or first[case_id]["annotated_rule_id"] != second[case_id]["annotated_rule_id"]
    ]
    payload = {
        "case_count": len(ids),
        "decision_and_rule_agreement": 1.0 - len(disagreements) / len(ids),
        "disagreement_count": len(disagreements),
        "disagreement_case_ids": disagreements,
        "requires_adjudication": bool(disagreements),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
