"""Build a blinded, balanced 120-case annotation pack."""

from __future__ import annotations

import json
from pathlib import Path

from integritybench.dataset import SLICES, build_cases


def main() -> None:
    selected = []
    cases = build_cases()
    for slice_name in SLICES:
        selected.extend([case for case in cases if case.slice_name == slice_name][:20])
    rows = [
        {
            "case_id": case.case_id,
            "policy_version": case.policy_version,
            "slice_name": case.slice_name,
            "language": case.language,
            "content": case.content,
            "annotated_decision": None,
            "annotated_rule_id": None,
            "supporting_passage": None,
            "reason": None,
            "remediation": None,
            "review_required": None,
            "annotation_confidence": None,
            "ambiguity_note": None,
        }
        for case in selected
    ]
    target = Path("data/annotation/integritybench_v0_2_blinded.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
