import json
from pathlib import Path

from integritybench.dataset import SLICES


def test_annotation_pack_is_blinded_and_balanced() -> None:
    rows = [
        json.loads(line)
        for line in Path("data/annotation/integritybench_v0_2_blinded.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 120
    assert len({row["case_id"] for row in rows}) == 120
    assert {row["slice_name"] for row in rows} == set(SLICES)
    assert all(sum(row["slice_name"] == name for row in rows) == 20 for name in SLICES)
    assert all(row["annotated_decision"] is None for row in rows)
    assert all("expected_decision" not in row for row in rows)
