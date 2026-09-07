import json
from pathlib import Path


def test_checked_in_public_evidence_is_honestly_blocked() -> None:
    root = Path(__file__).parents[1]
    beavertails = json.loads((root / "artifacts/beavertails_external_v1.json").read_text())
    assessment = json.loads((root / "artifacts/public_evidence_assessment_v1.json").read_text())
    assert beavertails["rows"] == 11088
    assert beavertails["false_acceptance_rate"] > 0.58
    assert assessment["decision"] == "blocked"
    assert not assessment["checks"]["beavertails_false_acceptance"]["passed"]
    assert not assessment["checks"]["author_error_audit"]["passed"]
