import json
from pathlib import Path

import pytest

from integritybench.civil_comments import CivilCommentsConfig, decision_from_toxicity, sample_rows
from integritybench.schemas import Decision
from scripts.train_civil_comments_baseline import expected_calibration_error


@pytest.mark.parametrize(
    "score,label", [(0.1, Decision.ALLOW), (0.2, Decision.ESCALATE), (0.5, Decision.REJECT)]
)
def test_label_mapping(score, label):
    assert decision_from_toxicity(score) is label


def test_sampling_is_deterministic():
    rows = [{"text": f"comment {i}", "toxicity": i / 20} for i in range(20)]
    config = CivilCommentsConfig(sample_size=6, seed=7)
    assert sample_rows(rows, config) == sample_rows(rows, config)


def test_calibration_error_is_zero_for_correct_certain_predictions():
    assert (
        expected_calibration_error(
            ["ALLOW", "REJECT"],
            ["ALLOW", "REJECT"],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        == 0.0
    )


def test_published_real_data_artifact_keeps_safety_evidence():
    artifact = json.loads(
        (Path(__file__).parents[1] / "artifacts/civil_comments_baseline_v1.json").read_text()
    )
    assert artifact["dataset_id"] == "google/civil_comments"
    assert artifact["contains_source_text"] is False
    test = artifact["evaluations"]["test"]
    assert test["rows"] == 97_320
    assert {
        "false_acceptance_rate",
        "false_rejection_rate",
        "expected_calibration_error",
    } <= test.keys()
    assert {"threat", "identity_attack", "sexual_explicit"} <= test["safety_slices"].keys()


def test_safety_thresholded_candidate_remains_honestly_rejected():
    root = Path(__file__).parents[1]
    candidate = json.loads((root / "artifacts/civil_comments_candidate_v2.json").read_text())
    release = json.loads((root / "artifacts/civil_comments_candidate_release_v2.json").read_text())
    assert candidate["evaluations"]["test"]["false_acceptance_rate"] < 0.05
    assert candidate["evaluations"]["test"]["escalation_rate"] > 0.45
    assert release["decision"] == "rejected"
