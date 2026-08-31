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
