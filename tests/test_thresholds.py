import numpy as np

from integritybench.thresholds import DecisionThresholds, select_thresholds, threshold_predictions
from scripts.train_civil_comments_candidate import decision_calibration_error


def test_threshold_predictions_prioritize_reject_and_escalate_uncertainty():
    probabilities = np.asarray([[0.9, 0.08, 0.02], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
    thresholds = DecisionThresholds(0.8, 0.6, 0.1)
    assert threshold_predictions(probabilities, thresholds).tolist() == [
        "ALLOW",
        "ESCALATE",
        "REJECT",
    ]


def test_selection_obeys_false_acceptance_constraint():
    expected = ["ALLOW", "ALLOW", "ESCALATE", "REJECT", "REJECT"]
    probabilities = np.asarray(
        [[0.9, 0.08, 0.02], [0.8, 0.15, 0.05], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )
    thresholds, evidence = select_thresholds(
        expected, probabilities, {"threat": np.asarray([False, False, False, True, True])}
    )
    predicted = threshold_predictions(probabilities, thresholds)
    assert not np.any(predicted[np.asarray(expected) == "REJECT"] == "ALLOW")
    assert evidence["feasible_candidates"] > 0


def test_decision_calibration_uses_probability_of_thresholded_class():
    expected = np.asarray(["ALLOW", "ESCALATE", "REJECT"])
    predicted = np.asarray(["ALLOW", "ESCALATE", "REJECT"])
    probabilities = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert decision_calibration_error(expected, predicted, probabilities) == 0
