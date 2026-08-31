"""Validation-only threshold selection for three-way moderation decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

LABELS = np.asarray(["ALLOW", "ESCALATE", "REJECT"])


@dataclass(frozen=True)
class DecisionThresholds:
    allow_minimum: float
    reject_minimum: float
    reject_probability_cap_for_allow: float


def threshold_predictions(probabilities: np.ndarray, thresholds: DecisionThresholds) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape [rows, 3]")
    prediction = np.full(len(values), "ESCALATE", dtype=object)
    reject = values[:, 2] >= thresholds.reject_minimum
    allow = (values[:, 0] >= thresholds.allow_minimum) & (
        values[:, 2] <= thresholds.reject_probability_cap_for_allow
    )
    prediction[allow] = "ALLOW"
    prediction[reject] = "REJECT"
    return prediction


def false_acceptance_rate(expected: np.ndarray, predicted: np.ndarray) -> float | None:
    unsafe = expected == "REJECT"
    return float(np.mean(predicted[unsafe] == "ALLOW")) if np.any(unsafe) else None


def select_thresholds(
    expected: Iterable[str],
    probabilities: np.ndarray,
    safety_masks: dict[str, np.ndarray],
    maximum_false_acceptance: float = 0.05,
) -> tuple[DecisionThresholds, dict[str, object]]:
    """Maximize macro F1 among candidates satisfying validation safety constraints."""
    expected_array = np.asarray(list(expected))
    best = None
    candidates = 0
    for allow_minimum in np.linspace(0.50, 0.95, 10):
        for reject_minimum in np.linspace(0.20, 0.80, 13):
            for reject_cap in (0.05, 0.10, 0.15, 0.20):
                thresholds = DecisionThresholds(
                    float(allow_minimum), float(reject_minimum), float(reject_cap)
                )
                prediction = threshold_predictions(probabilities, thresholds)
                overall = false_acceptance_rate(expected_array, prediction)
                slice_rates = {
                    name: false_acceptance_rate(expected_array[mask], prediction[mask])
                    for name, mask in safety_masks.items()
                }
                rates = [overall, *slice_rates.values()]
                if any(rate is not None and rate > maximum_false_acceptance for rate in rates):
                    continue
                candidates += 1
                score = float(
                    f1_score(
                        expected_array, prediction, labels=LABELS, average="macro", zero_division=0
                    )
                )
                key = (score, -float(np.mean(prediction == "ESCALATE")))
                if best is None or key > best[0]:
                    best = (key, thresholds, overall, slice_rates)
    if best is None:
        raise ValueError("no threshold candidate satisfies the validation safety constraints")
    _, thresholds, overall, slice_rates = best
    return thresholds, {
        "feasible_candidates": candidates,
        "validation_macro_f1": best[0][0],
        "validation_false_acceptance_rate": overall,
        "validation_slice_false_acceptance_rate": slice_rates,
    }
