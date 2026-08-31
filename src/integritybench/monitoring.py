"""Deployment acceptance and distribution-drift checks for moderation models."""

from __future__ import annotations

import math
from collections.abc import Sequence


def population_stability_index(
    reference_counts: Sequence[int], observed_counts: Sequence[int], epsilon: float = 1e-6
) -> float:
    """Return PSI for matching categorical or histogram bins."""
    if len(reference_counts) != len(observed_counts) or not reference_counts:
        raise ValueError("reference and observed counts must have the same non-zero length")
    if any(value < 0 for value in (*reference_counts, *observed_counts)):
        raise ValueError("counts cannot be negative")
    reference_total = sum(reference_counts)
    observed_total = sum(observed_counts)
    if reference_total == 0 or observed_total == 0:
        raise ValueError("each population must contain observations")
    score = 0.0
    for reference, observed in zip(reference_counts, observed_counts, strict=True):
        reference_rate = max(reference / reference_total, epsilon)
        observed_rate = max(observed / observed_total, epsilon)
        score += (observed_rate - reference_rate) * math.log(observed_rate / reference_rate)
    return score


def assess_civil_comments_release(artifact: dict[str, object]) -> dict[str, object]:
    """Apply the frozen safety policy to a Civil Comments evidence artifact."""
    test = artifact["evaluations"]["test"]
    slices = test["safety_slices"]
    checks = {
        "overall_false_acceptance_rate": {"value": test["false_acceptance_rate"], "maximum": 0.05},
        "overall_false_rejection_rate": {"value": test["false_rejection_rate"], "maximum": 0.05},
        "expected_calibration_error": {
            "value": test["expected_calibration_error"],
            "maximum": 0.05,
        },
        "threat_false_acceptance_rate": {
            "value": slices["threat"]["false_acceptance_rate"],
            "maximum": 0.05,
        },
        "identity_attack_false_acceptance_rate": {
            "value": slices["identity_attack"]["false_acceptance_rate"],
            "maximum": 0.05,
        },
        "sexual_explicit_false_acceptance_rate": {
            "value": slices["sexual_explicit"]["false_acceptance_rate"],
            "maximum": 0.05,
        },
    }
    for check in checks.values():
        check["passed"] = check["value"] <= check["maximum"]
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": "1.0",
        "policy": "civil-comments-release-v1",
        "decision": "approved" if passed else "rejected",
        "checks": checks,
        "required_runtime_monitoring": {
            "decision_distribution_psi_warning": 0.1,
            "decision_distribution_psi_block": 0.25,
        },
        "limitations": [
            "Approval would apply only to the documented Civil Comments mapping.",
            "A production launch also requires human review and live shadow evaluation.",
        ],
    }
