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


def assess_complete_release(
    candidate: dict[str, object],
    external_shift: dict[str, object] | None,
    human_agreement: dict[str, object] | None,
) -> dict[str, object]:
    """Require model, external-shift, and independent-review evidence together."""
    model = assess_civil_comments_release(candidate)
    external_rate = None if external_shift is None else external_shift.get("false_acceptance_rate")
    human_cases = None if human_agreement is None else human_agreement.get("case_count")
    human_rate = (
        None if human_agreement is None else human_agreement.get("decision_and_rule_agreement")
    )
    human_adjudication = (
        None if human_agreement is None else human_agreement.get("requires_adjudication")
    )
    checks = {
        "civil_comments_release": {
            "value": model["decision"], "required": "approved",
            "passed": model["decision"] == "approved",
        },
        "external_shift_false_acceptance": {
            "value": external_rate, "maximum": 0.1,
            "passed": external_rate is not None and external_rate <= 0.1,
        },
        "independent_human_review_size": {
            "value": human_cases, "minimum": 100,
            "passed": human_cases is not None and human_cases >= 100,
        },
        "independent_human_agreement": {
            "value": human_rate, "minimum": 0.8,
            "passed": human_rate is not None and human_rate >= 0.8,
        },
        "human_disagreements_adjudicated": {
            "value": human_adjudication, "required": False,
            "passed": human_adjudication is False,
        },
    }
    return {
        "schema_version": "1.0",
        "policy": "integritybench-complete-release-v1",
        "decision": "approved" if all(item["passed"] for item in checks.values()) else "blocked",
        "checks": checks,
        "civil_comments_assessment": model,
        "non_claim": "A blocked decision is not a production safety claim.",
    }
