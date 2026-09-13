from integritybench.monitoring import (
    assess_civil_comments_release,
    assess_complete_release,
    assess_public_evidence,
    population_stability_index,
)


def test_population_stability_index_detects_distribution_shift():
    assert population_stability_index([80, 15, 5], [80, 15, 5]) == 0
    assert population_stability_index([80, 15, 5], [40, 30, 30]) > 0.25


def test_release_policy_rejects_unsafe_slices():
    artifact = {
        "evaluations": {
            "test": {
                "false_acceptance_rate": 0.04,
                "false_rejection_rate": 0.02,
                "expected_calibration_error": 0.01,
                "safety_slices": {
                    "threat": {"false_acceptance_rate": 0.20},
                    "identity_attack": {"false_acceptance_rate": 0.04},
                    "sexual_explicit": {"false_acceptance_rate": 0.04},
                },
            }
        }
    }
    assessment = assess_civil_comments_release(artifact)
    assert assessment["decision"] == "rejected"
    assert not assessment["checks"]["threat_false_acceptance_rate"]["passed"]


def test_complete_release_blocks_missing_human_review_and_external_failure():
    candidate = {
        "evaluations": {
            "test": {
                "false_acceptance_rate": 0.01,
                "false_rejection_rate": 0.01,
                "expected_calibration_error": 0.01,
                "safety_slices": {
                    name: {"false_acceptance_rate": 0.01}
                    for name in ("threat", "identity_attack", "sexual_explicit")
                },
            }
        }
    }
    result = assess_complete_release(candidate, {"false_acceptance_rate": 0.59}, None)
    assert result["decision"] == "blocked"
    assert not result["checks"]["external_shift_false_acceptance"]["passed"]


def test_public_evidence_accepts_frozen_release_assessment():
    candidate = {
        "policy": "civil-comments-release-v1",
        "decision": "approved",
        "checks": {},
    }
    result = assess_public_evidence(
        candidate,
        {"false_acceptance_rate": 0.09},
        {"false_acceptance_rate": 0.08},
    )
    assert result["checks"]["civil_comments_release"]["passed"]
    assert result["checks"]["toxic_chat_false_acceptance"]["passed"]
    assert not result["checks"]["independent_human_review_size"]["passed"]


def test_public_evidence_uses_three_datasets_without_claiming_independent_review():
    candidate = {
        "evaluations": {
            "test": {
                "false_acceptance_rate": 0.01,
                "false_rejection_rate": 0.01,
                "expected_calibration_error": 0.01,
                "safety_slices": {
                    name: {"false_acceptance_rate": 0.01}
                    for name in ("threat", "identity_attack", "sexual_explicit")
                },
            }
        }
    }
    result = assess_public_evidence(
        candidate,
        {"false_acceptance_rate": 0.08},
        {"false_acceptance_rate": 0.09},
        {"case_count": 50, "complete": True},
    )
    assert result["decision"] == "supported"
    assert "not independent human review" in result["non_claims"][0]


def test_public_evidence_blocks_missing_beavertails_result():
    candidate = {
        "evaluations": {
            "test": {
                "false_acceptance_rate": 0.01,
                "false_rejection_rate": 0.01,
                "expected_calibration_error": 0.01,
                "safety_slices": {
                    name: {"false_acceptance_rate": 0.01}
                    for name in ("threat", "identity_attack", "sexual_explicit")
                },
            }
        }
    }
    result = assess_public_evidence(candidate, {"false_acceptance_rate": 0.08}, None)
    assert result["decision"] == "blocked"
    assert not result["checks"]["beavertails_false_acceptance"]["passed"]
