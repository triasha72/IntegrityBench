from integritybench.monitoring import assess_civil_comments_release, population_stability_index


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
