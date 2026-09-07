from integritybench.error_taxonomy import summarize_errors


def test_error_taxonomy_keeps_false_accepts_by_category() -> None:
    result = summarize_errors([
        {"expected_decision": "REJECT", "predicted_decision": "ALLOW", "categories": ["threat"]},
        {"expected_decision": "ALLOW", "predicted_decision": "REJECT", "categories": ["quoted_content"]},
        {"expected_decision": "REJECT", "predicted_decision": "ESCALATE", "categories": ["threat"]},
    ])
    assert result["categories"]["threat"] == {"escalated": 1, "false_accept": 1}
    assert result["categories"]["quoted_content"] == {"false_reject": 1}
