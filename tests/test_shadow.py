from integritybench.shadow import compare_shadow


class Predictor:
    def __init__(self, decision):
        self.decision = decision

    def __call__(self, _text):
        return {"decision": self.decision, "review_required": self.decision == "ESCALATE"}


def test_shadow_report_retains_no_text():
    result = compare_shadow(Predictor("ALLOW"), Predictor("ESCALATE"), ["private text"])
    assert result["agreement_rate"] == 0
    assert result["candidate_review_rate"] == 1
    assert result["contains_source_text"] is False
    assert "private text" not in str(result)
