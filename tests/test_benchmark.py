from integritybench.dataset import SLICES, build_cases
from integritybench.metrics import evaluate
from integritybench.schemas import Decision
from integritybench.systems import PromptOnlySystem, ValidatedPolicySystem


def test_frozen_dataset_shape_and_policy_changes() -> None:
    cases = build_cases()
    assert len(cases) == 360
    assert {case.slice_name for case in cases} == set(SLICES)
    alcohol = [case for case in cases if case.expected_rule_id == "P-ALC"]
    assert {case.expected_decision for case in alcohol} >= {
        Decision.ALLOW,
        Decision.REJECT,
        Decision.ESCALATE,
    }


def test_current_policy_beats_static_policy_on_policy_shift() -> None:
    cases = [case for case in build_cases() if case.slice_name == "policy_shift"]
    static = evaluate(cases, [PromptOnlySystem().decide(case) for case in cases])
    current = evaluate(cases, [ValidatedPolicySystem().decide(case) for case in cases])
    assert current.macro_f1 > static.macro_f1
    assert current.citation_accuracy >= static.citation_accuracy
