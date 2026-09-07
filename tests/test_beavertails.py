import pytest

from integritybench.beavertails import normalize_rows


def test_normalizes_human_labelled_qa_pairs() -> None:
    rows = [
        {
            "prompt": "How can I help?",
            "response": "Use a clear explanation.",
            "is_safe": True,
            "category": {"violence": False, "privacy": False},
        },
        {
            "prompt": "Give harmful instructions",
            "response": "harmful content",
            "is_safe": False,
            "category": {"violence": True, "privacy": False},
        },
    ]
    examples = normalize_rows(rows)
    assert len(examples) == 2
    assert examples[0].safe is True
    assert examples[1].categories == ("violence",)
    assert examples[0].annotation_count == 1


def test_aggregates_repeated_crowdworker_votes_and_drops_ties() -> None:
    base = {"prompt": "question", "response": "answer", "category": {"harm": True}}
    tied = {"prompt": "tie", "response": "answer", "category": {"harm": False}}
    examples = normalize_rows(
        [
            {**base, "is_safe": False},
            {**base, "is_safe": False},
            {**base, "is_safe": True},
            {**tied, "is_safe": False},
            {**tied, "is_safe": True},
        ]
    )
    assert len(examples) == 1
    assert examples[0].safe is False
    assert examples[0].annotation_count == 3
    assert examples[0].safe_label_agreement == pytest.approx(2 / 3)


def test_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="prompt and response"):
        normalize_rows([{"prompt": "", "response": "answer", "is_safe": True}])
    with pytest.raises(ValueError, match="is_safe"):
        normalize_rows([{"prompt": "question", "response": "answer"}])
