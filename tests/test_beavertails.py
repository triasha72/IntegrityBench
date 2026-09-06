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


def test_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="prompt and response"):
        normalize_rows([{"prompt": "", "response": "answer", "is_safe": True}])
    with pytest.raises(ValueError, match="is_safe"):
        normalize_rows([{"prompt": "question", "response": "answer"}])
