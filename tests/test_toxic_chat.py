import pytest

from integritybench.toxic_chat import normalize_human_rows


def test_normalization_keeps_only_explicit_human_annotations() -> None:
    rows = [
        {
            "user_input": "reviewed safe",
            "human_annotation": "true",
            "toxicity": "0",
            "jailbreaking": "0",
        },
        {
            "user_input": "model labeled",
            "human_annotation": "false",
            "toxicity": "1",
            "jailbreaking": "0",
        },
        {
            "user_input": "reviewed toxic",
            "human_annotation": "1",
            "toxicity": "1",
            "jailbreaking": "1",
        },
    ]
    examples = normalize_human_rows(rows)
    assert [example.text for example in examples] == ["reviewed safe", "reviewed toxic"]
    assert examples[1].toxic is True
    assert examples[1].jailbreaking is True


def test_normalization_rejects_unverified_and_deduplicates_matching_evidence() -> None:
    with pytest.raises(ValueError, match="No explicitly human-annotated"):
        normalize_human_rows([{"user_input": "x", "human_annotation": 0}])
    duplicate = {"user_input": "same", "human_annotation": 1, "toxicity": 0}
    assert len(normalize_human_rows([duplicate, duplicate])) == 1


def test_normalization_rejects_conflicting_human_labels() -> None:
    safe = {"user_input": "same", "human_annotation": 1, "toxicity": 0}
    toxic = {"user_input": "same", "human_annotation": 1, "toxicity": 1}
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_human_rows([safe, toxic])
