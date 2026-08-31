"""Real-world Civil Comments adapter with frozen three-way label mapping."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from integritybench.schemas import Decision, ModerationCase

DATASET_ID = "google/civil_comments"


@dataclass(frozen=True)
class CivilCommentsConfig:
    split: str = "validation"
    sample_size: int = 1_000
    seed: int = 42
    allow_max: float = 0.10
    reject_min: float = 0.50

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not 0 <= self.allow_max < self.reject_min <= 1:
            raise ValueError("Expected 0 <= allow_max < reject_min <= 1")


def decision_from_toxicity(
    score: float, allow_max: float = 0.10, reject_min: float = 0.50
) -> Decision:
    if not 0 <= score <= 1:
        raise ValueError("toxicity must be between zero and one")
    if score <= allow_max:
        return Decision.ALLOW
    if score >= reject_min:
        return Decision.REJECT
    return Decision.ESCALATE


def row_to_case(row: Mapping[str, Any], split: str, index: int) -> ModerationCase:
    text = str(row["text"]).strip()
    if not text:
        raise ValueError("comment text must not be empty")
    decision = decision_from_toxicity(float(row["toxicity"]))
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    restricted = decision is not Decision.ALLOW
    return ModerationCase(
        case_id=f"CC-{split}-{index:07d}-{digest}",
        policy_version="civil-comments-v1",
        slice_name="real_civil_comments",
        language="en",
        content=text,
        expected_decision=decision,
        expected_rule_id="P-TOX" if restricted else None,
        expected_remediation="Remove or rewrite the toxic content." if restricted else None,
    )


def sample_rows(
    rows: Iterable[Mapping[str, Any]], config: CivilCommentsConfig
) -> tuple[ModerationCase, ...]:
    rng = random.Random(config.seed)
    reservoir: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        if len(reservoir) < config.sample_size:
            reservoir.append((index, row))
        else:
            replacement = rng.randint(0, index)
            if replacement < config.sample_size:
                reservoir[replacement] = (index, row)
    if len(reservoir) < config.sample_size:
        raise ValueError("source split is smaller than requested sample")
    return tuple(row_to_case(row, config.split, index) for index, row in sorted(reservoir))
