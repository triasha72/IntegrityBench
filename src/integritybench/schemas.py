"""Strict benchmark and moderation-output contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class PolicyRule:
    version: str
    rule_id: str
    topic: str
    action: Decision
    terms: tuple[str, ...]
    text: str
    remediation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModerationCase:
    case_id: str
    policy_version: str
    slice_name: str
    language: str
    content: str
    expected_decision: Decision
    expected_rule_id: str | None
    expected_remediation: str | None
    label_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModerationResult:
    decision: Decision
    policy: str | None
    reason: str
    evidence: tuple[str, ...]
    confidence: float
    remediation: str | None
    review_required: bool
    latency_ms: float = 0.0
    tokens_used: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one.")
        if self.decision is Decision.ESCALATE and not self.review_required:
            raise ValueError("Escalated decisions require review.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
