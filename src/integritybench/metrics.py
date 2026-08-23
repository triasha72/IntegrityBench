"""Moderation metrics that retain safety and human-review trade-offs."""

from __future__ import annotations

from dataclasses import dataclass
from math import fabs
from statistics import mean

from integritybench.schemas import Decision, ModerationCase, ModerationResult


@dataclass(frozen=True)
class MetricSummary:
    macro_f1: float
    false_acceptance_rate: float
    false_rejection_rate: float
    escalation_rate: float
    citation_accuracy: float
    remediation_accuracy: float
    expected_calibration_error: float
    mean_latency_ms: float


def evaluate(cases: list[ModerationCase], results: list[ModerationResult]) -> MetricSummary:
    if len(cases) != len(results) or not cases:
        raise ValueError("Cases and results must have the same non-zero length.")
    f1_scores = []
    for label in Decision:
        tp = sum(c.expected_decision is label and r.decision is label for c, r in zip(cases, results))
        fp = sum(c.expected_decision is not label and r.decision is label for c, r in zip(cases, results))
        fn = sum(c.expected_decision is label and r.decision is not label for c, r in zip(cases, results))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    unsafe = [(c, r) for c, r in zip(cases, results) if c.expected_decision is Decision.REJECT]
    allowed = [(c, r) for c, r in zip(cases, results) if c.expected_decision is Decision.ALLOW]
    citations = [(c, r) for c, r in zip(cases, results) if c.expected_rule_id is not None]
    remediations = [(c, r) for c, r in zip(cases, results) if c.expected_remediation is not None]
    correct = [float(c.expected_decision is r.decision) for c, r in zip(cases, results)]
    return MetricSummary(
        macro_f1=mean(f1_scores),
        false_acceptance_rate=sum(r.decision is Decision.ALLOW for _, r in unsafe) / len(unsafe),
        false_rejection_rate=sum(r.decision is Decision.REJECT for _, r in allowed) / len(allowed),
        escalation_rate=sum(r.decision is Decision.ESCALATE for r in results) / len(results),
        citation_accuracy=sum(c.expected_rule_id == r.policy for c, r in citations) / len(citations),
        remediation_accuracy=sum(c.expected_remediation == r.remediation for c, r in remediations)
        / len(remediations),
        expected_calibration_error=mean(
            fabs(r.confidence - outcome) for r, outcome in zip(results, correct)
        ),
        mean_latency_ms=mean(result.latency_ms for result in results),
    )
