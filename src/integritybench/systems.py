"""Three comparable moderation systems with increasing policy grounding."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from integritybench.policies import policy_rules, rules_for_version
from integritybench.schemas import Decision, ModerationCase, ModerationResult, PolicyRule


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _matches(rule: PolicyRule, content: str) -> bool:
    normalized = normalize(content)
    return any(normalize(term) in normalized for term in rule.terms)


def _result(
    rule: PolicyRule | None, *, started: float, evidence: Sequence[str]
) -> ModerationResult:
    if rule is None:
        return ModerationResult(
            decision=Decision.ALLOW,
            policy=None,
            reason="No restricted policy topic was detected.",
            evidence=tuple(evidence),
            confidence=0.82,
            remediation=None,
            review_required=False,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return ModerationResult(
        decision=rule.action,
        policy=rule.rule_id,
        reason=f"The content matches {rule.rule_id} under policy {rule.version}.",
        evidence=tuple(evidence),
        confidence=0.88 if evidence else 0.68,
        remediation=rule.remediation,
        review_required=rule.action is Decision.ESCALATE,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


class PromptOnlySystem:
    """Static-policy baseline that cannot follow later policy changes."""

    name = "prompt_only"

    def decide(self, case: ModerationCase) -> ModerationResult:
        started = time.perf_counter()
        match = next(
            (rule for rule in rules_for_version("v1") if _matches(rule, case.content)), None
        )
        return _result(match, started=started, evidence=())


class RetrievedPolicySystem:
    """Retrieve the current version, then apply the top lexical rule."""

    name = "retrieved_policy"

    def decide(self, case: ModerationCase) -> ModerationResult:
        started = time.perf_counter()
        match = next(
            (
                rule
                for rule in rules_for_version(case.policy_version)
                if _matches(rule, case.content)
            ),
            None,
        )
        return _result(match, started=started, evidence=() if match is None else (match.text,))


class ValidatedPolicySystem:
    """Retrieve, rerank, check evidence, decide, and validate the contract."""

    name = "validated_policy"

    def decide(self, case: ModerationCase) -> ModerationResult:
        started = time.perf_counter()
        candidates = [rule for rule in policy_rules() if rule.version == case.policy_version]
        ranked = sorted(
            candidates,
            key=lambda rule: sum(normalize(term) in normalize(case.content) for term in rule.terms),
            reverse=True,
        )
        match = ranked[0] if ranked and _matches(ranked[0], case.content) else None
        result = _result(match, started=started, evidence=() if match is None else (match.text,))
        if result.policy and not result.evidence:
            return ModerationResult(
                decision=Decision.ESCALATE,
                policy=result.policy,
                reason="A rule matched, but policy evidence was insufficient.",
                evidence=(),
                confidence=0.5,
                remediation=result.remediation,
                review_required=True,
                latency_ms=result.latency_ms,
            )
        return result
