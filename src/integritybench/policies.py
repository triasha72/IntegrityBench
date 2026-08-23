"""Versioned policy definitions used by the frozen benchmark."""

from __future__ import annotations

from integritybench.schemas import Decision, PolicyRule


def policy_rules() -> tuple[PolicyRule, ...]:
    """Return deliberately changing policies with stable rule identifiers."""

    rows = [
        ("v1", "P-ALC", "alcohol", Decision.ALLOW, ("wine", "beer", "whisky"),
         "Alcohol advertising is allowed when it does not target minors.", "Remove youth targeting."),
        ("v2", "P-ALC", "alcohol", Decision.REJECT, ("wine", "beer", "whisky"),
         "Paid alcohol promotion is prohibited.", "Remove the alcohol promotion."),
        ("v3", "P-ALC", "alcohol", Decision.ESCALATE, ("wine", "beer", "whisky"),
         "Alcohol references require review unless clearly educational.", "Clarify the educational context."),
        ("v1", "P-GAM", "gambling", Decision.REJECT, ("casino", "bet", "jackpot"),
         "Gambling promotion is prohibited.", "Remove gambling promotion and incentives."),
        ("v2", "P-GAM", "gambling", Decision.ESCALATE, ("casino", "bet", "jackpot"),
         "Licensed gambling content requires human review.", "Add licensing and age-gating details."),
        ("v3", "P-GAM", "gambling", Decision.REJECT, ("casino", "bet", "jackpot"),
         "Gambling promotion and referral incentives are prohibited.", "Remove the gambling offer."),
        ("v1", "P-MED", "medical", Decision.ESCALATE, ("cure", "treatment", "diagnosis"),
         "Medical efficacy claims require substantiation review.", "Add evidence and remove guarantees."),
        ("v2", "P-MED", "medical", Decision.REJECT, ("cure", "treatment", "diagnosis"),
         "Guaranteed medical outcomes are prohibited.", "Replace the guarantee with a supported claim."),
        ("v3", "P-MED", "medical", Decision.ESCALATE, ("cure", "treatment", "diagnosis"),
         "Medical claims require review for evidence and risk disclosure.", "Add evidence and risk disclosure."),
    ]
    return tuple(PolicyRule(*row) for row in rows)


def rules_for_version(version: str) -> tuple[PolicyRule, ...]:
    return tuple(rule for rule in policy_rules() if rule.version == version)
