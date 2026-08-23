"""Deterministic construction of a 360-case policy-change benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from integritybench.policies import rules_for_version
from integritybench.schemas import Decision, ModerationCase

SLICES = ("standard", "policy_shift", "long_context", "multilingual", "adversarial", "low_resource")
LANGUAGES = ("en", "es", "fr", "hi")

TRANSLATIONS = {
    "en": "Special {term} promotion for adults.",
    "es": "Oferta especial de {term} para adultos.",
    "fr": "Offre spéciale de {term} réservée aux adultes.",
    "hi": "वयस्कों के लिए {term} का विशेष प्रचार।",
}


def _content(slice_name: str, term: str, index: int, language: str) -> str:
    base = f"Limited-time {term} promotion for adults. Reference {index}."
    if slice_name == "long_context":
        filler = "Product specifications and delivery details. " * 80
        return f"{filler}{base}{filler}"
    if slice_name == "multilingual":
        return TRANSLATIONS[language].format(term=term)
    if slice_name == "adversarial":
        return base.replace(term, " ".join(term)).replace("promotion", "pr0m0tion")
    return base


def build_cases() -> tuple[ModerationCase, ...]:
    cases: list[ModerationCase] = []
    for slice_index, slice_name in enumerate(SLICES):
        for version_index, version in enumerate(("v1", "v2", "v3")):
            for case_index in range(20):
                rules = rules_for_version(version)
                rule = rules[case_index % len(rules)]
                language = LANGUAGES[case_index % len(LANGUAGES)] if slice_name == "multilingual" else "en"
                term = rule.terms[case_index % len(rule.terms)]
                action = rule.action
                if case_index % 5 == 4:
                    action = Decision.ALLOW
                    content = f"A neutral post about hiking boots. Reference {case_index}."
                    rule_id = None
                    remediation = None
                else:
                    content = _content(slice_name, term, case_index, language)
                    rule_id = rule.rule_id
                    remediation = rule.remediation
                cases.append(
                    ModerationCase(
                        case_id=f"IB-{slice_index:02d}-{version_index:02d}-{case_index:03d}",
                        policy_version=version,
                        slice_name=slice_name,
                        language=language,
                        content=content,
                        expected_decision=action,
                        expected_rule_id=rule_id,
                        expected_remediation=remediation,
                        label_budget=(8, 32, 128, 256)[case_index % 4] if slice_name == "low_resource" else None,
                    )
                )
    if len(cases) != 360 or len({case.case_id for case in cases}) != 360:
        raise AssertionError("Benchmark construction must yield 360 unique cases.")
    return tuple(cases)


def write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(case.to_dict(), ensure_ascii=False) for case in build_cases()) + "\n",
        encoding="utf-8",
    )
