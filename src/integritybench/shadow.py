"""Aggregate candidate-versus-production shadow comparisons without retaining text."""

from __future__ import annotations

from collections import Counter


def compare_shadow(production, candidate, texts: list[str]) -> dict[str, object]:
    disagreements = Counter()
    production_review = candidate_review = 0
    for text in texts:
        current = production(text)
        proposed = candidate(text)
        disagreements[(current["decision"], proposed["decision"])] += 1
        production_review += int(current["review_required"])
        candidate_review += int(proposed["review_required"])
    total = len(texts)
    return {
        "rows": total,
        "agreement_rate": 1.0
        - sum(count for (left, right), count in disagreements.items() if left != right) / total,
        "production_review_rate": production_review / total,
        "candidate_review_rate": candidate_review / total,
        "decision_transitions": {
            f"{left}->{right}": count for (left, right), count in sorted(disagreements.items())
        },
        "contains_source_text": False,
    }
