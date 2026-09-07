"""Aggregate moderation decision errors by declared content category."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def summarize_errors(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize de-identified expected/predicted decisions without source text."""

    buckets: dict[str, Counter[str]] = {}
    total = 0
    for row in records:
        expected = row.get("expected_decision")
        predicted = row.get("predicted_decision")
        categories = row.get("categories", ["uncategorized"])
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise TypeError("Every record needs string expected_decision and predicted_decision")
        if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
            raise ValueError("categories must be a list of strings")
        outcome = (
            "false_accept" if expected == "REJECT" and predicted == "ALLOW"
            else "false_reject" if expected == "ALLOW" and predicted == "REJECT"
            else "escalated" if predicted == "ESCALATE"
            else "correct" if expected == predicted
            else "other_error"
        )
        for category in categories or ["uncategorized"]:
            buckets.setdefault(category, Counter())[outcome] += 1
        total += 1
    if not total:
        raise ValueError("At least one decision record is required")
    return {
        "schema_version": "integritybench.error-taxonomy.v1",
        "rows": total,
        "categories": {
            name: dict(sorted(counts.items())) for name, counts in sorted(buckets.items())
        },
        "contains_source_text": False,
    }
