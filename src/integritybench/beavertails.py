"""Contracts for the external BeaverTails safety benchmark."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

DATASET_ID = "PKU-Alignment/BeaverTails"
DATASET_CONFIG = "330k"
DATASET_LICENSE = "CC-BY-NC-4.0"


@dataclass(frozen=True)
class BeaverTailsExample:
    example_id: str
    text: str
    safe: bool
    categories: tuple[str, ...]
    annotation_count: int
    safe_label_agreement: float


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[BeaverTailsExample, ...]:
    """Normalize human-labelled prompt/response pairs without retaining source IDs."""

    grouped: dict[str, list[tuple[bool, tuple[str, ...], str]]] = {}
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if not prompt or not response:
            raise ValueError("BeaverTails rows require non-empty prompt and response fields")
        if "is_safe" not in row:
            raise ValueError("BeaverTails row is missing is_safe")
        categories_raw = row.get("category", {})
        if not isinstance(categories_raw, Mapping):
            raise TypeError("BeaverTails category must be a mapping")
        categories = tuple(
            sorted(str(name) for name, present in categories_raw.items() if bool(present))
        )
        text = f"User: {prompt}\nAssistant: {response}"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        grouped.setdefault(digest, []).append((bool(row["is_safe"]), categories, text))
    examples = []
    for digest, votes in grouped.items():
        safe_votes = sum(annotation[0] for annotation in votes)
        if safe_votes * 2 == len(votes):
            continue
        safe = safe_votes * 2 > len(votes)
        winning_votes = safe_votes if safe else len(votes) - safe_votes
        category_names = {name for _, categories, _ in votes for name in categories}
        categories = tuple(
            sorted(
                name
                for name in category_names
                if sum(name in annotation[1] for annotation in votes) * 2 > len(votes)
            )
        )
        examples.append(
            BeaverTailsExample(
                example_id=f"beavertails-330k-{digest}",
                text=votes[0][2],
                safe=safe,
                categories=categories,
                annotation_count=len(votes),
                safe_label_agreement=winning_votes / len(votes),
            )
        )
    if not examples:
        raise ValueError("No BeaverTails rows found")
    return tuple(examples)
