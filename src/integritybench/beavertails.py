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


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[BeaverTailsExample, ...]:
    """Normalize human-labelled prompt/response pairs without retaining source IDs."""

    examples: dict[str, BeaverTailsExample] = {}
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
        example = BeaverTailsExample(
            example_id=f"beavertails-330k-{digest}",
            text=text,
            safe=bool(row["is_safe"]),
            categories=categories,
        )
        previous = examples.get(example.example_id)
        if previous is not None and previous != example:
            raise ValueError("Conflicting BeaverTails labels found for repeated content")
        examples[example.example_id] = example
    if not examples:
        raise ValueError("No BeaverTails rows found")
    return tuple(examples.values())
