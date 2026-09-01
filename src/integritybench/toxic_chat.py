"""Pinned ToxicChat external-evaluation contracts.

ToxicChat is never folded into the Civil Comments training split.  It is used
only to measure how a frozen moderator transfers to human-reviewed user/chatbot
conversations collected from a different product setting.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

DATASET_ID = "lmsys/toxic-chat"
DATASET_VERSION = "0124"
DATASET_LICENSE = "CC-BY-NC-4.0"


@dataclass(frozen=True)
class ToxicChatExample:
    example_id: str
    text: str
    toxic: bool
    jailbreaking: bool


def normalize_human_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[ToxicChatExample, ...]:
    """Keep only rows explicitly identified as human annotated by the source."""

    examples_by_id: dict[str, ToxicChatExample] = {}
    for row in rows:
        if not _as_bool(row.get("human_annotation", False)):
            continue
        text = str(row.get("user_input", "")).strip()
        if not text:
            raise ValueError("Human-annotated ToxicChat row has empty user_input")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        example = ToxicChatExample(
            example_id=f"toxicchat-0124-{digest}",
            text=text,
            toxic=_as_bool(row.get("toxicity", 0)),
            jailbreaking=_as_bool(row.get("jailbreaking", 0)),
        )
        previous = examples_by_id.get(example.example_id)
        if previous is not None and previous != example:
            raise ValueError("Conflicting human labels found for repeated ToxicChat text")
        examples_by_id[example.example_id] = example
    if not examples_by_id:
        raise ValueError("No explicitly human-annotated ToxicChat rows found")
    return tuple(examples_by_id.values())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false", ""}:
        return False
    raise ValueError(f"Expected a boolean-like value, received {value!r}")
