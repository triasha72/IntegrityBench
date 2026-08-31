"""Metadata-only human review queue backed by SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class ReviewQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                case_id TEXT PRIMARY KEY, content_reference TEXT NOT NULL,
                model_id TEXT NOT NULL, proposed_decision TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', reviewer_decision TEXT,
                reviewer_id TEXT)"""
            )

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def enqueue(self, case_id: str, content_reference: str, model_id: str, decision: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO reviews(case_id, content_reference, model_id, proposed_decision) VALUES (?, ?, ?, ?)",
                (case_id, content_reference, model_id, decision),
            )

    def resolve(self, case_id: str, reviewer_decision: str, reviewer_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE reviews SET status='resolved', reviewer_decision=?, reviewer_id=? WHERE case_id=? AND status='pending'",
                (reviewer_decision, reviewer_id, case_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("pending review case was not found")

    def pending_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE status='pending'"
                ).fetchone()[0]
            )
