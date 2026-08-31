"""Fail-closed moderation API for an approved registry model."""

from __future__ import annotations

import hashlib
from collections import Counter
from threading import Lock
from time import perf_counter

from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    content_reference: str = Field(min_length=1, max_length=500)


class ReviewResolution(BaseModel):
    decision: str = Field(pattern="^(ALLOW|ESCALATE|REJECT)$")
    reviewer_id: str = Field(min_length=1, max_length=100)


class RuntimeMetrics:
    def __init__(self):
        self.lock = Lock()
        self.counts = Counter()
        self.latency_total_ms = 0.0

    def observe(self, decision: str, latency_ms: float) -> None:
        with self.lock:
            self.counts[decision] += 1
            self.latency_total_ms += latency_ms

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            total = sum(self.counts.values())
            return {
                "requests": total,
                "decisions": dict(self.counts),
                "mean_latency_ms": self.latency_total_ms / total if total else None,
            }


def create_app(predictor=None, review_queue=None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install IntegrityBench with the api extra") from exc
    app = FastAPI(title="IntegrityBench moderation candidate", version="0.1.0")
    metrics = RuntimeMetrics()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if predictor is None:
            raise HTTPException(status_code=503, detail="No approved production model is loaded")
        return {"status": "ready", "model_id": predictor.model_id}

    @app.get("/metrics")
    def runtime_metrics():
        return metrics.snapshot()

    @app.post("/v1/moderate")
    def moderate(request: ModerationRequest):
        if predictor is None:
            raise HTTPException(status_code=503, detail="No approved production model is loaded")
        started = perf_counter()
        result = predictor(request.text)
        latency_ms = (perf_counter() - started) * 1000
        metrics.observe(result["decision"], latency_ms)
        case_id = hashlib.sha256(request.text.encode()).hexdigest()[:16]
        if result["review_required"] and review_queue is not None:
            review_queue.enqueue(
                case_id,
                request.content_reference,
                result["model_id"],
                result["decision"],
            )
        return {**result, "case_id": case_id, "latency_ms": latency_ms}

    @app.post("/v1/reviews/{case_id}/resolve")
    def resolve_review(case_id: str, resolution: ReviewResolution):
        if review_queue is None:
            raise HTTPException(status_code=503, detail="Review queue is not configured")
        try:
            review_queue.resolve(case_id, resolution.decision, resolution.reviewer_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"case_id": case_id, "status": "resolved"}

    return app
