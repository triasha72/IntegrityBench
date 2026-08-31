"""Checkpoint-backed moderation inference with no source text logging."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from integritybench.thresholds import threshold_predictions


class ModerationPredictor:
    def __init__(self, model_path: Path, expected_sha256: str, model_id: str):
        path = Path(model_path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("model SHA-256 does not match registry")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.thresholds = bundle["thresholds"]
        self.model_id = model_id

    def __call__(self, text: str) -> dict[str, object]:
        probabilities = self.model.predict_proba([text])
        decision = str(threshold_predictions(probabilities, self.thresholds)[0])
        return {
            "decision": decision,
            "confidence": float(np.max(probabilities[0])),
            "probabilities": {
                label: float(probabilities[0, index])
                for index, label in enumerate(("ALLOW", "ESCALATE", "REJECT"))
            },
            "review_required": decision == "ESCALATE",
            "model_id": self.model_id,
        }
