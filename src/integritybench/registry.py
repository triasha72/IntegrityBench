"""Small file-backed model registry with approval and rollback rules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "schema_version": "1.0",
                "production": None,
                "previous_production": None,
                "models": {},
            }
        return json.loads(self.path.read_text())

    def _write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.path)

    def register(self, model_id: str, model_uri: str, sha256: str, evidence: str) -> None:
        payload = self.read()
        if model_id in payload["models"]:
            raise ValueError(f"model already registered: {model_id}")
        payload["models"][model_id] = {
            "model_uri": model_uri,
            "sha256": sha256,
            "evidence": evidence,
            "status": "candidate",
        }
        self._write(payload)

    def promote(self, model_id: str, model_path: Path, assessment_path: Path) -> None:
        payload = self.read()
        if model_id not in payload["models"]:
            raise ValueError(f"unknown model: {model_id}")
        assessment = json.loads(Path(assessment_path).read_text())
        if assessment.get("decision") != "approved":
            raise ValueError("release assessment has not approved this model")
        actual = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if actual != payload["models"][model_id]["sha256"]:
            raise ValueError("model SHA-256 does not match registry")
        previous = payload.get("production")
        if previous:
            payload["models"][previous]["status"] = "previous_production"
        payload["previous_production"] = previous
        payload["production"] = model_id
        payload["models"][model_id]["status"] = "production"
        self._write(payload)

    def rollback(self) -> str:
        payload = self.read()
        previous = payload.get("previous_production")
        current = payload.get("production")
        if not previous or not current:
            raise ValueError("no previous production model is available")
        payload["models"][current]["status"] = "rolled_back"
        payload["models"][previous]["status"] = "production"
        payload["production"] = previous
        payload["previous_production"] = current
        self._write(payload)
        return previous
