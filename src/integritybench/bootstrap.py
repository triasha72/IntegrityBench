"""Environment-driven API startup that loads only the registry production model."""

from __future__ import annotations

import os
from pathlib import Path

from integritybench.api import create_app
from integritybench.registry import ModelRegistry
from integritybench.review import ReviewQueue
from integritybench.runtime import ModerationPredictor


def predictor_from_environment():
    registry_path = os.environ.get("INTEGRITYBENCH_REGISTRY")
    if not registry_path:
        return None
    registry = ModelRegistry(Path(registry_path)).read()
    model_id = registry.get("production")
    if not model_id:
        return None
    record = registry["models"][model_id]
    model_path = Path(record["model_uri"])
    return ModerationPredictor(model_path, record["sha256"], model_id)


def review_queue_from_environment():
    path = os.environ.get("INTEGRITYBENCH_REVIEW_DB")
    return ReviewQueue(Path(path)) if path else None


app = create_app(predictor_from_environment(), review_queue_from_environment())
