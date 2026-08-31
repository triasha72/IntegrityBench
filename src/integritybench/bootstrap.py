"""Environment-driven API startup that loads only the registry production model."""

from __future__ import annotations

import os
from pathlib import Path

from integritybench.api import create_app
from integritybench.cloud import DynamoReviewQueue, download_s3
from integritybench.registry import ModelRegistry
from integritybench.review import ReviewQueue
from integritybench.runtime import ModerationPredictor


def predictor_from_environment():
    registry_source = os.environ.get("INTEGRITYBENCH_REGISTRY")
    if not registry_source:
        return None
    registry_path = (
        download_s3(registry_source, Path("/tmp/integritybench/registry.json"))
        if registry_source.startswith("s3://")
        else Path(registry_source)
    )
    registry = ModelRegistry(registry_path).read()
    model_id = registry.get("production")
    if not model_id:
        return None
    record = registry["models"][model_id]
    model_uri = record["model_uri"]
    model_path = (
        download_s3(model_uri, Path("/tmp/integritybench/model.joblib"))
        if model_uri.startswith("s3://")
        else Path(model_uri)
    )
    return ModerationPredictor(model_path, record["sha256"], model_id)


def review_queue_from_environment():
    table = os.environ.get("INTEGRITYBENCH_REVIEW_TABLE")
    if table:
        return DynamoReviewQueue(table)
    path = os.environ.get("INTEGRITYBENCH_REVIEW_DB")
    return ReviewQueue(Path(path)) if path else None


app = create_app(predictor_from_environment(), review_queue_from_environment())

if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
