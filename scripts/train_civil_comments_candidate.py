#!/usr/bin/env python3
"""Train and serialize a safety-thresholded Civil Comments candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from integritybench.thresholds import (
    DecisionThresholds,
    false_acceptance_rate,
    select_thresholds,
    threshold_predictions,
)
from scripts.train_civil_comments_baseline import (
    LABELS,
    SAFETY_ATTRIBUTES,
    digest,
    read_split,
)


def decision_calibration_error(expected, predicted, probabilities, bins: int = 10) -> float:
    class_index = {name: index for index, name in enumerate(LABELS)}
    confidence = np.asarray(
        [probabilities[row, class_index[label]] for row, label in enumerate(predicted)]
    )
    correct = np.asarray(expected) == np.asarray(predicted)
    value = 0.0
    for lower, upper in pairwise(np.linspace(0.0, 1.0, bins + 1)):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            value += np.mean(selected) * abs(
                np.mean(correct[selected]) - np.mean(confidence[selected])
            )
    return float(value)


def evaluate(model, frame, thresholds: DecisionThresholds) -> dict[str, object]:
    expected = frame.label.to_numpy()
    probabilities = model.predict_proba(frame.text.astype(str).tolist())
    predicted = threshold_predictions(probabilities, thresholds)
    allowed = expected == "ALLOW"
    return {
        "rows": len(frame),
        "classification_report": classification_report(
            expected, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        "false_acceptance_rate": false_acceptance_rate(expected, predicted),
        "false_rejection_rate": (
            float(np.mean(predicted[allowed] == "REJECT")) if np.any(allowed) else None
        ),
        "expected_calibration_error": decision_calibration_error(
            expected, predicted, probabilities
        ),
        "escalation_rate": float(np.mean(predicted == "ESCALATE")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()
    train = read_split(args.train, args.train_limit, args.seed)
    validation = read_split(args.validation, args.validation_limit, args.seed)
    test = read_split(args.test, None, args.seed)
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=3,
                    class_weight="balanced",
                    max_iter=300,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    if args.reuse_model and args.model_output.exists():
        existing = joblib.load(args.model_output)
        model = existing["model"] if isinstance(existing, dict) else existing
    else:
        model.fit(train.text.astype(str).tolist(), train.label.tolist())
    validation_probabilities = model.predict_proba(validation.text.astype(str).tolist())
    safety_masks = {
        attribute: validation[attribute].to_numpy() >= 0.5 for attribute in SAFETY_ATTRIBUTES
    }
    thresholds, selection = select_thresholds(
        validation.label.tolist(),
        validation_probabilities,
        safety_masks,
        maximum_false_acceptance=0.025,
    )
    evaluations = {
        "validation": evaluate(model, validation, thresholds),
        "test": evaluate(model, test, thresholds),
    }
    evaluations["test"]["safety_slices"] = {
        attribute: evaluate(model, test[test[attribute] >= 0.5], thresholds)
        for attribute in SAFETY_ATTRIBUTES
        if np.any(test[attribute] >= 0.5)
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": thresholds}, args.model_output)
    model_sha = hashlib.sha256(args.model_output.read_bytes()).hexdigest()
    payload = {
        "schema_version": "1.0",
        "dataset_id": "google/civil_comments",
        "dataset_license": "CC0-1.0",
        "model": "character TF-IDF + class-balanced logistic regression + safety thresholds",
        "seed": args.seed,
        "training_rows": len(train),
        "thresholds": thresholds.__dict__,
        "threshold_selection": selection,
        "source_sha256": {
            name: digest(getattr(args, name)) for name in ("train", "validation", "test")
        },
        "model_artifact": {"path": args.model_output.name, "sha256": model_sha},
        "evaluations": evaluations,
        "contains_source_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(evaluations["test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
