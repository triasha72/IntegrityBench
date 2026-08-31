#!/usr/bin/env python3
"""Train TF-IDF logistic regression on real Civil Comments data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from integritybench.civil_comments import decision_from_toxicity

LABELS = ["ALLOW", "ESCALATE", "REJECT"]


def read_split(path: Path, limit: int | None, seed: int):
    frame = pq.read_table(path, columns=["text", "toxicity"]).to_pandas().dropna()
    frame["label"] = frame.toxicity.map(lambda score: decision_from_toxicity(float(score)).value)
    if limit and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)
    return frame.text.astype(str).tolist(), frame.label.tolist()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/civil_comments_baseline_v1.json"))
    args = parser.parse_args()
    train_x, train_y = read_split(args.train, args.train_limit, args.seed)
    val_x, val_y = read_split(args.validation, args.validation_limit, args.seed)
    test_x, test_y = read_split(args.test, None, args.seed)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=2, class_weight="balanced", max_iter=300, random_state=args.seed)),
    ])
    model.fit(train_x, train_y)
    evaluations = {}
    for name, features, expected in (("validation", val_x, val_y), ("test", test_x, test_y)):
        predicted = model.predict(features)
        evaluations[name] = {
            "rows": len(expected),
            "classification_report": classification_report(expected, predicted, labels=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        }
    payload = {
        "schema_version": "1.0",
        "dataset_id": "google/civil_comments",
        "dataset_license": "CC0-1.0",
        "model": "TF-IDF bigrams + class-balanced logistic regression",
        "label_mapping": {"allow_max": 0.10, "reject_min": 0.50},
        "seed": args.seed,
        "training_rows": len(train_y),
        "source_sha256": {name: digest(getattr(args, name)) for name in ("train", "validation", "test")},
        "evaluations": evaluations,
        "contains_source_text": False,
        "limitations": ["Lexical baseline only; not a production moderation system.", "Training uses a deterministic sample from the first published training shard."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(evaluations["test"]["classification_report"]["macro avg"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
