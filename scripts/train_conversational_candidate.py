#!/usr/bin/env python3
"""Train a three-way candidate on Civil Comments and BeaverTails training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from integritybench.beavertails import DATASET_ID, DATASET_LICENSE, normalize_rows
from integritybench.thresholds import select_thresholds

try:
    from .train_civil_comments_baseline import SAFETY_ATTRIBUTES, digest, read_split
    from .train_civil_comments_candidate import evaluate
except ImportError:  # Direct execution: python scripts/train_conversational_candidate.py
    from train_civil_comments_baseline import SAFETY_ATTRIBUTES, digest, read_split
    from train_civil_comments_candidate import evaluate


def load_beavertails(path: Path, limit: int | None) -> pd.DataFrame:
    """Load only the public training split and map it to compatible end states."""

    if "test" in path.name.casefold():
        raise ValueError("Refusing to train on a file named as the official test split")
    with lzma.open(path, mode="rt", encoding="utf-8") as handle:
        examples = normalize_rows(json.loads(line) for line in handle if line.strip())
    if limit is not None:
        examples = examples[:limit]
    rows = [
        {"text": example.text, "label": "ALLOW" if example.safe else "REJECT"}
        for example in examples
    ]
    frame = pd.DataFrame(rows)
    if frame.empty or frame.label.nunique() != 2:
        raise ValueError("BeaverTails training data must contain safe and unsafe examples")
    return frame


def source_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("civil-train", "civil-validation", "civil-test", "beavertails-train"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--civil-train-limit", type=int, default=100_000)
    parser.add_argument("--civil-validation-limit", type=int, default=20_000)
    parser.add_argument("--beavertails-train-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    civil_train = read_split(args.civil_train, args.civil_train_limit, args.seed)
    civil_validation = read_split(
        args.civil_validation, args.civil_validation_limit, args.seed
    )
    civil_test = read_split(args.civil_test, None, args.seed)
    beavertails_train = load_beavertails(args.beavertails_train, args.beavertails_train_limit)
    train = pd.concat(
        [civil_train[["text", "label"]], beavertails_train], ignore_index=True
    ).sample(frac=1, random_state=args.seed)

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
    model.fit(train.text.astype(str).tolist(), train.label.tolist())

    validation_probabilities = model.predict_proba(civil_validation.text.astype(str).tolist())
    thresholds, selection = select_thresholds(
        civil_validation.label.tolist(),
        validation_probabilities,
        {
            attribute: civil_validation[attribute].to_numpy() >= 0.5
            for attribute in SAFETY_ATTRIBUTES
        },
        maximum_false_acceptance=0.025,
    )
    evaluations = {
        "civil_comments_validation": evaluate(model, civil_validation, thresholds),
        "civil_comments_test": evaluate(model, civil_test, thresholds),
    }
    evaluations["civil_comments_test"]["safety_slices"] = {
        attribute: evaluate(model, civil_test[civil_test[attribute] >= 0.5], thresholds)
        for attribute in SAFETY_ATTRIBUTES
        if np.any(civil_test[attribute] >= 0.5)
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": thresholds}, args.model_output)
    payload = {
        "schema_version": "1.0",
        "model": "character TF-IDF + class-balanced logistic regression + Civil validation thresholds",
        "seed": args.seed,
        "training_mix": {
            "civil_comments_rows": len(civil_train),
            "beavertails_rows": len(beavertails_train),
            "total_rows": len(train),
        },
        "sources": {
            "civil_comments": {"license": "CC0-1.0", "train_sha256": digest(args.civil_train)},
            "beavertails": {
                "dataset_id": DATASET_ID,
                "license": DATASET_LICENSE,
                "train_sha256": source_sha256(args.beavertails_train),
            },
        },
        "thresholds": thresholds.__dict__,
        "threshold_selection": selection,
        "model_artifact": {
            "path": args.model_output.name,
            "sha256": source_sha256(args.model_output),
        },
        "evaluations": evaluations,
        "contains_source_text": False,
        "limitations": [
            "BeaverTails contributes only ALLOW and REJECT examples; ESCALATE is calibrated on Civil Comments validation data.",
            "Thresholds are selected on Civil Comments validation data only and require external benchmark checks before any release claim.",
            "This lexical candidate is for reproducible comparison, not production moderation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluations["civil_comments_test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
