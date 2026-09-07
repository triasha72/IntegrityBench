#!/usr/bin/env python3
"""Train a BeaverTails-only safety candidate without touching its official test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from integritybench.beavertails import DATASET_ID, DATASET_LICENSE, normalize_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples(path: Path, limit: int | None) -> tuple[list[str], list[str]]:
    with lzma.open(path, mode="rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    examples = normalize_rows(rows)
    if limit is not None:
        examples = examples[:limit]
    texts = [example.text for example in examples]
    labels = ["ALLOW" if example.safe else "REJECT" for example in examples]
    if len(set(labels)) != 2:
        raise ValueError("Training data must contain both safe and unsafe examples")
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if "test" in args.train.name.casefold():
        raise ValueError("Refusing to train on a file named as the official test split")
    texts, labels = load_examples(args.train, args.train_limit)
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=75_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2,
                    class_weight="balanced",
                    max_iter=300,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(texts, labels)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, args.model_output)
    payload = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "dataset_license": DATASET_LICENSE,
        "model": "character TF-IDF + class-balanced binary logistic regression",
        "training_rows": len(texts),
        "seed": args.seed,
        "train_source_sha256": sha256(args.train),
        "model_sha256": sha256(args.model_output),
        "evaluation_policy": "Use only the official BeaverTails test split in evaluate_beavertails.py.",
        "contains_source_text": False,
        "limitations": [
            "This binary candidate cannot represent IntegrityBench's ESCALATE decision.",
            "It is a public-domain-shift experiment, not a replacement for policy-specific moderation.",
            "The CC-BY-NC-4.0 data cannot support a commercial release claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"training_rows": len(texts), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
