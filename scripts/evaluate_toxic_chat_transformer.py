#!/usr/bin/env python3
"""Evaluate a frozen transformer on human-reviewed ToxicChat rows.

The model directory must contain the tokenizer, classifier weights, and the
``decision_thresholds.json`` written by ``train_civil_comments_transformer.py``.
No model fitting or threshold selection happens in this command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from integritybench.thresholds import DecisionThresholds, threshold_predictions
from integritybench.toxic_chat import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_VERSION,
    normalize_human_rows,
)

LABELS = ["ALLOW", "ESCALATE", "REJECT"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def predict_decisions(model_path: Path, texts: list[str], batch_size: int) -> np.ndarray:
    """Run the frozen classifier and apply its saved three-way thresholds."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    thresholds_path = model_path / "decision_thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing saved decision thresholds: {thresholds_path}")
    thresholds = DecisionThresholds(**json.loads(thresholds_path.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    labels_by_id = {int(key): value for key, value in model.config.id2label.items()}
    if [labels_by_id[index] for index in range(len(LABELS))] != LABELS:
        raise ValueError(f"Expected classifier labels {LABELS}, got {labels_by_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(batches)
    return threshold_predictions(probabilities, thresholds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    with args.data.open(newline="", encoding="utf-8") as stream:
        examples = normalize_human_rows(csv.DictReader(stream))
    predicted = predict_decisions(args.model, [example.text for example in examples], args.batch_size)
    expected = np.asarray(["REJECT" if example.toxic else "ALLOW" for example in examples])
    toxic = expected == "REJECT"
    safe = expected == "ALLOW"
    payload = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "dataset_license": DATASET_LICENSE,
        "evaluation_role": "external_human_reviewed_distribution_shift",
        "model_type": "transformer",
        "rows": len(examples),
        "source_sha256": sha256(args.data),
        "model_sha256_tree": hash_model_directory(args.model),
        "classification_report": classification_report(
            expected, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        "false_acceptance_rate": float(np.mean(predicted[toxic] == "ALLOW")),
        "false_rejection_rate": float(np.mean(predicted[safe] == "REJECT")),
        "escalation_rate": float(np.mean(predicted == "ESCALATE")),
        "contains_source_text": False,
        "limitations": [
            "ToxicChat is conversational toxicity evidence, not a substitute for policy-specific human adjudication.",
            "The CC-BY-NC-4.0 source limits this track to non-commercial research use.",
            "Binary ToxicChat labels cannot validate IntegrityBench's three-way escalation policy directly.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("rows", "false_acceptance_rate", "false_rejection_rate", "escalation_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
