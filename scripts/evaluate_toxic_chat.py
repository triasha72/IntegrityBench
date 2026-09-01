#!/usr/bin/env python3
"""Evaluate a frozen Civil Comments model on human-reviewed ToxicChat rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from integritybench.toxic_chat import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_VERSION,
    normalize_human_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.data.open(newline="", encoding="utf-8") as stream:
        examples = normalize_human_rows(csv.DictReader(stream))
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    if not hasattr(model, "predict"):
        raise ValueError("Model artifact does not contain a predictor")
    predicted = np.asarray(model.predict([example.text for example in examples]))
    expected = np.asarray(["REJECT" if example.toxic else "ALLOW" for example in examples])
    labels = ["ALLOW", "ESCALATE", "REJECT"]
    toxic = expected == "REJECT"
    safe = expected == "ALLOW"
    payload = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "dataset_license": DATASET_LICENSE,
        "evaluation_role": "external_human_reviewed_distribution_shift",
        "rows": len(examples),
        "source_sha256": sha256(args.data),
        "model_sha256": sha256(args.model),
        "classification_report": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
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
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "rows",
                    "false_acceptance_rate",
                    "false_rejection_rate",
                    "escalation_rate",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
