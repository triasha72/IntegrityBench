#!/usr/bin/env python3
"""Evaluate a frozen moderator on human-labelled BeaverTails QA pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from integritybench.beavertails import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    normalize_rows,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="BeaverTails JSONL export")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    examples = normalize_rows(rows)
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    if not hasattr(model, "predict"):
        raise ValueError("Model artifact does not contain a predictor")
    predicted = np.asarray(model.predict([example.text for example in examples]))
    expected = np.asarray(["ALLOW" if example.safe else "REJECT" for example in examples])
    labels = ["ALLOW", "ESCALATE", "REJECT"]
    unsafe = expected == "REJECT"
    safe = expected == "ALLOW"
    category_totals: dict[str, int] = {}
    category_false_accepts: dict[str, int] = {}
    for example, decision in zip(examples, predicted, strict=True):
        if example.safe:
            continue
        for category in example.categories:
            category_totals[category] = category_totals.get(category, 0) + 1
            if decision == "ALLOW":
                category_false_accepts[category] = category_false_accepts.get(category, 0) + 1
    payload = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_license": DATASET_LICENSE,
        "evaluation_role": "external_human_labelled_qa_safety_shift",
        "rows": len(examples),
        "source_sha256": sha256(args.data),
        "model_sha256": sha256(args.model),
        "classification_report": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "false_acceptance_rate": float(np.mean(predicted[unsafe] == "ALLOW")),
        "false_rejection_rate": float(np.mean(predicted[safe] == "REJECT")),
        "escalation_rate": float(np.mean(predicted == "ESCALATE")),
        "false_acceptance_rate_by_harm_category": {
            category: category_false_accepts.get(category, 0) / total
            for category, total in sorted(category_totals.items())
        },
        "contains_source_text": False,
        "limitations": [
            "BeaverTails labels concern prompt-response safety, not IntegrityBench rule IDs.",
            "The CC-BY-NC-4.0 dataset is restricted to non-commercial research use.",
            "This external benchmark cannot replace two independent policy reviewers.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"rows": payload["rows"], "false_acceptance_rate": payload["false_acceptance_rate"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
