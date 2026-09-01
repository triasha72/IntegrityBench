#!/usr/bin/env python3
"""Fine-tune a transformer on the public Civil Comments three-way task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from integritybench.thresholds import (
    false_acceptance_rate,
    select_thresholds,
    threshold_predictions,
)

try:
    from .train_civil_comments_baseline import LABELS, SAFETY_ATTRIBUTES, digest, read_split
    from .train_civil_comments_candidate import decision_calibration_error
except ImportError:  # Direct execution: python scripts/train_civil_comments_transformer.py
    from train_civil_comments_baseline import LABELS, SAFETY_ATTRIBUTES, digest, read_split
    from train_civil_comments_candidate import decision_calibration_error


def hash_model_directory(path: Path) -> str:
    """Hash model files in stable relative-path order."""
    value = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        value.update(item.relative_to(path).as_posix().encode())
        value.update(item.read_bytes())
    return value.hexdigest()


def evaluate(expected: np.ndarray, probabilities: np.ndarray, thresholds) -> dict[str, object]:
    predicted = threshold_predictions(probabilities, thresholds)
    allowed = expected == "ALLOW"
    return {
        "rows": len(expected),
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
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Keep the large training stack optional for benchmark and API users.
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    label_to_id = {label: index for index, label in enumerate(LABELS)}
    set_seed(args.seed)
    train = read_split(args.train, args.train_limit, args.seed)
    validation = read_split(args.validation, args.validation_limit, args.seed)
    test = read_split(args.test, None, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class CommentDataset(Dataset):
        def __init__(self, frame):
            self.text = frame.text.astype(str).tolist()
            self.labels = [label_to_id[label] for label in frame.label]

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            encoded = tokenizer(self.text[index], truncation=True, max_length=args.max_length)
            encoded["labels"] = self.labels[index]
            return encoded

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=label_to_id,
    )
    training_args = TrainingArguments(
        output_dir=str(args.model_output.parent / "trainer-state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=CommentDataset(train),
        eval_dataset=CommentDataset(validation),
        processing_class=tokenizer,
    )
    trainer.train()

    def predict_probabilities(frame) -> np.ndarray:
        logits = trainer.predict(CommentDataset(frame)).predictions
        return torch.softmax(torch.as_tensor(logits), dim=1).numpy()

    validation_probabilities = predict_probabilities(validation)
    thresholds, selection = select_thresholds(
        validation.label.tolist(),
        validation_probabilities,
        {attribute: validation[attribute].to_numpy() >= 0.5 for attribute in SAFETY_ATTRIBUTES},
        maximum_false_acceptance=0.025,
    )
    test_probabilities = predict_probabilities(test)
    test_result = evaluate(test.label.to_numpy(), test_probabilities, thresholds)
    test_result["safety_slices"] = {
        attribute: evaluate(
            test.loc[test[attribute] >= 0.5, "label"].to_numpy(),
            test_probabilities[test[attribute].to_numpy() >= 0.5],
            thresholds,
        )
        for attribute in SAFETY_ATTRIBUTES
        if np.any(test[attribute] >= 0.5)
    }

    args.model_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_output))
    tokenizer.save_pretrained(str(args.model_output))
    (args.model_output / "decision_thresholds.json").write_text(
        json.dumps(thresholds.__dict__, indent=2) + "\n"
    )
    payload = {
        "schema_version": "1.0",
        "dataset_id": "google/civil_comments",
        "dataset_license": "CC0-1.0",
        "model": args.model_name,
        "task": "three-way moderation with validation-selected safety thresholds",
        "seed": args.seed,
        "training_rows": len(train),
        "thresholds": thresholds.__dict__,
        "threshold_selection": selection,
        "source_sha256": {
            name: digest(getattr(args, name)) for name in ("train", "validation", "test")
        },
        "model_artifact": {
            "path": args.model_output.name,
            "sha256_tree": hash_model_directory(args.model_output),
        },
        "evaluations": {
            "validation": evaluate(
                validation.label.to_numpy(), validation_probabilities, thresholds
            ),
            "test": test_result,
        },
        "contains_source_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(test_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
