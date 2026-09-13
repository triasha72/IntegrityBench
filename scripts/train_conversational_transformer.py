#!/usr/bin/env python3
"""Train a cross-domain transformer without reading a protected test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from integritybench.beavertails import DATASET_ID, DATASET_LICENSE, normalize_rows
from integritybench.thresholds import select_thresholds, threshold_predictions

try:
    from .train_civil_comments_baseline import LABELS, SAFETY_ATTRIBUTES, digest, read_split
    from .train_civil_comments_candidate import decision_calibration_error
except ImportError:
    from train_civil_comments_baseline import LABELS, SAFETY_ATTRIBUTES, digest, read_split
    from train_civil_comments_candidate import decision_calibration_error


def source_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load_beavertails_train(path: Path, limit: int | None) -> pd.DataFrame:
    if "test" in path.name.casefold():
        raise ValueError("Refusing to train on a file named as an official test split")
    with lzma.open(path, mode="rt", encoding="utf-8") as handle:
        examples = normalize_rows(json.loads(line) for line in handle if line.strip())
    if limit is not None:
        examples = examples[:limit]
    result = pd.DataFrame(
        {"text": example.text, "label": "ALLOW" if example.safe else "REJECT"}
        for example in examples
    )
    if result.empty or result.label.nunique() != 2:
        raise ValueError("BeaverTails training data must contain safe and unsafe rows")
    return result


def evaluate(expected: np.ndarray, probabilities: np.ndarray, thresholds) -> dict[str, object]:
    predicted = threshold_predictions(probabilities, thresholds)
    allow = expected == "ALLOW"
    return {
        "rows": len(expected),
        "macro_f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                expected, predicted, labels=LABELS, average="macro", zero_division=0
            )
        ),
        "false_acceptance_rate": float(np.mean(predicted[expected == "REJECT"] == "ALLOW")),
        "false_rejection_rate": float(np.mean(predicted[allow] == "REJECT")),
        "expected_calibration_error": decision_calibration_error(
            expected, predicted, probabilities
        ),
        "escalation_rate": float(np.mean(predicted == "ESCALATE")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--civil-train", type=Path, required=True)
    parser.add_argument("--civil-validation", type=Path, required=True)
    parser.add_argument("--beavertails-train", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--civil-train-limit", type=int, default=100_000)
    parser.add_argument("--civil-validation-limit", type=int, default=20_000)
    parser.add_argument("--beavertails-train-limit", type=int, default=100_000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")
    civil_train = read_split(args.civil_train, args.civil_train_limit, args.seed)
    validation = read_split(args.civil_validation, args.civil_validation_limit, args.seed)
    beavertails = load_beavertails_train(args.beavertails_train, args.beavertails_train_limit)
    train = pd.concat([civil_train[["text", "label"]], beavertails], ignore_index=True).sample(
        frac=1, random_state=args.seed
    )
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class Dataset(Dataset):
        def __init__(self, frame):
            self.text = frame.text.astype(str).tolist()
            self.labels = [label_to_id[label] for label in frame.label]

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            result = tokenizer(self.text[index], truncation=True, max_length=256)
            result["labels"] = self.labels[index]
            return result

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        label2id=label_to_id,
        id2label={value: key for key, value in label_to_id.items()},
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
            save_total_limit=1,
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
        ),
        train_dataset=Dataset(train),
        eval_dataset=Dataset(validation),
        processing_class=tokenizer,
    )
    trainer.train()
    logits = trainer.predict(Dataset(validation)).predictions
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    thresholds, selection = select_thresholds(
        validation.label.tolist(),
        probabilities,
        {attribute: validation[attribute].to_numpy() >= 0.5 for attribute in SAFETY_ATTRIBUTES},
        maximum_false_acceptance=0.025,
    )
    args.model_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_output))
    tokenizer.save_pretrained(str(args.model_output))
    (args.model_output / "decision_thresholds.json").write_text(
        json.dumps(thresholds.__dict__, indent=2) + "\n"
    )
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available()
        else None,
    }
    payload = {
        "schema_version": "1.0",
        "model": "distilbert-base-uncased",
        "task": "three-way cross-domain moderation with validation-selected thresholds",
        "training_mix": {
            "civil_comments_rows": len(civil_train),
            "beavertails_rows": len(beavertails),
        },
        "sources": {
            "civil_comments_train_sha256": digest(args.civil_train),
            "beavertails_dataset": DATASET_ID,
            "beavertails_license": DATASET_LICENSE,
            "beavertails_train_sha256": source_sha256(args.beavertails_train),
        },
        "runtime_environment": runtime,
        "thresholds": thresholds.__dict__,
        "threshold_selection": selection,
        "civil_comments_validation": evaluate(
            validation.label.to_numpy(), probabilities, thresholds
        ),
        "contains_source_text": False,
        "limitations": [
            "This command never reads a Civil Comments or BeaverTails test split.",
            "It must be evaluated once on a held-out external test before any release decision.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["civil_comments_validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
