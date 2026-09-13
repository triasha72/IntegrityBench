# Transformer GPU run

This run compares a compact transformer with the existing lexical candidate on
the same Civil Comments train, validation, and protected test files. It is a
research comparison, not a release approval.

## Environment

Use a temporary Linux session with an NVIDIA CUDA GPU, Python 3.11 or 3.12, and
enough disk for the public dataset and model checkpoint. Kaggle is suitable when
its notebook accelerator is set to GPU. Do not upload the downloaded comment
text or model weights to this repository.

## Exact command

```bash
git clone https://github.com/triasha72/IntegrityBench.git
cd IntegrityBench
python -m pip install --upgrade pip
python -m pip install -e '.[real-data,transformer]'

PYTHONPATH=src python scripts/download_civil_comments.py data/external/civil-comments

PYTHONPATH=src python scripts/train_civil_comments_transformer.py \
  --train data/external/civil-comments/train.parquet \
  --validation data/external/civil-comments/validation.parquet \
  --test data/external/civil-comments/test.parquet \
  --model-output /tmp/integritybench-transformer-model \
  --output artifacts/civil_comments_transformer_gpu_v1.json \
  --require-cuda
```

`--require-cuda` stops before training unless PyTorch detects a named NVIDIA
CUDA device. Keep the generated JSON, the Civil Comments `manifest.json`, and
the terminal output. Do not rerun the protected test split while changing model
choices or thresholds.

## Publication check

Before committing the text-free JSON, confirm all of the following:

- `runtime_environment.cuda_available` is `true`;
- `runtime_environment.cuda_device_count` is at least `1` and the device name
  is present;
- the source SHA-256 values match the retained `manifest.json`;
- `contains_source_text` is `false`;
- validation selected the thresholds; and
- the test result is evaluated once with those frozen thresholds.

Then run the existing Civil Comments release assessment. A passing training job
does not make the model releasable: the safety, external-shift, and human-review
gates remain independent.

## External-shift check

Use the frozen model directory and its saved thresholds to evaluate ToxicChat.
This command does not train the model or choose thresholds again:

```bash
PYTHONPATH=src python scripts/evaluate_toxic_chat_transformer.py \
  --data path/to/toxicchat_human_annotated_test.csv \
  --model /tmp/integritybench-transformer-model \
  --output artifacts/civil_comments_transformer_toxicchat_external_v1.json
```

Keep the source CSV out of Git. Commit only the text-free evaluation receipt.
