# BeaverTails external safety experiment

BeaverTails provides human-labelled prompt-response safety examples. This track
tests whether a simple model can learn the conversational format that Civil
Comments does not contain. It does not replace IntegrityBench's three-way policy
model, because BeaverTails only supplies a safe/unsafe label.

Download only the training split, train the candidate, then evaluate it once on
the official held-out test export:

```bash
PYTHONPATH=src python scripts/download_beavertails.py data/external/beavertails
PYTHONPATH=src python scripts/train_beavertails_candidate.py \
  --train data/external/beavertails/train.jsonl.xz \
  --model-output artifacts/beavertails_candidate_v1.joblib \
  --output artifacts/beavertails_candidate_v1.json
PYTHONPATH=src python scripts/evaluate_beavertails.py \
  --data data/external/beavertails/test.jsonl \
  --model artifacts/beavertails_candidate_v1.joblib \
  --output artifacts/beavertails_candidate_external_v1.json
```

The trainer refuses a filename that identifies itself as the official test split.
The reported model is binary, so it cannot be promoted as the three-way
IntegrityBench moderator. Its purpose is to measure whether the large transfer
failure came from using a comment-only source or from a broader safety-modeling
gap.
