# Mixed conversational candidate

This experiment asks a narrow question: does adding human-labelled
prompt-response data improve a three-way lexical safety candidate without using
any external test split for training or threshold selection?

It combines Civil Comments with the public BeaverTails training export. Civil
Comments provides `ALLOW`, `ESCALATE`, and `REJECT` labels derived from its
toxicity score. BeaverTails adds human-labelled prompt-response pairs as
`ALLOW` or `REJECT`. Thresholds are selected only on Civil Comments validation
data; BeaverTails and ToxicChat remain external checks.

## Reproduce

```bash
PYTHONPATH=src python scripts/download_civil_comments.py data/external/civil-comments
PYTHONPATH=src python scripts/download_beavertails.py data/external/beavertails
PYTHONPATH=src python scripts/train_conversational_candidate.py \
  --civil-train data/external/civil-comments/train.parquet \
  --civil-validation data/external/civil-comments/validation.parquet \
  --civil-test data/external/civil-comments/test.parquet \
  --beavertails-train data/external/beavertails/train.jsonl.xz \
  --output artifacts/conversational_candidate_v1.json \
  --model-output artifacts/conversational_candidate_v1.joblib
PYTHONPATH=src python scripts/evaluate_beavertails.py \
  --data data/external/beavertails/test.jsonl \
  --model artifacts/conversational_candidate_v1.joblib \
  --output artifacts/conversational_candidate_beavertails_external_v1.json
PYTHONPATH=src python scripts/evaluate_toxic_chat.py \
  --data data/external/toxic-chat-0124-test.csv \
  --model artifacts/conversational_candidate_v1.joblib \
  --output artifacts/conversational_candidate_toxicchat_external_v1.json
```

## Current run

The checked-in receipt used a 20,000-row training mix: 10,000 Civil Comments
rows and 10,000 BeaverTails rows. On the 97,320-row Civil Comments test set it
had a 2.29% false-acceptance rate and a 36.20% macro F1 score, while escalating
75.61% of cases. On the official 11,088-row BeaverTails test it had a 2.84%
false-acceptance rate, but rejected 61.97% of safe cases. On 2,802 human-reviewed
ToxicChat rows, false acceptance was 9.60%.

Those results are useful because they show the domain shift clearly. They are
not release evidence: the ToxicChat result and several Civil Comments safety
slices exceed the 2.5% safety target. The next experiment should improve the
representation or calibration, then repeat all three frozen checks. It must not
retune thresholds on the external test sets.

The receipts contain hashes, counts, thresholds, and aggregate metrics but no
source text. BeaverTails and ToxicChat are non-commercial research sources, so
this track cannot support a commercial moderation claim.
