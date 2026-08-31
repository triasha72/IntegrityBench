# IntegrityBench

[Portfolio case study](https://triasha72.github.io/Portfolio/case-integritybench.html)

IntegrityBench asks a narrow question: can a moderation system follow the policy
that applies now, explain its decision, and know when a person should review the
case?

The benchmark contains 360 deterministic cases across standard, policy-shift,
long-context, multilingual, adversarial, and low-resource slices. Policy rules
change across v1, v2, and v3, so memorizing a topic label is not enough. The
same alcohol promotion can be allowed, rejected, or escalated depending on the
active version.

## Systems compared

1. `PromptOnlySystem` uses a static v1 policy. It is intentionally unable to
   track later rule changes.
2. `RetrievedPolicySystem` retrieves the current policy before deciding.
3. `ValidatedPolicySystem` retrieves and reranks current rules, checks that a
   decision has policy evidence, and returns `ALLOW`, `REJECT`, or `ESCALATE`
   with a rule ID, reason, evidence, confidence, remediation, and review flag.

The deterministic systems make the benchmark runnable without an API key. They
are controls, not substitutes for the later LLM/RAG/LoRA experiment.

## Run it

```bash
python -m pip install -e ".[dev]"
python scripts/build_benchmark.py
python scripts/build_annotation_pack.py
python scripts/run_benchmark.py
pytest
```

## What is measured

Macro F1, false acceptance, false rejection, escalation, calibration error,
policy citation accuracy, remediation accuracy, and latency are reported overall
and by slice. Failures are retained rather than dropped from the denominator.

See `reports/policy_shift_evaluation_v0_1.md` for the experiment record and
limitations.

## Deterministic control result

On the generated v0.1 controls, the static-policy baseline reached 0.620 macro
F1 and a 0.286 false-acceptance rate. Retrieving the active policy reached 1.000
macro F1 with no false acceptances or false rejections. This large gap is a
construction check: the dataset was designed so that policy version changes the
correct action. It shows that the evaluator detects stale-policy behavior. It
does not estimate performance on real moderation traffic.

## Human-review phase

The v0.2 protocol samples 120 blinded cases, balanced across all six slices,
for two independent annotators. The annotation guide defines evidence,
escalation, ambiguity, and adjudication rules; the agreement script compares
decision and policy-rule labels without exposing the deterministic v0.1 answer
key during review.

No human-agreement number is claimed yet. That result will be published only
after both annotation files are complete and disagreements have been
adjudicated. See `docs/annotation-guide.md` for the protocol.

## Real-data result: Civil Comments

IntegrityBench now includes a separate real-world track using the CC0
`google/civil_comments` dataset. Crowd toxicity scores map to `ALLOW` at or
below 0.10, `REJECT` at or above 0.50, and `ESCALATE` in the ambiguous middle
band. This track does not replace the generated policy-shift control because
Civil Comments has no versioned policy labels.

A TF-IDF bigram plus class-balanced logistic-regression baseline was trained on
a deterministic 100,000-row sample from the first published training shard.
Model choices were checked on 20,000 validation rows, then evaluated on all
97,320 test rows.

| Test metric | Result |
|---|---:|
| Accuracy | 0.6803 |
| Macro F1 | 0.5796 |
| ALLOW F1 | 0.8010 |
| ESCALATE F1 | 0.3946 |
| REJECT F1 | 0.5433 |

The weak escalation score is retained as a real limitation rather than hidden
by binary relabeling. The checksummed, text-free experiment record is at
`artifacts/civil_comments_baseline_v1.json`; source comments are not committed.

Reproduce with:

```bash
python3 -m pip install -e ".[real-data]"
python3 scripts/train_civil_comments_baseline.py \
  --train data/external/train-00000-of-00002.parquet \
  --validation data/external/validation-00000-of-00001.parquet \
  --test data/external/test-00000-of-00001.parquet
```
