# IntegrityBench

[Portfolio case study](https://triasha72.github.io/Portfolio/case-integritybench.html)

IntegrityBench started with a practical moderation question: can a system apply
the policy that is active now, explain why it made a decision, and recognize
when a person needs to step in?

The project now covers both evaluation and deployment. Its main model is trained
on Google's public Civil Comments dataset. The generated policy-shift cases are
still useful for testing rule changes, but I keep those results separate from
the real-data results.

Accuracy is only part of the story here. Allowing a threat is much more serious
than sending an ambiguous comment to review, so the release checks include
false acceptance, false rejection, calibration, threat performance, and the
rate at which work is passed to a person.

## Project story

**Situation.** A moderation model can look accurate while still allowing a
dangerous share of harmful comments. Policies also change, and some comments are
too ambiguous for an automatic decision.

**Task.** I set out to build an evaluation and release process that treats those
risks separately. The system needed to follow the current policy, reserve
uncertain cases for people, and prevent a weak model from reaching production.

**Action.** I created versioned policy tests, then trained a class-balanced
TF-IDF and logistic-regression baseline on public Civil Comments data. I chose
decision thresholds on validation data, kept the full test set protected, and
added calibration and threat-specific gates. I also built a fail-closed FastAPI
service, a checksummed model registry, review and shadow paths, rollback, load
testing, telemetry hooks, and an AWS deployment plan.

**Result.** On 97,320 held-out comments, the thresholded candidate cut false
acceptance from `11.43%` to `1.84%`. It still failed release review because
threat false acceptance was `6.05%` and decision calibration error was `5.31%`.
The most important result is therefore the blocked release: the process caught
a model that improved overall safety but was not ready for real moderation.

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

## Transformer comparison

`scripts/train_civil_comments_transformer.py` fine-tunes a compact Hugging Face
classifier on the same public rows, labels, and protected test split as the
lexical candidate. It also uses the same validation-only threshold search and
release policy, so the more complex model does not get an easier test.

The training path is ready, but it has not produced a checked-in GPU experiment
record. No transformer result is claimed until that run finishes and the
text-free artifact passes the existing release gates.

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
| False acceptance rate | 0.1143 |
| False rejection rate | 0.0247 |
| Expected calibration error | 0.0158 |

The weak escalation score is retained as a real limitation rather than hidden
by binary relabeling. The checksummed, text-free experiment record is at
`artifacts/civil_comments_baseline_v1.json`; source comments are not committed.

### Independent conversational shift check

I evaluated the frozen candidate, without retraining or threshold tuning, on
the public ToxicChat 0124 test split. After retaining only rows explicitly
marked as human annotated and deduplicating repeated prompts, 2,802 examples
remained. The candidate falsely allowed `59.32%` of toxic prompts, rejected
`2.94%` of safe prompts, and escalated `13.81%` overall.

This is a large domain-transfer failure, not evidence that ToxicChat and Civil
Comments share the same moderation policy. ToxicChat is CC-BY-NC-4.0 and uses
binary toxicity labels, so it cannot replace the planned policy-specific
two-rater study. Source text remains outside Git; the checksummed, text-free
result is in `artifacts/toxic_chat_external_v1.json`.

Safety-attribute slices are also reported for comments with source annotation
scores of at least 0.50. False acceptance was 0.1746 on obscene content, 0.2279
on threats, 0.0929 on insults, 0.0837 on identity attacks, and 0.2066 on
sexually explicit content. These failures show why this lexical baseline must
not be used as a production moderator. The distributed Hugging Face schema does
not include demographic identity-membership columns, so these are content-type
slices rather than demographic fairness claims.

The frozen release policy rejects this baseline: overall and slice false
acceptance must each be at most 5%, while calibration error and false rejection
must also remain at most 5%. The current model fails overall false acceptance
and the threat, identity-attack, and sexual-explicit gates. The tracked decision
is `artifacts/civil_comments_release_assessment_v1.json`; CI regenerates it so a
metric or policy change cannot silently convert a research result into an
approved deployment. Runtime monitoring uses decision-distribution PSI warning
and blocking thresholds of 0.10 and 0.25.

### Safety-thresholded candidate

A second measured candidate uses character n-grams for greater robustness to
spelling variation and selects allow/reject thresholds exclusively on validation
data under stricter 2.5% false-acceptance constraints. On the protected test it
reduced overall false acceptance from `0.1143` to `0.0184` and false rejection
from `0.0247` to `0.0171`. This came with a large operational cost: `48.1%` of
test comments were escalated and macro F1 fell to `0.5605`. Threat false
acceptance was `0.0605` and decision ECE was `0.0531`, narrowly failing the
frozen 5% release limits. The release gate therefore correctly rejects it.

The result is tracked at `artifacts/civil_comments_candidate_v2.json`. The
serialized model is deliberately excluded from Git because its learned
vocabulary is derived from source text; deployment packages must be placed in
access-controlled artifact storage and verified by the recorded SHA-256.

Reproduce with:

```bash
python3 -m pip install -e ".[real-data]"
python3 scripts/train_civil_comments_baseline.py \
  --train data/external/train-00000-of-00002.parquet \
  --validation data/external/validation-00000-of-00001.parquet \
  --test data/external/test-00000-of-00001.parquet
```

## From experiment to service

The repository now includes the parts needed to move an approved model into a
small production service without bypassing the safety evidence:

- `model-candidate.yml` downloads the public Civil Comments files, trains a
  candidate, applies the frozen release policy, and uploads a short-lived model
  package. It never promotes a model automatically.
- `ModelRegistry` records candidate versions, verifies model hashes, refuses a
  rejected release assessment, and keeps the previous production version for
  rollback.
- The FastAPI service returns 503 until the registry names an approved
  production model. It validates request size, reports model version and
  latency, and sends `ESCALATE` decisions to a review queue.
- The local review queue stores a restricted content reference rather than the
  comment text. Shadow comparisons retain only aggregate decision transitions.

Run the service locally:

```bash
python -m pip install -e ".[api,real-data]"
export INTEGRITYBENCH_REGISTRY=deploy/registry.json
export INTEGRITYBENCH_REVIEW_DB=artifacts/local/reviews.sqlite
uvicorn integritybench.bootstrap:app --host 0.0.0.0 --port 8000
```

With no approved model, `/health` remains available while `/ready` and
`/v1/moderate` fail closed. This is the expected state of the repository today:
both measured candidates are rejected, so neither is quietly presented as a
production moderator.

## Cloud and operations path

The AWS reference deployment uses App Runner for the API, a private versioned
S3 bucket for registry and model objects, and DynamoDB for review metadata.
Terraform also sets least-privilege runtime access, autoscaling limits, health
checks, and a 5xx alarm. The CI job validates Terraform and builds the container;
the manual cloud workflow creates a reviewable plan but never applies it.

`scripts/canary_check.py` compares stable and canary services on a supplied real
shadow sample without saving comments. `scripts/load_test.py` reports success
rate and P50/P95/P99 latency. Deployment steps, rollback triggers, privacy rules,
incident exercises, and cost inputs are in `docs/production-runbook.md`.

No AWS deployment, uptime, latency, or cost claim is made from this code alone.
Those results require an account, an approved model, a named environment, and
retained monitoring evidence.
