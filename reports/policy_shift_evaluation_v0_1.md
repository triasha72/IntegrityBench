# Policy-shift evaluation v0.1

## Problem

A static classifier can look accurate while applying an obsolete rule. This
study tests whether retrieving the current policy improves decisions when the
topic stays the same but the required action changes.

## Hypothesis

Current-policy retrieval should beat a prompt-only v1 baseline on policy-shift
cases. Evidence validation should also improve policy citations and make
unsupported decisions escalate instead of passing silently.

## Setup and baselines

The frozen dataset has 360 cases, 60 per slice and 120 per policy version. The
three deterministic systems receive the same content and are evaluated against
the version attached to each case. No model output was used to author labels.

## Results

The checked run produced:

| System | Macro F1 | False acceptance | False rejection | Escalation | Citation accuracy | Remediation accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Static v1 prompt control | 0.620 | 0.286 | 0.000 | 0.250 | 1.000 | 0.333 |
| Retrieved current policy | 1.000 | 0.000 | 0.000 | 0.350 | 1.000 | 1.000 |
| Retrieved + validated policy | 1.000 | 0.000 | 0.000 | 0.350 | 1.000 | 1.000 |

The result proves that the benchmark catches stale-policy decisions and that
all six slices execute against the output contract. It does not establish model
quality: the cases and lexical controls share the same controlled vocabulary.
The validated system is slower than direct retrieval in this local run because
it reranks every current rule, but sub-millisecond Python timing is not a useful
production latency claim.

## Ablations

Remove version filtering, remove reranking, disable evidence validation, and
evaluate each language, policy version, label budget, and adversarial transform
separately.

## Failure analysis and limits

The generator controls policy shifts cleanly but does not capture the ambiguity
or cultural context of real moderation queues. Lexical controls are expected to
struggle with unseen paraphrases. Multilingual templates are parallel cases,
not a claim of broad language coverage. Human review is required before using
the benchmark for product decisions.

## Next experiment

Freeze a human-reviewed v0.2 set, add prompt-only/RAG/LoRA model outputs, and run
the preregistered 8/32/128/full-label comparison. Report negative transfer if
LoRA improves standard cases but hurts policy-shift generalization.
