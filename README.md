# IntegrityBench

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
