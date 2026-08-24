# IntegrityBench v0.2 annotation guide

Two annotators independently label the same blinded 120-case pack. They may
consult the versioned policy, but must not see v0.1 expected labels or each
other's decisions.

For every case, record the decision, rule ID, exact supporting passage, short
reason, remediation, review requirement, confidence, and any ambiguity. Use
`ESCALATE` when the available policy does not support a reliable allow/reject
decision. Do not infer a rule from a restricted word alone; quoted reporting,
education, and non-promotional discussion require their surrounding context.

After both files are complete, run:

```bash
python scripts/measure_annotation_agreement.py \
  data/annotation/annotator_a.jsonl \
  data/annotation/annotator_b.jsonl \
  --output artifacts/annotation/agreement_v0_2.json
```

Every disagreement must be adjudicated with a written note. The adjudicator
must not change policy text after seeing disagreements. The resulting v0.2 set
is not frozen until agreement, adjudication, and checksums are recorded.
