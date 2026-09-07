# Public evidence protocol

This track answers a narrower question than production approval: does the
moderator transfer across several independently human-labelled public datasets?

It uses Civil Comments, ToxicChat, and BeaverTails. Each stays in its original
role and retains its license. ToxicChat and BeaverTails are external tests, not
extra training rows. The combined check requires no more than 10% false
acceptance on either external set and preserves the existing Civil Comments
safety thresholds.

The project author also reviews a fixed 50-case error sample. That audit records
which errors occur and what changed afterward. It is useful engineering work,
but it is not independent review.

```bash
python scripts/build_author_error_audit.py \
  --output outputs/author_error_audit.jsonl

PYTHONPATH=src python scripts/assess_public_evidence.py \
  --candidate artifacts/civil_comments_candidate_v2.json \
  --toxic-chat artifacts/toxic_chat_external_v1.json \
  --beavertails artifacts/beavertails_external_v1.json \
  --author-audit outputs/author_error_audit_summary.json \
  --output artifacts/public_evidence_assessment_v1.json
```

The audit summary must state `reviewer_role: project_author`, `case_count: 50`,
and `complete: true`. This research assessment does not replace the stricter
production release gate or a live shadow test.
