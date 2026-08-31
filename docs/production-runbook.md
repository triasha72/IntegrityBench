# IntegrityBench production runbook

This runbook covers the moderation evaluation service, not a claim that the
current model is ready for production. Both checked-in candidates are rejected.

## Release sequence

1. Run the real-data candidate workflow and retain its data manifest, model,
   evaluation, and release assessment.
2. Stop if the release assessment says `rejected`.
3. Upload the model to the versioned private S3 bucket.
4. Register its URI and SHA-256. Promotion rechecks both the hash and approval.
5. Deploy a separate App Runner canary with automatic deployment disabled.
6. Replay a restricted real shadow sample with `scripts/canary_check.py`.
7. Check safety metrics, review load, 5xx errors, and P95/P99 latency.
8. Point the production registry at the new version only after review sign-off.

## Service objectives

- Availability target: 99.9% monthly for `/v1/moderate` after a model is loaded.
- P95 model-service latency target: 250 ms at the declared request size.
- 5xx alert: five responses in two consecutive one-minute periods.
- Safety and drift alerts use the stricter model-release and PSI thresholds in
  the tracked assessment, not availability as a substitute for model quality.

These are targets. They become measured claims only after a named environment
has retained monitoring data for at least 30 days.

## Rollback triggers

Roll back when any of these occurs:

- model hash or registry evidence cannot be verified;
- canary disagreement exceeds the approved change budget;
- false acceptance or calibration exceeds its release ceiling;
- decision-distribution PSI reaches 0.25;
- P95 latency breaches 250 ms for 15 minutes;
- review backlog exceeds the staffed capacity agreed before launch.

Run `integritybench-registry --registry <path> rollback`, redeploy, and verify
`/ready` reports the previous model ID. Preserve the failed version and incident
evidence; do not overwrite it.

## Incident exercise

Quarterly, stage a bad candidate with an intentionally incorrect hash. Confirm
that startup or promotion fails. Then stage an approved test fixture, promote a
second version, trigger the rollback command, and verify the original model is
restored. Record detection time, rollback time, and any manual steps.

## Privacy and access

The API does not return or log request text. The queue stores a restricted
content reference, model ID, and decisions. The referenced content belongs in a
separate access-controlled system with retention and deletion policies. S3
model objects are private and versioned. DynamoDB uses encryption and point-in-
time recovery.

## Cost inputs

Before deployment, use the AWS calculator with the configured one-vCPU/two-GB
App Runner instance, minimum and maximum instance counts, expected requests,
model storage, DynamoDB request volume, logs, and data transfer. Check the bill
after the first load test and again after the first week. No cost number is
claimed until an actual account and region are named.
