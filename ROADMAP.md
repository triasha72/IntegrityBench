# Where IntegrityBench goes next

[Read the project overview and measured results](README.md)

The first real-data candidate is deliberately still blocked. On 97,320 held-out
Civil Comments it reduced false acceptance from `11.43%` to `1.84%`, but threat
false acceptance was `6.05%` and decision calibration error was `5.31%`. Both
miss the release limits. I would rather keep that visible than turn one improved
number into a production claim.

## What is already in place

- A public Civil Comments training and evaluation path with frozen splits.
- Validation-only threshold selection and a protected test evaluation.
- Overall, calibration, and content-type safety reports.
- A fail-closed API and a checksummed model registry with explicit approval and
  rollback.
- A metadata-only human-review queue, shadow comparison, container, load-test
  tool, telemetry hooks, and an AWS infrastructure plan.

The generated policy-shift benchmark remains useful for checking whether a
system follows the current rule. It is a deterministic control, not a substitute
for real comments.

An external human-reviewed shift check is now complete. On 2,802 deduplicated
ToxicChat test prompts, the frozen candidate falsely allowed `59.32%` of toxic
examples. A future candidate must still be selected on Civil Comments
development data and then evaluated once on the unchanged ToxicChat test set.
The source's non-commercial license and binary label space remain part of this
result's boundary.

## The next experiment

The compact-transformer training path is now implemented for exactly the same
public data and release gates. Its GPU command now fails before training when
CUDA is unavailable, so a CPU run cannot be mistaken for GPU evidence. It still
needs a real GPU run and a checked-in experiment record. The comparison will not
be limited to macro F1; the main question is whether it lowers threat false
acceptance without creating an unreasonable review burden.

After that, two people need to annotate the blinded review set independently.
Agreement and adjudication will show whether the policy labels themselves are
clear enough to support a meaningful benchmark.

The unpaid-review pilot builder and recruitment guide are complete. The next
human step is to find two willing reviewers, run the 24-case pilot, and revise
only unclear instructions before freezing the full study.

An alternative public-evidence path is also implemented. It combines Civil
Comments, ToxicChat, BeaverTails, and a clearly labelled author error audit. It
can support offline research evidence but cannot be described as independent
review or production approval.

The BeaverTails-only training experiment is now complete: it lowered held-out
false acceptance from 58.29% to 18.79%, but raised false rejection to 16.43%
and has no escalation option. The next model experiment is therefore a
three-way conversational candidate, selected without using the ToxicChat or
BeaverTails tests.

The combined release gate is implemented and currently blocks the candidate.
It cannot pass when the external shift result or independent human review is
missing, even if the in-domain aggregate metrics improve.

## Work that requires a real operating environment

- Run the stable and candidate models on approved shadow traffic and store only
  aggregate transitions.
- Measure latency, availability, and cost in the chosen cloud environment.
- Exercise rollback during a staged incident drill.
- Review retention, deletion, and access rules with the people responsible for
  privacy and moderation operations.

None of these will be marked complete from a local fixture or an estimated cloud
number.
