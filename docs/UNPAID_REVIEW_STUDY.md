# Volunteer review study

IntegrityBench needs two people to review the same cases independently. Public
toxicity labels help with external testing, but they do not answer the project's
policy-specific questions. This study is the no-budget route to that evidence.

## Who to ask

Start with people who already have a reason to care about the work: university
AI clubs, responsible-AI reading groups, open-source safety communities, and
trust-and-safety practitioners in your professional network. Ask publicly rather
than pressuring classmates or direct reports. Do not use Prolific for unpaid
work; its platform requires paid studies.

Reviewers need fluent written English and must be comfortable seeing offensive
or upsetting text. They do not need to be ML engineers. Do not recruit anyone
under 18.

## Pilot

Create the two blinded 24-case files:

```bash
python scripts/build_unpaid_review_pilot.py --output-dir outputs/volunteer_pilot
```

Send one file and `docs/annotation-guide.md` to each reviewer. Reviewers must not
discuss cases until both files are returned. The pilot does not satisfy the
100-case release gate. It checks whether the instructions and workload make
sense before asking for the full review.

## Outreach note

> I'm testing an open-source moderation benchmark and need two volunteers to
> review a short pilot independently. It has 24 short English examples and takes
> about 20-30 minutes. Some examples contain offensive or upsetting language.
> There is no payment, and you can stop at any time. I will credit you by name or
> pseudonym if you want, but I will not publish your personal details or raw
> reviewer identity. Your feedback will help me fix the instructions before the
> full study.

Record affirmative consent, the policy version, start and finish dates, and a
pseudonymous reviewer ID. Never commit names, email addresses, consent messages,
or unredacted reviewer metadata. If the project becomes formal academic research
or the results are submitted for publication, check the relevant ethics or IRB
requirements before recruitment.
