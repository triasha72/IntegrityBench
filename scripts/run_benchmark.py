"""Run all deterministic baseline systems and write slice-level results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from integritybench.dataset import build_cases
from integritybench.metrics import evaluate
from integritybench.systems import PromptOnlySystem, RetrievedPolicySystem, ValidatedPolicySystem

cases = list(build_cases())
output: dict[str, object] = {}


def deterministic_metrics(selected_cases, selected_results):
    values = asdict(evaluate(selected_cases, selected_results))
    values.pop("mean_latency_ms")
    return values


for system in (PromptOnlySystem(), RetrievedPolicySystem(), ValidatedPolicySystem()):
    system_results = [system.decide(case) for case in cases]
    output[system.name] = {
        "overall": deterministic_metrics(cases, system_results),
        "by_slice": {
            slice_name: deterministic_metrics(
                [case for case in cases if case.slice_name == slice_name],
                [
                    result
                    for case, result in zip(cases, system_results)
                    if case.slice_name == slice_name
                ],
            )
            for slice_name in sorted({case.slice_name for case in cases})
        },
    }
path = Path("artifacts/integritybench_v0_1_results.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
