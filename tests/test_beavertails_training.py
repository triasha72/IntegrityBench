import json
import lzma
import subprocess
import sys
from pathlib import Path


def test_trainer_refuses_an_official_test_named_file(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl.xz"
    with lzma.open(source, "wt") as handle:
        handle.write(
            json.dumps({"prompt": "safe", "response": "fine", "is_safe": True, "category": {}})
        )
        handle.write("\n")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_beavertails_candidate.py",
            "--train",
            str(source),
            "--output",
            str(tmp_path / "record.json"),
            "--model-output",
            str(tmp_path / "model.joblib"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing" in result.stderr


def test_checked_in_beavertails_candidate_result_records_the_tradeoff() -> None:
    result = json.loads(
        Path("artifacts/beavertails_candidate_external_v1.json").read_text(encoding="utf-8")
    )
    assert result["rows"] == 11_088
    assert result["false_acceptance_rate"] < 0.20
    assert result["false_rejection_rate"] > 0.15
