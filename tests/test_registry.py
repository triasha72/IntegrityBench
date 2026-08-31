import hashlib
import json

import pytest

from integritybench.registry import ModelRegistry


def test_registry_requires_approval_and_supports_rollback(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    approved = tmp_path / "approved.json"
    approved.write_text('{"decision":"approved"}')
    rejected = tmp_path / "rejected.json"
    rejected.write_text('{"decision":"rejected"}')
    for model_id, path in (("first", first), ("second", second)):
        registry.register(
            model_id, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "evidence"
        )
    with pytest.raises(ValueError, match="not approved"):
        registry.promote("first", first, rejected)
    registry.promote("first", first, approved)
    registry.promote("second", second, approved)
    assert registry.read()["production"] == "second"
    assert registry.rollback() == "first"


def test_registry_file_is_valid_json_after_atomic_writes(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("candidate", "s3://bucket/model", "a" * 64, "result.json")
    assert json.loads(registry.path.read_text())["models"]["candidate"]["status"] == "candidate"
