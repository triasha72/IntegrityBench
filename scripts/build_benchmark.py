"""Write the frozen IntegrityBench JSONL dataset."""

from pathlib import Path

from integritybench.dataset import write_jsonl

write_jsonl(Path("data/integritybench_v0_1.jsonl"))
