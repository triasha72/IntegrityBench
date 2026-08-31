import pytest

from integritybench.review import ReviewQueue


def test_review_queue_stores_reference_and_resolution(tmp_path):
    queue = ReviewQueue(tmp_path / "reviews.sqlite")
    queue.enqueue("case-1", "s3://restricted/content/1", "model-1", "ESCALATE")
    assert queue.pending_count() == 1
    queue.resolve("case-1", "REJECT", "reviewer-7")
    assert queue.pending_count() == 0
    with pytest.raises(ValueError, match="not found"):
        queue.resolve("case-1", "ALLOW", "reviewer-8")
