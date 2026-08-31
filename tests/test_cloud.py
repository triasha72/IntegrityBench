from pathlib import Path

from integritybench.cloud import DynamoReviewQueue, download_s3, parse_s3_uri


class S3Client:
    def download_file(self, bucket, key, destination):
        assert (bucket, key) == ("models", "candidate/model.joblib")
        Path(destination).write_bytes(b"model")


class Table:
    def __init__(self):
        self.items = {}

    def put_item(self, Item, **_kwargs):
        self.items[Item["case_id"]] = Item

    def update_item(self, Key, ExpressionAttributeValues, **_kwargs):
        item = self.items[Key["case_id"]]
        if item["status"] != "pending":
            raise RuntimeError("conditional check failed")
        item["status"] = "resolved"
        item["reviewer_decision"] = ExpressionAttributeValues[":decision"]

    def scan(self, **_kwargs):
        return {"Count": sum(item["status"] == "pending" for item in self.items.values())}


def test_s3_download_and_uri_validation(tmp_path):
    assert parse_s3_uri("s3://models/candidate/model.joblib") == (
        "models",
        "candidate/model.joblib",
    )
    output = download_s3("s3://models/candidate/model.joblib", tmp_path / "model", S3Client())
    assert output.read_bytes() == b"model"


def test_dynamo_review_queue_contract():
    table = Table()
    queue = DynamoReviewQueue("reviews", table)
    queue.enqueue("case", "vault://case", "model", "ESCALATE")
    assert queue.pending_count() == 1
    queue.resolve("case", "REJECT", "reviewer")
    assert queue.pending_count() == 0
