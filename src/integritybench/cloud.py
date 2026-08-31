"""AWS adapters kept behind small interfaces for local testing."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("expected s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def download_s3(uri: str, destination: Path, client=None) -> Path:
    if client is None:
        import boto3

        client = boto3.client("s3")
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


class DynamoReviewQueue:
    def __init__(self, table_name: str, table=None):
        if table is None:
            import boto3

            table = boto3.resource("dynamodb").Table(table_name)
        self.table = table

    def enqueue(self, case_id: str, content_reference: str, model_id: str, decision: str) -> None:
        self.table.put_item(
            Item={
                "case_id": case_id,
                "content_reference": content_reference,
                "model_id": model_id,
                "proposed_decision": decision,
                "status": "pending",
            },
            ConditionExpression="attribute_not_exists(case_id)",
        )

    def resolve(self, case_id: str, reviewer_decision: str, reviewer_id: str) -> None:
        try:
            self.table.update_item(
                Key={"case_id": case_id},
                UpdateExpression=(
                    "SET #status=:resolved, reviewer_decision=:decision, reviewer_id=:reviewer"
                ),
                ConditionExpression="#status=:pending",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":resolved": "resolved",
                    ":pending": "pending",
                    ":decision": reviewer_decision,
                    ":reviewer": reviewer_id,
                },
            )
        except Exception as exc:
            raise ValueError("pending review case was not found") from exc

    def pending_count(self) -> int:
        response = self.table.scan(
            Select="COUNT",
            FilterExpression="#status=:pending",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":pending": "pending"},
        )
        return int(response["Count"])
