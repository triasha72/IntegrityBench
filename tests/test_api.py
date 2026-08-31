from fastapi.testclient import TestClient

from integritybench.api import create_app
from integritybench.review import ReviewQueue


class Predictor:
    model_id = "approved-v1"

    def __call__(self, _text):
        return {
            "decision": "ESCALATE",
            "confidence": 0.5,
            "probabilities": {"ALLOW": 0.2, "ESCALATE": 0.5, "REJECT": 0.3},
            "review_required": True,
            "model_id": self.model_id,
        }


def test_api_fails_closed_without_approved_model():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert (
        client.post("/v1/moderate", json={"text": "hello", "content_reference": "ref"}).status_code
        == 503
    )


def test_api_routes_escalations_without_exposing_text(tmp_path):
    queue = ReviewQueue(tmp_path / "reviews.sqlite")
    client = TestClient(create_app(Predictor(), queue))
    response = client.post(
        "/v1/moderate", json={"text": "sensitive content", "content_reference": "vault://1"}
    )
    assert response.status_code == 200
    assert response.json()["review_required"] is True
    assert "sensitive content" not in response.text
    assert queue.pending_count() == 1
    assert client.get("/metrics").json()["decisions"] == {"ESCALATE": 1}
