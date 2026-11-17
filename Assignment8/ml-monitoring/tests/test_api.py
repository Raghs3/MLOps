import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app  # noqa: E402

client = TestClient(app)
VALID = {"sepal_length": 5.1, "sepal_width": 3.5,
         "petal_length": 1.4, "petal_width": 0.2}


def metric(prefix):
    """Sum every sample on /metrics whose line starts with prefix."""
    total = 0.0
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(prefix):
            total += float(line.rsplit(" ", 1)[1])
    return total


def test_valid_prediction():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["prediction"] == "setosa"


def test_invalid_request_returns_422():
    assert client.post("/predict", json={"bad": 1}).status_code == 422


def test_metrics_exposed():
    text = client.get("/metrics").text
    for name in ("ml_prediction_requests_total", "ml_prediction_errors_total",
                 "ml_prediction_latency_seconds", "ml_predictions_total"):
        assert name in text


def test_errors_counter_increments_on_invalid_input():
    before = metric("ml_prediction_errors_total{")
    client.post("/predict", json={"bad": 1})
    assert metric("ml_prediction_errors_total{") == before + 1


def test_requests_counter_increments_on_valid_input():
    before = metric("ml_prediction_requests_total{")
    client.post("/predict", json=VALID)
    assert metric("ml_prediction_requests_total{") == before + 1
