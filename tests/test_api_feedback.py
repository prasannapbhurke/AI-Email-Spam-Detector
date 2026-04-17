"""Tests for feedback API endpoint."""

from fastapi.testclient import TestClient
from spam_detector.api.main import app


def test_feedback_endpoint_valid(client: TestClient):
    """Test submitting valid feedback."""
    # First make a prediction to get a prediction_id
    pred_response = client.post("/predict", json={"email_text": "test email"})
    assert pred_response.status_code == 200
    prediction_id = pred_response.json()["prediction_id"]

    # Submit feedback
    feedback_payload = {
        "prediction_id": prediction_id,
        "label": "spam"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == prediction_id
    assert data["label"] == "spam"
    assert "feedback_at" in data


def test_feedback_endpoint_invalid_prediction_id(client: TestClient):
    """Test feedback with non-existent prediction_id (valid format but not found)."""
    feedback_payload = {
        "prediction_id": "507f1f77bcf86cd799439011",  # valid ObjectId format, likely not in DB
        "label": "spam"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 404


def test_feedback_endpoint_invalid_label(client: TestClient):
    """Test feedback with invalid label."""
    pred_response = client.post("/predict", json={"email_text": "test email"})
    prediction_id = pred_response.json()["prediction_id"]

    feedback_payload = {
        "prediction_id": prediction_id,
        "label": "invalid_label"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 422


def test_feedback_endpoint_missing_field(client: TestClient):
    """Test feedback with missing required field."""
    payload = {"prediction_id": "some_id"}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 422
