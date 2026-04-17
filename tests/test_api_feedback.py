"""Tests for feedback API endpoint."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from fastapi.testclient import TestClient
from spam_detector.api.main import app

client = TestClient(app)


def test_feedback_endpoint_valid():
    """Test submitting valid feedback."""
    # First make a prediction to get a prediction_id
    pred_response = client.post("/predict", json={"email_text": "test email"})
    assert pred_response.status_code == 200
    prediction_id = pred_response.json()["prediction_id"]

    # Submit feedback
    feedback_payload = {
        "prediction_id": prediction_id,
        "correct_label": "spam"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_feedback_endpoint_invalid_prediction_id():
    """Test feedback with non-existent prediction_id."""
    feedback_payload = {
        "prediction_id": "invalid_id",
        "correct_label": "spam"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 404


def test_feedback_endpoint_invalid_label():
    """Test feedback with invalid label."""
    pred_response = client.post("/predict", json={"email_text": "test email"})
    prediction_id = pred_response.json()["prediction_id"]

    feedback_payload = {
        "prediction_id": prediction_id,
        "correct_label": "invalid_label"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 422


def test_feedback_endpoint_missing_field():
    """Test feedback with missing required field."""
    payload = {"prediction_id": "some_id"}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 422
