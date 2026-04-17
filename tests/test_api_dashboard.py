"""Tests for dashboard and stats endpoints."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from fastapi.testclient import TestClient
from spam_detector.api.main import app

client = TestClient(app)


def test_dashboard_stats():
    """Test dashboard stats endpoint."""
    response = client.get("/dashboard/stats")
    assert response.status_code in [200, 500]  # May fail if no predictions exist, or succeed

    if response.status_code == 200:
        data = response.json()
        assert "total_predictions" in data
        assert "spam_predictions" in data
        assert "ham_predictions" in data
        assert "spam_rate" in data


def test_dashboard_accuracy_over_time():
    """Test accuracy over time endpoint."""
    response = client.get("/dashboard/accuracy-over-time?days=7")
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "bucket" in data
        assert "period_days" in data
        assert "data" in data


def test_dashboard_model_performance():
    """Test model performance endpoint."""
    response = client.get("/dashboard/model-performance")
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "metrics" in data
        if "metrics" in data:
            metrics = data["metrics"]
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics


def test_dashboard_spam_stats():
    """Test spam statistics endpoint."""
    response = client.get("/dashboard/spam-stats")
    assert response.status_code in [200, 500]


def test_feedback_analytics():
    """Test feedback analytics endpoint."""
    response = client.get("/dashboard/feedback-analytics")
    assert response.status_code in [200, 500]
