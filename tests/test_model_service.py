"""Tests for the ModelService."""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from spam_detector.ml.model_service import ModelService, PredictionResult


def test_model_service_initialization():
    """Test ModelService can be instantiated."""
    model_path = project_root / "models" / "spam_detector.joblib"
    svc = ModelService(str(model_path))
    assert svc.model_path.exists()


def test_model_load():
    """Test model can be loaded."""
    model_path = project_root / "models" / "spam_detector.joblib"
    svc = ModelService(str(model_path))
    svc.load()
    assert svc.is_ready()
    assert svc.model is not None


def test_predict_spam():
    """Test prediction on spam email."""
    model_path = project_root / "models" / "spam_detector.joblib"
    svc = ModelService(str(model_path))
    svc.load()

    spam_text = "WIN FREE MONEY NOW! CLICK HERE TO CLAIM YOUR PRIZE!!!"
    result = svc.predict(spam_text)

    assert isinstance(result, PredictionResult)
    assert result.prediction_label in ("spam", "not_spam")
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.spam_probability <= 1.0
    # The model should classify this as spam with high confidence
    assert result.prediction_label == "spam"
    assert result.confidence > 0.5


def test_predict_ham():
    """Test prediction on legitimate email."""
    model_path = project_root / "models" / "spam_detector.joblib"
    svc = ModelService(str(model_path))
    svc.load()

    ham_text = "Hi John, can we meet tomorrow at 2pm to discuss the project?"
    result = svc.predict(ham_text)

    assert isinstance(result, PredictionResult)
    assert result.prediction_label in ("spam", "not_spam")
    # This should be classified as not spam
    assert result.prediction_label == "not_spam"
    assert result.confidence > 0.5
