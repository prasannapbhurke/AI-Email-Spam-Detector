from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np

from spam_detector.training.model_io import ModelMetadata, SpamDetectorModel, load_model


@dataclass(frozen=True)
class PredictionResult:
    prediction_label: str  # "spam" or "not_spam"
    confidence: float  # confidence in predicted class
    spam_probability: float  # P(spam)


class ModelService:
    def __init__(self, model_path: str | Path, *, spam_label: int = 1):
        self.model_path = Path(model_path)
        self.spam_label = int(spam_label)

        self.model: Optional[SpamDetectorModel] = None
        self.metadata: Optional[ModelMetadata] = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {str(self.model_path)}")

        model, metadata = load_model(self.model_path)

        # Trust wrapper's spam_label (but keep override for safety).
        model.spam_label = self.spam_label

        self.model = model
        self.metadata = metadata

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, email_text: str) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("ModelService is not loaded.")

        spam_proba = float(self.model.predict_proba([email_text])[0])
        predicted_int = int(self.model.predict([email_text])[0])

        if predicted_int == 1:
            return PredictionResult(
                prediction_label="spam",
                confidence=spam_proba,
                spam_probability=spam_proba,
            )

        return PredictionResult(
            prediction_label="not_spam",
            confidence=float(1.0 - spam_proba),
            spam_probability=spam_proba,
        )

