from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    vectorizer_type: str
    preprocess_spacy_model: str
    embeddings_spacy_model: Optional[str]
    threshold: float
    created_at_unix: float
    metrics: Optional[Dict[str, float]] = None


class SpamDetectorModel:
    """
    Thin wrapper around an sklearn pipeline that:
      - returns a tuned thresholded spam/ham decision
      - exposes `predict_proba` (spam probability) for transparency
    """

    def __init__(self, pipeline, threshold: float, spam_label: int = 1):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.spam_label = int(spam_label)

    def _spam_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError("Underlying pipeline does not support predict_proba().")

        proba = self.pipeline.predict_proba(texts)
        # pipeline classifier classes order should align with proba columns.
        classes = getattr(self.pipeline, "classes_", None)

        if classes is None:
            # For Pipeline, last step usually holds `classes_`.
            last_step = getattr(self.pipeline, "named_steps", {}).get("clf")
            classes = getattr(last_step, "classes_", None)

        if classes is None:
            raise RuntimeError("Could not determine classifier classes_ for probability mapping.")

        try:
            idx = int(np.where(np.asarray(classes) == self.spam_label)[0][0])
        except IndexError as e:
            raise RuntimeError(f"spam_label={self.spam_label} not present in classifier classes.") from e

        return proba[:, idx].astype(np.float32, copy=False)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self._spam_proba(list(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        spam_proba = self.predict_proba(texts)
        return (spam_proba >= self.threshold).astype(np.int64)


def save_model(
    model: SpamDetectorModel,
    metadata: ModelMetadata,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(bundle, output_path)


def load_model(model_path: str | Path) -> tuple[SpamDetectorModel, ModelMetadata]:
    model_path = Path(model_path)
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["metadata"]

