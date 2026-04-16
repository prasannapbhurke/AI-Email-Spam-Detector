from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Binary metrics for spam detection, assuming spam label is 1.
    """

    acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: int = 101,
) -> ThresholdResult:
    """
    Find a probability threshold that maximizes F1 for the positive class (spam=1).
    """

    if y_proba.ndim != 1:
        raise ValueError("y_proba must be a 1D array of spam probabilities.")

    if thresholds < 2:
        raise ValueError("thresholds must be >= 2")

    # Evaluate thresholds from 0.0 to 1.0 inclusive.
    grid = np.linspace(0.0, 1.0, thresholds, dtype=np.float64)

    best_threshold = 0.5
    best_f1 = -1.0

    for thr in grid:
        y_pred = (y_proba >= thr).astype(int)
        metrics = classification_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thr)

    return ThresholdResult(threshold=best_threshold, f1=best_f1)

