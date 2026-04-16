from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


ModelName = Literal["naive_bayes", "logistic_regression", "random_forest"]


@dataclass(frozen=True)
class ModelFactoryConfig:
    random_state: int = 42
    random_forest_estimators: int = 250
    random_forest_max_depth: Optional[int] = None
    random_forest_min_samples_split: int = 2

    logistic_regression_max_iter: int = 1000
    # saga handles TF-IDF sparse matrices well.
    logistic_regression_solver: str = "saga"


def get_model(model_name: ModelName, config: Optional[ModelFactoryConfig] = None):
    cfg = config or ModelFactoryConfig()

    if model_name == "naive_bayes":
        # Works well with non-negative sparse TF-IDF features.
        return MultinomialNB()

    if model_name == "logistic_regression":
        # Probability outputs required for threshold tuning.
        return LogisticRegression(
            max_iter=cfg.logistic_regression_max_iter,
            solver=cfg.logistic_regression_solver,
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    if model_name == "random_forest":
        # RandomForest expects dense features; training pipeline applies SVD.
        return RandomForestClassifier(
            n_estimators=cfg.random_forest_estimators,
            max_depth=cfg.random_forest_max_depth,
            min_samples_split=cfg.random_forest_min_samples_split,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")

