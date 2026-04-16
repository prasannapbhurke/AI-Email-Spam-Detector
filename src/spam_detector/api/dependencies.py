from __future__ import annotations

from typing import Generator

from fastapi import Request

from spam_detector.db.repositories.prediction_repo import PredictionRepository
from spam_detector.ml.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    svc = getattr(request.app.state, "model_service", None)
    if svc is None or not svc.is_ready():
        raise RuntimeError("Model service is not ready.")
    return svc


def get_prediction_repo(request: Request) -> PredictionRepository:
    repo = getattr(request.app.state, "prediction_repo", None)
    if repo is None:
        raise RuntimeError("Prediction repository is not initialized.")
    return repo

