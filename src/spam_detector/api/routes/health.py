from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    model_ready = bool(getattr(request.app.state, "model_service", None) and request.app.state.model_service.is_ready())
    db_ready = bool(getattr(request.app.state, "prediction_repo", None))

    return {
        "status": "ok" if model_ready and db_ready else "degraded",
        "model_ready": model_ready,
        "db_ready": db_ready,
    }

