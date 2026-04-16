from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.errors import PyMongoError

from spam_detector.api.dependencies import get_model_service, get_prediction_repo
from spam_detector.api.schemas import PredictRequest, PredictResponse
from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM
from spam_detector.ml.model_service import ModelService, PredictionResult

router = APIRouter()
logger = logging.getLogger(__name__)


def _to_response(prediction_id: str, result: PredictionResult, predicted_at: datetime) -> PredictResponse:
    return PredictResponse(
        prediction_id=prediction_id,
        prediction=prediction_label_to_str(result.prediction_label),
        confidence=float(result.confidence),
        spam_probability=float(result.spam_probability),
        predicted_at=predicted_at,
    )


def prediction_label_to_str(label: str) -> str:
    if label == "spam":
        return LABEL_SPAM
    return LABEL_NOT_SPAM


@router.post("/predict", response_model=PredictResponse)
def predict_email(
    req: PredictRequest,
    request: Request,
    model_service: ModelService = Depends(get_model_service),
    repo=Depends(get_prediction_repo),
):
    request_id = getattr(request.state, "request_id", "-")
    logger.info("predict_email start request_id=%s", request_id)

    email_text = req.email_text
    try:
        result = model_service.predict(email_text)
        now = datetime.now(timezone.utc)
        write_res = repo.insert_prediction(
            email_text=email_text,
            prediction=result.prediction_label,  # "spam" / "not_spam"
            confidence=result.confidence,
            spam_confidence=result.spam_probability,
            predicted_at=now,
            model_name=getattr(getattr(model_service, "metadata", None), "model_name", None),
        )

        logger.info(
            "predict_email success request_id=%s prediction_id=%s prediction=%s",
            request_id,
            write_res.inserted_id,
            result.prediction_label,
        )

        return _to_response(
            write_res.inserted_id,
            result,
            predicted_at=now,
        )
    except RuntimeError as e:
        logger.exception("predict_email runtime error request_id=%s", request_id)
        raise HTTPException(status_code=503, detail=str(e))
    except (PyMongoError, KeyError) as e:
        logger.exception("predict_email db error request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Failed to store prediction.")
    except Exception:
        logger.exception("predict_email unknown error request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Prediction failed.")

