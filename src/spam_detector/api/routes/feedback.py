from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.errors import PyMongoError

from spam_detector.api.dependencies import get_prediction_repo
from spam_detector.api.schemas import FeedbackRequest, FeedbackResponse
from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM

router = APIRouter()
logger = logging.getLogger(__name__)


def _normalize_label(label: str) -> str:
    if label == LABEL_SPAM:
        return LABEL_SPAM
    if label == LABEL_NOT_SPAM:
        return LABEL_NOT_SPAM
    # Be tolerant in case caller passes other spellings.
    if label.lower() in {"spam", "1", "true", "yes"}:
        return LABEL_SPAM
    return LABEL_NOT_SPAM


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    req: FeedbackRequest,
    request: Request,
    repo=Depends(get_prediction_repo),
):
    request_id = getattr(request.state, "request_id", "-")
    logger.info("submit_feedback start request_id=%s prediction_id=%s", request_id, req.prediction_id)

    try:
        label = _normalize_label(req.label)
        now = datetime.now(timezone.utc)
        repo.set_feedback(
            prediction_id=req.prediction_id,
            label=label,
            source=req.source,
            feedback_at=now,
        )

        logger.info("submit_feedback success request_id=%s prediction_id=%s label=%s", request_id, req.prediction_id, label)
        return FeedbackResponse(prediction_id=req.prediction_id, label=label, feedback_at=now)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prediction not found for feedback.")
    except PyMongoError:
        logger.exception("submit_feedback db error request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Failed to store feedback.")
    except Exception:
        logger.exception("submit_feedback unknown error request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Failed to submit feedback.")

