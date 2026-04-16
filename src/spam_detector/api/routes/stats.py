from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pymongo.errors import PyMongoError

from spam_detector.api.schemas import StatsResponse
from spam_detector.api.dependencies import get_prediction_repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=StatsResponse)
def stats(request: Request):
    request_id = getattr(request.state, "request_id", "-")
    repo = get_prediction_repo(request)

    try:
        last_hours = int(getattr(request.app.state, "settings").stats_last_hours)
        data = repo.get_stats(last_hours=last_hours)
        logger.info("stats success request_id=%s last_hours=%s", request_id, last_hours)
        return StatsResponse(**data)
    except PyMongoError:
        logger.exception("stats db error request_id=%s", request_id)
        # keep it simple: surface as 500
        raise
    except Exception:
        logger.exception("stats unknown error request_id=%s", request_id)
        raise

