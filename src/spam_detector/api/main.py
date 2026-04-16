from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError

from spam_detector.api.routes.dashboard import router as dashboard_router
from spam_detector.api.routes.feedback import router as feedback_router
from spam_detector.api.routes.health import router as health_router
from spam_detector.api.routes.predict import router as predict_router
from spam_detector.api.routes.stats import router as stats_router
from spam_detector.core.logging import configure_logging
from spam_detector.core.settings import Settings
from spam_detector.core.exceptions import ApiError
from spam_detector.db.mongo import create_mongo_client
from spam_detector.db.repositories.prediction_repo import PredictionRepository, create_indexes
from spam_detector.ml.model_service import ModelService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="AI Email Spam Detector Backend", version="1.0.0")
    app.state.settings = settings

    # Optional CORS for UI clients.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "-")
        logger.warning("validation error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "error": ApiError(code="validation_error", message="Invalid request.", detail=str(exc)).__dict__,
                "request_id": request_id,
            },
        )

    @app.exception_handler(PyMongoError)
    def mongo_exception_handler(request: Request, exc: PyMongoError):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("mongo error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": ApiError(code="mongo_error", message="Database error.").__dict__, "request_id": request_id},
        )

    @app.exception_handler(RuntimeError)
    def runtime_exception_handler(request: Request, exc: RuntimeError):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("runtime error request_id=%s", request_id)
        return JSONResponse(
            status_code=503,
            content={"error": ApiError(code="service_unavailable", message=str(exc)).__dict__, "request_id": request_id},
        )

    @app.exception_handler(Exception)
    def unknown_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unknown error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": ApiError(code="internal_error", message="Internal server error.").__dict__, "request_id": request_id},
        )

    @app.on_event("startup")
    def startup_event():
        # Load ML model.
        model_service = ModelService(
            settings.model_path,
            spam_label=settings.model_spam_label,
        )
        try:
            model_service.load()
            logger.info("model loaded model_path=%s", settings.model_path)
        except FileNotFoundError:
            logger.error("model artifact missing model_path=%s", settings.model_path)
            model_service = model_service  # keep not-ready
        except Exception:
            logger.exception("failed to load model")
            model_service = model_service

        app.state.model_service = model_service

        # Connect Mongo.
        if settings.enable_db_writes:
            client = create_mongo_client(settings.mongo_uri)
            collection = client[settings.mongo_db][settings.mongo_collection]
            prediction_repo = PredictionRepository(collection=collection)
            create_indexes(collection)
            app.state.prediction_repo = prediction_repo
            app.state.mongo_client = client
            logger.info("mongo connected db=%s collection=%s", settings.mongo_db, settings.mongo_collection)
        else:
            app.state.prediction_repo = None
            logger.warning("mongo writes disabled enable_db_writes=false")

        # Ensure repo and model states exist even if one fails.
        app.state.started_at = datetime.now(timezone.utc)

    app.include_router(predict_router)
    app.include_router(feedback_router)
    app.include_router(stats_router)
    app.include_router(health_router)
    app.include_router(dashboard_router)
    return app


app = create_app()

