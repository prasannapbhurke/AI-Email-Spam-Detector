from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    email_text: str = Field(..., min_length=1, max_length=200000)


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: Literal["spam", "not_spam"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    spam_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_at: datetime


class FeedbackRequest(BaseModel):
    prediction_id: str
    label: Literal["spam", "not_spam"]
    source: Optional[str] = Field(default=None, description="e.g. user, admin")


class FeedbackResponse(BaseModel):
    prediction_id: str
    label: Literal["spam", "not_spam"]
    feedback_at: datetime


class StatsResponse(BaseModel):
    total_predictions: int
    spam_predictions: int
    not_spam_predictions: int
    feedback_count: int
    spam_feedback_count: int
    not_spam_feedback_count: int
    predictions_last_hours: int
    feedback_last_hours: int

