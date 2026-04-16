from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from bson import ObjectId
from pymongo.collection import Collection

LABEL_SPAM = "spam"
LABEL_NOT_SPAM = "not_spam"


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_object_id(id_str: str) -> ObjectId:
    return ObjectId(id_str)


def create_indexes(collection: Collection) -> None:
    # Basic indexes for analytics & lookups.
    collection.create_index("predicted_at", name="idx_predicted_at")
    collection.create_index("label", name="idx_label")
    collection.create_index("feedback_at", name="idx_feedback_at")


@dataclass(frozen=True)
class PredictionWriteResult:
    inserted_id: str


class PredictionRepository:
    def __init__(self, collection: Collection):
        self.collection = collection

    def insert_prediction(
        self,
        *,
        email_text: str,
        prediction: str,
        confidence: float,
        spam_confidence: float,
        predicted_at: Optional[datetime] = None,
        model_name: Optional[str] = None,
    ) -> PredictionWriteResult:
        predicted_at = ensure_utc(predicted_at or datetime.now(timezone.utc))

        doc: Dict[str, Any] = {
            "email_text": email_text,
            "prediction": prediction,  # "spam" or "not_spam"
            "confidence": float(confidence),  # confidence in predicted class
            "spam_confidence": float(spam_confidence),  # model's P(spam)
            "predicted_at": predicted_at,
            "feedback": None,
            "model_name": model_name,
        }

        res = self.collection.insert_one(doc)
        return PredictionWriteResult(inserted_id=str(res.inserted_id))

    def set_feedback(
        self,
        *,
        prediction_id: str,
        label: str,
        source: Optional[str] = None,
        feedback_at: Optional[datetime] = None,
    ) -> None:
        feedback_at = ensure_utc(feedback_at or datetime.now(timezone.utc))
        _id = to_object_id(prediction_id)

        result = self.collection.update_one(
            {"_id": _id},
            {
                "$set": {
                    "label": label,
                    "feedback": {
                        "label": label,
                        "source": source,
                        "feedback_at": feedback_at,
                    },
                    "feedback_at": feedback_at,
                }
            },
        )

        if result.matched_count == 0:
            raise KeyError(f"Prediction id not found: {prediction_id}")

    def get_by_id(self, prediction_id: str) -> Dict[str, Any]:
        _id = to_object_id(prediction_id)
        doc = self.collection.find_one({"_id": _id})
        if not doc:
            raise KeyError(f"Prediction id not found: {prediction_id}")
        return doc

    def get_stats(self, last_hours: int = 24) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        since = now.timestamp() - float(last_hours) * 3600.0
        since_dt = datetime.fromtimestamp(since, tz=timezone.utc)

        total_predictions = self.collection.count_documents({})

        spam_count = self.collection.count_documents({"prediction": LABEL_SPAM})
        ham_count = self.collection.count_documents({"prediction": LABEL_NOT_SPAM})

        feedback_count = self.collection.count_documents({"label": {"$in": [LABEL_SPAM, LABEL_NOT_SPAM]}})
        spam_feedback_count = self.collection.count_documents({"label": LABEL_SPAM})
        ham_feedback_count = self.collection.count_documents({"label": LABEL_NOT_SPAM})

        preds_last_24h = self.collection.count_documents({"predicted_at": {"$gte": since_dt}})
        feedback_last_24h = self.collection.count_documents({"feedback_at": {"$gte": since_dt}})

        return {
            "total_predictions": int(total_predictions),
            "spam_predictions": int(spam_count),
            "not_spam_predictions": int(ham_count),
            "feedback_count": int(feedback_count),
            "spam_feedback_count": int(spam_feedback_count),
            "not_spam_feedback_count": int(ham_feedback_count),
            "predictions_last_hours": int(preds_last_24h),
            "feedback_last_hours": int(feedback_last_24h),
        }

