"""
Dashboard API routes for spam detection analytics.

Provides:
- Accuracy over time
- Spam detection statistics
- Model performance metrics
- Feedback analytics
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from pymongo.collection import Collection

from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def get_collection(request) -> Collection:
    """Get MongoDB collection from app state."""
    from spam_detector.db.mongo import get_collection as _get_col
    return _get_col()


@router.get("/stats")
def get_dashboard_stats(request) -> dict[str, Any]:
    """
    Get overall dashboard statistics.

    Returns aggregated metrics for spam detection.
    """
    collection = get_collection(request)

    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # Current totals
    total = collection.count_documents({})
    spam = collection.count_documents({"prediction": LABEL_SPAM})
    ham = collection.count_documents({"prediction": LABEL_NOT_SPAM})

    # With feedback
    with_feedback = collection.count_documents({"feedback": {"$exists": True, "$ne": None}})
    feedback_correct = collection.count_documents({
        "feedback": {"$exists": True, "$ne": None},
        "$expr": {"$eq": ["$prediction", "$feedback.label"]},
    })

    # Time-based counts
    last_24h = collection.count_documents({"predicted_at": {"$gte": day_ago}})
    last_week = collection.count_documents({"predicted_at": {"$gte": week_ago}})
    last_month = collection.count_documents({"predicted_at": {"$gte": month_ago}})

    # Feedback time-based
    feedback_24h = collection.count_documents({"feedback.feedback_at": {"$gte": day_ago}})

    # Calculate accuracy from feedback
    accuracy = (feedback_correct / with_feedback * 100) if with_feedback > 0 else None

    return {
        "total_predictions": total,
        "spam_predictions": spam,
        "ham_predictions": ham,
        "spam_rate": round(spam / total * 100, 2) if total > 0 else 0,
        "with_feedback": with_feedback,
        "feedback_correct": feedback_correct,
        "accuracy_percent": round(accuracy, 2) if accuracy else None,
        "time_ranges": {
            "last_24h": last_24h,
            "last_week": last_week,
            "last_month": last_month,
        },
        "feedback_24h": feedback_24h,
    }


@router.get("/accuracy-over-time")
def get_accuracy_over_time(
    request,
    days: int = Query(default=30, ge=1, le=90),
    bucket: str = Query(default="day", regex="^(hour|day|week)$"),
) -> dict[str, Any]:
    """
    Get accuracy metrics over time.

    Args:
        days: Number of days to look back.
        bucket: Time bucket granularity (hour, day, week).

    Returns:
        Time series of accuracy metrics.
    """
    collection = get_collection(request)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    # Build date grouping
    if bucket == "hour":
        date_format = {"$dateToString": {"format": "%Y-%m-%d %H:00", "date": "$predicted_at"}}
        delta = timedelta(hours=1)
    elif bucket == "day":
        date_format = {"$dateToString": {"format": "%Y-%m-%d", "date": "$predicted_at"}}
        delta = timedelta(days=1)
    else:  # week
        date_format = {"$dateToString": {"format": "%Y-W%V", "date": "$predicted_at"}}
        delta = timedelta(weeks=1)

    pipeline = [
        # Filter to date range and feedback exists
        {
            "$match": {
                "predicted_at": {"$gte": start_date},
                "feedback": {"$exists": True, "$ne": None},
            }
        },
        # Group by time bucket
        {
            "$group": {
                "_id": date_format,
                "total": {"$sum": 1},
                "correct": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$prediction", "$feedback.label"]},
                            1,
                            0,
                        ]
                    }
                },
                "spam_total": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_SPAM]}, 1, 0]}
                },
                "ham_total": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_NOT_SPAM]}, 1, 0]}
                },
            }
        },
        # Add accuracy percentage
        {
            "$addFields": {
                "accuracy": {
                    "$cond": [
                        {"$gt": ["$total", 0]},
                        {"$divide": [{"$multiply": ["$correct", 100]}, "$total"]},
                        0,
                    ]
                }
            }
        },
        {"$sort": {"_id": 1}},
    ]

    results = list(collection.aggregate(pipeline))

    return {
        "bucket": bucket,
        "period_days": days,
        "data": [
            {
                "period": r["_id"],
                "total_predictions": r["total"],
                "correct": r["correct"],
                "incorrect": r["total"] - r["correct"],
                "accuracy": round(r["accuracy"], 2),
                "spam_count": r["spam_total"],
                "ham_count": r["ham_total"],
            }
            for r in results
        ],
    }


@router.get("/spam-stats")
def get_spam_stats(
    request,
    days: int = Query(default=7, ge=1, le=90),
) -> dict[str, Any]:
    """
    Get spam detection statistics.

    Args:
        days: Number of days to look back.

    Returns:
        Spam detection metrics.
    """
    collection = get_collection(request)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    # Hourly spam distribution
    hourly_pipeline = [
        {"$match": {"predicted_at": {"$gte": start_date}}},
        {
            "$group": {
                "_id": {
                    "$hour": {"date": "$predicted_at", "timezone": "UTC"}
                },
                "spam_count": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_SPAM]}, 1, 0]}
                },
                "ham_count": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_NOT_SPAM]}, 1, 0]}
                },
                "total": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    hourly = list(collection.aggregate(hourly_pipeline))

    # Daily spam distribution
    daily_pipeline = [
        {"$match": {"predicted_at": {"$gte": start_date}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$predicted_at"}},
                "spam_count": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_SPAM]}, 1, 0]}
                },
                "ham_count": {
                    "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_NOT_SPAM]}, 1, 0]}
                },
                "total": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    daily = list(collection.aggregate(daily_pipeline))

    # Confidence distribution
    confidence_pipeline = [
        {"$match": {"predicted_at": {"$gte": start_date}}},
        {
            "$bucket": {
                "groupBy": "$confidence",
                "boundaries": [0, 0.2, 0.4, 0.6, 0.8, 1.01],
                "default": "other",
                "output": {
                    "count": {"$sum": 1},
                    "spam_in_bucket": {
                        "$sum": {"$cond": [{"$eq": ["$prediction", LABEL_SPAM]}, 1, 0]}
                    },
                }
            }
        },
    ]

    confidence_buckets = list(collection.aggregate(confidence_pipeline))

    return {
        "period_days": days,
        "hourly_distribution": [
            {"hour": h["_id"], "spam": h["spam_count"], "ham": h["ham_count"], "total": h["total"]}
            for h in hourly
        ],
        "daily_distribution": [
            {
                "date": d["_id"],
                "spam": d["spam_count"],
                "ham": d["ham_count"],
                "total": d["total"],
                "avg_confidence": round(d["avg_confidence"], 3) if d.get("avg_confidence") else None,
            }
            for d in daily
        ],
        "confidence_distribution": [
            {
                "bucket": b["_id"],
                "count": b["count"],
                "spam_in_bucket": b["spam_in_bucket"],
                "ham_in_bucket": b["count"] - b["spam_in_bucket"],
            }
            for b in confidence_buckets
        ],
    }


@router.get("/model-performance")
def get_model_performance(
    request,
    days: int = Query(default=7, ge=1, le=90),
) -> dict[str, Any]:
    """
    Get model performance metrics.

    Args:
        days: Number of days to look back.

    Returns:
        Precision, recall, F1 estimates based on feedback.
    """
    collection = get_collection(request)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    # Get confusion matrix components
    pipeline = [
        {"$match": {
            "predicted_at": {"$gte": start_date},
            "feedback": {"$exists": True, "$ne": None},
        }},
        {
            "$group": {
                "_id": None,
                "true_positives": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$prediction", LABEL_SPAM]},
                                {"$eq": ["$feedback.label", LABEL_SPAM]},
                            ]},
                            1, 0,
                        ]
                    }
                },
                "true_negatives": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$prediction", LABEL_NOT_SPAM]},
                                {"$eq": ["$feedback.label", LABEL_NOT_SPAM]},
                            ]},
                            1, 0,
                        ]
                    }
                },
                "false_positives": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$prediction", LABEL_SPAM]},
                                {"$eq": ["$feedback.label", LABEL_NOT_SPAM]},
                            ]},
                            1, 0,
                        ]
                    }
                },
                "false_negatives": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$prediction", LABEL_NOT_SPAM]},
                                {"$eq": ["$feedback.label", LABEL_SPAM]},
                            ]},
                            1, 0,
                        ]
                    }
                },
                "total": {"$sum": 1},
            }
        },
    ]

    results = list(collection.aggregate(pipeline))

    if not results:
        return {
            "period_days": days,
            "sample_size": 0,
            "message": "No feedback data available for this period",
        }

    r = results[0]
    tp = r["true_positives"]
    tn = r["true_negatives"]
    fp = r["false_positives"]
    fn = r["false_negatives"]
    total = r["total"]

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    accuracy = (tp + tn) / total if total > 0 else None

    return {
        "period_days": days,
        "sample_size": total,
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "metrics": {
            "accuracy": round(accuracy * 100, 2) if accuracy else None,
            "precision": round(precision * 100, 2) if precision else None,
            "recall": round(recall * 100, 2) if recall else None,
            "f1_score": round(f1 * 100, 2) if f1 else None,
        },
    }


@router.get("/feedback-analytics")
def get_feedback_analytics(
    request,
    days: int = Query(default=7, ge=1, le=90),
) -> dict[str, Any]:
    """
    Get feedback analytics.

    Args:
        days: Number of days to look back.

    Returns:
        Feedback source breakdown and trends.
    """
    collection = get_collection(request)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    # Source breakdown
    source_pipeline = [
        {"$match": {
            "feedback": {"$exists": True, "$ne": None},
            "feedback.feedback_at": {"$gte": start_date},
        }},
        {
            "$group": {
                "_id": "$feedback.source",
                "count": {"$sum": 1},
                "correct": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$prediction", "$feedback.label"]},
                            1, 0,
                        ]
                    }
                },
            }
        },
        {"$sort": {"count": -1}},
    ]

    sources = list(collection.aggregate(source_pipeline))

    # Agreement rate by source
    agreement_by_source = {}
    for s in sources:
        source = s["_id"] or "unknown"
        agreement = s["correct"] / s["count"] * 100 if s["count"] > 0 else 0
        agreement_by_source[source] = {
            "total": s["count"],
            "correct": s["correct"],
            "agreement_rate": round(agreement, 2),
        }

    # Daily feedback trend
    daily_pipeline = [
        {"$match": {
            "feedback": {"$exists": True, "$ne": None},
            "feedback.feedback_at": {"$gte": start_date},
        }},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$feedback.feedback_at"}},
                "count": {"$sum": 1},
                "spam_correct": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$prediction", LABEL_SPAM]},
                                {"$eq": ["$feedback.label", LABEL_SPAM]},
                            ]},
                            1, 0,
                        ]
                    }
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    daily = list(collection.aggregate(daily_pipeline))

    # Total feedback stats
    total_feedback = collection.count_documents({
        "feedback": {"$exists": True, "$ne": None},
        "feedback.feedback_at": {"$gte": start_date},
    })

    return {
        "period_days": days,
        "total_feedback": total_feedback,
        "by_source": agreement_by_source,
        "daily_trend": [
            {
                "date": d["_id"],
                "count": d["count"],
                "spam_correct": d["spam_correct"],
                "accuracy": round(d["spam_correct"] / d["count"] * 100, 2) if d["count"] > 0 else 0,
            }
            for d in daily
        ],
    }


@router.get("/retraining-history")
def get_retraining_history(
    request,
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Get model retraining history.

    Args:
        limit: Number of recent retraining events to return.

    Returns:
        List of past retraining events with metrics.
    """
    # In production, this would query a retraining_events collection
    # For now, return placeholder structure
    return {
        "message": "Retraining history tracking not yet implemented",
        "note": "Track retraining events by adding to a 'retraining_events' collection",
        "suggested_schema": {
            "timestamp": "datetime",
            "model_version": "string",
            "feedback_samples_used": "int",
            "metrics_before": {"accuracy": "float", "f1": "float"},
            "metrics_after": {"accuracy": "float", "f1": "float"},
            "status": "success|failed",
        },
    }
