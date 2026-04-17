# API Documentation

The FastAPI backend provides a comprehensive REST API for spam detection, feedback collection, and analytics.

## Base URL

- Development: `http://localhost:8000`
- Production: `[your-domain]/api` (if behind reverse proxy)

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to explore and test the API directly from your browser.

---

## Endpoints

### Health Check

**GET** `/health`

Check if the API is running and model is loaded.

**Response 200:**
```json
{
  "status": "healthy",
  "model_ready": true,
  "timestamp": "2026-04-17T08:30:00Z"
}
```

---

### Predict Spam

**POST** `/predict`

Classify an email as spam or legitimate.

**Request Body:**
```json
{
  "email_text": "Your email content here..."
}
```

**Response 200:**
```json
{
  "prediction_id": "507f1f77bcf86cd799439011",
  "prediction": "spam",
  "confidence": 0.95,
  "spam_probability": 0.95,
  "predicted_at": "2026-04-17T08:30:15Z"
}
```

**Response 503:**
```json
{
  "error": {
    "code": "service_unavailable",
    "message": "ModelService is not loaded."
  }
}
```

---

### Submit Feedback

**POST** `/feedback`

Submit feedback on a prediction to improve the model.

**Request Body:**
```json
{
  "prediction_id": "507f1f77bcf86cd799439011",
  "correct_label": "spam"
}
```

**Parameters:**
- `prediction_id` (string, UUID): The prediction ID from a previous `/predict` response
- `correct_label` (string): Either `"spam"` or `"ham"` (case-insensitive)

**Response 200:**
```json
{
  "success": true,
  "message": "Feedback recorded"
}
```

**Response 404:**
```json
{
  "error": {
    "code": "not_found",
    "message": "Prediction not found"
  }
```

---

### Dashboard Statistics

**GET** `/dashboard/stats`

Get overall system statistics.

**Query Parameters:**
- `hours` (integer, default: 24): Lookback period in hours

**Response 200:**
```json
{
  "total_predictions": 15420,
  "spam_predictions": 3245,
  "ham_predictions": 12175,
  "spam_rate": 21.05,
  "with_feedback": 890,
  "feedback_correct": 842,
  "accuracy_percent": 94.61,
  "time_ranges": {
    "last_24h": 234,
    "last_week": 1523,
    "last_month": 8934
  }
}
```

---

### Accuracy Over Time

**GET** `/dashboard/accuracy-over-time`

Get accuracy trend broken down by time buckets.

**Query Parameters:**
- `days` (integer, default: 30): Lookback period
- `bucket` (string, default: "day"): One of `"hour"`, `"day"`, `"week"`, `"month"`

**Response 200:**
```json
{
  "bucket": "day",
  "period_days": 30,
  "data": [
    {
      "period": "2024-01-15",
      "total": 156,
      "correct": 148,
      "accuracy": 94.87
    }
  ]
}
```

---

### Spam Statistics

**GET** `/dashboard/spam-stats`

Get spam/class distribution over time.

**Query Parameters:**
- `days` (integer, default: 7): Lookback period

**Response 200:**
```json
{
  "period_days": 7,
  "daily_counts": [
    {
      "date": "2024-01-15",
      "spam": 45,
      "ham": 234,
      "total": 279
    }
  ]
}
```

---

### Model Performance

**GET** `/dashboard/model-performance`

Get precision, recall, F1 metrics over time.

**Query Parameters:**
- `days` (integer, default: 7): Lookback period

**Response 200:**
```json
{
  "period_days": 7,
  "sample_size": 523,
  "confusion_matrix": {
    "true_positives": 234,
    "true_negatives": 267,
    "false_positives": 12,
    "false_negatives": 10
  },
  "metrics": {
    "accuracy": 95.79,
    "precision": 95.12,
    "recall": 95.90,
    "f1_score": 95.51
  }
}
```

---

### Feedback Analytics

**GET** `/dashboard/feedback-analytics`

Analyze feedback patterns and sources.

**Query Parameters:**
- `days` (integer, default: 7): Lookback period

**Response 200:**
```json
{
  "period_days": 7,
  "total_feedback": 156,
  "by_source": {
    "user": 89,
    "admin": 45,
    "gmail": 22
  },
  "by_correctness": {
    "agreed": 142,
    "disagreed": 14
  },
  "feedback_rate": 0.89
}
```

---

## Error Responses

All endpoints may return these standard error formats:

**400 Bad Request:**
```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid request."
  }
}
```

**404 Not Found:**
```json
{
  "error": {
    "code": "not_found",
    "message": "Resource not found."
  }
}
```

**422 Unprocessable Entity:**
```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid request.",
    "detail": "Field required"
  }
}
```

**500 Internal Server Error:**
```json
{
  "error": {
    "code": "internal_error",
    "message": "Internal server error."
  }
}
```

**503 Service Unavailable:**
```json
{
  "error": {
    "code": "service_unavailable",
    "message": "Model not loaded."
  }
}
```

All responses include an `X-Request-ID` header for tracing.

---

## Rate Limits

Currently, no rate limits are enforced. For production deployments, consider adding:

- Per-IP rate limiting
- API key authentication
- Query quotas

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Predict
response = requests.post(f"{BASE_URL}/predict", json={
    "email_text": "Test email content"
})
result = response.json()
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")

# Feedback
requests.post(f"{BASE_URL}/feedback", json={
    "prediction_id": result['prediction_id'],
    "correct_label": "spam"
})

# Dashboard
stats = requests.get(f"{BASE_URL}/dashboard/stats").json()
print(f"Total predictions: {stats['total_predictions']}")
```

---

## cURL Examples

```bash
# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "WIN FREE MONEY NOW!!!"}'

# Feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "abc123", "correct_label": "spam"}'

# Stats
curl http://localhost:8000/dashboard/stats

# Accuracy over time
curl "http://localhost:8000/dashboard/accuracy-over-time?days=30"

# Model performance
curl http://localhost:8000/dashboard/model-performance
```

---

## WebSocket (Future)

Future versions may include WebSocket for real-time predictions and streaming analytics.

---

## Support

- Issues: [GitHub Issues](https://github.com/prasannapbhurke/AI-Email-Spam-Detector/issues)
- Documentation: [README.md](README.md)
- Email: prasannapbhurke@github.com
