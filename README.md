# Email Spam Detector (NLP + ML)

[![Tests](https://github.com/prasannapbhurke/AI-Email-Spam-Detector/actions/workflows/tests.yml/badge.svg)](https://github.com/prasannapbhurke/AI-Email-Spam-Detector/actions/workflows/tests.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> AI-powered email spam detection system using NLP and Machine Learning

This project trains, persistence, and serves a production-style email spam detector using:
- spaCy (lemmatization)
- NLTK (stopword removal)
- scikit-learn (TF-IDF + model training)
- joblib (model persistence)

## Dataset format

Provide a CSV file with at least two columns:
- `text` (email body/combined text)
- `label` (`spam`/`ham` or `1`/`0`)

Optional: you can override column names with `--text-col` and `--label-col`.

## Train (TF-IDF features)

```bash
python scripts/train_spam_detector.py --data path/to/dataset.csv --output models/spam_detector.joblib
```

## Train (spaCy word vectors embeddings)

Requires a spaCy model with vectors (e.g. `en_core_web_md`).

```bash
python scripts/train_spam_detector.py --data path/to/dataset.csv --feature-type embeddings --output models/spam_detector.joblib
```

### Gmail Integration

Automatically check your Gmail inbox for spam:

```bash
# 1. Get credentials from Google Cloud Console (see docs/GMAIL_INTEGRATION.md)
# 2. Run the monitor
python scripts/run_gmail_spam_detector.py --credentials credentials.json --dry-run --max-emails 10

# 3. Once confident, run for real
python scripts/run_gmail_spam_detector.py --credentials credentials.json
```

See [docs/GMAIL_INTEGRATION.md](docs/GMAIL_INTEGRATION.md) for full setup guide.

---

## Quick Test

```bash
# 1. Start the API server
python scripts/run_api.py --port 8000

# 2. In another terminal, test a prediction
python -c "
from spam_detector.ml.model_service import ModelService
svc = ModelService('models/spam_detector.joblib')
svc.load()
result = svc.predict('WIN FREE MONEY NOW!!!')
print(f'Spam? {result.prediction_label} ({result.confidence:.2%})')
"
```

Or use the interactive checker:
```bash
python scripts/check_email.py
```
Paste your email content and get instant result.

### Expected Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.9% |
| Precision | 99.8% |
| Recall | 100.0% |
| F1-Score | 99.9% |

*Performance based on the current trained model on the included dataset.*

## Gmail Integration

The spam detector can be integrated with Gmail API for automatic email classification.

### Prerequisites

1. A Google Cloud project with Gmail API enabled
2. OAuth 2.0 credentials (Desktop app type)
3. Python 3.9+

### Gmail API Setup

#### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Name your project (e.g., "spam-detector")
4. Click "Create"

#### Step 2: Enable Gmail API

1. In the sidebar, go to "APIs & Services" → "Library"
2. Search for "Gmail API"
3. Click on "Gmail API" and click "Enable"

#### Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" → "OAuth consent screen"
2. Select "External" and click "Create"
3. Fill in app name and user support email
4. Under "Scopes", click "Add or Remove Scopes"
5. Add these scopes:
   - `../auth/gmail.readonly`
   - `../auth/gmail.modify`
   - `../auth/gmail.labels`
6. Add test users (your Google email)
7. Click "Save and Continue"

#### Step 4: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Application type: "Desktop app"
4. Name it (e.g., "Spam Detector")
5. Click "Create"
6. Download the JSON file and save as `credentials.json`

#### Step 5: Run the Gmail Spam Detector

```bash
# Install additional dependencies
pip install google-api-python-client google-auth requests

# First run (triggers OAuth flow)
python scripts/run_gmail_spam_detector.py --credentials credentials.json

# Subsequent runs (uses cached token)
python scripts/run_gmail_spam_detector.py --credentials credentials.json

# Dry run (preview actions without modifying emails)
python scripts/run_gmail_spam_detector.py --credentials credentials.json --dry-run

# With custom prediction API
python scripts/run_gmail_spam_detector.py --credentials credentials.json \
  --prediction-url http://localhost:8000

# Continuous mode (runs every 5 minutes)
python scripts/run_gmail_spam_detector.py --credentials credentials.json \
  --continuous --interval 300
```

### Gmail Spam Detector Features

- **OAuth Authentication**: Secure OAuth2 flow with automatic token refresh
- **Unread Email Fetching**: Processes unread inbox emails
- **ML Classification**: Sends emails to prediction API for spam detection
- **Automatic Actions**:
  - Spam (high probability) → Moves to Spam folder
  - Promotional content → Labels as "PROMOTIONS"
  - Important emails → Labels as "IMPORTANT"
  - Uncertain → Left for manual review
- **Rate Limiting**: Handles Gmail API rate limits with exponential backoff
- **Dry Run Mode**: Preview actions without modifying emails

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PREDICTION_API_URL` | Spam prediction API URL | `http://localhost:8000` |
| `PREDICTION_API_KEY` | API key for prediction endpoint | None |
| `GMAIL_TOKEN_FILE` | OAuth token storage path | `gmail_token.json` |

### Gmail Labels

The processor automatically creates these labels if they don't exist:
- `SPAM` - For emails classified as spam
- `IMPORTANT` - For important non-spam emails
- `PROMOTIONS` - For promotional/marketing emails

### Running as a Service

For continuous operation, use systemd (Linux) or Task Scheduler (Windows):

**Linux (systemd):**
```ini
[Unit]
Description=Gmail Spam Detector
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 scripts/run_gmail_spam_detector.py \
  --credentials /path/to/credentials.json \
  --continuous --interval 300
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Windows (Task Scheduler):**
```
schtasks /create /tn "Gmail Spam Detector" /tr "python scripts\run_gmail_spam_detector.py --credentials credentials.json --continuous" /sc minute /mo 5
```

---

## Self-Learning & Advanced AI Features

### Overview

The spam detection system includes self-learning capabilities that automatically improve
from user feedback, along with advanced AI features for better detection.

### Installation

```bash
# Install additional ML and AI dependencies
pip install lime shap langdetect
```

---

## Auto-Learning System

### Feedback Collection

User feedback is automatically stored when predictions are corrected:

```bash
# Submit feedback via API
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "abc123", "correct_label": "spam"}'
```

Feedback data is stored in MongoDB with:
- Original prediction and confidence
- Correct label
- Feedback source (user/admin/gmail)
- Timestamps

### Retraining Pipeline

The model automatically retrains when enough feedback accumulates:

```bash
# Check if retraining is needed
python scripts/retrain_model.py --dry-run

# Run single retraining
python scripts/retrain_model.py --original-data data/original.csv

# Run continuous retraining (check every hour)
python scripts/retrain_model.py --continuous --interval 3600

# Export feedback data
python scripts/retrain_model.py --export-feedback --feedback-output data/feedback.csv
```

### Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Gmail/     │     │  Prediction  │     │  Feedback       │
│  User       │───▶│  API         │────▶│  Collection     │
│  Submission │     │  (stored)    │     │  (MongoDB)      │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  Retraining         │
         │  Pipeline           │
         │  (when ≥50 samples) │
         └──────────┬──────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐     ┌─────────────────┐
│ Validate      │     │ Deploy          │
│ (accuracy/F1) │     │ (swap model)    │
└───────────────┘     └─────────────────┘
```

### Incremental Learning

For real-time updates without full retraining:

```python
from spam_detector.learning import IncrementalLearner

learner = IncrementalLearner(model)
learner.partial_fit(
    texts=["new spam email"],
    labels=[1],  # 1=spam, 0=ham
)
```

---

## Advanced AI Features

### LIME Explainability

Understand why the model classified an email:

```python
from spam_detector.advanced import LIMEExplainer

explainer = LIMEExplainer(model_service, vectorizer)
explanation = explainer.explain(email_text)

print(explanation.prediction)      # "spam" or "not_spam"
print(explanation.confidence)      # 0.0-1.0
print(explanation.explanation)    # Human-readable
print(explanation.warnings)       # ["Suspicious URL", ...]
```

### Phishing URL Detection

Detect malicious links in emails:

```python
from spam_detector.advanced import PhishingDetector

detector = PhishingDetector()
result = detector.analyze_text(email_text)

print(result["has_urls"])           # True/False
print(result["overall_score"])       # 0.0-1.0 (higher = more suspicious)
print(result["high_risk_urls"])      # List of suspicious URLs with reasons
```

**Detected Patterns:**
- Shortened URLs (bit.ly, tinyurl, etc.)
- IP addresses in URLs
- Typosquatting (paypa1.com, g00gle.com)
- Suspicious TLDs (.tk, .xyz, .top)
- Login/account keywords in unusual contexts
- URL obfuscation (encoding)

### Multi-Language Support

Detects and handles spam in multiple languages:

```python
from spam_detector.advanced import MultiLanguageDetector

detector = MultiLanguageDetector()
lang = detector.detect_language(email_text)

analysis = detector.analyze_multilingual_spam(email_text, spam_probability)
print(analysis["language"])          # ISO code (en, es, fr, zh, etc.)
print(analysis["warnings"])          # Language-specific warnings
print(analysis["adjusted_probability"])  # Adjusted for language
```

**Supported Languages:**
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Russian (ru)
- Hindi (hi)

---

## Dashboard API

Real-time analytics and model performance:

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /dashboard/stats` | Overall statistics |
| `GET /dashboard/accuracy-over-time` | Accuracy trend |
| `GET /dashboard/spam-stats` | Spam detection distribution |
| `GET /dashboard/model-performance` | Precision, recall, F1 |
| `GET /dashboard/feedback-analytics` | Feedback source breakdown |

### Usage

```bash
# Get overall stats
curl http://localhost:8000/dashboard/stats

# Get accuracy over last 30 days
curl "http://localhost:8000/dashboard/accuracy-over-time?days=30&bucket=day"

# Get spam stats for last week
curl "http://localhost:8000/dashboard/spam-stats?days=7"

# Get model performance metrics
curl "http://localhost:8000/dashboard/model-performance?days=7"

# Get feedback analytics
curl "http://localhost:8000/dashboard/feedback-analytics?days=7"
```

### Response Examples

**Dashboard Stats:**
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

**Accuracy Over Time:**
```json
{
  "bucket": "day",
  "period_days": 30,
  "data": [
    {"period": "2024-01-15", "total": 156, "correct": 148, "accuracy": 94.87},
    {"period": "2024-01-16", "total": 189, "correct": 181, "accuracy": 95.77}
  ]
}
```

**Model Performance:**
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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gmail Spam Detector                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Gmail     │───▶│   Gmail      │───▶│   GmailSpam      │  │
│  │   API       │    │   Client     │    │   Processor      │  │
│  └─────────────┘    └──────────────┘    └─────────┬─────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────┘           │
│  │                                                               │
│  ▼                                                               │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │   Prediction API    │◀───│   FastAPI                      │  │
│  │   (FastAPI)         │    │   - /predict                   │  │
│  └──────────┬──────────┘    │   - /feedback                  │  │
│             │               │   - /dashboard/*               │  │
│             │               └────────────────────────────────┘  │
│             │                            │                      │
│             ▼                            ▼                      │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │   ModelService      │    │   MongoDB                       │  │
│  │   (TF-IDF + sklearn)│    │   - Predictions                 │  │
│  └──────────┬──────────┘    │   - Feedback                    │  │
│             │               └────────────────────────────────┘  │
│             │                            │                      │
│             ▼                            ▼                      │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │   Advanced AI       │    │   Self-Learning                 │  │
│  │   - LIME Explainer  │    │   - FeedbackCollector          │  │
│  │   - PhishingDetector│    │   - RetrainingPipeline         │  │
│  │   - MultiLang       │    │   - ScheduledRetrainer         │  │
│  └─────────────────────┘    └────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file | `models/spam_detector.joblib` |
| `MONGO_URI` | MongoDB connection URI | `mongodb://localhost:27017` |
| `MONGO_DB` | Database name | `spam_detector` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RETRAIN_MIN_FEEDBACK` | Min feedback to trigger retrain | `50` |
| `RETRAIN_INTERVAL` | Check interval (seconds) | `3600` |

### Learning Configuration

```python
from spam_detector.learning import LearningConfig

config = LearningConfig(
    min_feedback_for_retrain=50,      # Trigger at 50 feedback samples
    retrain_threshold_percent=0.1,     # Retrain when 10% of predictions have feedback
    enable_incremental=True,          # Enable partial_fit updates
    incremental_batch_size=32,        # Batch size for incremental learning
)
```
