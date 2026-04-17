# Deployment Guide

This guide covers deploying the Email Spam Detector in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production (Manual)](#production-manual)
- [Environment Variables](#environment-variables)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.9+ (if not using Docker)
- MongoDB 4.4+ (local or cloud)
- 512MB RAM minimum, 1GB+ recommended
- Disk space for models and logs (~100MB)

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/prasannapbhurke/AI-Email-Spam-Detector.git
cd email-spam-editor
```

### 2. Install Dependencies

```bash
make install-dev
# or
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Setup NLP Resources

```bash
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('stopwords')"
```

### 4. Train Model (Optional)

```bash
make run-train
# or
python scripts/train_spam_detector.py --data data/spam_dataset.csv --output models/spam_detector.joblib
```

### 5. Start MongoDB

#### Option A: Docker
```bash
docker run -d -p 27017:27017 --name mongodb mongo:7
```

#### Option B: Native Installation
Download from [MongoDB Community Server](https://www.mongodb.com/try/download/community)

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local)
```

### 7. Run API

```bash
make run-api
# or
python scripts/run_api.py --port 8000
```

API available at: `http://localhost:8000`

---

## Docker Deployment

### Quick Start (All-in-One)

```bash
# Clone and cd into repo
cd email-spam-detector

# Start all services (MongoDB + API)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

This starts:
- MongoDB on `localhost:27017`
- FastAPI on `http://localhost:8000`

### Custom Configuration

Edit `docker-compose.yml`:

```yaml
environment:
  MODEL_PATH: /app/models/spam_detector.joblib
  MONGO_URI: mongodb://mongo:27017/spam_detector
  LOG_LEVEL: INFO
  ENABLE_DB_WRITES: "true"
```

### Building Image Manually

```bash
docker build -t email-spam-detector -f Dockerfile .
docker run -p 8000:8000 email-spam-detector
```

### Docker without Compose

```bash
# Start MongoDB first
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  mongo:7

# Start API
docker run -d \
  --name spam-detector \
  -p 8000:8000 \
  --link mongodb:mongodb \
  -e MONGO_URI=mongodb://admin:secret@mongodb:27017/spam_detector \
  email-spam-detector

# View logs
docker logs -f spam-detector
```

---

## Production (Manual)

### Systemd Service (Linux)

Create `/etc/systemd/system/spam-detector.service`:

```ini
[Unit]
Description=Email Spam Detector API
After=network.target mongodb.service
Wants=mongodb.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/email-spam-detector
Environment="PATH=/opt/email-spam-detector/venv/bin"
ExecStart=/opt/email-spam-detector/venv/bin/python scripts/run_api.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable spam-detector
sudo systemctl start spam-detector
sudo systemctl status spam-detector
```

### Using Gunicorn (Recommended for Production)

FastAPI runs on uvicorn by default. For production, use gunicorn with uvicorn workers:

```bash
pip install gunicorn

# Run with gunicorn
gunicorn spam_detector.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile /var/log/spam-detector/access.log \
  --error-logfile /var/log/spam-detector/error.log
```

Add to systemd service:
```ini
ExecStart=/opt/email-spam-detector/venv/bin/gunicorn \
  spam_detector.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/spam-detector`:

```nginx
server {
    listen 80;
    server_name spam.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/spam-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL with Let's Encrypt

```bash
sudo certbot --nginx -d spam.example.com
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model | `models/spam_detector.joblib` |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGO_DB` | Database name | `spam_detector` |
| `MONGO_COLLECTION` | Collection name | `predictions` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENABLE_DB_WRITES` | Enable DB persistence | `true` |
| `STATS_LAST_HOURS` | Stats lookback window | `24` |
| `MODEL_SPAM_LABEL` | Integer label for spam class | `1` |
| `RETRAIN_MIN_FEEDBACK` | Min feedback to retrain | `50` |
| `RETRAIN_INTERVAL` | Check interval (seconds) | `3600` |

### Setting Environment Variables

**Linux (systemd):**
```ini
Environment="MODEL_PATH=/app/models/spam_detector.joblib"
Environment="MONGO_URI=mongodb://localhost:27017"
```

**Docker:**
```yaml
environment:
  - MODEL_PATH=/app/models/spam_detector.joblib
  - MONGO_URI=mongodb://mongo:27017/spam_detector
```

**Shell:**
```bash
export MONGO_URI="mongodb://localhost:27017"
python scripts/run_api.py
```

**.env file** (automatically loaded by pydantic-settings):
```
MODEL_PATH=models/spam_detector.joblib
MONGO_URI=mongodb://localhost:27017
MONGO_DB=spam_detector
LOG_LEVEL=INFO
ENABLE_DB_WRITES=true
```

---

## Monitoring

### Health Checks

Configure monitoring to poll `/health` every 30s:

```bash
curl -f http://localhost:8000/health || echo "Service down"
```

### Prometheus Metrics (Future)

Add Prometheus exporter for metrics:
- Request rate
- Latency percentiles
- Prediction distribution
- Error rates

### Logs

Logs are structured JSON by default. Configure logging level via `LOG_LEVEL` env var.

Example log entry:
```json
{
  "level": "INFO",
  "message": "predict_email success",
  "request_id": "abc123",
  "prediction_id": "def456",
  "prediction": "spam",
  "timestamp": "2026-04-17T08:30:15.123Z"
}
```

---

## Scaling

### Horizontal Scaling

Run multiple API instances behind a load balancer:

```bash
# Start 4 instances on different ports
python scripts/run_api.py --port 8000 &
python scripts/run_api.py --port 8001 &
python scripts/run_api.py --port 8002 &
python scripts/run_api.py --port 8003 &
```

Use nginx or HAProxy to load balance.

### Database Connection Pooling

Tune MongoDB connection pool size in production:

```python
from pymongo import MongoClient

client = MongoClient(
    "mongodb://localhost:27017",
    maxPoolSize=50,
    minPoolSize=10,
    maxIdleTimeMS=30000
)
```

---

## Backup & Restore

### MongoDB Backup

```bash
# Backup
mongodump --uri="mongodb://localhost:27017/spam_detector" --out=backup/

# Restore
mongorestore --uri="mongodb://localhost:27017/spam_detector" backup/
```

### Model Backup

The retraining pipeline automatically creates `.joblib.backup` files.

---

## Security

### Production Checklist

- [ ] Use HTTPS (TLS/SSL) in frontend/load balancer
- [ ] Set strong MongoDB credentials (not default)
- [ ] Enable authentication on API (future: API keys)
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Use secrets manager for credentials
- [ ] Disable `ENABLE_DB_WRITES` if only serving predictions

### Gmail API Credentials

- Store `credentials.json` outside version control
- Use environment-specific OAuth credentials
- Rotate client secrets periodically
- Monitor Google Cloud Console for suspicious activity

---

## Troubleshooting

### Model Not Loading

```
ERROR: Model artifact missing: models/spam_detector.joblib
```

**Solution:** Train the model first:
```bash
make run-train
```

### MongoDB Connection Failed

```
ERROR: Failed to connect to MongoDB
```

**Solution:** Ensure MongoDB is running:
```bash
# Docker
docker start mongodb

# Systemd
sudo systemctl start mongod

# Check status
mongo --eval "db.adminCommand('ping')"
```

### Port Already in Use

```
OSError: [Errno 98] Address already in use
```

**Solution:** Change port or kill existing process:
```bash
# Find process
lsof -i :8000

# Kill
kill -9 <PID>

# Or use different port
python scripts/run_api.py --port 8001
```

### Tests Fail on Windows

**Solution:** Fix line endings:
```bash
git config core.autocrlf false
```

### Out of Memory

**Solution:** Reduce batch size in training or use smaller model:
```bash
python scripts/train_spam_detector.py --feature-type tfidf --tfidf-max-features 25000
```

---

## Upgrading

### Minor Version Upgrade

1. Backup database and model
2. Pull latest code: `git pull`
3. Install new dependencies: `pip install -r requirements.txt`
4. Migrate data if needed (check migration docs)
5. Restart services

### Major Version Upgrade

Follow release notes and migration guides carefully. May require:

- Database schema migrations
- Model retraining
- Configuration changes

---

## Support

- Documentation: [README.md](README.md)
- Issues: [GitHub Issues](https://github.com/prasannapbhurke/AI-Email-Spam-Detector/issues)
- Email: prasannapbhurke@github.com

---

*Last updated: April 17, 2026*
