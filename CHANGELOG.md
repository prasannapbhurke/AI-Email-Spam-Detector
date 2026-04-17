# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-17

### Added

- Initial release of AI Email Spam Detector
- Machine Learning pipeline with spaCy + scikit-learn
- TF-IDF feature extraction with configurable n-grams
- Pre-trained model with 99.9% accuracy
- FastAPI REST API with prediction, feedback, and analytics endpoints
- MongoDB integration for prediction persistence
- Self-learning system with feedback collection
- Automated retraining pipeline (trigger @ 50 feedback samples)
- Incremental learning support (partial_fit)
- Gmail API integration for automatic spam filtering
- Advanced AI features:
  - LIME explainability framework
  - Phishing URL detection
  - Multi-language support (10+ languages)
- Dashboard analytics with time-series metrics
- Comprehensive test suite (7+ tests)
- GitHub Actions CI/CD
- Docker deployment (Dockerfile + docker-compose)
- Interactive HTML frontend for testing
- CLI scripts for training, prediction, retraining, evaluation
- Complete documentation (README, API docs, deployment guide)
- Makefile for common development tasks
- Pre-commit hooks for code quality
- pyproject.toml for modern Python packaging

### Technical Details

- **Framework:** FastAPI 0.135
- **ML Library:** scikit-learn 1.8
- **NLP:** spaCy 3.8, NLTK 3.9
- **Database:** MongoDB 7
- **Python Support:** 3.9+
- **License:** MIT

---

## [Unreleased]

### Planned

- User authentication and API keys
- WebSocket for real-time predictions
- Prometheus metrics endpoint
- Advanced model explainability UI
- Batch prediction endpoint
- Email threading analysis
- Attachment content scanning
- Integration with more email providers (Outlook, IMAP)
- Model versioning and A/B testing
- Web-based admin dashboard
- REST API rate limiting
- Request/response caching
