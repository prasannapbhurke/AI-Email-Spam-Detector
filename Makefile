.PHONY: help install test lint format clean run-api run-train docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests with pytest"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Run code quality checks (black, isort, flake8, mypy)"
	@echo "  make format       Auto-format code with black and isort"
	@echo "  make clean        Clean up compiled files and caches"
	@echo "  make run-api      Start FastAPI server"
	@echo "  make run-train    Train the spam detection model"
	@echo "  make retrain      Run retraining pipeline"
	@echo "  make docker-up    Start services with docker-compose"
	@echo "  make docker-down  Stop docker-compose services"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	python -m pytest -v

test-cov:
	python -m pytest --cov=src --cov-report=html

lint:
	black --check src tests scripts
	isort --check-only src tests scripts
	flake8 src tests scripts
	mypy src

format:
	black src tests scripts
	isort src tests scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ 2>/dev/null || true

run-api:
	python scripts/run_api.py --port 8000

run-train:
	python scripts/train_spam_detector.py --data data/spam_dataset.csv --output models/spam_detector.joblib

retrain:
	python scripts/retrain_model.py --dry-run
	python scripts/retrain_model.py --original-data data/spam_dataset.csv

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
