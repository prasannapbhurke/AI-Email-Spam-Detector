"""Pytest configuration and fixtures for the Email Spam Detector tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from spam_detector.api.main import app


@pytest.fixture(scope="session")
def client():
    """TestClient fixture with proper lifespan handling."""
    with TestClient(app) as client:
        yield client
