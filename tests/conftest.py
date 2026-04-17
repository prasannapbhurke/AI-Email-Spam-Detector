"""Email Spam Detector - Complete Test Suite."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
