"""Tests for dataset loading."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from spam_detector.training.dataset import load_dataset_from_csv, DatasetConfig, _default_label_to_int


def test_default_label_mapping():
    """Test label conversion function."""
    cfg = DatasetConfig()

    # Spam values
    assert _default_label_to_int("spam", cfg) == 1
    assert _default_label_to_int("1", cfg) == 1
    assert _default_label_to_int("yes", cfg) == 1
    assert _default_label_to_int("true", cfg) == 1

    # Ham values
    assert _default_label_to_int("ham", cfg) == 0
    assert _default_label_to_int("0", cfg) == 0
    assert _default_label_to_int("no", cfg) == 0
    assert _default_label_to_int("false", cfg) == 0

    # Numeric
    assert _default_label_to_int(1, cfg) == 1
    assert _default_label_to_int(0, cfg) == 0


def test_load_dataset():
    """Test loading the spam dataset CSV."""
    data_path = project_root / "data" / "spam_dataset.csv"
    assert data_path.exists(), f"Dataset not found at {data_path}"

    texts, labels = load_dataset_from_csv(str(data_path))

    assert len(texts) > 0
    assert len(labels) > 0
    assert len(texts) == len(labels)
    # Labels should be binary 0/1
    assert set(labels).issubset({0, 1})


def test_dataset_balanced():
    """Check dataset has both spam and ham."""
    data_path = project_root / "data" / "spam_dataset.csv"
    texts, labels = load_dataset_from_csv(str(data_path))

    unique_labels = set(labels)
    assert 0 in unique_labels, "No ham (0) labels in dataset"
    assert 1 in unique_labels, "No spam (1) labels in dataset"
