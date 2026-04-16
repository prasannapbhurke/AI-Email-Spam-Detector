from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np


LabelValue = Union[int, str]


@dataclass(frozen=True)
class DatasetConfig:
    text_col: str = "text"
    label_col: str = "label"
    # Common binary mapping for spam detection.
    spam_label_values: Tuple[str, ...] = ("spam", "1", "true", "yes")
    ham_label_values: Tuple[str, ...] = ("ham", "0", "false", "no")


def _default_label_to_int(value: LabelValue, cfg: DatasetConfig) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)

    s = str(value).strip().lower()
    if s in cfg.spam_label_values:
        return 1
    if s in cfg.ham_label_values:
        return 0
    # If user already supplies numeric-like labels, try coercion.
    try:
        return int(float(s))
    except ValueError as e:
        raise ValueError(f"Unrecognized label value: {value!r}") from e


def load_dataset_from_csv(
    path: Union[str, os.PathLike],
    config: Optional[DatasetConfig] = None,
    label_to_int: Optional[Callable[[LabelValue, DatasetConfig], int]] = None,
    encoding: str = "utf-8",
) -> tuple[list[str], np.ndarray]:
    """
    Load a labeled dataset from CSV.

    Expected columns:
      - text_col (default: 'text')
      - label_col (default: 'label')

    label mapping:
      - spam-like strings -> 1
      - ham-like strings -> 0
    """

    cfg = config or DatasetConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {str(path)}")

    mapper = label_to_int or _default_label_to_int

    texts: list[str] = []
    labels: list[int] = []

    # Use csv module to avoid forcing a heavy dependency in this training core.
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No headers found in CSV: {str(path)}")
        if cfg.text_col not in reader.fieldnames or cfg.label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{cfg.text_col}' and '{cfg.label_col}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            raw_text = row.get(cfg.text_col, "")
            raw_label = row.get(cfg.label_col, "")
            texts.append("" if raw_text is None else str(raw_text))
            labels.append(mapper(raw_label, cfg))

    if not texts:
        raise ValueError("Dataset is empty.")

    return texts, np.asarray(labels, dtype=np.int64)


def load_sample_dataset() -> tuple[list[str], np.ndarray]:
    """
    Small built-in dataset for smoke testing the pipeline.

    For real training, use a dataset CSV and call `load_dataset_from_csv`.
    """

    samples = [
        ("Congratulations! You have won a free lottery ticket. Claim now.", 1),
        ("Limited time offer!!! Buy now and save 50% on all items.", 1),
        ("Your invoice is attached. Please review and let us know.", 0),
        ("Meeting scheduled for tomorrow at 10am. Please confirm.", 0),
        ("Earn money fast by working from home. No experience needed.", 1),
        ("Could you please review the attached document and send feedback?", 0),
        ("Get cheap meds delivered to your door. Order today.", 1),
        ("Lunch at noon? Let me know if you are available.", 0),
    ]

    texts = [t for t, _ in samples]
    y = np.asarray([lbl for _, lbl in samples], dtype=np.int64)
    return texts, y

