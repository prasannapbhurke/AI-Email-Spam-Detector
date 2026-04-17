#!/usr/bin/env python3
"""Evaluate the trained spam detection model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spam_detector.ml.model_service import ModelService
from spam_detector.training.dataset import load_dataset_from_csv
from spam_detector.training.evaluation import classification_metrics
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate spam detection model")
    parser.add_argument("--model", type=str, default="models/spam_detector.joblib",
                       help="Path to saved model")
    parser.add_argument("--data", type=str, default="data/spam_dataset.csv",
                       help="Path to evaluation dataset")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    svc = ModelService(args.model)
    svc.load()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    texts, y_true = load_dataset_from_csv(args.data)

    # Predict
    print("Running predictions...")
    y_pred = []
    y_proba = []
    for text in texts:
        result = svc.predict(text)
        y_pred.append(1 if result.prediction_label == "spam" else 0)
        y_proba.append(result.spam_probability)

    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    # Metrics
    metrics = classification_metrics(y_true, y_pred)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset size: {len(texts)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("="*50)

    # Save results
    results_file = ROOT / "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Dataset size: {len(texts)}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")

    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
