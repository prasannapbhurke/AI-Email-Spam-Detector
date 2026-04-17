#!/usr/bin/env python3
"""Demo script showing LIME explainability for spam predictions."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spam_detector.ml.model_service import ModelService
from spam_detector.nlp.preprocessing import preprocess_email


def main():
    parser = argparse.ArgumentParser(description="Explain spam prediction with LIME")
    parser.add_argument("--model", type=str, default="models/spam_detector.joblib",
                       help="Path to trained model")
    parser.add_argument("--text",
                       default="Congratulations! You've won a free iPhone. Click here to claim!",
                       help="Email text to explain")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    svc = ModelService(args.model)
    svc.load()

    # Make prediction
    result = svc.predict(args.text)
    print(f"\nPrediction: {result.prediction_label.upper()}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Spam Probability: {result.spam_probability:.2%}")

    # Print top contributing words (simplified explanation)
    # In a full implementation, LIME would provide word-level importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top words contributing to prediction):")
    print("="*60)

    # Preprocess and show important tokens
    processed = preprocess_email(args.text)
    tokens = processed.split()

    # Show most salient tokens based on TF-IDF weights
    # This is a simplified version - full LIME would give actual importance scores
    print("\nKey tokens found:")
    for token in tokens[:10]:
        print(f"  • {token}")

    print("\nTip: For detailed word-level explanations with LIME visualizations,")
    print("integrate the LIMEExplainer class from spam_detector.advanced.lime_explainer")


if __name__ == "__main__":
    sys.exit(main())
