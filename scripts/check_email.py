#!/usr/bin/env python3
"""Quick interactive test - check if an email is spam without Gmail integration."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from spam_detector.ml.model_service import ModelService

def main():
    print("=" * 60)
    print("Email Spam Checker - Quick Test")
    print("=" * 60)
    print()
    
    # Load model
    model_path = project_root / "models" / "spam_detector.joblib"
    print(f"Loading model from {model_path}...")
    svc = ModelService(str(model_path))
    svc.load()
    print("✓ Model loaded\n")
    
    print("Paste your email content below.")
    print("Press Enter twice (empty line) when done:\n")
    
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    email_text = "\n".join(lines).strip()
    
    if not email_text:
        print("No email content provided.")
        return
    
    print()
    print("-" * 60)
    print("Email content:")
    print("-" * 60)
    print(email_text[:500] + ("..." if len(email_text) > 500 else ""))
    print("-" * 60)
    print()
    
    # Predict
    result = svc.predict(email_text)
    
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Prediction:  {result.prediction_label.upper()}")
    print(f"Confidence:  {result.confidence:.2%}")
    print(f"Spam Score:  {result.spam_probability:.2%}")
    print("=" * 60)
    print()
    
    if result.prediction_label == "spam":
        print("⚠️  This email is likely SPAM. Be cautious!")
    else:
        print("✓  This email appears to be legitimate.")
    print()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(130)
