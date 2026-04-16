#!/usr/bin/env python3
"""
Model retraining script with self-learning capabilities.

Usage:
    python scripts/retrain_model.py --feedback-only

    python scripts/retrain_model.py --original-data data/original.csv --feedback-data data/feedback.csv

    python scripts/retrain_model.py --continuous --interval 3600
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from spam_detector.core.logging import configure_logging as setup_logging
from spam_detector.db.mongo import get_collection
from spam_detector.learning import FeedbackCollector, LearningConfig, RetrainingPipeline, ScheduledRetrainer
from spam_detector.ml.model_service import ModelService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain spam detection model with user feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--original-data",
        type=str,
        default="data/spam_dataset.csv",
        help="Path to original training data CSV",
    )
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Train only on feedback data (no original dataset)",
    )
    parser.add_argument(
        "--min-feedback",
        type=int,
        default=50,
        help="Minimum feedback samples to trigger retraining (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/spam_detector.joblib",
        help="Output model path (default: models/spam_detector.joblib)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously with scheduled retraining",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Check interval in seconds for continuous mode (default: 3600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check if retraining is needed without running it",
    )
    parser.add_argument(
        "--export-feedback",
        action="store_true",
        help="Export feedback to CSV and exit",
    )
    parser.add_argument(
        "--feedback-output",
        type=str,
        default=None,
        help="Output path for feedback export",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def get_model_service() -> ModelService:
    """Initialize model service."""
    model_path = os.getenv("MODEL_PATH", "models/spam_detector.joblib")
    spam_label = int(os.getenv("MODEL_SPAM_LABEL", "1"))
    svc = ModelService(model_path, spam_label=spam_label)
    svc.load()
    return svc


def main() -> int:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")

    logger = logging.getLogger(__name__)

    # Initialize components
    config = LearningConfig(
        min_feedback_for_retrain=args.min_feedback,
        feedback_data_dir="data/feedback",
    )

    try:
        collection = get_collection()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", e)
        print(f"Error: MongoDB connection failed: {e}")
        return 1

    feedback_collector = FeedbackCollector(collection, config)

    # Export feedback if requested
    if args.export_feedback:
        output = feedback_collector.export_to_csv(args.feedback_output)
        print(f"Exported feedback to: {output}")
        stats = feedback_collector.get_feedback_stats()
        print("\nFeedback Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    # Check if retraining is needed
    pipeline = RetrainingPipeline(
        feedback_collector=feedback_collector,
        model_service=get_model_service(),
        config=config,
    )

    should_retrain, reason = pipeline.should_retrain()

    if args.dry_run:
        print(f"Retraining check: {reason}")
        print(f"Should retrain: {should_retrain}")
        if should_retrain:
            stats = feedback_collector.get_feedback_stats()
            print("\nFeedback Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        return 0

    # Continuous mode
    if args.continuous:
        logger.info("Starting continuous retraining mode (interval=%ds)", args.interval)
        print(f"Running in continuous mode (checking every {args.interval}s)")
        print("Press Ctrl+C to stop.\n")

        retrainer = ScheduledRetrainer(pipeline, check_interval=args.interval)

        try:
            retrainer.start(daemon=True)
            while True:
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            retrainer.stop()
        return 0

    # Single run
    if not should_retrain:
        logger.info("Retraining not needed: %s", reason)
        print(f"Retraining not needed: {reason}")
        return 0

    logger.info("Starting retraining: %s", reason)
    print(f"\nStarting retraining: {reason}\n")

    results = pipeline.run(
        original_dataset_path=None if args.feedback_only else args.original_data,
        validate_before_deploy=True,
        backup_model=True,
    )

    if results["status"] == "success":
        print("\n" + "=" * 50)
        print("RETRAINING SUCCESSFUL")
        print("=" * 50)
        print(f"  Version: {results.get('model_version')}")
        print(f"  Duration: {results.get('total_duration', 0):.2f}s")
        print("\nSteps:")
        for step in results.get("steps", []):
            status = "✓" if step["status"] == "success" else "✗"
            print(f"  {status} {step['step']}: {step['status']} ({step.get('duration', 0):.2f}s)")

        if results.get("steps"):
            training_step = next((s for s in results["steps"] if s["step"] == "training"), None)
            if training_step and training_step.get("metrics"):
                print("\nTraining Metrics:")
                for key, value in training_step["metrics"].items():
                    print(f"  {key}: {value}")

        return 0
    else:
        print("\n" + "=" * 50)
        print("RETRAINING FAILED")
        print("=" * 50)
        print(f"  Error: {results.get('error')}")
        print("\nSteps:")
        for step in results.get("steps", []):
            status = "✓" if step["status"] == "success" else "✗"
            print(f"  {status} {step['step']}: {step['status']}")
            if step.get("stderr"):
                print(f"    Error: {step['stderr'][:500]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
