"""
Self-learning module for spam detection.

Handles:
- Feedback storage and retrieval
- Incremental learning with partial_fit
- Retraining pipeline
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Threshold for auto-include feedback in training
MIN_FEEDBACK_CONFIDENCE = 0.7


@dataclass
class LearningConfig:
    """Configuration for self-learning module."""

    # Minimum feedback samples before triggering retraining
    min_feedback_for_retrain: int = 50

    # Auto-retrain when this % of model predictions have feedback
    retrain_threshold_percent: float = 0.1

    # Maximum feedback samples to keep per model version
    max_feedback_per_version: int = 10000

    # Confidence threshold for high-quality feedback
    high_confidence_threshold: float = 0.85

    # Path to store retraining data
    feedback_data_dir: str = "data/feedback"

    # Enable incremental learning (partial_fit)
    enable_incremental: bool = True

    # Batch size for incremental learning
    incremental_batch_size: int = 32


class FeedbackCollector:
    """
    Collects and manages user feedback on predictions.

    Stores feedback in MongoDB and exports to training dataset.
    """

    def __init__(
        self,
        collection,
        config: LearningConfig | None = None,
    ):
        from spam_detector.db.repositories.prediction_repo import PredictionRepository

        self._repo = PredictionRepository(collection)
        self.config = config or LearningConfig()
        self._setup_data_dir()

    def _setup_data_dir(self) -> None:
        """Ensure feedback data directory exists."""
        Path(self.config.feedback_data_dir).mkdir(parents=True, exist_ok=True)

    def submit_feedback(
        self,
        prediction_id: str,
        correct_label: Literal["spam", "not_spam"],
        source: str = "user",
    ) -> bool:
        """
        Submit feedback for a prediction.

        Args:
            prediction_id: MongoDB ID of the prediction.
            correct_label: Ground truth label (spam/not_spam).
            source: Feedback source (user/admin/gmail).

        Returns:
            True if feedback was stored successfully.
        """
        try:
            self._repo.set_feedback(
                prediction_id=prediction_id,
                label=correct_label,
                source=source,
            )
            logger.info(
                "Feedback stored: prediction_id=%s label=%s source=%s",
                prediction_id,
                correct_label,
                source,
            )
            return True
        except KeyError:
            logger.error("Prediction not found: %s", prediction_id)
            return False
        except Exception as e:
            logger.error("Failed to store feedback: %s", e)
            return False

    def get_pending_feedback(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get predictions that have user feedback but model hasn't been retrained.

        Args:
            limit: Maximum number of samples to return.

        Returns:
            List of prediction documents with feedback.
        """
        from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM

        # Query predictions with feedback
        collection = self._repo.collection
        cursor = collection.find(
            {
                "feedback": {"$exists": True, "$ne": None},
                "$or": [
                    {"model_version_retrained_at": {"$exists": False}},
                    {
                        "feedback_at": {"$gt": "$model_version_retrained_at"}
                    },
                ],
            }
        ).sort("feedback_at", -1).limit(limit)

        return list(cursor)

    def get_feedback_stats(self) -> dict[str, Any]:
        """
        Get statistics about collected feedback.

        Returns:
            Dictionary with feedback counts and metrics.
        """
        from spam_detector.db.repositories.prediction_repo import LABEL_SPAM, LABEL_NOT_SPAM

        collection = self._repo.collection

        total = collection.count_documents({"feedback": {"$exists": True, "$ne": None}})
        spam = collection.count_documents({"feedback.label": LABEL_SPAM})
        ham = collection.count_documents({"feedback.label": LABEL_NOT_SPAM})

        # High confidence feedback (user strongly disagreed)
        high_conf = collection.count_documents(
            {
                "feedback": {"$exists": True, "$ne": None},
                "$expr": {
                    "$gt": [
                        {"$subtract": [1, {"$abs": {"$subtract": ["$confidence", "$spam_confidence"]}}]},
                        0.7,
                    ]
                },
            }
        )

        # Recent feedback (last 24h)
        from datetime import timedelta

        day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        recent = collection.count_documents(
            {"feedback.feedback_at": {"$gte": day_ago}}
        )

        return {
            "total_feedback": total,
            "spam_feedback": spam,
            "ham_feedback": ham,
            "high_confidence_disagreement": high_conf,
            "recent_24h": recent,
        }

    def export_to_csv(self, output_path: str | None = None) -> str:
        """
        Export feedback data to CSV for retraining.

        Args:
            output_path: Custom output path. If None, generates timestamped name.

        Returns:
            Path to exported CSV file.
        """
        from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config.feedback_data_dir}/feedback_{timestamp}.csv"

        collection = self._repo.collection
        cursor = collection.find(
            {"feedback": {"$exists": True, "$ne": None}},
            {"email_text": 1, "feedback.label": 1},
        )

        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()

            for doc in cursor:
                label = doc.get("feedback", {}).get("label")
                if label:
                    writer.writerow({
                        "text": doc.get("email_text", ""),
                        "label": label,
                    })

        logger.info("Exported feedback to %s", output_path)
        return output_path

    def mark_retrained(self, retrain_time: datetime | None = None) -> None:
        """
        Mark all current feedback as incorporated into model.

        Args:
            retrain_time: Time of retraining. Defaults to now.
        """
        from spam_detector.db.repositories.prediction_repo import LABEL_NOT_SPAM, LABEL_SPAM

        retrain_time = retrain_time or datetime.now(timezone.utc)

        self._repo.collection.update_many(
            {"feedback": {"$exists": True, "$ne": None}},
            {"$set": {"model_version_retrained_at": retrain_time}},
        )
        logger.info("Marked all feedback as incorporated at %s", retrain_time)


class IncrementalLearner:
    """
    Performs incremental/online learning using partial_fit.

    Updates model weights without full retraining.
    """

    def __init__(
        self,
        model,
        config: LearningConfig | None = None,
    ):
        """
        Initialize incremental learner.

        Args:
            model: sklearn-compatible model with partial_fit method.
            config: Learning configuration.
        """
        self.model = model
        self.config = config or LearningConfig()
        self._classes = np.array([0, 1])  # spam=1, not_spam=0
        self._batches_processed = 0

    def partial_fit(
        self,
        texts: list[str],
        labels: list[int],
        preprocess_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """
        Incrementally train on new data.

        Args:
            texts: Raw email texts.
            labels: Integer labels (0=ham, 1=spam).
            preprocess_fn: Optional text preprocessing function.

        Returns:
            Training metrics dictionary.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")

        if not texts:
            return {"status": "skipped", "reason": "no_data"}

        # Apply preprocessing if provided
        if preprocess_fn:
            texts = [preprocess_fn(t) for t in texts]

        X = np.array(texts)
        y = np.array(labels)

        # Process in batches
        batch_size = self.config.incremental_batch_size
        metrics = {"samples_processed": 0, "batches": 0}

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            try:
                self.model.partial_fit(batch_X, batch_y, classes=self._classes)
                metrics["samples_processed"] += len(batch_X)
                metrics["batches"] += 1
                self._batches_processed += 1
            except Exception as e:
                logger.error("Partial fit error on batch %d: %s", i, e)
                metrics["error"] = str(e)

        logger.info(
            "Incremental learning complete: %d samples in %d batches",
            metrics["samples_processed"],
            metrics["batches"],
        )

        return metrics


class RetrainingPipeline:
    """
    Orchestrates model retraining with feedback data.

    Workflow:
    1. Collect feedback from MongoDB
    2. Export to CSV
    3. Retrain model with combined dataset
    4. Evaluate and validate
    5. Deploy new model version
    """

    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        model_service: Any,  # ModelService
        config: LearningConfig | None = None,
    ):
        self.feedback = feedback_collector
        self.model_service = model_service
        self.config = config or LearningConfig()

    def should_retrain(self) -> tuple[bool, str]:
        """
        Check if model should be retrained.

        Returns:
            Tuple of (should_retrain, reason).
        """
        stats = self.feedback.get_feedback_stats()

        total = stats["total_feedback"]
        if total < self.config.min_feedback_for_retrain:
            return False, f"Only {total} feedback samples (need {self.config.min_feedback_for_retrain})"

        # Check if enough feedback accumulated relative to total predictions
        # (simplified - would need total predictions from stats)
        recent_ratio = stats["recent_24h"] / max(total, 1)
        if stats["recent_24h"] > 10 and recent_ratio > 0.3:
            return True, f"High feedback velocity: {stats['recent_24h']} in last 24h"

        return True, f"Sufficient feedback accumulated: {total} samples"

    def run(
        self,
        original_dataset_path: str | None = None,
        validate_before_deploy: bool = True,
        backup_model: bool = True,
    ) -> dict[str, Any]:
        """
        Run full retraining pipeline.

        Args:
            original_dataset_path: Path to original training data CSV.
            validate_before_deploy: Run validation before switching model.
            backup_model: Backup current model before deploying.

        Returns:
            Dictionary with retraining results and metrics.
        """
        logger.info("Starting retraining pipeline...")
        start_time = time.time()

        results = {
            "status": "started",
            "start_time": start_time,
            "steps": [],
        }

        # Step 1: Export feedback to CSV
        step_start = time.time()
        try:
            feedback_csv = self.feedback.export_to_csv()
            results["steps"].append({
                "step": "export_feedback",
                "status": "success",
                "output": feedback_csv,
                "duration": time.time() - step_start,
            })
        except Exception as e:
            results["status"] = "failed"
            results["error"] = f"Export failed: {e}"
            return results

        # Step 2: Run training script with combined data
        step_start = time.time()
        try:
            if original_dataset_path:
                # Combine original + feedback for training
                combined_csv = self._combine_datasets(original_dataset_path, feedback_csv)
            else:
                combined_csv = feedback_csv

            training_results = self._run_training(combined_csv)
            results["steps"].append({
                "step": "training",
                "status": "success" if training_results["success"] else "failed",
                "metrics": training_results.get("metrics"),
                "duration": time.time() - step_start,
            })

            if not training_results["success"]:
                results["status"] = "failed"
                results["error"] = "Training failed"
                return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = f"Training error: {e}"
            return results

        # Step 3: Validate if requested
        if validate_before_deploy:
            step_start = time.time()
            validation = self._validate_model(training_results["model_path"])
            results["steps"].append({
                "step": "validation",
                "status": "success" if validation["passed"] else "failed",
                "metrics": validation,
                "duration": time.time() - step_start,
            })

            if not validation["passed"]:
                results["status"] = "failed"
                results["error"] = f"Validation failed: {validation.get('reason')}"
                return results

        # Step 4: Deploy new model
        step_start = time.time()
        try:
            self._deploy_model(training_results["model_path"], backup=backup_model)
            results["steps"].append({
                "step": "deploy",
                "status": "success",
                "duration": time.time() - step_start,
            })
        except Exception as e:
            results["status"] = "failed"
            results["error"] = f"Deploy failed: {e}"
            return results

        # Step 5: Mark feedback as incorporated
        self.feedback.mark_retrained()

        results["status"] = "success"
        results["total_duration"] = time.time() - start_time
        results["model_version"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Retraining pipeline complete: %s", results)
        return results

    def _combine_datasets(self, original: str, feedback: str) -> str:
        """Combine original dataset with feedback data."""
        import csv
        import shutil

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined = f"{self.config.feedback_data_dir}/combined_{timestamp}.csv"

        # Copy original
        shutil.copy(original, combined)

        # Append feedback
        with open(feedback, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            with open(combined, "a", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=["text", "label"])
                for row in reader:
                    writer.writerow(row)

        logger.info("Combined datasets: %s + %s -> %s", original, feedback, combined)
        return combined

    def _run_training(self, dataset_path: str) -> dict[str, Any]:
        """Run the training script."""
        script_path = Path(__file__).parent.parent / "scripts" / "train_spam_detector.py"
        output_path = Path(self.config.feedback_data_dir) / "models" / f"retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--data", dataset_path,
            "--output", str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            success = result.returncode == 0
            metrics = {}

            if success:
                # Parse metrics from output
                for line in result.stdout.split("\n"):
                    if ":" in line and "threshold" in line.lower():
                        parts = line.split(":")
                        if len(parts) >= 2:
                            metrics[parts[0].strip()] = parts[1].strip()

            return {
                "success": success,
                "model_path": str(output_path) if success else None,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-2000:],
                "metrics": metrics,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Training timeout (>1h)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_model(self, model_path: str) -> dict[str, Any]:
        """Validate new model before deployment."""
        # Basic validation - load and run predictions
        try:
            loaded, _ = self.model_service.load_model(model_path)
            test_texts = [
                "WIN FREE MONEY!!! Click here now!!!",
                "Hi, let's meet for lunch tomorrow",
            ]
            preds = loaded.predict(test_texts)
            return {
                "passed": True,
                "predictions": list(preds),
            }
        except Exception as e:
            return {
                "passed": False,
                "reason": str(e),
            }

    def _deploy_model(
        self,
        new_model_path: str,
        backup: bool = True,
    ) -> None:
        """Deploy new model, optionally backing up current."""
        current_path = Path(self.model_service.model_path)

        if backup and current_path.exists():
            backup_path = current_path.with_suffix(".joblib.backup")
            shutil.copy(current_path, backup_path)
            logger.info("Backed up current model to %s", backup_path)

        # Copy new model to production path
        shutil.copy(new_model_path, current_path)
        logger.info("Deployed new model from %s to %s", new_model_path, current_path)

        # Reload model service
        self.model_service.load()


class ScheduledRetrainer:
    """
    Runs retraining on a schedule.

    Integrates with cron/scheduler for automated learning.
    """

    def __init__(
        self,
        pipeline: RetrainingPipeline,
        check_interval: int = 3600,  # Check every hour
    ):
        self.pipeline = pipeline
        self.check_interval = check_interval
        self._running = False

    def start(self, daemon: bool = True) -> None:
        """
        Start scheduled retraining loop.

        Args:
            daemon: If True, runs in background thread.
        """
        import threading

        self._running = True

        if daemon:
            thread = threading.Thread(target=self._loop, daemon=True)
            thread.start()
            logger.info("Scheduled retrainer started (daemon)")
        else:
            self._loop()

    def stop(self) -> None:
        """Stop the retraining loop."""
        self._running = False

    def _loop(self) -> None:
        """Main scheduling loop."""
        import time

        while self._running:
            try:
                should_retrain, reason = self.pipeline.should_retrain()
                if should_retrain:
                    logger.info("Triggering retrain: %s", reason)
                    results = self.pipeline.run()
                    if results["status"] == "success":
                        logger.info("Retrain successful: version=%s", results.get("model_version"))
                    else:
                        logger.error("Retrain failed: %s", results.get("error"))
                else:
                    logger.debug("Skipping retrain: %s", reason)
            except Exception as e:
                logger.error("Error in retrain loop: %s", e)

            time.sleep(self.check_interval)
