"""
Spam processor that integrates Gmail API with ML spam detection.

Orchestrates the full pipeline: fetch emails, classify, and apply actions.
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import requests

from spam_detector.gmail.auth import GmailAuthenticator
from spam_detector.gmail.client import GmailClient, EmailMessage, LABEL_IMPORTANT, LABEL_PROMOTIONS, LABEL_SPAM
from spam_detector.gmail.exceptions import GmailAPIError, RateLimitError

logger = logging.getLogger(__name__)


class SpamClassification(Enum):
    """Spam classification result categories."""

    SPAM = "spam"
    NOT_SPAM = "not_spam"
    UNCERTAIN = "uncertain"


@dataclass
class ProcessingResult:
    """Result of processing a single email."""

    message_id: str
    classification: SpamClassification
    confidence: float
    spam_probability: float
    action_taken: str
    error: str | None = None


@dataclass
class ProcessingStats:
    """Statistics for a batch of processed emails."""

    total: int = 0
    spam: int = 0
    not_spam: int = 0
    uncertain: int = 0
    errors: int = 0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "spam": self.spam,
            "not_spam": self.not_spam,
            "uncertain": self.uncertain,
            "errors": self.errors,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
        }


class GmailSpamProcessor:
    """
    Processes Gmail inbox for spam detection and classification.

    Integrates with ML prediction API to classify emails and apply
    appropriate Gmail labels/actions based on predictions.
    """

    # Confidence thresholds for classification
    SPAM_THRESHOLD = 0.7       # Above this → mark as spam
    NOT_SPAM_THRESHOLD = 0.3   # Below this → marked as important/promotions
    UNCERTAIN_BAND = 0.1       # Within band → uncertain (no action)

    def __init__(
        self,
        authenticator: GmailAuthenticator,
        prediction_api_url: str,
        api_key: str | None = None,
        confidence_threshold: float | None = None,
        uncertain_threshold: float | None = None,
        dry_run: bool = False,
        http_session: requests.Session | None = None,
    ):
        """
        Initialize Gmail spam processor.

        Args:
            authenticator: GmailAuthenticator for API access.
            prediction_api_url: URL of the ML prediction API.
            api_key: Optional API key for prediction endpoint.
            confidence_threshold: Override for spam classification threshold.
            uncertain_threshold: Override for uncertain band size.
            dry_run: If True, don't modify emails (for testing).
            http_session: Optional requests session for HTTP calls.
        """
        self._authenticator = authenticator
        self._client = GmailClient(authenticator)
        self._prediction_api_url = prediction_api_url.rstrip("/")
        self._api_key = api_key or os.getenv("PREDICTION_API_KEY")
        self._dry_run = dry_run

        if confidence_threshold is not None:
            self.SPAM_THRESHOLD = confidence_threshold
        if uncertain_threshold is not None:
            self.UNCERTAIN_BAND = uncertain_threshold

        self._session = http_session or requests.Session()
        if api_key:
            self._session.headers.update({"X-API-Key": api_key})

        self._rate_limit_delay = 0.5  # Delay between API calls

    def process_unread(
        self,
        max_emails: int = 100,
        progress_callback: Callable[[ProcessingResult], None] | None = None,
    ) -> ProcessingStats:
        """
        Process all unread emails in inbox.

        Args:
            max_emails: Maximum number of emails to process.
            progress_callback: Optional callback for each processed email.

        Returns:
            ProcessingStats with processing statistics.
        """
        import time

        start_time = time.time()
        stats = ProcessingStats()

        logger.info(
            "Starting spam processing (dry_run=%s, max_emails=%d)",
            self._dry_run,
            max_emails,
        )

        try:
            for email in self._client.get_unread_messages(max_results=max_emails):
                stats.total += 1

                result = self._process_single_email(email)
                stats += self._result_to_stats(result)

                if progress_callback:
                    progress_callback(result)

                # Rate limiting between emails
                time.sleep(self._rate_limit_delay)

        except RateLimitError:
            logger.error("Rate limit hit during processing, returning partial stats")
        except GmailAPIError as e:
            logger.error("Gmail API error during processing: %s", e)

        stats.processing_time_seconds = time.time() - start_time
        logger.info(
            "Processing complete: %s (took %.2fs)",
            stats.to_dict(),
            stats.processing_time_seconds,
        )

        return stats

    def _process_single_email(self, email: EmailMessage) -> ProcessingResult:
        """
        Process a single email through the spam detection pipeline.

        Args:
            email: EmailMessage to process.

        Returns:
            ProcessingResult with classification and action taken.
        """
        logger.debug(
            "Processing email %s: subject='%s'",
            email.id,
            email.subject[:50],
        )

        try:
            # Get spam prediction
            spam_prob = self._call_prediction_api(email)
            confidence = abs(spam_prob - 0.5) * 2  # Convert to 0-1 confidence

            # Classify
            if spam_prob >= self.SPAM_THRESHOLD:
                classification = SpamClassification.SPAM
            elif spam_prob <= self.NOT_SPAM_THRESHOLD:
                classification = SpamClassification.NOT_SPAM
            else:
                classification = SpamClassification.UNCERTAIN

            # Determine action
            action = self._apply_classification_action(email, classification)

            return ProcessingResult(
                message_id=email.id,
                classification=classification,
                confidence=confidence,
                spam_probability=spam_prob,
                action_taken=action,
                error=None,
            )

        except RateLimitError:
            return ProcessingResult(
                message_id=email.id,
                classification=SpamClassification.UNCERTAIN,
                confidence=0.0,
                spam_probability=0.5,
                action_taken="rate_limited",
                error="Rate limit hit",
            )
        except Exception as e:
            logger.error("Error processing email %s: %s", email.id, e)
            return ProcessingResult(
                message_id=email.id,
                classification=SpamClassification.UNCERTAIN,
                confidence=0.0,
                spam_probability=0.5,
                action_taken="error",
                error=str(e),
            )

    def _call_prediction_api(self, email: EmailMessage) -> float:
        """
        Send email to prediction API and get spam probability.

        Args:
            email: EmailMessage to classify.

        Returns:
            Spam probability between 0 and 1.
        """
        # Combine subject and body for prediction
        text = f"Subject: {email.subject}\n\n{email.body_text}"

        payload = {"email_text": text}

        # Retry logic for API calls
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self._session.post(
                    f"{self._prediction_api_url}/predict",
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", base_delay * 2))
                    logger.warning(
                        "Prediction API rate limited, waiting %ds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        max_retries,
                    )
                    if attempt == max_retries - 1:
                        raise RateLimitError("Prediction API rate limit exceeded")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                result = response.json()

                # Extract spam probability from response
                spam_prob = result.get("spam_probability", result.get("spam_confidence", 0.5))
                return float(spam_prob)

            except requests.exceptions.Timeout:
                logger.warning(
                    "Prediction API timeout (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )
                if attempt == max_retries - 1:
                    raise GmailAPIError("Prediction API timeout after retries")
                time.sleep(base_delay * (2**attempt))
            except requests.exceptions.RequestException as e:
                logger.error("Prediction API error: %s", e)
                raise GmailAPIError(f"Prediction API error: {e}") from e

        return 0.5  # Default to uncertain on failure

    def _apply_classification_action(
        self,
        email: EmailMessage,
        classification: SpamClassification,
    ) -> str:
        """
        Apply the appropriate Gmail action based on classification.

        Args:
            email: EmailMessage to modify.
            classification: Classification result.

        Returns:
            Description of action taken.
        """
        if self._dry_run:
            logger.info("[DRY RUN] Would %s for %s", classification.value, email.id)
            return f"dry_run:{classification.value}"

        try:
            if classification == SpamClassification.SPAM:
                # Mark as spam and move to spam folder
                self._client.mark_as_spam(email.id)
                return f"marked_spam:{email.id}"

            elif classification == SpamClassification.NOT_SPAM:
                # Apply category labels based on content
                action = self._apply_category_labels(email)
                return action

            else:  # UNCERTAIN
                # Don't modify - leave for manual review
                logger.info("Email %s uncertain, leaving for manual review", email.id)
                return f"skipped_uncertain:{email.id}"

        except GmailAPIError as e:
            logger.error("Failed to apply action for %s: %s", email.id, e)
            return f"error:{e}"

    def _apply_category_labels(self, email: EmailMessage) -> str:
        """
        Apply category labels (Important/Promotions) to non-spam email.

        Args:
            email: EmailMessage to label.

        Returns:
            Description of labels applied.
        """
        labels_added = []

        # Heuristic: promotional keywords suggest Promotions label
        promo_keywords = [
            "offer", "sale", "discount", "promo", "deal",
            "subscribe", "newsletter", "unsubscribe", "advertisement",
        ]

        combined_text = f"{email.subject} {email.body_text}".lower()

        if any(kw in combined_text for kw in promo_keywords):
            if LABEL_PROMOTIONS not in email.labels:
                self._client.add_labels(email.id, [LABEL_PROMOTIONS])
                labels_added.append(LABEL_PROMOTIONS)
        else:
            # Mark as important if no promo indicators
            if LABEL_IMPORTANT not in email.labels:
                self._client.add_labels(email.id, [LABEL_IMPORTANT])
                labels_added.append(LABEL_IMPORTANT)

        if labels_added:
            return f"added_labels:{labels_added}"

        return f"no_labels_needed:{email.id}"

    def _result_to_stats(self, result: ProcessingResult) -> ProcessingStats:
        """Convert ProcessingResult to stats increment."""
        stats = ProcessingStats()
        if result.error:
            stats.errors = 1
        elif result.classification == SpamClassification.SPAM:
            stats.spam = 1
        elif result.classification == SpamClassification.NOT_SPAM:
            stats.not_spam = 1
        else:
            stats.uncertain = 1
        return stats

    def setup_labels(self) -> None:
        """
        Ensure required labels exist in Gmail.

        Creates SPAM, IMPORTANT, and PROMOTIONS labels if they don't exist.
        """
        existing_labels = {label["name"] for label in self._client.get_labels()}
        required_labels = [LABEL_SPAM, LABEL_IMPORTANT, LABEL_PROMOTIONS]

        for label_name in required_labels:
            if label_name not in existing_labels:
                label_id = self._client.create_label(label_name)
                if label_id:
                    logger.info("Created label: %s", label_name)
                else:
                    logger.warning("Failed to create label: %s", label_name)
            else:
                logger.debug("Label already exists: %s", label_name)
