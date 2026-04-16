"""
Gmail API client for fetching and modifying emails.

Handles all Gmail API operations with proper error handling and rate limiting.
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator

from google.api_core import exceptions as google_exceptions
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from spam_detector.gmail.auth import GmailAuthenticator
from spam_detector.gmail.exceptions import (
    EmailFetchError,
    EmailModifyError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Gmail API constants
GMAIL_API_VERSION = "v1"
GMAIL_SERVICE_NAME = "gmail"

# Batch size for fetching messages
FETCH_BATCH_SIZE = 100

# Standard Gmail labels
LABEL_SPAM = "SPAM"
LABEL_IMPORTANT = "IMPORTANT"
LABEL_PROMOTIONS = "PROMOTIONS"
LABEL_INBOX = "INBOX"


@dataclass
class EmailMessage:
    """
    Represents a Gmail email message with extracted content.
    """

    id: str
    thread_id: str
    subject: str
    sender: str
    recipient: str
    date: str
    snippet: str
    body_text: str
    body_html: str | None
    labels: list[str]

    @property
    def has_html(self) -> bool:
        return self.body_html is not None


@dataclass
class EmailMetadata:
    """
    Lightweight email metadata without full body extraction.
    """

    id: str
    thread_id: str
    subject: str
    sender: str
    snippet: str
    labels: list[str]


class GmailClient:
    """
    Client for interacting with Gmail API.

    Provides methods for fetching emails, applying labels, and managing spam.
    """

    def __init__(
        self,
        authenticator: GmailAuthenticator,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize Gmail client.

        Args:
            authenticator: GmailAuthenticator for handling credentials.
            max_retries: Maximum retry attempts for failed requests.
            base_delay: Base delay in seconds for exponential backoff.
        """
        self._authenticator = authenticator
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._service = None

    @property
    def service(self):
        """Lazy initialization of Gmail service."""
        if self._service is None:
            creds = self._authenticator.get_credentials()
            self._service = build(
                GMAIL_SERVICE_NAME,
                GMAIL_API_VERSION,
                credentials=creds,
                cache_discovery=False,
            )
            logger.info("Gmail service initialized")
        return self._service

    def _execute_with_retry(self, request, label: str = "operation"):
        """
        Execute API request with exponential backoff retry.

        Args:
            request: Google API request object.
            label: Label for logging purposes.

        Returns:
            API response dictionary.

        Raises:
            RateLimitError: When rate limit is exceeded after retries.
            GmailAPIError: For other API errors.
        """
        last_exception = None

        for attempt in range(self._max_retries):
            try:
                return request.execute()
            except HttpError as e:
                status_code = e.resp.status
                if status_code == 429:  # Rate limit
                    retry_after = int(e.headers.get("Retry-After", self._base_delay * 2))
                    logger.warning(
                        "Rate limit hit on %s (attempt %d/%d), waiting %ds",
                        label,
                        attempt + 1,
                        self._max_retries,
                        retry_after,
                    )
                    if attempt == self._max_retries - 1:
                        raise RateLimitError(f"Rate limit exceeded for {label}")
                    time.sleep(retry_after)
                elif status_code in (500, 503):
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Server error %d on %s (attempt %d/%d), waiting %.1fs",
                        status_code,
                        label,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    if attempt == self._max_retries - 1:
                        raise EmailFetchError(f"Server error after {self._max_retries} retries")
                    time.sleep(delay)
                else:
                    raise EmailFetchError(f"HTTP {status_code}: {e}") from e
            except google_exceptions.GoogleAPICallError as e:
                last_exception = e
                delay = self._base_delay * (2**attempt)
                logger.warning(
                    "API call failed on %s (attempt %d/%d): %s",
                    label,
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                if attempt == self._max_retries - 1:
                    raise EmailFetchError(f"API call failed after {self._max_retries} retries") from e
                time.sleep(delay)

        raise EmailFetchError("Max retries exceeded") from last_exception

    def get_unread_messages(
        self,
        max_results: int = 100,
        label_ids: list[str] | None = None,
    ) -> Generator[EmailMessage, None, None]:
        """
        Fetch unread messages from inbox.

        Args:
            max_results: Maximum number of messages to fetch.
            label_ids: Optional list of label IDs to filter (e.g., ['INBOX']).

        Yields:
            EmailMessage objects for each unread email.
        """
        query = "is:unread"
        if label_ids:
            # Build label filter
            label_filter = " OR ".join(f"label:{lid}" for lid in label_ids)
            query = f"({label_filter}) is:unread"

        logger.info("Fetching unread messages with query: %s", query)

        page_token = None
        fetched = 0

        while fetched < max_results:
            request = self.service.users().messages().list(
                userId="me",
                q=query,
                maxResults=min(FETCH_BATCH_SIZE, max_results - fetched),
                pageToken=page_token,
            )

            response = self._execute_with_retry(request, "list_messages")

            messages = response.get("messages", [])
            next_token = response.get("nextPageToken")

            logger.debug("Fetched %d messages, total fetched so far: %d", len(messages), fetched)

            for msg_ref in messages:
                if fetched >= max_results:
                    break

                message = self._get_full_message(msg_ref["id"])
                if message:
                    yield message
                    fetched += 1

            if not next_token:
                break

            page_token = next_token

        logger.info("Finished fetching %d unread messages", fetched)

    def _get_full_message(self, msg_id: str) -> EmailMessage | None:
        """
        Fetch full message details by ID.

        Args:
            msg_id: Gmail message ID.

        Returns:
            EmailMessage object or None if fetch fails.
        """
        try:
            request = self.service.users().messages().get(
                userId="me",
                id=msg_id,
                format="full",
            )
            raw_message = self._execute_with_retry(request, f"get_message:{msg_id}")
            return self._parse_message(raw_message)
        except Exception as e:
            logger.error("Failed to fetch message %s: %s", msg_id, e)
            return None

    def _parse_message(self, raw: dict) -> EmailMessage:
        """
        Parse raw Gmail message into EmailMessage.

        Args:
            raw: Raw message dict from Gmail API.

        Returns:
            Parsed EmailMessage object.
        """
        msg_id = raw["id"]
        thread_id = raw["threadId"]
        label_ids = raw.get("labelIds", [])
        payload = raw["payload"]

        # Extract headers
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

        subject = headers.get("subject", "(no subject)")
        sender = self._extract_email(headers.get("from", ""))
        recipient = self._extract_email(headers.get("to", ""))
        date = headers.get("date", "")

        # Extract body
        body_text, body_html = self._extract_body(payload)

        # Get snippet
        snippet = raw.get("snippet", "")

        return EmailMessage(
            id=msg_id,
            thread_id=thread_id,
            subject=subject,
            sender=sender,
            recipient=recipient,
            date=date,
            snippet=snippet,
            body_text=body_text,
            body_html=body_html,
            labels=label_ids,
        )

    def _extract_email(self, header_value: str) -> str:
        """Extract email address from header value like 'Name <email@example.com>'."""
        if "<" in header_value:
            start = header_value.find("<") + 1
            end = header_value.find(">")
            return header_value[start:end].strip()
        return header_value.strip()

    def _extract_body(self, payload: dict) -> tuple[str, str | None]:
        """
        Extract text and HTML body from message payload.

        Args:
            payload: Message payload dict.

        Returns:
            Tuple of (plain_text, html_body).
        """
        body_text = ""
        body_html = None

        # Handle simple single-part messages
        if "parts" not in payload:
            body_data = payload.get("body", {}).get("data", "")
            mime_type = payload.get("mimeType", "")

            if mime_type == "text/html" and body_data:
                body_html = self._decode_body(body_data)
            elif mime_type == "text/plain" and body_data:
                body_text = self._decode_body(body_data)

            return body_text, body_html

        # Handle multi-part messages
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")

            if mime_type == "text/plain":
                body_text = self._decode_body(part["body"].get("data", ""))
            elif mime_type == "text/html":
                body_html = self._decode_body(part["body"].get("data", ""))

            # Recurse into nested parts
            if "parts" in part:
                nested_text, nested_html = self._extract_body(part)
                if nested_text:
                    body_text = nested_text
                if nested_html:
                    body_html = nested_html

        return body_text, body_html

    def _decode_body(self, data: str) -> str:
        """Decode base64url encoded message body."""
        import base64

        try:
            # Gmail uses URL-safe base64
            decoded = base64.urlsafe_b64decode(data.encode("ASCII") + b"==")
            # Try UTF-8, fall back to latin-1
            try:
                return decoded.decode("utf-8")
            except UnicodeDecodeError:
                return decoded.decode("latin-1", errors="replace")
        except Exception as e:
            logger.warning("Failed to decode body: %s", e)
            return ""

    def add_labels(self, message_id: str, labels: list[str]) -> None:
        """
        Add labels to a message.

        Args:
            message_id: Gmail message ID.
            labels: List of label names to add.
        """
        try:
            request = self.service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"addLabelIds": labels, "removeLabelIds": []},
            )
            self._execute_with_retry(request, f"add_labels:{message_id}")
            logger.debug("Added labels %s to message %s", labels, message_id)
        except Exception as e:
            logger.error("Failed to add labels to message %s: %s", message_id, e)
            raise EmailModifyError(f"Failed to add labels: {e}") from e

    def remove_labels(self, message_id: str, labels: list[str]) -> None:
        """
        Remove labels from a message.

        Args:
            message_id: Gmail message ID.
            labels: List of label names to remove.
        """
        try:
            request = self.service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"removeLabelIds": labels, "addLabelIds": []},
            )
            self._execute_with_retry(request, f"remove_labels:{message_id}")
            logger.debug("Removed labels %s from message %s", labels, message_id)
        except Exception as e:
            logger.error("Failed to remove labels from message %s: %s", message_id, e)
            raise EmailModifyError(f"Failed to remove labels: {e}") from e

    def trash_message(self, message_id: str) -> None:
        """
        Move message to trash.

        Args:
            message_id: Gmail message ID.
        """
        try:
            request = self.service.users().messages().trash(
                userId="me",
                id=message_id,
            )
            self._execute_with_retry(request, f"trash:{message_id}")
            logger.info("Trashed message %s", message_id)
        except Exception as e:
            logger.error("Failed to trash message %s: %s", message_id, e)
            raise EmailModifyError(f"Failed to trash message: {e}") from e

    def mark_as_spam(self, message_id: str) -> None:
        """
        Mark message as spam (moves to spam folder).

        Args:
            message_id: Gmail message ID.
        """
        self.add_labels(message_id, [LABEL_SPAM])
        logger.info("Marked message %s as spam", message_id)

    def create_label(self, label_name: str) -> str | None:
        """
        Create a new label.

        Args:
            label_name: Name for the new label.

        Returns:
            Label ID if created successfully, None otherwise.
        """
        try:
            label_obj = {
                "name": label_name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            }
            request = self.service.users().labels().create(
                userId="me",
                body=label_obj,
            )
            result = self._execute_with_retry(request, "create_label")
            logger.info("Created label '%s' with ID %s", label_name, result["id"])
            return result["id"]
        except Exception as e:
            logger.error("Failed to create label '%s': %s", label_name, e)
            return None

    def get_labels(self) -> list[dict]:
        """
        Get all labels in the user's account.

        Returns:
            List of label dictionaries.
        """
        try:
            request = self.service.users().labels().list(userId="me")
            response = self._execute_with_retry(request, "list_labels")
            return response.get("labels", [])
        except Exception as e:
            logger.error("Failed to list labels: %s", e)
            return []
