"""
Gmail API integration for spam detection.

Provides OAuth authentication, email fetching, and automated spam classification.
"""

from spam_detector.gmail.auth import GmailAuthenticator, TokenManager
from spam_detector.gmail.client import GmailClient
from spam_detector.gmail.processor import GmailSpamProcessor
from spam_detector.gmail.exceptions import (
    GmailAPIError,
    AuthenticationError,
    RateLimitError,
    TokenRefreshError,
)

__all__ = [
    "GmailAuthenticator",
    "TokenManager",
    "GmailClient",
    "GmailSpamProcessor",
    "GmailAPIError",
    "AuthenticationError",
    "RateLimitError",
    "TokenRefreshError",
]
