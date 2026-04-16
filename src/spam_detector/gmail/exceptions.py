"""
Custom exceptions for Gmail API integration.
"""


class GmailAPIError(Exception):
    """Base exception for Gmail API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(GmailAPIError):
    """Raised when OAuth authentication fails."""

    def __init__(self, message: str = "OAuth authentication failed"):
        super().__init__(message, status_code=401)


class TokenRefreshError(GmailAPIError):
    """Raised when token refresh fails."""

    def __init__(self, message: str = "Failed to refresh access token"):
        super().__init__(message, status_code=401)


class RateLimitError(GmailAPIError):
    """Raised when Gmail API rate limit is exceeded."""

    def __init__(self, message: str = "Gmail API rate limit exceeded"):
        super().__init__(message, status_code=429)
        self.retry_after: int | None = None


class EmailFetchError(GmailAPIError):
    """Raised when email fetching fails."""

    def __init__(self, message: str = "Failed to fetch emails"):
        super().__init__(message)


class EmailModifyError(GmailAPIError):
    """Raised when email modification (label, archive, etc.) fails."""

    def __init__(self, message: str = "Failed to modify email"):
        super().__init__(message)
