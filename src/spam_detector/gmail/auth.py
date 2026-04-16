"""
Gmail API OAuth authentication and token management.

Handles OAuth2 flow, token storage, and automatic token refresh.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from spam_detector.gmail.exceptions import AuthenticationError, TokenRefreshError

logger = logging.getLogger(__name__)

# Gmail API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

# Token storage file name
TOKEN_FILE = "gmail_token.json"


@dataclass
class TokenManager:
    """
    Manages OAuth tokens with automatic refresh and persistence.

    Handles loading/saving tokens and automatic refresh when expired.
    """

    credentials_path: str
    token_path: str = field(default_factory=lambda: TOKEN_FILE)
    scopes: list[str] = field(default_factory=lambda: GMAIL_SCOPES)

    _credentials: Credentials | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._token_file = Path(self.token_path)

    @property
    def credentials(self) -> Credentials | None:
        return self._credentials

    def load_token(self) -> Credentials | None:
        """
        Load token from file if it exists.

        Returns:
            Credentials object if token file exists and is valid, None otherwise.
        """
        if not self._token_file.exists():
            logger.debug("Token file not found at %s", self._token_file)
            return None

        try:
            creds = Credentials.from_authorized_user_file(
                str(self._token_file), scopes=self.scopes
            )
            logger.info("Loaded credentials from %s", self._token_file)
            return creds
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Invalid token file: %s", e)
            self._remove_token_file()
            return None

    def save_token(self, credentials: Credentials) -> None:
        """
        Save credentials to token file.

        Args:
            credentials: Google OAuth2 Credentials object to persist.
        """
        try:
            with open(self._token_file, "w") as f:
                f.write(credentials.to_json())
            logger.info("Token saved to %s", self._token_file)
        except OSError as e:
            logger.error("Failed to save token file: %s", e)
            raise AuthenticationError(f"Failed to save token: {e}") from e

    def _remove_token_file(self) -> None:
        """Remove invalid token file."""
        try:
            if self._token_file.exists():
                self._token_file.unlink()
                logger.info("Removed invalid token file")
        except OSError:
            pass

    def get_valid_credentials(self) -> Credentials | None:
        """
        Get valid credentials, refreshing if necessary.

        Returns:
            Valid Credentials object or None if authentication not possible.
        """
        creds = self.load_token()

        if creds is None:
            logger.info("No existing credentials found")
            return None

        if creds.expired and creds.refresh_token:
            return self._refresh_credentials(creds)

        if not creds.valid:
            logger.warning("Credentials are invalid but not expired")
            return None

        self._credentials = creds
        return creds

    def _refresh_credentials(self, creds: Credentials) -> Credentials | None:
        """
        Refresh expired credentials.

        Args:
            creds: Expired credentials to refresh.

        Returns:
            Refreshed credentials or None if refresh fails.
        """
        try:
            logger.info("Refreshing expired credentials...")
            creds.refresh(Request())
            self._credentials = creds
            self.save_token(creds)
            logger.info("Credentials refreshed successfully")
            return creds
        except Exception as e:
            logger.error("Failed to refresh credentials: %s", e)
            self._remove_token_file()
            raise TokenRefreshError(f"Failed to refresh token: {e}") from e


class GmailAuthenticator:
    """
    Handles Gmail OAuth2 authentication flow.

    Supports both default browser-based OAuth and manual token generation.
    """

    def __init__(
        self,
        credentials_path: str,
        token_path: str = TOKEN_FILE,
        scopes: list[str] | None = None,
    ):
        """
        Initialize Gmail authenticator.

        Args:
            credentials_path: Path to OAuth client secrets JSON file.
            token_path: Path where token will be stored.
            scopes: Gmail API scopes to request.
        """
        self._credentials_path = Path(credentials_path)
        self._scopes = scopes or GMAIL_SCOPES
        self._token_manager = TokenManager(
            credentials_path=str(self._credentials_path),
            token_path=token_path,
            scopes=self._scopes,
        )

    def authenticate(self, use_port: int = 0) -> Credentials:
        """
        Run full OAuth2 flow to obtain new credentials.

        Args:
            use_port: Port for localhost callback. Use 0 for auto-select.

        Returns:
            Valid Credentials object.

        Raises:
            AuthenticationError: If OAuth flow fails.
        """
        if not self._credentials_path.exists():
            raise AuthenticationError(
                f"Credentials file not found: {self._credentials_path}\n"
                "Please download OAuth client secrets from Google Cloud Console."
            )

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._credentials_path), scopes=self._scopes
            )
            creds = flow.run_local_server(port=use_port, prompt="consent")
            self._token_manager.save_token(creds)
            logger.info("OAuth authentication completed successfully")
            return creds
        except Exception as e:
            logger.error("OAuth authentication failed: %s", e)
            raise AuthenticationError(f"OAuth authentication failed: {e}") from e

    def get_credentials(self) -> Credentials:
        """
        Get valid credentials, authenticating if necessary.

        Returns:
            Valid Credentials object.

        Raises:
            AuthenticationError: If authentication is required but fails.
        """
        creds = self._token_manager.get_valid_credentials()

        if creds is not None:
            return creds

        logger.info("Starting new OAuth authentication flow...")
        return self.authenticate()

    def is_authenticated(self) -> bool:
        """
        Check if currently authenticated with valid credentials.

        Returns:
            True if valid credentials exist, False otherwise.
        """
        creds = self._token_manager.get_valid_credentials()
        return creds is not None

    def revoke(self) -> None:
        """
        Revoke current credentials and remove token file.
        """
        creds = self._token_manager.credentials
        if creds:
            try:
                creds.revoke(Request())
                logger.info("Credentials revoked successfully")
            except Exception as e:
                logger.warning("Error revoking credentials: %s", e)
        self._token_manager._remove_token_file()
