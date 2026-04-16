#!/usr/bin/env python3
"""
Gmail Spam Detection Runner

Production-ready script to run Gmail spam detection with OAuth authentication.

Usage:
    python scripts/run_gmail_spam_detector.py --credentials credentials.json

Environment Variables:
    PREDICTION_API_URL: URL of the spam prediction API (default: http://localhost:8000)
    PREDICTION_API_KEY: Optional API key for prediction endpoint
    GMAIL_TOKEN_FILE: Path to token file (default: gmail_token.json)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spam_detector.gmail.auth import GmailAuthenticator
from spam_detector.gmail.processor import GmailSpamProcessor, ProcessingStats


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gmail Spam Detection - Fetch and classify emails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run (will trigger OAuth flow):
  python scripts/run_gmail_spam_detector.py --credentials credentials.json

  # Subsequent runs (uses cached token):
  python scripts/run_gmail_spam_detector.py --credentials credentials.json --dry-run

  # With custom prediction API:
  python scripts/run_gmail_spam_detector.py --credentials creds.json \\
    --prediction-url https://spam-api.example.com

  # Process only 10 emails:
  python scripts/run_gmail_spam_detector.py --credentials creds.json --max-emails 10
        """,
    )

    parser.add_argument(
        "--credentials",
        type=str,
        required=True,
        help="Path to OAuth client secrets JSON file",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default=os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
        help="Path to store OAuth token (default: gmail_token.json)",
    )
    parser.add_argument(
        "--prediction-url",
        type=str,
        default=os.getenv("PREDICTION_API_URL", "http://localhost:8000"),
        help="Spam prediction API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("PREDICTION_API_KEY"),
        help="API key for prediction endpoint",
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=100,
        help="Maximum emails to process (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify emails, just report what would happen",
    )
    parser.add_argument(
        "--setup-labels",
        action="store_true",
        help="Create required Gmail labels before processing",
    )
    parser.add_argument(
        "--spam-threshold",
        type=float,
        default=0.7,
        help="Spam probability threshold (default: 0.7)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously with sleep interval between runs",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between runs in continuous mode (default: 300)",
    )

    return parser.parse_args()


def print_stats(stats: ProcessingStats) -> None:
    """Print formatted processing statistics."""
    print("\n" + "=" * 50)
    print("PROCESSING RESULTS")
    print("=" * 50)
    print(f"  Total processed: {stats.total}")
    print(f"  Marked as spam:  {stats.spam}")
    print(f"  Marked as safe:   {stats.not_spam}")
    print(f"  Uncertain:        {stats.uncertain}")
    print(f"  Errors:           {stats.errors}")
    print(f"  Time taken:       {stats.processing_time_seconds:.2f}s")
    print("=" * 50)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Gmail Spam Detector starting...")

    # Validate credentials file exists
    credentials_path = Path(args.credentials)
    if not credentials_path.exists():
        logger.error("Credentials file not found: %s", credentials_path)
        print(f"Error: Credentials file not found: {credentials_path}")
        print("\nTo get credentials:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project or select existing")
        print("3. Enable Gmail API")
        print("4. Create OAuth 2.0 credentials (Desktop app)")
        print("5. Download JSON and pass path with --credentials")
        return 1

    # Initialize authenticator
    authenticator = GmailAuthenticator(
        credentials_path=str(credentials_path),
        token_path=args.token_file,
    )

    # Check authentication
    logger.info("Checking OAuth authentication...")
    if not authenticator.is_authenticated():
        print("\n" + "=" * 50)
        print("OAuth Authentication Required")
        print("=" * 50)
        print("A browser window will open to complete Google sign-in.")
        print("The token will be cached for future runs.\n")
        time.sleep(1)

        try:
            authenticator.get_credentials()
            print("\nAuthentication successful!")
        except Exception as e:
            logger.error("Authentication failed: %s", e)
            print(f"\nAuthentication failed: {e}")
            return 1
    else:
        logger.info("Already authenticated with valid token")

    # Initialize processor
    processor = GmailSpamProcessor(
        authenticator=authenticator,
        prediction_api_url=args.prediction_url,
        api_key=args.api_key,
        confidence_threshold=args.spam_threshold,
        dry_run=args.dry_run,
    )

    # Setup labels if requested
    if args.setup_labels:
        logger.info("Setting up Gmail labels...")
        print("\nEnsuring required labels exist in Gmail...")
        processor.setup_labels()
        print("Label setup complete.")

    # Main processing loop
    if args.continuous:
        logger.info("Running in continuous mode (interval: %ds)", args.interval)
        print(f"\nRunning continuously with {args.interval}s interval...")
        print("Press Ctrl+C to stop.\n")

        while True:
            try:
                stats = processor.process_unread(max_emails=args.max_emails)
                print_stats(stats)

                if stats.total == 0:
                    logger.info("No new emails to process")

                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                print("\nShutting down...")
                break
    else:
        logger.info("Processing emails (dry_run=%s)", args.dry_run)
        if args.dry_run:
            print("\nRunning in DRY RUN mode - no emails will be modified.\n")

        stats = processor.process_unread(max_emails=args.max_emails)
        print_stats(stats)

        if args.dry_run and stats.spam > 0:
            print(f"\n{stats.spam} emails would be marked as spam")

    logger.info("Gmail Spam Detector finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
