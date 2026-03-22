"""
helpers.py
----------
General utility functions used across the project.

Includes:
  - configure_logging: Set up consistent logging for all modules
  - load_env: Load .env file with dotenv
  - sanitise_phone: Basic phone number cleaning
  - format_currency: Format INR amounts for display
"""

import logging
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a clean, readable format.
    Call this once at application startup.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(__file__).resolve().parent.parent / "app.log",
                encoding="utf-8",
            ),
        ],
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env file if dotenv is installed and .env exists."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logging.getLogger(__name__).info(f".env loaded from {env_path}")
        else:
            logging.getLogger(__name__).warning(
                ".env file not found. Make sure environment variables are set."
            )
    except ImportError:
        logging.getLogger(__name__).warning(
            "python-dotenv not installed. Skipping .env load."
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sanitise_phone(raw: str) -> str:
    """Strip non-digit characters and ensure 10-digit Indian mobile format."""
    digits = re.sub(r"\D", "", raw or "")
    # Strip country code prefix (91 or 0)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits if len(digits) == 10 else raw  # return original if uncertain


def format_currency(amount_str: str) -> str:
    """
    Convert a raw amount string to a readable INR format.
    e.g. "32000000" → "₹3.20 Cr"
         "3.2 cr"   → "₹3.20 Cr"
    """
    if not amount_str:
        return amount_str
    # If already looks formatted, return as-is
    if "cr" in amount_str.lower() or "₹" in amount_str:
        return amount_str
    try:
        value = float(re.sub(r"[^\d.]", "", amount_str))
        crore = value / 1_00_00_000
        return f"₹{crore:.2f} Cr"
    except ValueError:
        return amount_str


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate a string to max_len characters for display/logging."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
