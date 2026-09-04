"""Structured logging setup (scaffold stub)."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure basic logging. Full JSON/structlog wiring comes later."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
