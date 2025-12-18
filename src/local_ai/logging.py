"""Logging configuration for local-ai using Loguru.

Provides bounded loggers with component prefixes and late binding support.
Log messages use {} placeholders for lazy evaluation - values are only
formatted when the log level is enabled.

Example:
    from local_ai.logging import get_logger

    logger = get_logger("ServerManager")
    logger.info("Server started on port {}", port)  # Lazy evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _base_logger

if TYPE_CHECKING:
    from loguru import Logger

# Default log directory
LOG_DIR = Path.home() / ".local" / "state" / "local-ai" / "logs"


def _ensure_log_dir() -> Path:
    """Ensure log directory exists and return path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def configure_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    console: bool = True,
) -> None:
    """Configure logging for local-ai.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to ~/.local/state/local-ai/logs/
        console: Whether to also log to console (stderr).
    """
    global LOG_DIR
    if log_dir is not None:
        LOG_DIR = log_dir

    log_path = _ensure_log_dir()

    # Remove default handler
    _base_logger.remove()

    # Console handler with colors
    if console:
        _base_logger.add(
            sys.stderr,
            level=log_level,
            format="<level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | {message}",
            colorize=True,
        )

    # File handler with rotation
    _base_logger.add(
        log_path / "local-ai.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[component]}]: {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    # Separate error log
    _base_logger.add(
        log_path / "local-ai.error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[component]}]: {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )


def get_logger(component: str) -> Logger:
    """Get a bounded logger for a specific component.

    The component name is prepended to all log messages as [component]:

    Args:
        component: Name of the component (e.g., "ServerManager", "CLI", "Config").

    Returns:
        A loguru Logger bound to the component name.

    Example:
        logger = get_logger("ServerManager")
        logger.info("Starting server on port {}", 8080)
        # Output: INFO | [ServerManager]: Starting server on port 8080
    """
    return _base_logger.bind(component=component)


# Initialize with default component for module-level usage
logger = get_logger("local-ai")
