"""local-ai: Offline AI coding assistant for Apple Silicon."""

__version__ = "0.1.0"

# Default server constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

from local_ai.logging import configure_logging, get_logger

__all__ = ["__version__", "DEFAULT_HOST", "DEFAULT_PORT", "configure_logging", "get_logger"]
