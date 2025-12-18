"""Main entry point for local-ai custom server.

This module provides the main entry point for running the custom server
that combines MLX Omni Server with local-ai welcome page.
"""

import argparse
import sys
from pathlib import Path

from local_ai.config.loader import load_config
from local_ai.logging import configure_logging, get_logger
from local_ai.server.custom_server import CustomServer

_logger = get_logger("ServerMain")


def main():
    """Main entry point for local-ai server."""
    parser = argparse.ArgumentParser(
        description="local-ai custom server with welcome page and model tester"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config TOML file"
    )
    parser.add_argument(
        "--model",
        help="Model path or name (optional)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(log_level=args.log_level)

    try:
        # Load configuration
        settings = load_config(
            config_path=args.config,
            model=args.model,
            port=args.port,
            host=args.host
        )

        _logger.info(
            "Starting local-ai custom server: host={}, port={}, model={}",
            settings.server.host,
            settings.server.port,
            settings.model.path or "none (dynamic loading)"
        )

        # Create and run custom server
        server = CustomServer(settings)
        server.run(settings.server.host, settings.server.port)

    except Exception as e:
        _logger.error("Failed to start server: {}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
