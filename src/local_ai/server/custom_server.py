"""Custom server that combines MLX Omni Server with local-ai welcome page.

This module creates a unified FastAPI server that includes both the MLX Omni Server
endpoints and our custom welcome page with model tester.
"""

import importlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from local_ai.config.schema import LocalAISettings
from local_ai.logging import get_logger
from local_ai.server.welcome import WelcomeApp

_logger = get_logger("CustomServer")


class CustomServer:
    """Custom server combining MLX Omni Server with local-ai features."""

    def __init__(self, settings: LocalAISettings):
        """Initialize custom server with settings.

        Args:
            settings: LocalAI configuration settings
        """
        self.settings = settings
        self.app = None
        self.welcome_app = WelcomeApp(settings)

    def create_app(self) -> FastAPI:
        """Create the combined FastAPI application.

        Returns:
            FastAPI app with MLX Omni Server routes and welcome page
        """
        try:
            # Import MLX Omni Server's FastAPI app
            mlx_omni_server = importlib.import_module("mlx_omni_server.main")
            mlx_app = mlx_omni_server.app

            # Get our welcome app
            welcome_app = self.welcome_app.get_fastapi_app()

            # Add welcome page routes to MLX app
            # This approach preserves all MLX routes, middlewares, and functionality
            for route in welcome_app.routes:
                # Copy route to MLX app - this is the proper way to add routes
                mlx_app.routes.append(route)

            # Mount static files for CSS/JS assets
            static_dir = Path(__file__).parent / "templates"
            if static_dir.exists():
                mlx_app.mount("/static", StaticFiles(directory=static_dir), name="static")

            _logger.info("Successfully created custom server with welcome page")
            return mlx_app

        except ImportError as e:
            _logger.error("Failed to import MLX Omni Server: {}", e)
            raise RuntimeError("MLX Omni Server not available") from e

        except Exception as e:
            message: str = f"Failed to create custom server: {e}"
            _logger.exception(message)
            raise RuntimeError(message) from e

    def run(self, host: str, port: int):
        """Run the custom server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        try:
            app = self.create_app()

            _logger.info("Starting custom server on {}:{}", host, port)

            uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)

        except Exception as e:
            _logger.error("Failed to start custom server: {}", e)
            raise
