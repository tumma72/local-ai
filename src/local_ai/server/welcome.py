"""Welcome page and model tester for local-ai server.

Provides a web interface for testing models and viewing server status.
"""

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from local_ai.config.schema import LocalAISettings
from local_ai.logging import get_logger
from local_ai.server.health import get_models
from local_ai.server.manager import ServerManager

_logger = get_logger("WelcomePage")

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


class WelcomeApp:
    """FastAPI application for welcome page and model testing."""

    def __init__(self, settings: LocalAISettings):
        """Initialize welcome app with server settings.

        Args:
            settings: LocalAI configuration settings
        """
        self.settings = settings
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory=TEMPLATES_DIR)

        # Setup routes
        self._setup_routes()

        _logger.info(
            "Welcome app initialized with host={}, port={}",
            settings.server.host,
            settings.server.port,
        )

    def _setup_routes(self) -> None:
        """Setup all FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def welcome_page(request: Request):
            """Serve welcome page with server status and model tester."""
            try:
                # Get server status
                manager = ServerManager(self.settings)
                status = manager.status()

                # Don't fetch models server-side - let JavaScript do it client-side
                # This ensures the page loads immediately without waiting for model discovery
                models = []  # Start empty, JavaScript will populate dynamically
                
                return self.templates.TemplateResponse(
                    "welcome.html", {"request": request, "status": status, "models": models}
                )

            except Exception as e:
                _logger.error("Failed to generate welcome page: {}", e)
                # Page loads even if we can't get server status
                models = []  # Empty list, JavaScript will handle errors
                return self.templates.TemplateResponse(
                    "welcome.html",
                    {"request": request, "status": None, "models": models, "error": str(e)},
                )

        @self.app.post("/api/chat")
        async def proxy_chat(request: Request):
            """Proxy chat requests to MLX Omni Server.

            This endpoint allows the welcome page to test models without CORS issues.
            """
            try:
                data = await request.json()
                model = data.get("model")

                if not model:
                    raise HTTPException(status_code=400, detail="Model is required")

                _logger.debug("Proxying chat request to model: {}", model)

                # Forward request to MLX Omni Server
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"http://{self.settings.server.host}:{self.settings.server.port}/v1/chat/completions",
                        json=data,
                        timeout=30.0,
                    )

                # Check for errors
                if response.status_code != 200:
                    error_detail = response.json().get("detail", response.text)
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"MLX Omni Server error: {error_detail}",
                    )

                return response.json()

            except httpx.RequestError as e:
                _logger.error("Failed to proxy chat request: {}", e)
                raise HTTPException(
                    status_code=502, detail=f"Failed to connect to MLX Omni Server: {str(e)}"
                )
            except HTTPException:
                # Re-raise HTTPExceptions as-is
                raise
            except Exception as e:
                _logger.error("Chat proxy error: {}", e)
                raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


    def get_fastapi_app(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app
