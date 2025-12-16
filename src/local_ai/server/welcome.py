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

                # Get models from server status (already fetched by ServerManager)
                models = []
                if status.models and status.models != "none available":
                    models = [m.strip() for m in status.models.split(",") if m.strip()]
                
                # If no models, use fallback
                if not models:
                    models = [
                        "mlx-community/Orchestrator-8B-8bit",
                        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
                        "mlx-community/Llama-3.2-1B-Instruct-4bit",
                    ]
                
                return self.templates.TemplateResponse(
                    "welcome.html", {"request": request, "status": status, "models": models}
                )

            except Exception as e:
                _logger.error("Failed to generate welcome page: {}", e)
                # Provide default models even on error
                default_models = [
                    "mlx-community/Orchestrator-8B-8bit",
                    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
                    "mlx-community/Llama-3.2-1B-Instruct-4bit",
                ]
                return self.templates.TemplateResponse(
                    "welcome.html",
                    {"request": request, "status": None, "models": default_models, "error": str(e)},
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
