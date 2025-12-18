"""Tests for welcome page functionality."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from local_ai.config.schema import LocalAISettings
from local_ai.server.welcome import WelcomeApp


class TestWelcomePage:
    """Test welcome page functionality."""

    def test_welcome_app_initialization(self, welcome_app: WelcomeApp) -> None:
        """Test that welcome app initializes correctly."""
        assert welcome_app.settings is not None
        assert welcome_app.app is not None
        assert len(welcome_app.app.routes) > 0

    def test_welcome_page_route_exists(self, welcome_app: WelcomeApp) -> None:
        """Test that welcome page route is registered."""
        routes = [route.path for route in welcome_app.app.routes]
        assert "/" in routes

    def test_chat_proxy_route_exists(self, welcome_app: WelcomeApp) -> None:
        """Test that chat proxy route is registered."""
        routes = [route.path for route in welcome_app.app.routes]
        assert "/api/chat" in routes

    def test_models_api_route_removed(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/models route has been removed (use /v1/models instead)."""
        routes = [route.path for route in welcome_app.app.routes]
        assert "/api/models" not in routes  # Should not have duplicate endpoint
        assert "/" in routes  # Should have welcome page
        assert "/api/chat" in routes  # Should have chat proxy

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_html_response(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test that welcome page returns HTML."""
        # Setup mocks
        mock_status = Mock()
        mock_status.host = "127.0.0.1"
        mock_status.port = 8080
        mock_status.running = True
        mock_status.pid = 12345
        mock_status.health = "healthy"
        mock_status.models = "test-model-1, test-model-2"

        mock_manager_instance = Mock()
        mock_manager_instance.status.return_value = mock_status
        mock_server_manager.return_value = mock_manager_instance

        # Test the endpoint
        client = TestClient(welcome_app.app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "local-ai Server" in response.text
        # Models are now loaded dynamically via JavaScript, so they won't be in initial HTML
        assert "test-model-1" not in response.text
        assert "Loading available models..." in response.text

    def test_welcome_page_with_empty_models(
        self,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page when no models are available."""
        # Models are now loaded dynamically via JavaScript
        client = TestClient(welcome_app.app)
        response = client.get("/")

        assert response.status_code == 200
        # Should show empty model list, not hardcoded defaults
        assert "Orchestrator-8B-8bit" not in response.text
        # Page loads with placeholder for dynamic model loading
        assert "Loading available models..." in response.text

    def test_models_api_endpoint_removed(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/models endpoint has been removed (use /v1/models instead)."""
        client = TestClient(welcome_app.app)
        response = client.get("/api/models")

        # Should return 404 since we removed this endpoint
        assert response.status_code == 404

    def test_chat_proxy_endpoint_basic(self, welcome_app: WelcomeApp) -> None:
        """Test basic chat proxy endpoint structure."""
        # Just test that the endpoint exists and has proper validation
        client = TestClient(welcome_app.app)

        # Test missing model validation
        response = client.post(
            "/api/chat",
            json={"messages": []}  # Missing model
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "required" in data["detail"]

    def test_chat_proxy_endpoint_validation(self, welcome_app: WelcomeApp) -> None:
        """Test chat proxy endpoint validation."""
        client = TestClient(welcome_app.app)

        # Test with valid model but empty messages
        response = client.post(
            "/api/chat",
            json={"model": "test-model", "messages": []}
        )

        # Should get an error because we can't connect to MLX Omni Server in tests
        # But the validation should pass
        assert response.status_code in [500, 502]  # Connection error expected


class TestWelcomePageIntegration:
    """Integration tests for welcome page with server manager."""

    def test_welcome_app_creation_with_settings(self) -> None:
        """Test creating welcome app with various settings."""
        # Test with default settings
        settings = LocalAISettings()
        app = WelcomeApp(settings)
        assert app.settings.server.host == "127.0.0.1"
        assert app.settings.server.port == 8080

        # Test with custom settings
        from local_ai.config.schema import ServerConfig
        custom_settings = LocalAISettings(
            server=ServerConfig(host="0.0.0.0", port=9000)
        )
        app = WelcomeApp(custom_settings)
        assert app.settings.server.host == "0.0.0.0"
        assert app.settings.server.port == 9000

    def test_get_fastapi_app_method(self) -> None:
        """Test getting FastAPI app instance."""
        settings = LocalAISettings()
        welcome_app = WelcomeApp(settings)

        fastapi_app = welcome_app.get_fastapi_app()
        assert fastapi_app is not None
        assert hasattr(fastapi_app, 'routes')
        assert len(fastapi_app.routes) > 0
