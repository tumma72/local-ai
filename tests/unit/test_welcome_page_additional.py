"""Additional tests for welcome page functionality to improve coverage."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from local_ai.server.welcome import WelcomeApp


class TestWelcomePageEdgeCases:
    """Test welcome page edge cases and error handling."""

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_with_server_manager_error(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page when ServerManager raises an error."""
        # Setup mocks to raise error
        mock_server_manager.side_effect = Exception("Server manager error")

        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load page with empty models (dynamic loading via JavaScript)
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "Orchestrator-8B-8bit" not in response.text

    def test_welcome_page_renders_with_dynamic_model_loading(
        self,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page renders with JavaScript-based model loading."""
        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load page with placeholder for dynamic model loading
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "Orchestrator-8B-8bit" not in response.text

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_with_empty_status_models(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page when status.models is empty string."""
        # Setup mocks
        mock_status = Mock()
        mock_status.host = "127.0.0.1"
        mock_status.port = 8080
        mock_status.running = True
        mock_status.pid = 12345
        mock_status.health = "healthy"
        mock_status.models = ""  # Empty string

        mock_manager_instance = Mock()
        mock_manager_instance.status.return_value = mock_status
        mock_server_manager.return_value = mock_manager_instance

        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load with empty models (dynamic loading via JavaScript)
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "Orchestrator-8B-8bit" not in response.text

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_with_none_status_models(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page when status.models is None."""
        # Setup mocks
        mock_status = Mock()
        mock_status.host = "127.0.0.1"
        mock_status.port = 8080
        mock_status.running = True
        mock_status.pid = 12345
        mock_status.health = "healthy"
        mock_status.models = None  # None

        mock_manager_instance = Mock()
        mock_manager_instance.status.return_value = mock_status
        mock_server_manager.return_value = mock_manager_instance

        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load with empty models (dynamic loading via JavaScript)
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "Orchestrator-8B-8bit" not in response.text

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_with_single_model(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page with single model."""
        # Setup mocks
        mock_status = Mock()
        mock_status.host = "127.0.0.1"
        mock_status.port = 8080
        mock_status.running = True
        mock_status.pid = 12345
        mock_status.health = "healthy"
        mock_status.models = "single-model"

        mock_manager_instance = Mock()
        mock_manager_instance.status.return_value = mock_status
        mock_server_manager.return_value = mock_manager_instance

        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load with empty models (dynamic loading via JavaScript)
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "single-model" not in response.text

    @patch('local_ai.server.welcome.ServerManager')
    def test_welcome_page_with_multiple_models_in_status(
        self,
        mock_server_manager: Mock,
        welcome_app: WelcomeApp
    ) -> None:
        """Test welcome page with multiple models in status."""
        # Setup mocks
        mock_status = Mock()
        mock_status.host = "127.0.0.1"
        mock_status.port = 8080
        mock_status.running = True
        mock_status.pid = 12345
        mock_status.health = "healthy"
        mock_status.models = "model1, model2, model3"

        mock_manager_instance = Mock()
        mock_manager_instance.status.return_value = mock_status
        mock_server_manager.return_value = mock_manager_instance

        client = TestClient(welcome_app.app)
        response = client.get("/")

        # Should load with empty models (dynamic loading via JavaScript)
        assert response.status_code == 200
        assert "Loading available models..." in response.text
        assert "model1" not in response.text
        assert "model2" not in response.text
        assert "model3" not in response.text


class TestWelcomePageModelsAPI:
    """Test welcome page models API endpoint."""

    # Uses welcome_app fixture from conftest.py

    def test_models_api_removed(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/models endpoint has been removed (use /v1/models instead)."""
        client = TestClient(welcome_app.app)
        response = client.get("/api/models")

        # Should return 404 since we removed this endpoint
        assert response.status_code == 404

    def test_models_api_empty_removed(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/models endpoint has been removed."""
        client = TestClient(welcome_app.app)
        response = client.get("/api/models")

        # Should return 404 since we removed this endpoint
        assert response.status_code == 404

    def test_models_api_error_removed(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/models endpoint has been removed."""
        client = TestClient(welcome_app.app)
        response = client.get("/api/models")

        # Should return 404 since we removed this endpoint
        assert response.status_code == 404


class TestWelcomePageChatProxy:
    """Test welcome page chat proxy endpoint."""

    # Uses welcome_app fixture from conftest.py

    def test_chat_proxy_missing_model(self, welcome_app: WelcomeApp) -> None:
        """Test chat proxy when model is missing."""
        client = TestClient(welcome_app.app)
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "test"}]}  # Missing model
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "required" in data["detail"]

    def test_chat_proxy_empty_messages(self, welcome_app: WelcomeApp) -> None:
        """Test chat proxy with empty messages."""
        client = TestClient(welcome_app.app)
        response = client.post(
            "/api/chat",
            json={"model": "test-model", "messages": []}
        )

        # Should get connection error since MLX Omni Server isn't running in tests
        assert response.status_code in [500, 502]

    def test_chat_proxy_invalid_json(self, welcome_app: WelcomeApp) -> None:
        """Test chat proxy with invalid JSON."""
        client = TestClient(welcome_app.app)
        response = client.post(
            "/api/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # The invalid JSON causes a parsing error which is caught as 500
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Expecting value" in data["detail"]
