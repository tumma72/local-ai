"""Additional tests for welcome page functionality to improve coverage."""

from pathlib import Path
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

    def test_chat_proxy_returns_http_error_from_mlx_server(
        self,
        welcome_app: WelcomeApp
    ) -> None:
        """Test chat proxy returns appropriate error when MLX server returns HTTP error."""
        import httpx
        from unittest.mock import AsyncMock

        async def mock_post(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Model not loaded"}
            mock_response.text = "Internal Server Error"
            return mock_response

        with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_enter.return_value = mock_client

            with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                client = TestClient(welcome_app.app)
                response = client.post(
                    "/api/chat",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]}
                )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "MLX Omni Server error" in data["detail"]

    def test_chat_proxy_success_returns_model_response(
        self,
        welcome_app: WelcomeApp
    ) -> None:
        """Test chat proxy successfully returns response from MLX server."""
        import httpx
        from unittest.mock import AsyncMock

        expected_response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "model": "test-model"
        }

        async def mock_post(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            return mock_response

        with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_enter.return_value = mock_client

            with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                client = TestClient(welcome_app.app)
                response = client.post(
                    "/api/chat",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]}
                )

        assert response.status_code == 200
        data = response.json()
        assert data == expected_response

    def test_chat_proxy_handles_connection_error(
        self,
        welcome_app: WelcomeApp
    ) -> None:
        """Test chat proxy handles connection errors gracefully."""
        # Without mocking, this will naturally fail to connect to MLX Omni Server
        # since it's not running in tests
        client = TestClient(welcome_app.app)
        response = client.post(
            "/api/chat",
            json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]}
        )

        # Connection fails, should get either 502 (RequestError) or 500 (other error)
        assert response.status_code in [500, 502]
        data = response.json()
        assert "detail" in data


class TestWelcomePageStatusEndpoint:
    """Test welcome page /api/status endpoint."""

    def test_status_endpoint_exists(self, welcome_app: WelcomeApp) -> None:
        """Test that /api/status endpoint is registered."""
        routes = [route.path for route in welcome_app.app.routes]
        assert "/api/status" in routes

    def test_status_returns_json_with_expected_fields(
        self, welcome_app: WelcomeApp
    ) -> None:
        """Test status endpoint returns JSON with expected fields."""
        client = TestClient(welcome_app.app)
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert "pid" in data
        assert "memory_mb" in data
        assert "uptime_seconds" in data
        assert "loaded_model" in data

    def test_status_reads_pid_and_memory_from_files(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status endpoint reads PID and gets memory from state files."""
        import os
        import time

        # Create state files in temp directory
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))  # Use current process PID

        start_time_file = tmp_path / "server.start_time"
        start_time_file.write_text(str(time.time() - 100))  # Started 100 seconds ago

        # Patch STATE_DIR to use our temp directory
        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            client = TestClient(welcome_app.app)
            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        # Current process PID should work
        assert data["pid"] == os.getpid()
        # Should have memory info for current process
        assert data["memory_mb"] is not None
        # Should have uptime
        assert data["uptime_seconds"] is not None
        assert data["uptime_seconds"] >= 100

    def test_status_handles_invalid_pid_file(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles invalid PID file content gracefully."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("not-a-number")

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            client = TestClient(welcome_app.app)
            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        # Should have null PID when file is invalid
        assert data["pid"] is None

    def test_status_handles_invalid_start_time_file(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles invalid start time file content gracefully."""
        start_time_file = tmp_path / "server.start_time"
        start_time_file.write_text("not-a-timestamp")

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            client = TestClient(welcome_app.app)
            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        # Should have null uptime when file is invalid
        assert data["uptime_seconds"] is None

    def test_status_handles_nonexistent_pid(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles non-existent PID gracefully."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("999999999")  # Very high PID that likely doesn't exist

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            client = TestClient(welcome_app.app)
            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        # Should have PID but null memory (process doesn't exist)
        assert data["pid"] == 999999999
        assert data["memory_mb"] is None

    def test_status_queries_mlx_server_for_loaded_model(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status endpoint queries MLX server for loaded model."""
        import httpx
        from unittest.mock import AsyncMock

        async def mock_get(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"id": "mlx-community/Qwen3-8B-4bit"}]
            }
            return mock_response

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_enter.return_value = mock_client

                with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                    client = TestClient(welcome_app.app)
                    response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["loaded_model"] == "mlx-community/Qwen3-8B-4bit"

    def test_status_handles_mlx_server_not_responding(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles MLX server connection failures gracefully."""
        import httpx
        from unittest.mock import AsyncMock

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_enter.return_value = mock_client

                with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                    client = TestClient(welcome_app.app)
                    response = client.get("/api/status")

        # Should still return 200 with null loaded_model
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_model"] is None

    def test_status_handles_empty_models_response(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles empty models response gracefully."""
        import httpx
        from unittest.mock import AsyncMock

        async def mock_get(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            return mock_response

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_enter.return_value = mock_client

                with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                    client = TestClient(welcome_app.app)
                    response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["loaded_model"] is None

    def test_status_handles_non_200_from_mlx_server(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles non-200 response from MLX server gracefully."""
        import httpx
        from unittest.mock import AsyncMock

        async def mock_get(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 500
            return mock_response

        with patch('local_ai.server.welcome.STATE_DIR', tmp_path):
            with patch.object(httpx.AsyncClient, '__aenter__', new_callable=AsyncMock) as mock_enter:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_enter.return_value = mock_client

                with patch.object(httpx.AsyncClient, '__aexit__', new_callable=AsyncMock):
                    client = TestClient(welcome_app.app)
                    response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        # loaded_model should be null since server returned error
        assert data["loaded_model"] is None

    def test_status_handles_unexpected_exception(
        self,
        welcome_app: WelcomeApp,
        tmp_path: Path
    ) -> None:
        """Test status handles unexpected exceptions gracefully."""
        # Force an exception by making STATE_DIR access fail
        with patch('local_ai.server.welcome.STATE_DIR') as mock_dir:
            # Make the / operator raise an exception
            mock_dir.__truediv__ = Mock(side_effect=RuntimeError("Unexpected error"))

            client = TestClient(welcome_app.app)
            response = client.get("/api/status")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Internal error" in data["detail"]
