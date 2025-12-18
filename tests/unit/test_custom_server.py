"""Tests for custom server functionality."""

from unittest.mock import Mock, patch

import pytest

from local_ai.config.schema import LocalAISettings
from local_ai.server.custom_server import CustomServer


class TestCustomServer:
    """Test custom server functionality."""

    def test_custom_server_initialization(self, settings: LocalAISettings) -> None:
        """Test that custom server initializes correctly."""
        server = CustomServer(settings)
        assert server.settings == settings
        assert server.app is None  # App created on demand
        assert server.welcome_app is not None

    @patch('local_ai.server.custom_server.importlib.import_module')
    def test_create_app_success(self, mock_import: Mock, settings: LocalAISettings) -> None:
        """Test successful app creation."""
        # Mock MLX Omni Server app
        mock_mlx_app = Mock()
        mock_mlx_app.routes = []
        mock_mlx_app.mount = Mock()

        mock_module = Mock()
        mock_module.app = mock_mlx_app
        mock_import.return_value = mock_module

        server = CustomServer(settings)
        app = server.create_app()

        assert app is not None
        mock_import.assert_called_once_with("mlx_omni_server.main")

    @patch('local_ai.server.custom_server.importlib.import_module')
    def test_create_app_import_error(self, mock_import: Mock, settings: LocalAISettings) -> None:
        """Test app creation when MLX Omni Server import fails."""
        mock_import.side_effect = ImportError("MLX Omni Server not found")

        server = CustomServer(settings)

        with pytest.raises(RuntimeError, match="MLX Omni Server not available"):
            server.create_app()

    @patch('local_ai.server.custom_server.importlib.import_module')
    def test_create_app_general_error(self, mock_import: Mock, settings: LocalAISettings) -> None:
        """Test app creation when general error occurs."""
        mock_import.side_effect = Exception("Unexpected error")

        server = CustomServer(settings)

        with pytest.raises(RuntimeError, match="Failed to create custom server"):
            server.create_app()

    def test_custom_server_with_different_settings(self) -> None:
        """Test custom server with various settings."""
        from local_ai.config.schema import ServerConfig

        # Test with custom host/port
        settings = LocalAISettings(
            server=ServerConfig(host="0.0.0.0", port=9000)
        )
        server = CustomServer(settings)
        assert server.settings.server.host == "0.0.0.0"
        assert server.settings.server.port == 9000


class TestCustomServerIntegration:
    """Integration tests for custom server."""

    def test_welcome_app_integration(self) -> None:
        """Test that welcome app is properly integrated."""
        settings = LocalAISettings()
        server = CustomServer(settings)

        # Welcome app should be initialized
        assert server.welcome_app is not None
        assert server.welcome_app.settings == settings

        # Welcome app should have routes
        welcome_fastapi_app = server.welcome_app.get_fastapi_app()
        assert len(welcome_fastapi_app.routes) > 0

    @patch('local_ai.server.custom_server.importlib.import_module')
    @patch('local_ai.server.custom_server.uvicorn.run')
    def test_run_method(self, mock_uvicorn: Mock, mock_import: Mock) -> None:
        """Test server run method."""
        # Mock MLX Omni Server app
        mock_mlx_app = Mock()
        mock_mlx_app.routes = []
        mock_mlx_app.mount = Mock()

        mock_module = Mock()
        mock_module.app = mock_mlx_app
        mock_import.return_value = mock_module

        settings = LocalAISettings()
        server = CustomServer(settings)

        # Run should call uvicorn.run
        server.run("127.0.0.1", 8080)

        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["host"] == "127.0.0.1"
        assert call_args[1]["port"] == 8080

    @patch('local_ai.server.custom_server.importlib.import_module')
    @patch('local_ai.server.custom_server.uvicorn.run')
    def test_run_method_error_handling(self, mock_uvicorn: Mock, mock_import: Mock) -> None:
        """Test server run method error handling."""
        mock_import.side_effect = ImportError("MLX Omni Server not found")

        settings = LocalAISettings()
        server = CustomServer(settings)

        with pytest.raises(RuntimeError):
            server.run("127.0.0.1", 8080)


class TestServerMain:
    """Test server main entry point."""

    @patch('local_ai.server.__main__.load_config')
    @patch('local_ai.server.__main__.CustomServer')
    @patch('local_ai.server.__main__.configure_logging')
    def test_main_success(
        self,
        mock_logging: Mock,
        mock_custom_server: Mock,
        mock_load_config: Mock
    ) -> None:
        """Test main function successful execution."""
        # Mock dependencies
        mock_settings = Mock()
        mock_settings.server.host = "127.0.0.1"
        mock_settings.server.port = 8080
        mock_settings.model.path = None

        mock_server_instance = Mock()
        mock_custom_server.return_value = mock_server_instance

        mock_load_config.return_value = mock_settings

        # Mock sys.argv for argparse
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["local_ai.server", "--port", "8080"]

            # Import and run main
            from local_ai.server.__main__ import main

            # Should not raise an error
            main()

            # Verify calls
            mock_load_config.assert_called_once()
            mock_custom_server.assert_called_once_with(mock_settings)
            mock_server_instance.run.assert_called_once_with("127.0.0.1", 8080)

        finally:
            sys.argv = original_argv

    @patch('local_ai.server.__main__.load_config')
    @patch('local_ai.server.__main__.CustomServer')
    @patch('local_ai.server.__main__.configure_logging')
    def test_main_error_handling(
        self,
        mock_logging: Mock,
        mock_custom_server: Mock,
        mock_load_config: Mock
    ) -> None:
        """Test main function error handling."""
        # Mock dependencies to raise error
        mock_load_config.side_effect = Exception("Config error")

        # Mock sys.argv for argparse
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["local_ai.server", "--port", "8080"]

            # Import and run main
            from local_ai.server.__main__ import main

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

        finally:
            sys.argv = original_argv
