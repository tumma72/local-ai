"""Behavioral tests for CLI server commands.

Tests verify the public CLI interface behavior:
- `local-ai server start` spawns server and reports success/failure
- `local-ai server stop` terminates server and reports result
- `local-ai server status` reports current server state

Tests mock ServerManager to avoid real process management.
Tests focus on CLI output and exit codes, not implementation details.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.config.loader import ConfigError
from local_ai.server.manager import ServerStatus, StartResult, StopResult


class TestVersionCommand:
    """Verify `local-ai --version` command behavior."""

    def test_version_flag_returns_exit_code_0_and_prints_version(
        self, cli_runner: CliRunner
    ) -> None:
        """--version should return exit code 0 and print version string."""
        result = cli_runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "local-ai version" in result.stdout
        assert "0.1.0" in result.stdout

    def test_version_short_flag_returns_exit_code_0_and_prints_version(
        self, cli_runner: CliRunner
    ) -> None:
        """-V should return exit code 0 and print version string."""
        result = cli_runner.invoke(app, ["-V"])

        assert result.exit_code == 0
        assert "local-ai version" in result.stdout


class TestServerStartCommand:
    """Verify `local-ai server start` command behavior."""

    def test_start_success_returns_exit_code_0_and_prints_port(
        self, cli_runner: CliRunner
    ) -> None:
        """start command should return exit code 0 and print success message with port."""
        mock_result = StartResult(success=True, pid=12345, error=None)
        mock_manager = MagicMock()
        mock_manager.start.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(
                app, ["server", "start", "--model", "test-model"]
            )

        assert result.exit_code == 0
        # Should indicate success and include port information
        assert "8080" in result.stdout or "started" in result.stdout.lower()

    def test_start_when_already_running_returns_exit_code_1_and_prints_error(
        self, cli_runner: CliRunner
    ) -> None:
        """start command should return exit code 1 when server is already running."""
        mock_result = StartResult(
            success=False, pid=12345, error="Server already running with PID 12345"
        )
        mock_manager = MagicMock()
        mock_manager.start.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(
                app, ["server", "start", "--model", "test-model"]
            )

        assert result.exit_code == 1
        # Should indicate the server is already running
        assert "already running" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_start_failure_with_multiline_error_shows_panel(
        self, cli_runner: CliRunner
    ) -> None:
        """start command should show error panel when error has multiple lines."""
        multiline_error = (
            "Server did not become healthy within 30.0s\n\n"
            "Server log:\n"
            "Loading model...\n"
            "Error: Repository Not Found"
        )
        mock_result = StartResult(success=False, pid=None, error=multiline_error)
        mock_manager = MagicMock()
        mock_manager.start.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(
                app, ["server", "start", "--model", "test-model"]
            )

        assert result.exit_code == 1
        # Should show the error content (panel output includes the text)
        assert "Server did not become healthy" in result.stdout
        assert "Repository Not Found" in result.stdout


class TestServerStopCommand:
    """Verify `local-ai server stop` command behavior."""

    def test_stop_success_returns_exit_code_0_and_prints_stopped(
        self, cli_runner: CliRunner
    ) -> None:
        """stop command should return exit code 0 and print stopped message."""
        mock_result = StopResult(success=True, error=None)
        mock_manager = MagicMock()
        mock_manager.stop.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "stop"])

        assert result.exit_code == 0
        assert "stopped" in result.stdout.lower() or "success" in result.stdout.lower()

    def test_stop_failure_returns_exit_code_1_and_prints_error(
        self, cli_runner: CliRunner
    ) -> None:
        """stop command should return exit code 1 when stop fails."""
        mock_result = StopResult(success=False, error="Operation not permitted")
        mock_manager = MagicMock()
        mock_manager.stop.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "stop"])

        assert result.exit_code == 1
        error_shown = "error" in result.stdout.lower()
        permission_error = "operation not permitted" in result.stdout.lower()
        assert error_shown or permission_error


class TestServerStatusCommand:
    """Verify `local-ai server status` command behavior."""

    def test_status_when_running_returns_exit_code_0_and_shows_pid(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should return exit code 0 and show running status with PID."""
        mock_status = ServerStatus(
            running=True,
            pid=12345,
            host="127.0.0.1",
            port=8080,
            models="test-model",
            uptime_seconds=120.0,
            health="healthy",
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        # Should show running status and PID
        assert "running" in result.stdout.lower()
        assert "12345" in result.stdout

    def test_status_shows_unhealthy_health_in_red(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should display unhealthy status with appropriate styling."""
        mock_status = ServerStatus(
            running=True,
            pid=12345,
            host="127.0.0.1",
            port=8080,
            models="test-model",
            uptime_seconds=60.0,
            health="unhealthy",
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        assert "unhealthy" in result.stdout.lower()

    def test_status_shows_unknown_health_in_yellow(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should display unknown health status with yellow styling."""
        mock_status = ServerStatus(
            running=True,
            pid=12345,
            host="127.0.0.1",
            port=8080,
            models="test-model",
            uptime_seconds=60.0,
            health="checking",
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        # Should show the health status
        assert "checking" in result.stdout.lower()

    def test_status_shows_uptime_when_available(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should show formatted uptime when server has uptime_seconds."""
        mock_status = ServerStatus(
            running=True,
            pid=12345,
            host="127.0.0.1",
            port=8080,
            models="test-model",
            uptime_seconds=7265.0,  # 2h 1m 5s
            health="healthy",
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        # Should show uptime formatted
        assert "2h" in result.stdout
        assert "Uptime" in result.stdout

    def test_status_when_not_running_returns_exit_code_0_and_shows_not_running(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should return exit code 0 and show not running message."""
        mock_status = ServerStatus(
            running=False,
            pid=None,
            host=None,
            port=None,
            models=None,
            uptime_seconds=None,
            health=None,
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        assert "not running" in result.stdout.lower() or "stopped" in result.stdout.lower()

    def test_status_when_running_shows_all_status_fields(
        self, cli_runner: CliRunner
    ) -> None:
        """status command should display host, port, model, and health when server is running."""
        mock_status = ServerStatus(
            running=True,
            pid=54321,
            host="0.0.0.0",
            port=9090,
            models="mlx-community/Llama-3.2-1B-Instruct-4bit",
            uptime_seconds=300.0,
            health="healthy",
        )
        mock_manager = MagicMock()
        mock_manager.status.return_value = mock_status

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config"):
            result = cli_runner.invoke(app, ["server", "status"])

        assert result.exit_code == 0
        # Verify all status fields are displayed
        assert "0.0.0.0" in result.stdout  # host
        assert "9090" in result.stdout  # port
        assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in result.stdout  # model
        assert "healthy" in result.stdout  # health


class TestServerStartOptionPropagation:
    """Verify CLI options are correctly passed to configuration loader."""

    def test_start_passes_cli_options_to_load_config(
        self, cli_runner: CliRunner
    ) -> None:
        """start command should pass port, host, and config options to load_config."""
        mock_result = StartResult(success=True, pid=12345, error=None)
        mock_manager = MagicMock()
        mock_manager.start.return_value = mock_result

        with patch(
            "local_ai.cli.server.ServerManager", return_value=mock_manager
        ), patch("local_ai.cli.server.load_config") as mock_load_config:
            cli_runner.invoke(
                app,
                [
                    "server",
                    "start",
                    "--port",
                    "9090",
                    "--host",
                    "0.0.0.0",
                    "--config",
                    "/path/to/config.toml",
                    "--model",
                    "test-model",
                ],
            )

        # Verify load_config was called with the correct CLI arguments
        mock_load_config.assert_called_once()
        call_kwargs = mock_load_config.call_args
        assert call_kwargs.kwargs["port"] == 9090
        assert call_kwargs.kwargs["host"] == "0.0.0.0"
        assert call_kwargs.kwargs["config_path"] == Path("/path/to/config.toml")
        assert call_kwargs.kwargs["model"] == "test-model"


class TestServerStartValidation:
    """Verify CLI validates required configuration."""

    def test_start_without_model_exits_with_error_when_no_model_configured(
        self, cli_runner: CliRunner
    ) -> None:
        """start command should exit with code 1 and raise ConfigError when model is missing."""
        with patch("local_ai.cli.server.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigError(
                "Model is required but not specified. Provide it via:\n"
                "  1. --model CLI argument\n"
                "  2. [model] section in config.toml\n"
                "  3. LOCAL_AI_MODEL environment variable"
            )
            result = cli_runner.invoke(app, ["server", "start"])

        assert result.exit_code == 1
        # Verify the exception indicates missing model
        assert result.exception is not None
        assert isinstance(result.exception, ConfigError)
        assert "model" in str(result.exception).lower()


class TestConfigShowCommand:
    """Verify `local-ai config show` command behavior."""

    def test_config_show_returns_exit_code_0_and_displays_settings(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should return exit code 0 and display configuration values."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should display configuration sections
        assert "Configuration" in result.stdout
        assert "Server" in result.stdout
        assert "Model" in result.stdout

    def test_config_show_displays_default_values_when_no_config_file(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should indicate defaults when no config file found."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should indicate no config file and using defaults
        assert "No configuration file found" in result.stdout
        assert "Using default values" in result.stdout

    def test_config_show_with_explicit_config_file_displays_file_path(
        self, cli_runner: CliRunner, sample_config_toml: Path
    ) -> None:
        """config show should display the explicit config file path when provided."""
        result = cli_runner.invoke(
            app, ["config", "show", "--config", str(sample_config_toml)]
        )

        assert result.exit_code == 0
        # Should display the config file path
        assert str(sample_config_toml) in result.stdout
        assert "explicit path" in result.stdout

    def test_config_show_displays_server_settings(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should display server host, port, and log level."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show default server settings
        assert "host" in result.stdout
        assert "port" in result.stdout
        assert "log_level" in result.stdout

    def test_config_show_displays_model_settings(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should display model configuration settings."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show model settings section
        assert "Model" in result.stdout
        assert "path" in result.stdout
        assert "trust_remote_code" in result.stdout


class TestConfigShowSourceDisplay:
    """Verify config show displays correct source for each setting."""

    def test_config_show_indicates_file_source_when_config_loaded(
        self, cli_runner: CliRunner, sample_config_toml: Path
    ) -> None:
        """config show should indicate 'File' source when loading from config file."""
        result = cli_runner.invoke(
            app, ["config", "show", "--config", str(sample_config_toml)]
        )

        assert result.exit_code == 0
        # Should indicate values come from file
        assert "File" in result.stdout

    def test_config_show_indicates_default_source_when_no_config(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should indicate 'Default' source when using defaults."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should indicate values are defaults
        assert "Default" in result.stdout


class TestConfigShowSearchPaths:
    """Verify config show displays configuration search paths."""

    def test_config_show_displays_search_paths(
        self, cli_runner: CliRunner
    ) -> None:
        """config show should list the config file search paths."""
        with patch("local_ai.cli.config._discover_config_path", return_value=None):
            result = cli_runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show search paths
        assert "Config search paths" in result.stdout
        assert "./config.toml" in result.stdout
        assert "~/.config/local-ai/config.toml" in result.stdout
