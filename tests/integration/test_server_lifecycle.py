"""Integration tests for server lifecycle management.

Tests verify the integration between CLI, config loader, and ServerManager:
- Config file loading flows through to server start
- Server start/stop lifecycle with PID file management
- Error handling for already-running servers
- Status reporting for running and stopped servers

Unlike unit tests, these test real component interactions but mock
system-level operations (subprocess, os.kill) to avoid starting real servers.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.config.loader import load_config
from local_ai.server.manager import ServerManager


@pytest.fixture
def integration_config_toml(temp_dir: Path) -> Path:
    """Create a config.toml with specific server settings for integration testing."""
    config_path = temp_dir / "config.toml"
    config_path.write_text("""
[server]
host = "0.0.0.0"
port = 9090
log_level = "DEBUG"

[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[generation]
max_tokens = 2048
temperature = 0.7
""")
    return config_path


class TestConfigToCLIIntegration:
    """Verify config file values flow through CLI to ServerManager.start()."""

    def test_config_file_values_passed_to_server_start(
        self, cli_runner: CliRunner, integration_config_toml: Path, temp_dir: Path
    ) -> None:
        """Config file host/port should be used in subprocess.Popen command."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process is running

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(Path, "home", return_value=temp_dir), \
             patch("local_ai.server.manager.check_health", return_value="healthy"):
            result = cli_runner.invoke(
                app,
                ["server", "start", "--config", str(integration_config_toml), "--timeout", "5"],
            )

        # Verify CLI succeeded
        assert result.exit_code == 0

        # Verify Popen was called with config values
        # Note: mlx-omni-server loads models dynamically, so --model is not in command
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]  # First positional argument is the command list

        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "9090" in cmd


class TestServerStartStopLifecycle:
    """Verify full server lifecycle: start creates PID, stop removes it."""

    def test_start_creates_pid_file_stop_removes_it(
        self, temp_dir: Path
    ) -> None:
        """Server start should create PID file, stop should remove it."""
        state_dir = temp_dir / "state"
        state_dir.mkdir()

        # Create settings directly (avoiding config discovery)
        settings = load_config(model="test-model")
        manager = ServerManager(settings, state_dir=state_dir)

        mock_process = MagicMock()
        mock_process.pid = 99999
        mock_process.poll.return_value = None

        # Start server (mock health check to return healthy immediately)
        with patch("subprocess.Popen", return_value=mock_process), \
             patch("local_ai.server.manager.check_health", return_value="healthy"):
            start_result = manager.start(startup_timeout=5.0)

        # Verify start succeeded and PID file created
        assert start_result.success is True
        assert start_result.pid == 99999
        pid_file = state_dir / "server.pid"
        assert pid_file.exists()
        assert pid_file.read_text().strip() == "99999"

        # Stop server (mock process exists for the stop check)
        with patch("os.kill") as mock_kill:
            # First call (signal 0) checks if process exists
            # Second call (SIGTERM) actually stops it
            mock_kill.side_effect = [None, None]
            stop_result = manager.stop()

        # Verify stop succeeded and PID file removed
        assert stop_result.success is True
        assert not pid_file.exists()

    def test_status_when_running_shows_running_state(
        self, temp_dir: Path
    ) -> None:
        """Status should report running=True when server is running."""
        state_dir = temp_dir / "state"
        state_dir.mkdir()

        settings = load_config(model="status-test-model", host="127.0.0.1", port=8080)
        manager = ServerManager(settings, state_dir=state_dir)

        mock_process = MagicMock()
        mock_process.pid = 77777
        mock_process.poll.return_value = None

        # Start server (mock health check to return healthy immediately)
        with patch("subprocess.Popen", return_value=mock_process), \
             patch("local_ai.server.manager.check_health", return_value="healthy"):
            manager.start(startup_timeout=5.0)

        # Check status (mock os.kill, health check and models query)
        with patch("os.kill") as mock_kill, \
             patch("local_ai.server.manager.check_health", return_value="healthy"), \
             patch("local_ai.server.manager.get_models", return_value=["status-test-model"]):
            mock_kill.return_value = None  # Process exists
            status = manager.status()

        assert status.running is True
        assert status.pid == 77777
        assert status.host == "127.0.0.1"
        assert status.port == 8080
        assert status.models == "status-test-model"
        assert status.health == "healthy"


class TestServerAlreadyRunningError:
    """Verify error handling when attempting to start an already-running server."""

    def test_start_when_already_running_returns_error(
        self, temp_dir: Path
    ) -> None:
        """Second start attempt should fail with 'already running' error."""
        state_dir = temp_dir / "state"
        state_dir.mkdir()

        settings = load_config(model="duplicate-test-model")
        manager = ServerManager(settings, state_dir=state_dir)

        mock_process = MagicMock()
        mock_process.pid = 55555
        mock_process.poll.return_value = None

        # First start (mock health check to return healthy immediately)
        with patch("subprocess.Popen", return_value=mock_process), \
             patch("local_ai.server.manager.check_health", return_value="healthy"):
            first_result = manager.start(startup_timeout=5.0)

        assert first_result.success is True

        # Second start (process still running)
        with patch("os.kill") as mock_kill, \
             patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            mock_kill.return_value = None  # Process exists
            second_result = manager.start(startup_timeout=5.0)

        # Verify second start failed
        assert second_result.success is False
        assert second_result.error is not None
        assert "already running" in second_result.error.lower()
        assert "55555" in second_result.error

        # Verify Popen was NOT called for second attempt
        mock_popen.assert_not_called()


class TestStatusWhenNotRunning:
    """Verify status reporting when no server is running."""

    def test_status_without_running_server_shows_not_running(
        self, temp_dir: Path
    ) -> None:
        """Status should report running=False when no server is running."""
        state_dir = temp_dir / "state"
        state_dir.mkdir()

        settings = load_config(model="no-server-model")
        manager = ServerManager(settings, state_dir=state_dir)

        # No server started, no PID file
        status = manager.status()

        assert status.running is False
        assert status.pid is None
        assert status.host is None
        assert status.port is None
        assert status.models is None
        assert status.health is None
