"""Behavioral tests for ServerManager module.

Tests verify public behavior of the ServerManager class:
- start() spawns server process and returns result
- stop() terminates running server process
- status() reports server state
- is_running() checks process existence

Tests mock subprocess and OS interactions for isolation.
Tests are implementation-agnostic and should survive refactoring.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_ai.config.schema import LocalAISettings
from local_ai.server.manager import (
    ServerManager,
    ServerStatus,
    StartResult,
    StopResult,
)


@pytest.fixture
def state_dir(temp_dir: Path) -> Path:
    """Create a temporary state directory for PID/log files."""
    state = temp_dir / "state" / "local-ai"
    state.mkdir(parents=True)
    return state


class TestServerStart:
    """Verify ServerManager.start() behavior for spawning server process."""

    def test_start_with_valid_settings_returns_success_with_pid(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """start() should spawn subprocess and return StartResult with success=True and PID."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process still running

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch("local_ai.server.manager.check_health", return_value="healthy"):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.start(startup_timeout=5.0)

        assert isinstance(result, StartResult)
        assert result.success is True
        assert result.pid == 12345
        assert result.error is None
        mock_popen.assert_called_once()

    def test_start_when_already_running_returns_failure(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """start() should return failure when server is already running."""
        # Create PID file to simulate running server
        pid_file = state_dir / "server.pid"
        pid_file.write_text("12345")

        # Mock os.kill to indicate process exists
        with patch("os.kill") as mock_kill:
            mock_kill.return_value = None  # No exception = process exists
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.start()

        assert isinstance(result, StartResult)
        assert result.success is False
        assert result.error is not None
        assert "already running" in result.error.lower()

    def test_start_when_process_exits_immediately_returns_failure(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """start() should return failure when spawned process exits immediately."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = 1  # Process exited with error code
        mock_process.returncode = 1  # Set explicit return code

        with patch("subprocess.Popen", return_value=mock_process):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.start(startup_timeout=5.0)

        assert isinstance(result, StartResult)
        assert result.success is False
        assert result.error is not None
        assert "exited with code" in result.error.lower()


class TestServerStop:
    """Verify ServerManager.stop() behavior for terminating server process."""

    def test_stop_when_running_terminates_and_returns_success(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """stop() should send SIGTERM to running process and return success."""
        # Create PID file to simulate running server
        pid_file = state_dir / "server.pid"
        pid_file.write_text("12345")

        with patch("os.kill"):
            # First call checks existence, subsequent calls for termination
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.stop()

        assert isinstance(result, StopResult)
        assert result.success is True
        assert result.error is None
        # PID file should be removed after successful stop
        assert not pid_file.exists()

    def test_stop_when_not_running_returns_appropriate_result(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """stop() should handle gracefully when no server is running."""
        # No PID file exists - server not running
        manager = ServerManager(settings, state_dir=state_dir)
        result = manager.stop()

        assert isinstance(result, StopResult)
        # Should not fail, just indicate nothing to stop
        assert result.success is True or "not running" in (result.error or "").lower()

    def test_stop_when_os_kill_fails_returns_error(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """stop() should return StopResult with error when os.kill raises OSError."""
        # Create PID file to simulate running server
        pid_file = state_dir / "server.pid"
        pid_file.write_text("12345")

        def kill_side_effect(pid: int, sig: int) -> None:
            if sig == 0:
                # Signal 0 just checks if process exists - allow it
                return
            # For actual kill signals, raise permission error
            raise OSError("Operation not permitted")

        with patch("os.kill", side_effect=kill_side_effect):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.stop()

        assert isinstance(result, StopResult)
        assert result.success is False
        assert result.error is not None
        assert "Operation not permitted" in result.error


class TestServerStatus:
    """Verify ServerManager.status() behavior for reporting server state."""

    def test_status_when_running_returns_running_state_with_details(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """status() should return ServerStatus with running=True and process details."""
        # Create PID file to simulate running server
        pid_file = state_dir / "server.pid"
        pid_file.write_text("12345")

        with patch("os.kill") as mock_kill, \
             patch("local_ai.server.manager.check_health", return_value="healthy"), \
             patch("local_ai.server.manager.get_models", return_value=["mlx-community/test-model"]):
            mock_kill.return_value = None  # Process exists
            manager = ServerManager(settings, state_dir=state_dir)
            status = manager.status()

        assert isinstance(status, ServerStatus)
        assert status.running is True
        assert status.pid == 12345
        assert status.host == "127.0.0.1"
        assert status.port == 8080
        assert status.models == "mlx-community/test-model"
        assert status.health == "healthy"

    def test_status_when_not_running_returns_not_running_state(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """status() should return ServerStatus with running=False when no server running."""
        # No PID file exists
        manager = ServerManager(settings, state_dir=state_dir)
        status = manager.status()

        assert isinstance(status, ServerStatus)
        assert status.running is False
        assert status.pid is None
        assert status.host is None
        assert status.port is None
        assert status.models is None
        assert status.health is None


class TestServerStaleState:
    """Verify ServerManager handles stale state (PID file exists but process does not)."""

    def test_stop_cleans_up_stale_pid_file_when_process_not_running(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """stop() should clean up stale PID file when process no longer exists."""
        # Create PID file pointing to non-existent process
        pid_file = state_dir / "server.pid"
        pid_file.write_text("99999")

        def kill_raises_process_not_found(pid: int, sig: int) -> None:
            # Process doesn't exist - raise OSError
            raise OSError("No such process")

        with patch("os.kill", side_effect=kill_raises_process_not_found):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.stop()

        assert isinstance(result, StopResult)
        assert result.success is True
        # Stale PID file should be cleaned up
        assert not pid_file.exists()

    def test_is_running_returns_false_with_invalid_pid_file(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """is_running() should return False when PID file contains invalid data."""
        # Create PID file with non-numeric content
        pid_file = state_dir / "server.pid"
        pid_file.write_text("not-a-pid")

        manager = ServerManager(settings, state_dir=state_dir)
        result = manager.is_running()

        assert result is False


class TestServerStartupTimeout:
    """Verify ServerManager handles startup timeout scenarios."""

    def test_start_timeout_returns_failure_with_log_content(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """start() should return failure with log content when timeout exceeded."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process still running
        log_file = state_dir / "server.log"

        def health_never_ready(host: str, port: int, timeout: float = 5.0) -> str:
            # Simulate subprocess writing to log file
            log_file.write_text("Loading model...\nInitializing...\n")
            return "unknown"

        with patch("subprocess.Popen", return_value=mock_process), \
             patch("local_ai.server.manager.check_health", side_effect=health_never_ready), \
             patch("os.kill"):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.start(startup_timeout=0.1, health_interval=0.05)

        assert isinstance(result, StartResult)
        assert result.success is False
        assert result.error is not None
        assert "did not become healthy" in result.error.lower()
        assert "Loading model" in result.error

    def test_start_process_exit_includes_error_log(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """start() should include error log content when process exits with error."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = 1
        log_file = state_dir / "server.log"

        def poll_side_effect() -> int | None:
            # Simulate subprocess writing error to log before exiting
            log_file.write_text("Starting server...\nError: Model not found\n")
            return 1

        mock_process.poll.side_effect = poll_side_effect

        with patch("subprocess.Popen", return_value=mock_process):
            manager = ServerManager(settings, state_dir=state_dir)
            result = manager.start(startup_timeout=5.0, health_interval=0.01)

        assert isinstance(result, StartResult)
        assert result.success is False
        assert result.error is not None
        assert "exited with code" in result.error.lower()
        assert "Error: Model not found" in result.error


class TestServerLogReading:
    """Verify ServerManager log file reading behavior."""

    def test_get_last_log_lines_when_log_file_missing(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """_get_last_log_lines() should return empty string when log file doesn't exist."""
        manager = ServerManager(settings, state_dir=state_dir)
        # Ensure no log file exists
        log_file = state_dir / "server.log"
        if log_file.exists():
            log_file.unlink()

        # Access private method for testing edge case
        result = manager._get_last_log_lines(10)
        assert result == ""

    def test_get_last_log_lines_returns_last_n_lines(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """_get_last_log_lines() should return last N lines from log file."""
        manager = ServerManager(settings, state_dir=state_dir)
        log_file = state_dir / "server.log"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = manager._get_last_log_lines(3)
        assert "line3" in result
        assert "line4" in result
        assert "line5" in result
        assert "line1" not in result

    def test_get_last_log_lines_handles_read_error(
        self, settings: LocalAISettings, state_dir: Path
    ) -> None:
        """_get_last_log_lines() should return empty string when read fails."""
        manager = ServerManager(settings, state_dir=state_dir)
        log_file = state_dir / "server.log"
        log_file.write_text("some content")

        # Mock read_text to raise OSError
        with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            result = manager._get_last_log_lines(10)

        assert result == ""
