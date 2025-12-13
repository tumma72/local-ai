"""Server process lifecycle management for local-ai.

Handles starting, stopping, and monitoring the MLX LM server process.
"""

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_ai.config.schema import LocalAISettings
from local_ai.logging import get_logger


@dataclass
class StartResult:
    """Result of a server start operation."""

    success: bool
    pid: int | None = None
    error: str | None = None


@dataclass
class StopResult:
    """Result of a server stop operation."""

    success: bool
    error: str | None = None


@dataclass
class ServerStatus:
    """Current state of the server process."""

    running: bool
    pid: int | None
    host: str | None
    port: int | None
    model: str | None
    uptime_seconds: float | None
    health: str | None  # "healthy" | "unhealthy" | "unknown"


class ServerManager:
    """Manages the MLX LM server process lifecycle."""

    def __init__(
        self, settings: LocalAISettings, state_dir: Path | None = None
    ) -> None:
        """Initialize ServerManager with settings and state directory.

        Args:
            settings: LocalAI configuration settings.
            state_dir: Directory for PID and log files. Defaults to ~/.local/state/local-ai/
        """
        self._settings = settings
        self._state_dir = state_dir or Path.home() / ".local" / "state" / "local-ai"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("ServerManager")
        self._logger.debug("Initialized with state_dir={}", self._state_dir)

    @property
    def _pid_file(self) -> Path:
        """Path to the server PID file."""
        return self._state_dir / "server.pid"

    @property
    def _log_file(self) -> Path:
        """Path to the server log file."""
        return self._state_dir / "server.log"

    def _read_pid(self) -> int | None:
        """Read PID from file if it exists."""
        if not self._pid_file.exists():
            return None
        try:
            return int(self._pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def _process_exists(self, pid: int) -> bool:
        """Check if a process with the given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def is_running(self) -> bool:
        """Check if the server process is currently running."""
        pid = self._read_pid()
        if pid is None:
            return False
        return self._process_exists(pid)

    def start(self, foreground: bool = False) -> StartResult:
        """Start the server process.

        Args:
            foreground: If True, run in foreground (not implemented).

        Returns:
            StartResult with success status and PID or error.
        """
        self._logger.info(
            "Starting server: host={}, port={}, model={}",
            self._settings.server.host,
            self._settings.server.port,
            self._settings.model.path,
        )

        if self.is_running():
            pid = self._read_pid()
            self._logger.warning("Server already running with PID {}", pid)
            return StartResult(
                success=False,
                pid=pid,
                error=f"Server already running with PID {pid}",
            )

        cmd = [
            "python",
            "-m",
            "mlx_lm.server",
            "--host",
            self._settings.server.host,
            "--port",
            str(self._settings.server.port),
            "--model",
            self._settings.model.path,
        ]

        self._logger.debug("Executing command: {}", " ".join(cmd))

        with open(self._log_file, "a") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        # Check if process started successfully
        if process.poll() is not None:
            self._logger.error("Server process exited immediately")
            return StartResult(
                success=False,
                error="Server process exited immediately",
            )

        # Write PID file
        self._pid_file.write_text(str(process.pid))
        self._logger.info("Server started successfully with PID {}", process.pid)

        return StartResult(success=True, pid=process.pid)

    def stop(self, force: bool = False, timeout: float = 10.0) -> StopResult:
        """Stop the server process.

        Args:
            force: If True, use SIGKILL instead of SIGTERM.
            timeout: Seconds to wait for graceful shutdown.

        Returns:
            StopResult with success status or error.
        """
        self._logger.info("Stopping server (force={})", force)

        pid = self._read_pid()
        if pid is None:
            self._logger.debug("No PID file found, server not running")
            return StopResult(success=True)

        if not self._process_exists(pid):
            # Process not running, clean up stale PID file
            self._logger.debug("Stale PID file found, cleaning up")
            self._pid_file.unlink(missing_ok=True)
            return StopResult(success=True)

        sig = signal.SIGKILL if force else signal.SIGTERM
        sig_name = "SIGKILL" if force else "SIGTERM"
        self._logger.debug("Sending {} to PID {}", sig_name, pid)

        try:
            os.kill(pid, sig)
        except OSError as e:
            self._logger.error("Failed to stop server: {}", e)
            return StopResult(success=False, error=str(e))

        self._pid_file.unlink(missing_ok=True)
        self._logger.info("Server stopped successfully (PID {})", pid)
        return StopResult(success=True)

    def status(self) -> ServerStatus:
        """Get current server status.

        Returns:
            ServerStatus with process and configuration details.
        """
        self._logger.debug("Checking server status")
        pid = self._read_pid()
        running = pid is not None and self._process_exists(pid)

        if not running:
            self._logger.debug("Server is not running")
            return ServerStatus(
                running=False,
                pid=None,
                host=None,
                port=None,
                model=None,
                uptime_seconds=None,
                health=None,
            )

        self._logger.debug(
            "Server running: PID={}, host={}, port={}",
            pid,
            self._settings.server.host,
            self._settings.server.port,
        )
        return ServerStatus(
            running=True,
            pid=pid,
            host=self._settings.server.host,
            port=self._settings.server.port,
            model=self._settings.model.path,
            uptime_seconds=None,  # Would require tracking start time
            health="unknown",  # Would require HTTP health check
        )
