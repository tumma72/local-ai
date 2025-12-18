"""Server process lifecycle management for local-ai.

Handles starting, stopping, and monitoring the MLX Omni Server process.
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.config.schema import LocalAISettings
from local_ai.logging import get_logger
from local_ai.server.health import check_health, get_models


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
    models: str | None  # Available models from /v1/models (not loaded, just cached locally)
    uptime_seconds: float | None
    health: str | None  # "healthy" | "unhealthy" | "unknown"


class ServerManager:
    """Manages the MLX Omni Server process lifecycle."""

    def __init__(
        self,
        settings: LocalAISettings | None = None,
        state_dir: Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize ServerManager with settings and state directory.

        Args:
            settings: LocalAI configuration settings. Required for start().
            state_dir: Directory for PID and log files. Defaults to ~/.local/state/local-ai/
            host: Server host for status checks. Overrides settings if provided.
            port: Server port for status checks. Overrides settings if provided.

        Note:
            For stop() and status(), settings is not required - only host/port
            are needed for health checks.

            Priority order for host/port:
            1. Explicit host/port parameters (highest priority)
            2. Settings from configuration
            3. Default constants (lowest priority)
        """
        self._settings = settings
        # Initialize host and port with correct priority:
        # 1. Explicit parameters override everything
        # 2. Settings override defaults
        # 3. Default constants as fallback
        if host is not None:
            self._host = host
        elif settings is not None:
            self._host = settings.server.host
        else:
            self._host = DEFAULT_HOST

        if port is not None:
            self._port = port
        elif settings is not None:
            self._port = settings.server.port
        else:
            self._port = DEFAULT_PORT

        self._state_dir = state_dir or Path.home() / ".local" / "state" / "local-ai"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("ServerManager")
        self._logger.debug(
            "Initialized with state_dir={}, host={}, port={}",
            self._state_dir,
            self._host,
            self._port,
        )

    @property
    def _pid_file(self) -> Path:
        """Path to the server PID file."""
        return self._state_dir / "server.pid"

    @property
    def _log_file(self) -> Path:
        """Path to the server log file."""
        return self._state_dir / "server.log"

    @property
    def _start_time_file(self) -> Path:
        """Path to the server start time tracking file."""
        return self._state_dir / "server.start_time"

    def _read_start_time(self) -> float | None:
        """Read start time from file if it exists."""
        if not self._start_time_file.exists():
            return None
        try:
            return float(self._start_time_file.read_text().strip())
        except (ValueError, OSError):
            return None

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

    def _get_last_log_lines(self, num_lines: int = 20) -> str:
        """Read last N lines from server log file."""
        if not self._log_file.exists():
            return ""
        try:
            lines = self._log_file.read_text().strip().split("\n")
            return "\n".join(lines[-num_lines:])
        except OSError:
            return ""

    def start(
        self,
        foreground: bool = False,
        startup_timeout: float = 30.0,
        health_interval: float = 0.5,
    ) -> StartResult:
        """Start the server process.

        Args:
            foreground: If True, run in foreground (not implemented).
            startup_timeout: Seconds to wait for server to become healthy.
            health_interval: Seconds between health checks.

        Returns:
            StartResult with success status and PID or error.

        Raises:
            ValueError: If settings were not provided to constructor.
        """
        if self._settings is None:
            raise ValueError("Settings required for start(). Initialize with settings=...")

        model_info = (
            f"model={self._settings.model.path}"
            if self._settings.model.path
            else "no specific model (dynamic loading)"
        )
        self._logger.info(
            "Starting server: host={}, port={}, {}",
            self._settings.server.host,
            self._settings.server.port,
            model_info,
        )

        if self.is_running():
            pid = self._read_pid()
            self._logger.warning("Server already running with PID {}", pid)
            return StartResult(
                success=False,
                pid=pid,
                error=f"Server already running with PID {pid}",
            )

        # Use custom local-ai server that combines MLX Omni Server with welcome page
        cmd = [
            sys.executable,
            "-m",
            "local_ai.server",
            "--host",
            self._settings.server.host,
            "--port",
            str(self._settings.server.port),
        ]

        # Add model if specified
        # Add model if specified        # Add model if specified
        if self._settings.model.path:
            cmd.extend(["--model", self._settings.model.path])

        self._logger.debug("Executing command: {}", " ".join(cmd))

        # Clear log file for fresh start
        self._log_file.write_text("")

        with open(self._log_file, "a") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        # Write PID file and start time early so we can track the process
        self._pid_file.write_text(str(process.pid))
        self._start_time_file.write_text(str(time.time()))
        self._logger.debug("Process started with PID {}, waiting for health...", process.pid)

        # Wait for server to become healthy
        start_time = time.monotonic()
        host = self._settings.server.host
        port = self._settings.server.port

        while True:
            elapsed = time.monotonic() - start_time

            # Check if process is still alive
            if process.poll() is not None:
                # Process exited - read log for error details
                self._pid_file.unlink(missing_ok=True)
                self._start_time_file.unlink(missing_ok=True)
                log_tail = self._get_last_log_lines(30)
                self._logger.error(
                    "Server process exited with code {} after {:.1f}s",
                    process.returncode,
                    elapsed,
                )
                error_msg = f"Server process exited with code {process.returncode}"
                if log_tail and ("Error" in log_tail or "error" in log_tail):
                    error_msg += f"\n\nServer log:\n{log_tail}"
                return StartResult(success=False, error=error_msg)

            # Check health
            health = check_health(host, port, timeout=2.0)
            if health == "healthy":
                self._logger.info(
                    "Server started successfully with PID {} (healthy after {:.1f}s)",
                    process.pid,
                    elapsed,
                )
                return StartResult(success=True, pid=process.pid)

            # Check timeout
            if elapsed >= startup_timeout:
                # Timeout - kill the process and report failure
                self._logger.error(
                    "Server did not become healthy within {}s, terminating",
                    startup_timeout,
                )
                with contextlib.suppress(OSError):
                    os.kill(process.pid, signal.SIGTERM)
                self._pid_file.unlink(missing_ok=True)
                self._start_time_file.unlink(missing_ok=True)
                log_tail = self._get_last_log_lines(30)
                error_msg = f"Server did not become healthy within {startup_timeout}s"
                if log_tail:
                    error_msg += f"\n\nServer log:\n{log_tail}"
                return StartResult(success=False, error=error_msg)

            time.sleep(health_interval)

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
            self._start_time_file.unlink(missing_ok=True)
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
        self._start_time_file.unlink(missing_ok=True)
        self._logger.info("Server stopped successfully (PID {})", pid)
        return StopResult(success=True)

    def status(self) -> ServerStatus:
        """Get current server status.

        Queries the server for actual health status and loaded models.

        Returns:
            ServerStatus with process and server details.
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
                models=None,
                uptime_seconds=None,
                health=None,
            )

        # Query server for available models (cached locally, not necessarily loaded in memory)
        health = check_health(self._host, self._port, timeout=2.0)
        available_models = get_models(self._host, self._port, timeout=2.0)
        models_str = ", ".join(available_models) if available_models else "none available"

        # Calculate uptime from start time file
        uptime_seconds: float | None = None
        start_time = self._read_start_time()
        if start_time is not None:
            uptime_seconds = time.time() - start_time

        self._logger.debug(
            "Server running: PID={}, host={}, port={}, health={}, uptime={:.1f}s, models={}",
            pid,
            self._host,
            self._port,
            health,
            uptime_seconds or 0,
            available_models,
        )
        return ServerStatus(
            running=True,
            pid=pid,
            host=self._host,
            port=self._port,
            models=models_str,
            uptime_seconds=uptime_seconds,
            health=health,
        )
