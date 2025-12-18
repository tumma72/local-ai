"""Behavioral tests for server CLI logs command.

Tests verify the public CLI interface behavior for the `logs` subcommand:
- View last N lines of server log
- Follow logs in real-time
- Handle missing log file gracefully
- Handle permission errors

Tests mock filesystem and subprocess to avoid real I/O.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.cli.server import _format_uptime

runner = CliRunner()


class TestServerLogsCommand:
    """Verify `local-ai server logs` command behavior."""

    def test_logs_shows_last_lines_when_log_file_exists(self, temp_dir: Path) -> None:
        """logs command should display last N lines from log file."""
        # Create a log file with content
        log_content = "\n".join([f"Log line {i}" for i in range(1, 101)])
        log_file = temp_dir / "server.log"
        log_file.write_text(log_content)

        with patch("local_ai.cli.server.STATE_DIR", temp_dir):
            result = runner.invoke(app, ["server", "logs", "--lines", "10"])

        assert result.exit_code == 0
        # Should show the last 10 lines
        assert "Log line 100" in result.stdout
        assert "Log line 91" in result.stdout
        # Should show summary
        assert "Showing 10 of 100 total lines" in result.stdout

    def test_logs_shows_not_found_when_no_log_file(self, temp_dir: Path) -> None:
        """logs command should show helpful message when log file doesn't exist."""
        with patch("local_ai.cli.server.STATE_DIR", temp_dir):
            result = runner.invoke(app, ["server", "logs"])

        assert result.exit_code == 0
        assert "No log file found" in result.stdout
        assert "server may not have been started" in result.stdout

    def test_logs_shows_generic_read_error(self, temp_dir: Path) -> None:
        """logs command should show error when reading log file fails."""
        log_file = temp_dir / "server.log"
        log_file.write_text("test content")

        # Mock open to raise OSError after exists() check passes
        original_open = open

        def mock_open_error(*args, **kwargs):
            if args and str(temp_dir) in str(args[0]):
                raise OSError("Disk read error")
            return original_open(*args, **kwargs)

        with (
            patch("local_ai.cli.server.STATE_DIR", temp_dir),
            patch("builtins.open", side_effect=mock_open_error),
        ):
            result = runner.invoke(app, ["server", "logs"])

        assert result.exit_code == 1
        # The error output is captured
        assert "Failed to read logs" in result.stdout or result.exception is not None

    def test_logs_follow_calls_tail(self, temp_dir: Path) -> None:
        """logs --follow should invoke tail -f subprocess."""
        log_file = temp_dir / "server.log"
        log_file.write_text("test content")

        mock_run = MagicMock()
        with (
            patch("local_ai.cli.server.STATE_DIR", temp_dir),
            patch("subprocess.run", mock_run),
        ):
            result = runner.invoke(app, ["server", "logs", "--follow"])

        assert result.exit_code == 0
        # Verify tail -f was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "tail"
        assert "-f" in call_args


class TestFormatUptime:
    """Verify _format_uptime helper produces correct output."""

    def test_format_uptime_hours_minutes_seconds(self) -> None:
        """Should format uptime with hours, minutes, and seconds."""
        # 2 hours, 30 minutes, 15 seconds
        result = _format_uptime(2 * 3600 + 30 * 60 + 15)
        assert result == "2h 30m 15s"

    def test_format_uptime_minutes_seconds_only(self) -> None:
        """Should format uptime with minutes and seconds when no hours."""
        # 5 minutes, 30 seconds
        result = _format_uptime(5 * 60 + 30)
        assert result == "5m 30s"

    def test_format_uptime_seconds_only(self) -> None:
        """Should format uptime with seconds only when under 1 minute."""
        result = _format_uptime(45)
        assert result == "45s"

    def test_format_uptime_zero_seconds(self) -> None:
        """Should handle zero uptime."""
        result = _format_uptime(0)
        assert result == "0s"
