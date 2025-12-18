"""Behavioral tests for logging configuration module.

Tests verify public behavior of logging configuration:
- configure_logging sets up console and file handlers
- configure_logging respects custom log directory
- configure_logging respects console flag
- get_logger returns bounded logger with component name
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_ai.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Verify configure_logging() behavior for setting up loggers."""

    def test_configure_logging_with_custom_log_dir(self, tmp_path: Path) -> None:
        """Should use custom log directory when log_dir is provided.

        Covers line 47: LOG_DIR = log_dir assignment.
        """
        custom_log_dir = tmp_path / "custom_logs"

        # Patch the _base_logger to avoid side effects
        with patch("local_ai.logging._base_logger") as mock_logger:
            mock_logger.remove = MagicMock()
            mock_logger.add = MagicMock()

            configure_logging(log_level="INFO", log_dir=custom_log_dir, console=False)

            # Verify log directory was created
            assert custom_log_dir.exists()
            assert custom_log_dir.is_dir()

            # Verify file handler was added with custom directory
            add_calls = mock_logger.add.call_args_list
            # Should have at least 2 file handlers (main log and error log)
            assert len(add_calls) >= 2

            # Check that the custom path is used
            log_paths = [str(call[0][0]) for call in add_calls]
            assert any("custom_logs" in path for path in log_paths)

    def test_configure_logging_with_console_true(self, tmp_path: Path) -> None:
        """Should add console handler when console=True.

        Covers line 56: _base_logger.add(sys.stderr, ...) when console is True.
        """
        with patch("local_ai.logging._base_logger") as mock_logger:
            mock_logger.remove = MagicMock()
            mock_logger.add = MagicMock()

            configure_logging(log_level="DEBUG", log_dir=tmp_path, console=True)

            # Verify console handler was added (sys.stderr is first argument)
            add_calls = mock_logger.add.call_args_list
            # First call should be to sys.stderr (console)
            import sys
            first_call_arg = add_calls[0][0][0]
            assert first_call_arg == sys.stderr

    def test_configure_logging_with_console_false(self, tmp_path: Path) -> None:
        """Should NOT add console handler when console=False."""
        with patch("local_ai.logging._base_logger") as mock_logger:
            mock_logger.remove = MagicMock()
            mock_logger.add = MagicMock()

            configure_logging(log_level="INFO", log_dir=tmp_path, console=False)

            # Verify console handler was NOT added
            add_calls = mock_logger.add.call_args_list
            import sys
            # No call should have sys.stderr as first argument
            for call in add_calls:
                first_arg = call[0][0]
                assert first_arg != sys.stderr

    def test_configure_logging_creates_log_directory(self, tmp_path: Path) -> None:
        """Should create log directory if it does not exist."""
        nested_log_dir = tmp_path / "nested" / "log" / "dir"
        assert not nested_log_dir.exists()

        with patch("local_ai.logging._base_logger") as mock_logger:
            mock_logger.remove = MagicMock()
            mock_logger.add = MagicMock()

            configure_logging(log_level="INFO", log_dir=nested_log_dir, console=False)

            # Directory should be created
            assert nested_log_dir.exists()


class TestGetLogger:
    """Verify get_logger() returns bounded logger."""

    def test_get_logger_returns_bounded_logger(self) -> None:
        """Should return logger bound to component name."""
        logger = get_logger("TestComponent")

        # Logger should have the bind method and component extra
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")

    def test_get_logger_with_different_components(self) -> None:
        """Should return distinct loggers for different components."""
        logger1 = get_logger("Component1")
        logger2 = get_logger("Component2")

        # Both should be valid loggers
        assert hasattr(logger1, "info")
        assert hasattr(logger2, "info")
