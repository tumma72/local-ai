"""Tests for output console module."""

from unittest.mock import MagicMock, patch

import pytest

from local_ai.output.console import (
    format_downloads,
    print_error,
    print_info,
    print_success,
    print_warning,
)


class TestOutputConsole:
    """Test output console functionality."""

    def test_format_downloads_millions(self) -> None:
        """Test formatting large download counts in millions."""
        assert format_downloads(1_500_000) == "1.5M"
        assert format_downloads(999_999) == "1000.0K"  # Just under 1M
        assert format_downloads(1_000_000) == "1.0M"

    def test_format_downloads_thousands(self) -> None:
        """Test formatting medium download counts in thousands."""
        assert format_downloads(5_000) == "5.0K"
        assert format_downloads(999) == "999"  # Just under 1K
        assert format_downloads(1_000) == "1.0K"

    def test_format_downloads_small_numbers(self) -> None:
        """Test formatting small download counts as plain numbers."""
        assert format_downloads(0) == "0"
        assert format_downloads(1) == "1"
        assert format_downloads(999) == "999"

    @patch('local_ai.output.console.console.print')
    def test_print_success(self, mock_print: MagicMock) -> None:
        """Test printing success messages."""
        print_success("Operation completed")
        mock_print.assert_called_once_with("[green]✓[/green] Operation completed")

    @patch('local_ai.output.console.console.print')
    def test_print_error_without_suggestion(self, mock_print: MagicMock) -> None:
        """Test printing error messages without suggestion."""
        print_error("File not found")
        mock_print.assert_called_once_with("[red]✗ File not found[/red]")

    @patch('local_ai.output.console.console.print')
    def test_print_error_with_suggestion(self, mock_print: MagicMock) -> None:
        """Test printing error messages with suggestion."""
        print_error("File not found", "Try specifying a different path")
        calls = mock_print.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "[red]✗ File not found[/red]"
        assert calls[1][0][0] == "\nTry specifying a different path"

    @patch('local_ai.output.console.console.print')
    def test_print_warning(self, mock_print: MagicMock) -> None:
        """Test printing warning messages."""
        print_warning("This may take a while")
        mock_print.assert_called_once_with("[yellow]⚠ This may take a while[/yellow]")

    @patch('local_ai.output.console.console.print')
    def test_print_info(self, mock_print: MagicMock) -> None:
        """Test printing informational messages."""
        print_info("Processing request")
        mock_print.assert_called_once_with("[blue]ℹ[/blue] Processing request")
