"""Tests for CLI status commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.hardware.apple_silicon import AppleSiliconInfo


class TestStatusCLI:
    """Test status CLI functionality."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Provide a Typer CLI test runner."""
        return CliRunner()

    @patch('local_ai.cli.status.detect_hardware')
    @patch('local_ai.cli.status.get_max_model_size_gb')
    def test_status_command_success(
        self,
        mock_get_max_model_size_gb: MagicMock,
        mock_detect_hardware: MagicMock,
        cli_runner: CliRunner
    ) -> None:
        """Test status command shows hardware info and exits successfully."""
        # Mock hardware detection
        mock_hardware = AppleSiliconInfo(
            chip_name="Apple M1",
            chip_generation=1,
            chip_tier="base",
            memory_gb=16.0,
            cpu_cores=8,
            cpu_performance_cores=4,
            cpu_efficiency_cores=4,
            gpu_cores=8,
            neural_engine_cores=16
        )
        mock_detect_hardware.return_value = mock_hardware
        mock_get_max_model_size_gb.return_value = 7.0

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Apple M1" in result.output
        assert "16 GB" in result.output
        assert "8 cores" in result.output
        assert "~7 GB" in result.output

    @patch('local_ai.cli.status.detect_hardware')
    def test_status_command_hardware_detection_failure(
        self,
        mock_detect_hardware: MagicMock,
        cli_runner: CliRunner
    ) -> None:
        """Test status command handles hardware detection failure."""
        mock_detect_hardware.side_effect = RuntimeError("Hardware detection failed")

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "Hardware detection failed" in result.output

    def test_status_help_shows_usage(self, cli_runner: CliRunner) -> None:
        """Test status help command shows usage information."""
        result = cli_runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "Show system status and hardware info" in result.output
        assert "--log-level" in result.output
