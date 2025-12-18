"""Integration tests for models workflow.

Tests verify the complete workflow from search to download to serving.
These tests use real external dependencies but with controlled inputs.
"""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from local_ai.cli.main import app


class TestModelsWorkflow:
    """Verify complete models workflow from search to download."""

    def test_search_and_download_workflow(
        self, cli_runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test complete workflow: search for model, then download it."""
        from local_ai.models.huggingface import SearchResults
        from local_ai.models.schema import ModelSearchResult

        # Mock search results
        mock_search_result = ModelSearchResult(
            id="mlx-community/test-model-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2024-01-01",
            is_mlx_community=True,
            tags=["mlx", "test"],
            size_bytes=2_000_000_000,
        )

        search_results = SearchResults(
            top_models=[],
            mlx_models=[mock_search_result],
        )

        # Step 1: Search for model
        with patch(
            "local_ai.cli.models.search_models_enhanced", return_value=search_results
        ):
            search_result = cli_runner.invoke(app, ["models", "search", "test"])

        assert search_result.exit_code == 0
        assert "mlx-community/test-model-4bit" in search_result.stdout

        # Step 2: Download the model
        mock_cache_path = temp_dir / "cache"
        mock_cache_path.mkdir()

        def get_size_side_effect(model_id: str) -> int | None:
            # Return None first (not cached), then return size after download
            if not hasattr(get_size_side_effect, "called"):
                get_size_side_effect.called = True
                return None
            return 2_000_000_000

        with patch(
            "huggingface_hub.snapshot_download", return_value=str(mock_cache_path)
        ), patch(
            "local_ai.cli.models.get_local_model_size", side_effect=get_size_side_effect
        ):
            download_result = cli_runner.invoke(
                app, ["models", "download", "mlx-community/test-model-4bit"]
            )

        assert download_result.exit_code == 0
        assert "Downloaded to:" in download_result.stdout

    def test_conversion_workflow_with_hardware_detection(
        self, cli_runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test conversion workflow with hardware detection and quantization."""
        from local_ai.hardware.apple_silicon import AppleSiliconInfo, ChipTier

        # Mock hardware detection
        mock_hardware = AppleSiliconInfo(
            chip_name="Apple M4 Max",
            chip_generation=4,
            chip_tier=ChipTier.MAX,
            memory_gb=128.0,
            cpu_cores=12,
            cpu_performance_cores=8,
            cpu_efficiency_cores=4,
            gpu_cores=40,
            neural_engine_cores=16,
        )

        # Mock conversion process
        mock_convert_path = temp_dir / "converted"
        mock_convert_path.mkdir()

        with patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware
        ), patch(
            "local_ai.cli.models.estimate_model_params_from_name", return_value=8.0
        ), patch(
            "local_ai.cli.models.get_recommended_quantization", return_value="4bit"
        ), patch(
            "mlx_lm.convert"
        ), patch(
            "local_ai.cli.models.get_local_model_size", return_value=None  # Not cached
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mistralai/test-model", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 0
        assert "Auto-detected: 8B params → 4bit recommended" in result.stdout
        assert "Converting mistralai/test-model to MLX format" in result.stdout

    def test_info_command_for_converted_model(
        self, cli_runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test info command for locally converted models."""
        from local_ai.models.schema import ModelSearchResult

        # Create a mock converted model directory
        model_dir = temp_dir / "models" / "mistralai_test-model-4bit-mlx"
        model_dir.mkdir(parents=True)

        # Create a dummy file to simulate model files
        (model_dir / "config.json").write_text('{"model_type": "test"}')

        mock_model = ModelSearchResult(
            id="local/mistralai_test-model-4bit-mlx",
            author="local",
            is_mlx_community=False,
            size_bytes=2_000_000_000,
        )

        with patch(
            "local_ai.cli.models.get_converted_model_info", return_value=mock_model
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "local/mistralai_test-model-4bit-mlx"]
            )

        assert result.exit_code == 0
        assert "mistralai_test-model-4bit-mlx" in result.stdout
        assert "Author: local" in result.stdout
        assert "Location: " in result.stdout


class TestErrorHandling:
    """Verify error handling in models workflow."""

    def test_search_handles_api_errors_gracefully(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that search command handles API errors gracefully."""
        with patch(
            "local_ai.cli.models.search_models_enhanced",
            side_effect=Exception("Network timeout"),
        ):
            result = cli_runner.invoke(app, ["models", "search", "nonexistent"])

        assert result.exit_code == 1
        assert "Search failed" in result.stdout
        assert "Network timeout" in result.stdout

    def test_download_validates_model_format_before_conversion(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that download command validates format before attempting conversion."""
        result = cli_runner.invoke(
            app, ["models", "download", "model/gguf-format", "--convert"]
        )

        assert result.exit_code == 1
        assert "Cannot convert GGUF/GGML models to MLX format" in result.stdout

    def test_info_handles_nonexistent_models(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that info command handles nonexistent models gracefully."""
        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=None
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "nonexistent/model"]
            )

        assert result.exit_code == 1
        assert "Model not found" in result.stdout


class TestHelpAndDocumentation:
    """Verify help text and documentation."""

    def test_models_help_shows_all_subcommands(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that models help shows all available subcommands."""
        result = cli_runner.invoke(app, ["models", "--help"])

        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "info" in result.stdout
        assert "download" in result.stdout

    def test_search_help_shows_sort_options(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that search help shows available sort options."""
        result = cli_runner.invoke(app, ["models", "search", "--help"])

        assert result.exit_code == 0
        assert "--sort" in result.stdout
        assert "downloads" in result.stdout
        assert "likes" in result.stdout

    def test_download_help_shows_conversion_options(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that download help shows conversion and quantization options."""
        result = cli_runner.invoke(app, ["models", "download", "--help"])

        assert result.exit_code == 0
        assert "--convert" in result.stdout
        assert "--quantize" in result.stdout
        assert "auto" in result.stdout
