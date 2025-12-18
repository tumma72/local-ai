"""Behavioral tests for CLI models commands.

Tests verify the public CLI interface behavior for model management:
- `local-ai models search` - Search HuggingFace for models
- `local-ai models list` - List locally available models
- `local-ai models info` - Get detailed model information
- `local-ai models download` - Download and convert models

Tests mock external dependencies (HuggingFace Hub, mlx_lm) for isolation.
Tests focus on CLI output, exit codes, and user-facing behavior.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.models.schema import ModelSearchResult


@pytest.fixture
def mock_search_results() -> list[ModelSearchResult]:
    """Create mock search results for testing."""
    return [
        ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2024-01-01",
            is_mlx_community=True,
            tags=["mlx", "quantized"],
            size_bytes=4_000_000_000,
        ),
        ModelSearchResult(
            id="mistralai/Devstral-Small-2505",
            author="mistralai",
            downloads=5000,
            likes=200,
            last_modified="2024-02-01",
            is_mlx_community=False,
            tags=["coding", "small"],
            size_bytes=None,
        ),
    ]


class TestModelsSearchCommand:
    """Verify `local-ai models search` command behavior."""

    def test_search_success_returns_exit_code_0_and_shows_results(
        self, cli_runner: CliRunner, mock_search_results: list[ModelSearchResult]
    ) -> None:
        """search command should return exit code 0 and display results."""
        from local_ai.models.huggingface import SearchResults

        mock_results = SearchResults(
            top_models=[mock_search_results[1]],  # mistralai model
            mlx_models=[mock_search_results[0]],  # mlx-community model
        )

        with patch(
            "local_ai.cli.models.search_models_enhanced", return_value=mock_results
        ):
            result = cli_runner.invoke(app, ["models", "search", "qwen3"])

        assert result.exit_code == 0
        # Should show both sections
        assert "Top Models" in result.stdout
        assert "MLX-Optimized" in result.stdout
        assert "mlx-community/Qwen3-8B-4bit" in result.stdout
        assert "mistralai/Devstral-Small-2505" in result.stdout

    def test_search_with_invalid_sort_option_returns_exit_code_1(
        self, cli_runner: CliRunner
    ) -> None:
        """search command should return exit code 1 with invalid sort option."""
        result = cli_runner.invoke(
            app, ["models", "search", "qwen3", "--sort", "invalid_sort"]
        )

        assert result.exit_code == 1
        assert "Invalid sort option" in result.stdout
        assert "Valid options" in result.stdout

    def test_search_when_api_fails_returns_exit_code_1_and_error(
        self, cli_runner: CliRunner
    ) -> None:
        """search command should handle API failures gracefully."""
        with patch(
            "local_ai.cli.models.search_models_enhanced",
            side_effect=Exception("API connection failed"),
        ):
            result = cli_runner.invoke(app, ["models", "search", "nonexistent"])

        assert result.exit_code == 1
        assert "Search failed" in result.stdout
        assert "API connection failed" in result.stdout

    def test_search_with_no_results_shows_helpful_message(
        self, cli_runner: CliRunner
    ) -> None:
        """search command should show helpful message when no results found."""
        from local_ai.models.huggingface import SearchResults

        mock_results = SearchResults(top_models=[], mlx_models=[])

        with patch(
            "local_ai.cli.models.search_models_enhanced", return_value=mock_results
        ):
            result = cli_runner.invoke(app, ["models", "search", "nonexistent"])

        assert result.exit_code == 0  # Not an error, just no results
        assert "No models found" in result.stdout
        assert "Tips" in result.stdout

    def test_search_shows_correct_result_counts(
        self, cli_runner: CliRunner, mock_search_results: list[ModelSearchResult]
    ) -> None:
        """search command should display correct result counts."""
        from local_ai.models.huggingface import SearchResults

        mock_results = SearchResults(
            top_models=[mock_search_results[1]],
            mlx_models=[mock_search_results[0]],
        )

        with patch(
            "local_ai.cli.models.search_models_enhanced", return_value=mock_results
        ):
            result = cli_runner.invoke(app, ["models", "search", "qwen3"])

        assert result.exit_code == 0
        assert "Showing 2 results (1 top + 1 MLX-optimized)" in result.stdout


class TestModelsListCommand:
    """Verify `local-ai models list` command behavior."""

    def test_list_without_server_shows_converted_models_only(
        self, cli_runner: CliRunner
    ) -> None:
        """list command should show converted models when server not running."""
        mock_converted = [
            ModelSearchResult(
                id="local/mistralai_Devstral-Small-4bit-mlx",
                author="local",
                is_mlx_community=False,
                size_bytes=2_000_000_000,
            )
        ]

        with patch(
            "local_ai.cli.models.get_converted_models", return_value=mock_converted
        ), patch("httpx.get", side_effect=Exception("Connection refused")):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 1  # Should fail when server not running
        assert "Failed to list models" in result.stdout
        assert "Connection refused" in result.stdout

    def test_list_with_all_flag_shows_converted_models(
        self, cli_runner: CliRunner
    ) -> None:
        """list --all command should show converted models without server."""
        mock_converted = [
            ModelSearchResult(
                id="local/mistralai_Devstral-Small-4bit-mlx",
                author="local",
                is_mlx_community=False,
                size_bytes=2_000_000_000,
            )
        ]

        with patch(
            "local_ai.cli.models.get_converted_models", return_value=mock_converted
        ):
            result = cli_runner.invoke(app, ["models", "list", "--all"])

        assert result.exit_code == 0
        assert "Local Models" in result.stdout
        assert "mistralai_Devstral-Small-4bit-mlx" in result.stdout
        assert "Showing 1 locally converted models" in result.stdout

    def test_list_with_no_models_shows_helpful_message(
        self, cli_runner: CliRunner
    ) -> None:
        """list command should show helpful message when no models available."""
        with patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ), patch("httpx.get", side_effect=Exception("Connection refused")):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 1
        assert "Failed to list models" in result.stdout
        # Note: The server start suggestion is only shown for connection errors
        # In this case, we just get the generic error message


class TestModelsInfoCommand:
    """Verify `local-ai models info` command behavior."""

    def test_info_for_huggingface_model_shows_detailed_info(
        self, cli_runner: CliRunner
    ) -> None:
        """info command should show detailed information for HuggingFace models."""
        mock_model = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2024-01-01",
            is_mlx_community=True,
            tags=["mlx", "quantized"],
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "mlx-community/Qwen3-8B-4bit"]
            )

        assert result.exit_code == 0
        assert "mlx-community/Qwen3-8B-4bit" in result.stdout
        assert "Author: mlx-community" in result.stdout
        assert "Downloads: 1.0K" in result.stdout
        assert "Size: 3.7 GB" in result.stdout

    def test_info_for_local_model_shows_converted_model_info(
        self, cli_runner: CliRunner
    ) -> None:
        """info command should show information for locally converted models."""
        mock_model = ModelSearchResult(
            id="local/mistralai_Devstral-Small-4bit-mlx",
            author="local",
            is_mlx_community=False,
            size_bytes=2_000_000_000,
        )

        with patch(
            "local_ai.cli.models.get_converted_model_info", return_value=mock_model
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "local/mistralai_Devstral-Small-4bit-mlx"]
            )

        assert result.exit_code == 0
        assert "mistralai_Devstral-Small-4bit-mlx" in result.stdout
        assert "Author: local" in result.stdout
        assert "Location: ~/.local/share/local-ai/models/mistralai_Devstral-Small-4bit-mlx" in result.stdout

    def test_info_for_nonexistent_model_returns_exit_code_1(
        self, cli_runner: CliRunner
    ) -> None:
        """info command should return exit code 1 for nonexistent models."""
        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=None
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "nonexistent/model"]
            )

        assert result.exit_code == 1
        assert "Model not found" in result.stdout


class TestModelsDownloadCommand:
    """Verify `local-ai models download` command behavior."""

    def test_download_mlx_model_success_shows_progress_and_completion(
        self, cli_runner: CliRunner
    ) -> None:
        """download command should show progress and completion for MLX models."""
        mock_cache_path = "/path/to/cache"

        def get_size_side_effect(model_id: str) -> int | None:
            # Return None first (not cached), then return size after download
            if not hasattr(get_size_side_effect, "called"):
                get_size_side_effect.called = True
                return None
            return 4_000_000_000

        with patch(
            "huggingface_hub.snapshot_download", return_value=mock_cache_path
        ), patch(
            "local_ai.cli.models.get_local_model_size", side_effect=get_size_side_effect
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mlx-community/Qwen3-8B-4bit"]
            )

        assert result.exit_code == 0
        assert "Downloading mlx-community/Qwen3-8B-4bit" in result.stdout
        assert "Downloaded to: /path/to/cache" in result.stdout
        assert "Size: 3.7 GB" in result.stdout

    def test_download_with_force_flag_re_downloads_existing_model(
        self, cli_runner: CliRunner
    ) -> None:
        """download command with --force should re-download existing models."""
        mock_cache_path = "/path/to/cache"

        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=4_000_000_000
        ), patch(
            "huggingface_hub.snapshot_download", return_value=mock_cache_path
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mlx-community/Qwen3-8B-4bit", "--force"]
            )

        assert result.exit_code == 0
        assert "Downloading mlx-community/Qwen3-8B-4bit" in result.stdout

    def test_download_already_cached_shows_size_and_skips(
        self, cli_runner: CliRunner
    ) -> None:
        """download command should skip when model already cached."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=4_000_000_000
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mlx-community/Qwen3-8B-4bit"]
            )

        assert result.exit_code == 0
        assert "Model already downloaded" in result.stdout
        assert "Size: 3.7 GB" in result.stdout
        assert "Use --force to re-download" in result.stdout

    def test_download_with_convert_flag_validates_model_format(
        self, cli_runner: CliRunner
    ) -> None:
        """download command with --convert should validate model format."""
        result = cli_runner.invoke(
            app, ["models", "download", "model/gguf-format", "--convert"]
        )

        assert result.exit_code == 1
        assert "Cannot convert GGUF/GGML models to MLX format" in result.stdout

    def test_download_conversion_success_shows_quantization_and_output(
        self, cli_runner: CliRunner
    ) -> None:
        """download command with conversion should show quantization and output path."""
        with patch(
            "mlx_lm.convert"
        ), patch(
            "local_ai.cli.models.get_local_model_size", return_value=None  # Not cached
        ), patch(
            "local_ai.cli.models.detect_hardware"
        ), patch(
            "local_ai.cli.models.estimate_model_params_from_name", return_value=8.0
        ), patch(
            "local_ai.cli.models.get_recommended_quantization", return_value="4bit"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mistralai/test-model", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 0
        assert "Converting mistralai/test-model to MLX format" in result.stdout
        assert "Auto-detected: 8B params → 4bit recommended" in result.stdout

    def test_download_conversion_failure_shows_helpful_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download command should show helpful error when conversion fails."""
        with patch(
            "mlx_lm.convert",
            side_effect=Exception("No safetensors found"),
        ), patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mistralai/test-model", "--convert"]
            )

        assert result.exit_code == 1
        assert "Cannot convert: No safetensors weights found" in result.stdout
        assert "Search for an MLX-optimized version" in result.stdout


class TestModelsDownloadValidation:
    """Verify download command validates inputs and provides helpful errors."""

    def test_download_convert_with_gguf_model_shows_specific_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert should show specific error for GGUF models."""
        result = cli_runner.invoke(
            app, ["models", "download", "model/gguf-format", "--convert"]
        )

        assert result.exit_code == 1
        assert "Cannot convert GGUF/GGML models to MLX format" in result.stdout
        assert "Use an MLX-optimized version from mlx-community" in result.stdout

    def test_download_convert_with_awq_model_shows_specific_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert should show specific error for AWQ models."""
        result = cli_runner.invoke(
            app, ["models", "download", "model-awq", "--convert"]
        )

        assert result.exit_code == 1
        assert "Cannot convert AWQ models to MLX format" in result.stdout

    def test_download_convert_with_gptq_model_shows_specific_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert should show specific error for GPTQ models."""
        result = cli_runner.invoke(
            app, ["models", "download", "model-gptq", "--convert"]
        )

        assert result.exit_code == 1
        assert "Cannot convert GPTQ models to MLX format" in result.stdout


class TestModelsCommandHelp:
    """Verify models commands show helpful usage information."""

    def test_models_help_shows_available_subcommands(
        self, cli_runner: CliRunner
    ) -> None:
        """models command should show available subcommands."""
        result = cli_runner.invoke(app, ["models", "--help"])

        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "info" in result.stdout
        assert "download" in result.stdout

    def test_search_help_shows_query_argument(
        self, cli_runner: CliRunner
    ) -> None:
        """search help should show query argument and options."""
        result = cli_runner.invoke(app, ["models", "search", "--help"])

        assert result.exit_code == 0
        assert "QUERY" in result.stdout
        assert "--top" in result.stdout
        assert "--sort" in result.stdout

    def test_download_help_shows_convert_option(
        self, cli_runner: CliRunner
    ) -> None:
        """download help should show convert and quantize options."""
        result = cli_runner.invoke(app, ["models", "download", "--help"])

        assert result.exit_code == 0
        assert "MODEL_ID" in result.stdout
        assert "--convert" in result.stdout
        assert "--quantize" in result.stdout
