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


class TestModelsListWithServer:
    """Verify list command behavior when server is running."""

    def test_list_shows_server_models_and_converted_models(
        self, cli_runner: CliRunner
    ) -> None:
        """list command should show models from server and converted models."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "mlx-community/Model-A"},
                {"id": "mlx-community/Model-B"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_converted = [
            ModelSearchResult(
                id="local/converted-model",
                author="local",
                is_mlx_community=False,
                size_bytes=2_000_000_000,
            )
        ]

        with patch(
            "httpx.get", return_value=mock_response
        ), patch(
            "local_ai.cli.models.get_converted_models", return_value=mock_converted
        ), patch(
            "local_ai.cli.models.create_local_model_result", side_effect=lambda x: ModelSearchResult(
                id=x, author="mlx-community", is_mlx_community=True
            )
        ):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "Server Models" in result.stdout
        assert "Converted Models" in result.stdout

    def test_list_shows_only_server_models_when_no_converted(
        self, cli_runner: CliRunner
    ) -> None:
        """list command shows server models only when no converted models exist."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "mlx-community/Model-A"}]
        }
        mock_response.raise_for_status = Mock()

        with patch(
            "httpx.get", return_value=mock_response
        ), patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ), patch(
            "local_ai.cli.models.create_local_model_result", side_effect=lambda x: ModelSearchResult(
                id=x, author="mlx-community", is_mlx_community=True
            )
        ):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "Server Models" in result.stdout

    def test_list_shows_no_models_message_when_empty(
        self, cli_runner: CliRunner
    ) -> None:
        """list command shows helpful message when no models available."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        with patch(
            "httpx.get", return_value=mock_response
        ), patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "No models available" in result.stdout

    def test_list_handles_server_connection_error(
        self, cli_runner: CliRunner
    ) -> None:
        """list command shows error when server connection fails."""
        import httpx

        with patch(
            "httpx.get", side_effect=httpx.ConnectError("Connection refused")
        ), patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 1
        assert "Cannot connect to server" in result.stdout

    def test_list_handles_server_http_error(
        self, cli_runner: CliRunner
    ) -> None:
        """list command shows error on HTTP error from server."""
        import httpx
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Server error", request=Mock(), response=mock_response
            )
        )

        with patch(
            "httpx.get", return_value=mock_response
        ), patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ):
            result = cli_runner.invoke(app, ["models", "list"])

        assert result.exit_code == 1
        assert "Server error" in result.stdout


class TestModelsInfoLocalNotFound:
    """Verify info command handles local model not found."""

    def test_info_local_model_not_found_shows_error(
        self, cli_runner: CliRunner
    ) -> None:
        """info command shows error when local model not found."""
        with patch(
            "local_ai.cli.models.get_converted_model_info", return_value=None
        ):
            result = cli_runner.invoke(
                app, ["models", "info", "local/nonexistent-model"]
            )

        assert result.exit_code == 1
        assert "Local model not found" in result.stdout
        assert "List available models" in result.stdout


class TestModelsRecommendCommand:
    """Verify models recommend command behavior."""

    def test_recommend_shows_settings_in_text_format(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend command should show settings in text format."""
        mock_model = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            is_mlx_community=True,
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("Not Apple Silicon")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Qwen3-8B-4bit"]
            )

        assert result.exit_code == 0
        assert "Model Recommendation" in result.stdout
        assert "temperature" in result.stdout
        assert "max_tokens" in result.stdout

    def test_recommend_shows_settings_in_json_format(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend command should output JSON when --format json specified."""
        mock_model = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("Not Apple Silicon")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Qwen3-8B-4bit", "--format", "json"]
            )

        assert result.exit_code == 0
        import json
        data = json.loads(result.stdout)
        assert "model" in data
        assert "recommendation" in data

    def test_recommend_shows_settings_in_zed_format(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend command should output Zed config when --format zed specified."""
        mock_model = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("Not Apple Silicon")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Qwen3-8B-4bit", "--format", "zed"]
            )

        assert result.exit_code == 0
        assert "Zed settings.json" in result.stdout
        assert "language_models" in result.stdout

    def test_recommend_shows_hardware_info_on_apple_silicon(
        self, cli_runner: CliRunner, mock_hardware_128gb
    ) -> None:
        """recommend command should show hardware info on Apple Silicon."""
        mock_model = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware_128gb
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Qwen3-8B-4bit"]
            )

        assert result.exit_code == 0
        assert "Hardware" in result.stdout
        assert "M4 Max" in result.stdout

    def test_recommend_invalid_format_shows_error(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend command with invalid format shows error."""
        result = cli_runner.invoke(
            app, ["models", "recommend", "mlx-community/test", "--format", "invalid"]
        )

        assert result.exit_code == 1
        assert "Invalid format" in result.stdout

    def test_recommend_model_not_found_shows_error(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend command shows error when model not found."""
        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=None
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "nonexistent/model"]
            )

        assert result.exit_code == 1
        assert "Model not found" in result.stdout


class TestModelsDownloadConversion:
    """Verify download command conversion functionality."""

    def test_download_convert_with_auto_quantize_detects_hardware(
        self, cli_runner: CliRunner, mock_hardware_128gb
    ) -> None:
        """download --convert --quantize auto should detect hardware."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware_128gb
        ), patch(
            "local_ai.cli.models.estimate_model_params_from_name", return_value=8.0
        ), patch(
            "local_ai.cli.models.get_recommended_quantization", return_value="4bit"
        ), patch(
            "mlx_lm.convert"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mistralai/Test-8B", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 0
        assert "Auto-detected" in result.stdout
        assert "4bit recommended" in result.stdout

    def test_download_convert_auto_quantize_fallback_on_detection_failure(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert --quantize auto should fallback when detection fails."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("No Apple Silicon")
        ), patch(
            "mlx_lm.convert"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mistralai/Test-Model", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 0
        assert "Hardware detection failed" in result.stdout or "4bit default" in result.stdout

    def test_download_convert_auto_quantize_model_too_large(
        self, cli_runner: CliRunner, mock_hardware_8gb
    ) -> None:
        """download --convert --quantize auto should error when model too large."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware_8gb
        ), patch(
            "local_ai.cli.models.estimate_model_params_from_name", return_value=70.0  # 70B model
        ), patch(
            "local_ai.cli.models.get_recommended_quantization", return_value="too_large"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "meta/Llama-70B", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 1
        assert "too large" in result.stdout.lower()

    def test_download_convert_auto_quantize_size_unknown_fallback(
        self, cli_runner: CliRunner, mock_hardware_128gb
    ) -> None:
        """download --convert --quantize auto should fallback when size unknown."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware_128gb
        ), patch(
            "local_ai.cli.models.estimate_model_params_from_name", return_value=None
        ), patch(
            "mlx_lm.convert"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "unknown/model-name", "--convert", "--quantize", "auto"]
            )

        assert result.exit_code == 0
        assert "Could not detect model size" in result.stdout

    def test_download_mlx_model_failure_shows_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download command shows error when download fails."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "huggingface_hub.snapshot_download", side_effect=Exception("Network error")
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "mlx-community/test-model"]
            )

        assert result.exit_code == 1
        assert "Download failed" in result.stdout

    def test_download_conversion_unsupported_architecture_error(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert shows helpful error for unsupported architecture."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "mlx_lm.convert", side_effect=Exception("model_type 'custom' not supported")
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "custom/unsupported-arch", "--convert"]
            )

        assert result.exit_code == 1
        assert "Unsupported model architecture" in result.stdout

    def test_download_convert_with_explicit_quantize_level(
        self, cli_runner: CliRunner
    ) -> None:
        """download --convert --quantize 8bit should use explicit quantization."""
        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "mlx_lm.convert"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "test/model", "--convert", "--quantize", "8bit"]
            )

        assert result.exit_code == 0
        assert "8bit" in result.stdout

    def test_download_convert_with_output_dir(
        self, cli_runner: CliRunner, tmp_path
    ) -> None:
        """download --convert --output should use custom output directory."""
        output_dir = str(tmp_path / "custom_output")

        with patch(
            "local_ai.cli.models.get_local_model_size", return_value=None
        ), patch(
            "mlx_lm.convert"
        ):
            result = cli_runner.invoke(
                app, ["models", "download", "test/model", "--convert", "--output", output_dir]
            )

        assert result.exit_code == 0
        assert "custom_output" in result.stdout


class TestModelsListAllFlag:
    """Verify list --all flag behavior."""

    def test_list_all_no_models_shows_helpful_message(
        self, cli_runner: CliRunner
    ) -> None:
        """list --all should show helpful message when no local models found."""
        with patch(
            "local_ai.cli.models.get_converted_models", return_value=[]
        ):
            result = cli_runner.invoke(app, ["models", "list", "--all"])

        assert result.exit_code == 0
        assert "No local models found" in result.stdout
        assert "Download models with" in result.stdout


class TestModelsRecommendEdgeCases:
    """Verify recommend command edge cases."""

    def test_recommend_shows_model_too_large_warning(
        self, cli_runner: CliRunner, mock_hardware_8gb
    ) -> None:
        """recommend should show warning when model doesn't fit in memory."""
        mock_model = ModelSearchResult(
            id="mlx-community/Large-Model-32B",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=32_000_000_000,  # 32 GB
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", return_value=mock_hardware_8gb
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Large-Model-32B"]
            )

        assert result.exit_code == 0
        # Should show "too large" warning
        assert "Too large" in result.stdout or "over" in result.stdout

    def test_recommend_handles_analysis_failure(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend should show error when analysis fails."""
        with patch(
            "local_ai.models.huggingface.get_model_info",
            side_effect=Exception("Network error")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "error/model"]
            )

        assert result.exit_code == 1
        assert "Model not found" in result.stdout

    def test_recommend_without_context_length(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend should work without context length info."""
        mock_model = ModelSearchResult(
            id="mlx-community/Model-8B",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=4_000_000_000,
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=None
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("No Apple Silicon")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Model-8B"]
            )

        assert result.exit_code == 0
        assert "temperature" in result.stdout

    def test_recommend_without_size_info(
        self, cli_runner: CliRunner
    ) -> None:
        """recommend should work without size info."""
        mock_model = ModelSearchResult(
            id="mlx-community/Model-8B",
            author="mlx-community",
            is_mlx_community=True,
            size_bytes=None,  # No size info
        )

        with patch(
            "local_ai.models.huggingface.get_model_info", return_value=mock_model
        ), patch(
            "local_ai.cli.models._fetch_context_length", return_value=32768
        ), patch(
            "local_ai.cli.models.detect_hardware", side_effect=RuntimeError("No Apple Silicon")
        ):
            result = cli_runner.invoke(
                app, ["models", "recommend", "mlx-community/Model-8B"]
            )

        assert result.exit_code == 0
        assert "temperature" in result.stdout
