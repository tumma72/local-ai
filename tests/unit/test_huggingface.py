"""Behavioral tests for HuggingFace Hub integration.

Tests verify the public behavior of model discovery functions:
- search_models() returns MLX-optimized models from HuggingFace
- search_models_enhanced() returns both top models and MLX versions
- get_model_info() returns detailed model information
- Local model functions handle filesystem operations correctly

Tests mock HuggingFace API to avoid network calls.
Tests focus on WHAT the functions return, not HOW they query the API.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_ai.models.huggingface import (
    SearchResults,
    create_local_model_result,
    get_converted_model_info,
    get_converted_models,
    get_local_model_size,
    get_model_info,
    search_models,
    search_models_enhanced,
)


@pytest.fixture
def mock_hf_model() -> MagicMock:
    """Create a mock HuggingFace ModelInfo object."""
    model = MagicMock()
    model.id = "mlx-community/Qwen3-8B-4bit"
    model.author = "mlx-community"
    model.downloads = 5000
    model.likes = 100
    model.last_modified = "2024-01-15T12:00:00Z"
    model.tags = ["mlx", "quantized", "text-generation"]
    model.siblings = None
    return model


@pytest.fixture
def mock_original_model() -> MagicMock:
    """Create a mock for an original (non-MLX) model."""
    model = MagicMock()
    model.id = "Qwen/Qwen3-8B"
    model.author = "Qwen"
    model.downloads = 50000
    model.likes = 1000
    model.last_modified = "2024-01-10T12:00:00Z"
    model.tags = ["transformers", "text-generation"]
    model.siblings = None
    return model


class TestSearchModels:
    """Verify search_models() returns appropriate MLX model results."""

    def test_search_returns_mlx_community_models_first(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models should prioritize mlx-community models in results."""
        mock_api = MagicMock()
        mock_api.list_models.return_value = [mock_hf_model]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("qwen3", limit=10)

        assert len(results) == 1
        assert results[0].id == "mlx-community/Qwen3-8B-4bit"
        assert results[0].is_mlx_community is True

    def test_search_respects_limit_parameter(self, mock_hf_model: MagicMock) -> None:
        """search_models should not return more results than the limit."""
        # Create multiple mock models
        models = []
        for i in range(10):
            m = MagicMock()
            m.id = f"mlx-community/model-{i}"
            m.author = "mlx-community"
            m.downloads = 1000 - i
            m.likes = 50
            m.last_modified = "2024-01-01"
            m.tags = ["mlx"]
            models.append(m)

        mock_api = MagicMock()
        mock_api.list_models.return_value = models

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("model", limit=5)

        assert len(results) == 5

    def test_search_handles_api_failure_gracefully(self) -> None:
        """search_models should return empty list when API fails."""
        mock_api = MagicMock()
        mock_api.list_models.side_effect = Exception("Network error")

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("qwen3", limit=10)

        assert results == []

    def test_search_deduplicates_results(self, mock_hf_model: MagicMock) -> None:
        """search_models should not return duplicate model IDs."""
        mock_api = MagicMock()
        # Return same model from both searches
        mock_api.list_models.return_value = [mock_hf_model]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("qwen3", limit=10, include_all_mlx=True)

        # Should only have one result despite being returned twice
        assert len(results) == 1


class TestSearchModelsEnhanced:
    """Verify search_models_enhanced() returns structured results."""

    def test_enhanced_search_returns_separate_sections(
        self, mock_hf_model: MagicMock, mock_original_model: MagicMock
    ) -> None:
        """search_models_enhanced should separate top models from MLX models."""
        mock_api = MagicMock()
        # First call returns original models, second returns MLX models
        mock_api.list_models.side_effect = [
            [mock_original_model],  # Top models search
            [mock_hf_model],  # MLX community search
            [],  # Additional MLX search
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models_enhanced("qwen3", top_limit=3, mlx_limit=10)

        assert isinstance(results, SearchResults)
        assert len(results.top_models) == 1
        assert len(results.mlx_models) == 1
        assert results.top_models[0].author == "Qwen"
        assert results.mlx_models[0].is_mlx_community is True

    def test_enhanced_search_excludes_mlx_community_from_top_models(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models_enhanced should not include mlx-community in top_models."""
        mock_api = MagicMock()
        mock_api.list_models.side_effect = [
            [mock_hf_model],  # Top models would return MLX model
            [mock_hf_model],  # MLX search
            [],
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models_enhanced("qwen3")

        # MLX community model should be excluded from top_models
        assert len(results.top_models) == 0
        assert len(results.mlx_models) == 1


class TestGetModelInfo:
    """Verify get_model_info() returns detailed model information."""

    def test_get_model_info_returns_model_details(
        self, mock_hf_model: MagicMock
    ) -> None:
        """get_model_info should return complete model information."""
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_hf_model

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            result = get_model_info("mlx-community/Qwen3-8B-4bit")

        assert result is not None
        assert result.id == "mlx-community/Qwen3-8B-4bit"
        assert result.downloads == 5000
        assert result.likes == 100

    def test_get_model_info_calculates_size_from_files(self) -> None:
        """get_model_info should calculate total size from file metadata."""
        mock_file1 = MagicMock()
        mock_file1.size = 2_000_000_000
        mock_file2 = MagicMock()
        mock_file2.size = 1_500_000_000

        mock_model = MagicMock()
        mock_model.id = "mlx-community/test-model"
        mock_model.author = "mlx-community"
        mock_model.downloads = 100
        mock_model.likes = 10
        mock_model.last_modified = "2024-01-01"
        mock_model.tags = ["mlx"]
        mock_model.siblings = [mock_file1, mock_file2]

        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_model

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            result = get_model_info("mlx-community/test-model")

        assert result is not None
        assert result.size_bytes == 3_500_000_000

    def test_get_model_info_returns_none_for_nonexistent_model(self) -> None:
        """get_model_info should return None when model not found."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("Repository Not Found")

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            result = get_model_info("nonexistent/model")

        assert result is None


class TestLocalModelFunctions:
    """Verify functions for locally cached and converted models."""

    def test_get_local_model_size_returns_none_when_not_cached(
        self, temp_dir: Path
    ) -> None:
        """get_local_model_size should return None when model not in cache."""
        with patch.object(Path, "home", return_value=temp_dir):
            result = get_local_model_size("mlx-community/Qwen3-8B-4bit")

        assert result is None

    def test_get_local_model_size_returns_size_when_cached(
        self, temp_dir: Path
    ) -> None:
        """get_local_model_size should return total size of cached model files."""
        # Create mock cache structure
        cache_dir = temp_dir / ".cache" / "huggingface" / "hub"
        model_blobs = cache_dir / "models--mlx-community--Qwen3-8B-4bit" / "blobs"
        model_blobs.mkdir(parents=True)

        # Create test files
        (model_blobs / "file1.safetensors").write_bytes(b"x" * 1000)
        (model_blobs / "file2.safetensors").write_bytes(b"x" * 500)

        with patch.object(Path, "home", return_value=temp_dir):
            result = get_local_model_size("mlx-community/Qwen3-8B-4bit")

        assert result == 1500

    def test_create_local_model_result_extracts_author(self) -> None:
        """create_local_model_result should extract author from model ID."""
        with patch(
            "local_ai.models.huggingface.get_local_model_size", return_value=None
        ):
            result = create_local_model_result("mlx-community/Qwen3-8B-4bit")

        assert result.id == "mlx-community/Qwen3-8B-4bit"
        assert result.author == "mlx-community"
        assert result.is_mlx_community is True

    def test_get_converted_models_returns_empty_when_no_directory(
        self, temp_dir: Path
    ) -> None:
        """get_converted_models should return empty list when models dir doesn't exist."""
        with patch(
            "local_ai.models.huggingface.CONVERTED_MODELS_DIR",
            temp_dir / "nonexistent",
        ):
            results = get_converted_models()

        assert results == []

    def test_get_converted_models_returns_local_models(self, temp_dir: Path) -> None:
        """get_converted_models should return list of locally converted models."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()

        # Create mock converted model directory
        model_dir = models_dir / "mistralai_Devstral-Small-8bit-mlx"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x" * 2000)

        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", models_dir):
            results = get_converted_models()

        assert len(results) == 1
        assert results[0].id == "local/mistralai_Devstral-Small-8bit-mlx"
        assert results[0].author == "local"
        assert results[0].size_bytes == 2000

    def test_get_converted_model_info_strips_local_prefix(
        self, temp_dir: Path
    ) -> None:
        """get_converted_model_info should handle both prefixed and unprefixed names."""
        models_dir = temp_dir / "models"
        model_dir = models_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.safetensors").write_bytes(b"x" * 1000)

        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", models_dir):
            # Both should work
            result1 = get_converted_model_info("local/test-model")
            result2 = get_converted_model_info("test-model")

        assert result1 is not None
        assert result2 is not None
        assert result1.id == result2.id == "local/test-model"

    def test_get_converted_model_info_returns_none_when_not_found(
        self, temp_dir: Path
    ) -> None:
        """get_converted_model_info should return None for nonexistent model."""
        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", temp_dir):
            result = get_converted_model_info("nonexistent-model")

        assert result is None


class TestSearchModelsApiFailures:
    """Verify search_models handles various API failure scenarios gracefully."""

    def test_search_with_include_all_mlx_returns_non_community_models(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models with include_all_mlx should return non-mlx-community MLX models."""
        # Create a non-mlx-community MLX model
        other_mlx_model = MagicMock()
        other_mlx_model.id = "user/custom-mlx-model"
        other_mlx_model.author = "user"
        other_mlx_model.downloads = 100
        other_mlx_model.likes = 5
        other_mlx_model.last_modified = "2024-01-01"
        other_mlx_model.tags = ["mlx"]

        mock_api = MagicMock()
        # First call (mlx-community) returns empty, second call (all MLX) returns model
        mock_api.list_models.side_effect = [[], [other_mlx_model]]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("custom", limit=10, include_all_mlx=True)

        assert len(results) == 1
        assert results[0].id == "user/custom-mlx-model"
        assert results[0].is_mlx_community is False

    def test_search_all_mlx_api_failure_returns_partial_results(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models should return mlx-community results even if second search fails."""
        mock_api = MagicMock()
        # First call succeeds, second call (all MLX) fails
        mock_api.list_models.side_effect = [
            [mock_hf_model],
            Exception("Secondary search failed"),
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models("qwen3", limit=10, include_all_mlx=True)

        # Should still return the mlx-community results
        assert len(results) == 1
        assert results[0].id == "mlx-community/Qwen3-8B-4bit"


class TestSearchModelsEnhancedApiFailures:
    """Verify search_models_enhanced handles API failures gracefully."""

    def test_enhanced_search_handles_top_models_api_failure(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models_enhanced should return MLX results even if top models search fails."""
        mock_api = MagicMock()
        # First call (top models) fails, rest succeed
        mock_api.list_models.side_effect = [
            Exception("Top models search failed"),
            [mock_hf_model],  # MLX community search
            [],  # Additional MLX search
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models_enhanced("qwen3", top_limit=3, mlx_limit=10)

        assert len(results.top_models) == 0  # Failed
        assert len(results.mlx_models) == 1  # Still works

    def test_enhanced_search_includes_other_mlx_when_community_insufficient(
        self, mock_hf_model: MagicMock
    ) -> None:
        """search_models_enhanced should fetch other MLX models when mlx-community is insufficient."""
        # Create models for each search type
        other_mlx_model = MagicMock()
        other_mlx_model.id = "user/other-mlx-model"
        other_mlx_model.author = "user"
        other_mlx_model.downloads = 100
        other_mlx_model.likes = 5
        other_mlx_model.last_modified = "2024-01-01"
        other_mlx_model.tags = ["mlx"]

        mock_api = MagicMock()
        mock_api.list_models.side_effect = [
            [],  # Top models (empty)
            [mock_hf_model],  # MLX community (only 1, less than limit)
            [other_mlx_model],  # Other MLX search
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models_enhanced("model", top_limit=3, mlx_limit=10)

        # Should have both MLX models
        assert len(results.mlx_models) == 2
        model_ids = {m.id for m in results.mlx_models}
        assert "mlx-community/Qwen3-8B-4bit" in model_ids
        assert "user/other-mlx-model" in model_ids

    def test_enhanced_search_handles_mlx_section_api_failure(
        self, mock_original_model: MagicMock
    ) -> None:
        """search_models_enhanced should return top models even if MLX search fails."""
        mock_api = MagicMock()
        # First call (top models) succeeds, MLX section fails
        mock_api.list_models.side_effect = [
            [mock_original_model],  # Top models search
            Exception("MLX search failed"),  # MLX community search fails
        ]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            results = search_models_enhanced("qwen3", top_limit=3, mlx_limit=10)

        assert len(results.top_models) == 1  # Should still have top models
        assert len(results.mlx_models) == 0  # MLX search failed


class TestLocalModelSizeEdgeCases:
    """Verify get_local_model_size handles filesystem edge cases."""

    def test_get_local_model_size_handles_permission_error(
        self, temp_dir: Path
    ) -> None:
        """get_local_model_size should return None when cache cannot be read."""
        # Create cache structure
        cache_dir = temp_dir / ".cache" / "huggingface" / "hub"
        model_blobs = cache_dir / "models--mlx-community--test-model" / "blobs"
        model_blobs.mkdir(parents=True)

        # Mock iterdir to raise an exception
        with patch.object(Path, "home", return_value=temp_dir), patch.object(
            Path, "iterdir", side_effect=PermissionError("Access denied")
        ):
            result = get_local_model_size("mlx-community/test-model")

        assert result is None


class TestConvertedModelsEdgeCases:
    """Verify converted model functions handle edge cases."""

    def test_get_converted_models_skips_files_in_models_dir(
        self, temp_dir: Path
    ) -> None:
        """get_converted_models should skip non-directory entries."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()

        # Create a valid model directory
        model_dir = models_dir / "valid-model"
        model_dir.mkdir()
        (model_dir / "weights.safetensors").write_bytes(b"x" * 1000)

        # Create a file (not a directory) that should be skipped
        (models_dir / "readme.txt").write_text("This is a readme")

        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", models_dir):
            results = get_converted_models()

        # Should only include the valid directory
        assert len(results) == 1
        assert results[0].id == "local/valid-model"

    def test_get_converted_models_handles_size_calculation_error(
        self, temp_dir: Path
    ) -> None:
        """get_converted_models should return 0 size when calculation fails."""
        models_dir = temp_dir / "models"
        model_dir = models_dir / "broken-model"
        model_dir.mkdir(parents=True)

        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", models_dir), patch(
            "pathlib.Path.rglob", side_effect=PermissionError("Access denied")
        ):
            results = get_converted_models()

        # Should still return the model but with None size
        assert len(results) == 1
        assert results[0].id == "local/broken-model"
        assert results[0].size_bytes is None

    def test_get_converted_model_info_handles_size_calculation_error(
        self, temp_dir: Path
    ) -> None:
        """get_converted_model_info should return 0 size when calculation fails."""
        models_dir = temp_dir / "models"
        model_dir = models_dir / "broken-model"
        model_dir.mkdir(parents=True)

        with patch("local_ai.models.huggingface.CONVERTED_MODELS_DIR", models_dir), patch(
            "pathlib.Path.rglob", side_effect=PermissionError("Access denied")
        ):
            result = get_converted_model_info("broken-model")

        assert result is not None
        assert result.id == "local/broken-model"
        assert result.size_bytes is None
