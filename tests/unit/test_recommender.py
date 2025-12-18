"""Behavioral tests for model recommendation engine.

Tests verify public behavior of the recommender module:
- Model type detection from name/tags
- Temperature recommendations based on model type
- max_tokens calculation (25% of context, 2048 floor, 32768 ceiling)
- Memory fit analysis
- Recommendation output structure

Tests are implementation-agnostic and should survive refactoring.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from local_ai.hardware.apple_silicon import AppleSiliconInfo


class TestModelTypeDetection:
    """Verify model type detection from name patterns."""

    @pytest.mark.parametrize(
        "model_id,expected_type",
        [
            # Reasoning models
            ("mlx-community/DeepSeek-R1-Qwen3-8B-4bit", "reasoning"),
            ("deepseek-ai/DeepSeek-R1-0528", "reasoning"),
            ("mlx-community/QwQ-32B-4bit", "reasoning"),  # QwQ is a reasoning model
            # Chat/Instruct models
            ("mlx-community/Qwen3-8B-Instruct-4bit", "chat"),
            ("meta-llama/Llama-3.3-70B-Instruct", "chat"),
            ("mlx-community/gemma-3-27b-it-4bit", "chat"),  # "it" = instruct-tuned
            # Code models
            ("mlx-community/Devstral-Small-4bit", "code"),  # Mistral's code model
            ("bigcode/starcoder2-15b", "code"),
            ("mlx-community/Qwen3-Coder-30B-4bit", "code"),
            ("deepseek-ai/DeepSeek-Coder-V2", "code"),
            # General models (no specific type indicators)
            ("mlx-community/Qwen3-30B-A3B-4bit", "general"),
            ("mlx-community/Mistral-7B-v0.3-4bit", "general"),
        ],
        ids=[
            "deepseek-r1-reasoning",
            "deepseek-r1-original",
            "qwq-reasoning",
            "qwen-instruct-chat",
            "llama-instruct-chat",
            "gemma-it-chat",
            "devstral-code",
            "starcoder-code",
            "qwen-coder",
            "deepseek-coder",
            "qwen-general",
            "mistral-general",
        ],
    )
    def test_detects_model_type_from_name(
        self, model_id: str, expected_type: str
    ) -> None:
        """detect_model_type should identify model category from name patterns."""
        from local_ai.models.recommender import detect_model_type

        detected = detect_model_type(model_id)
        assert detected == expected_type


class TestTemperatureRecommendation:
    """Verify temperature recommendations based on model type."""

    @pytest.mark.parametrize(
        "model_type,expected_temp",
        [
            ("reasoning", 0.0),  # Reasoning models need deterministic output
            ("code", 0.2),  # Code needs low temperature for accuracy
            ("chat", 0.7),  # Chat models can be more creative
            ("creative", 1.0),  # Creative writing needs high temperature
            ("general", 0.7),  # Default for unspecified types
        ],
    )
    def test_recommends_temperature_by_type(
        self, model_type: str, expected_temp: float
    ) -> None:
        """get_recommended_temperature should return appropriate value for each type."""
        from local_ai.models.recommender import get_recommended_temperature

        temp = get_recommended_temperature(model_type)
        assert temp == expected_temp


class TestMaxTokensRecommendation:
    """Verify max_tokens calculation from context length."""

    @pytest.mark.parametrize(
        "context_length,expected_max_tokens",
        [
            # Standard scaling: 25% of context
            (131072, 32768),  # 128K context -> 32K max (capped at ceiling)
            (32768, 8192),  # 32K context -> 8K max
            (16384, 4096),  # 16K context -> 4K max
            (8192, 2048),  # 8K context -> 2K max (floor)
            # Floor enforcement
            (4096, 2048),  # 4K context -> 2K (floor)
            (2048, 2048),  # 2K context -> 2K (floor)
            (1024, 2048),  # 1K context -> 2K (floor)
            # Ceiling enforcement
            (262144, 32768),  # 256K context -> 32K (ceiling)
            (524288, 32768),  # 512K context -> 32K (ceiling)
        ],
        ids=[
            "128k-context",
            "32k-context",
            "16k-context",
            "8k-context",
            "4k-floor",
            "2k-floor",
            "1k-floor",
            "256k-ceiling",
            "512k-ceiling",
        ],
    )
    def test_calculates_max_tokens_from_context(
        self, context_length: int, expected_max_tokens: int
    ) -> None:
        """get_recommended_max_tokens should scale to 25% with floor/ceiling."""
        from local_ai.models.recommender import get_recommended_max_tokens

        max_tokens = get_recommended_max_tokens(context_length)
        assert max_tokens == expected_max_tokens

    def test_returns_default_when_context_unknown(self) -> None:
        """get_recommended_max_tokens should return sensible default for None."""
        from local_ai.models.recommender import get_recommended_max_tokens

        max_tokens = get_recommended_max_tokens(None)
        assert max_tokens == 4096  # Safe default


class TestMemoryFitAnalysis:
    """Verify memory fit calculation for models on hardware."""

    def test_small_model_fits_in_large_memory(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """check_memory_fit should return True for small models on high-memory hardware."""
        from local_ai.models.recommender import check_memory_fit

        # 5GB model on 128GB machine
        fits, headroom = check_memory_fit(
            model_size_gb=5.0, hardware=mock_hardware_128gb
        )

        assert fits is True
        assert headroom > 50  # Plenty of headroom

    def test_large_model_does_not_fit(
        self, mock_hardware_8gb: "AppleSiliconInfo"
    ) -> None:
        """check_memory_fit should return False for models too large for hardware."""
        from local_ai.models.recommender import check_memory_fit

        # 20GB model on 8GB machine
        fits, headroom = check_memory_fit(model_size_gb=20.0, hardware=mock_hardware_8gb)

        assert fits is False
        assert headroom < 0  # Negative headroom

    def test_borderline_model_considers_overhead(
        self, mock_hardware_16gb: "AppleSiliconInfo"
    ) -> None:
        """check_memory_fit should account for system/inference overhead."""
        from local_ai.models.recommender import check_memory_fit

        # Model that would fit if we only looked at raw memory
        # but doesn't fit with overhead considered
        # 16GB machine with 25% overhead = ~12GB usable
        # 10GB model should fit, but with reduced headroom
        fits, headroom = check_memory_fit(
            model_size_gb=10.0, hardware=mock_hardware_16gb
        )

        assert fits is True
        assert 0 < headroom < 5  # Should fit but with limited headroom


class TestModelRecommendation:
    """Verify complete recommendation generation."""

    def test_generates_recommendation_for_reasoning_model(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """recommend_settings should return complete recommendation for reasoning model."""
        from local_ai.models.recommender import ModelRecommendation, recommend_settings

        # Create minimal model info
        recommendation = recommend_settings(
            model_id="mlx-community/DeepSeek-R1-Qwen3-8B-4bit",
            model_size_gb=5.0,
            context_length=131072,
            hardware=mock_hardware_128gb,
        )

        assert isinstance(recommendation, ModelRecommendation)
        assert recommendation.model_id == "mlx-community/DeepSeek-R1-Qwen3-8B-4bit"
        assert recommendation.model_type == "reasoning"
        assert recommendation.temperature == 0.0
        assert recommendation.max_tokens == 32768
        assert recommendation.top_p == 1.0
        assert recommendation.fits_in_memory is True

    def test_generates_recommendation_for_code_model(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """recommend_settings should return appropriate settings for code model."""
        from local_ai.models.recommender import recommend_settings

        recommendation = recommend_settings(
            model_id="mlx-community/Devstral-Small-4bit",
            model_size_gb=4.0,
            context_length=32768,
            hardware=mock_hardware_128gb,
        )

        assert recommendation.model_type == "code"
        assert recommendation.temperature == 0.2
        assert recommendation.max_tokens == 8192

    def test_recommendation_includes_reasoning(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """recommend_settings should include explanations for recommendations."""
        from local_ai.models.recommender import recommend_settings

        recommendation = recommend_settings(
            model_id="mlx-community/DeepSeek-R1-Qwen3-8B-4bit",
            model_size_gb=5.0,
            context_length=131072,
            hardware=mock_hardware_128gb,
        )

        assert recommendation.temperature_reason is not None
        assert len(recommendation.temperature_reason) > 0
        assert recommendation.max_tokens_reason is not None
        assert len(recommendation.max_tokens_reason) > 0


class TestEstimateModelParams:
    """Verify parameter estimation from model name."""

    @pytest.mark.parametrize(
        "model_name,expected_params",
        [
            ("Qwen3-8B-4bit", 8.0),
            ("DeepSeek-R1-Qwen3-8B", 8.0),
            ("Llama-3.3-70B-Instruct", 70.0),
            ("gemma-3-27b-it", 27.0),
            ("Mistral-7B-v0.3", 7.0),
            ("starcoder2-15b", 15.0),
            ("Qwen3-30B-A3B", 30.0),
            ("phi-3.5-mini-128k-instruct", None),  # No size in name
        ],
        ids=[
            "qwen-8b",
            "deepseek-8b",
            "llama-70b",
            "gemma-27b",
            "mistral-7b",
            "starcoder-15b",
            "qwen-30b",
            "phi-no-size",
        ],
    )
    def test_extracts_param_count_from_name(
        self, model_name: str, expected_params: float | None
    ) -> None:
        """estimate_params_from_name should extract parameter count from model name."""
        from local_ai.hardware.apple_silicon import estimate_model_params_from_name

        params = estimate_model_params_from_name(model_name)
        assert params == expected_params


class TestCreativeModelDetection:
    """Verify creative model type detection."""

    @pytest.mark.parametrize(
        "model_id,expected_type",
        [
            ("mlx-community/creative-writing-7b", "creative"),
            ("user/story-generator-3b", "creative"),
            ("mlx-community/writing-assistant-8b", "creative"),
        ],
        ids=[
            "creative-keyword",
            "story-keyword",
            "writing-keyword",
        ],
    )
    def test_detects_creative_models(
        self, model_id: str, expected_type: str
    ) -> None:
        """detect_model_type should identify creative models from name patterns."""
        from local_ai.models.recommender import detect_model_type

        detected = detect_model_type(model_id)
        assert detected == expected_type


class TestRecommendSettingsHardwareFailure:
    """Verify recommend_settings handles hardware detection failures."""

    def test_recommend_settings_when_hardware_detection_fails(self) -> None:
        """recommend_settings should use defaults when not on Apple Silicon."""
        from unittest.mock import patch

        from local_ai.models.recommender import recommend_settings

        # Mock detect_hardware to raise RuntimeError (not on Apple Silicon)
        # The import happens inside recommend_settings, so we patch the source module
        with patch(
            "local_ai.hardware.apple_silicon.detect_hardware",
            side_effect=RuntimeError("Not running on Apple Silicon"),
        ):
            recommendation = recommend_settings(
                model_id="mlx-community/Qwen3-8B-4bit",
                model_size_gb=5.0,
                context_length=32768,
                hardware=None,  # Trigger auto-detection
            )

        # Should still return a valid recommendation with defaults
        assert recommendation is not None
        assert recommendation.model_id == "mlx-community/Qwen3-8B-4bit"
        # When hardware is None (detection failed), fits_in_memory defaults to True
        assert recommendation.fits_in_memory is True
        # Temperature and max_tokens should still be computed
        assert recommendation.temperature == 0.7  # general model
        assert recommendation.max_tokens == 8192  # 25% of 32768


class TestMaxTokensReasonEdgeCases:
    """Verify max_tokens reason generation handles edge cases."""

    def test_max_tokens_reason_for_floor_value(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """recommend_settings should explain when max_tokens hits floor."""
        from local_ai.models.recommender import recommend_settings

        recommendation = recommend_settings(
            model_id="mlx-community/small-model",
            model_size_gb=1.0,
            context_length=4096,  # 25% = 1024, which is below floor of 2048
            hardware=mock_hardware_128gb,
        )

        assert recommendation.max_tokens == 2048
        assert "Minimum" in recommendation.max_tokens_reason or "minimum" in recommendation.max_tokens_reason.lower()

    def test_max_tokens_reason_when_context_unknown(
        self, mock_hardware_128gb: "AppleSiliconInfo"
    ) -> None:
        """recommend_settings should explain when context length is unknown."""
        from local_ai.models.recommender import recommend_settings

        recommendation = recommend_settings(
            model_id="mlx-community/unknown-model",
            model_size_gb=5.0,
            context_length=None,  # Unknown context length
            hardware=mock_hardware_128gb,
        )

        assert recommendation.max_tokens == 4096  # Default
        assert "unknown" in recommendation.max_tokens_reason.lower()


# Hardware fixtures are defined in conftest.py:
# - mock_hardware_128gb: M4 Max with 128GB
# - mock_hardware_16gb: M2 with 16GB
# - mock_hardware_8gb: M1 with 8GB
