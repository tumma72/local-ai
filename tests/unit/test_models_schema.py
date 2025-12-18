"""Tests for models schema module."""


from local_ai.models.schema import (
    ModelSearchResult,
    Quantization,
    extract_quantization,
)


class TestModelsSchema:
    """Test models schema functionality."""

    def test_extract_quantization_4bit(self) -> None:
        """Test extracting 4-bit quantization from model names."""
        assert extract_quantization("mlx-community/Qwen3-8B-4bit") == Quantization.Q4
        assert extract_quantization("model-4b-quant") == Quantization.Q4
        assert extract_quantization("model-q4-km") == Quantization.Q4

    def test_extract_quantization_8bit(self) -> None:
        """Test extracting 8-bit quantization from model names."""
        assert extract_quantization("mlx-community/Qwen3-8B-8bit") == Quantization.Q8
        assert extract_quantization("model-8b-quant") == Quantization.Q8
        assert extract_quantization("model-q8-0") == Quantization.Q8

    def test_extract_quantization_bf16(self) -> None:
        """Test extracting BF16 quantization from model names."""
        assert extract_quantization("model-bf16") == Quantization.BF16

    def test_extract_quantization_fp16(self) -> None:
        """Test extracting FP16 quantization from model names."""
        assert extract_quantization("model-fp16") == Quantization.FP16
        assert extract_quantization("model-f16") == Quantization.FP16

    def test_extract_quantization_fp32(self) -> None:
        """Test extracting FP32 quantization from model names."""
        assert extract_quantization("model-fp32") == Quantization.FP32
        assert extract_quantization("model-f32") == Quantization.FP32

    def test_extract_quantization_dwq(self) -> None:
        """Test extracting DWQ quantization from model names.

        Note: DWQ patterns like "4bit-dwq" are caught by the explicit
        "4bit" check first, so this tests the behavior users would see.
        """
        # These patterns contain "Xbit" which matches earlier patterns
        assert extract_quantization("model-4bit-dwq") == Quantization.Q4
        assert extract_quantization("model-5bit-dwq") == Quantization.Q5
        assert extract_quantization("model-6bit-dwq") == Quantization.Q6
        assert extract_quantization("model-8bit-dwq") == Quantization.Q8

    def test_extract_quantization_5bit_patterns(self) -> None:
        """Test extracting 5-bit quantization from model names."""
        assert extract_quantization("model-5bit") == Quantization.Q5
        assert extract_quantization("model-5b-quant") == Quantization.Q5
        assert extract_quantization("model-q5-km") == Quantization.Q5

    def test_extract_quantization_6bit_patterns(self) -> None:
        """Test extracting 6-bit quantization from model names."""
        assert extract_quantization("model-6bit") == Quantization.Q6
        assert extract_quantization("model-6b-quant") == Quantization.Q6
        assert extract_quantization("model-q6-km") == Quantization.Q6

    def test_extract_quantization_unusual_dwq_returns_unknown(self) -> None:
        """Test that unusual bit-width DWQ patterns return unknown.

        The DWQ regex handler only explicitly handles 4, 5, 6, 8 bit patterns.
        Other bit widths should return UNKNOWN.
        """
        # This pattern has "3bit-dwq" which won't match explicit patterns
        # (no "3bit" check) and will reach DWQ handler but bits != 4,5,6,8
        assert extract_quantization("model-3bit-dwq") == Quantization.UNKNOWN
        # Similarly for 7-bit
        assert extract_quantization("model-7bit-dwq") == Quantization.UNKNOWN

    def test_extract_quantization_unknown(self) -> None:
        """Test handling unknown quantization."""
        assert extract_quantization("model-unknown") == Quantization.UNKNOWN
        assert extract_quantization("model") == Quantization.UNKNOWN

    def test_model_search_result_name_extraction(self) -> None:
        """Test extracting model name from full ID."""
        result = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm", "quantized"],
            size_bytes=4_000_000_000
        )

        assert result.name == "Qwen3-8B-4bit"

    def test_model_search_result_quantization_detection(self) -> None:
        """Test detecting quantization from model ID."""
        result = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm", "quantized"],
            size_bytes=4_000_000_000
        )

        assert result.quantization == Quantization.Q4

    def test_model_search_result_source_label(self) -> None:
        """Test getting source label for display."""
        mlx_result = ModelSearchResult(
            id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm", "quantized"],
            size_bytes=4_000_000_000
        )

        other_result = ModelSearchResult(
            id="other-org/model-name",
            author="other-org",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=False,
            tags=["llm"],
            size_bytes=4_000_000_000
        )

        assert mlx_result.source_label == "★ MLX"
        assert other_result.source_label == "mlx"

    def test_model_search_result_size_formatting(self) -> None:
        """Test formatting model size for display."""
        # Test large size (>10GB)
        large_result = ModelSearchResult(
            id="mlx-community/large-model",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm"],
            size_bytes=15_000_000_000  # ~15 GB
        )

        # Test small size (<10GB)
        small_result = ModelSearchResult(
            id="mlx-community/small-model",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm"],
            size_bytes=3_500_000_000  # ~3.5 GB
        )

        # Test None size
        none_result = ModelSearchResult(
            id="mlx-community/unknown-model",
            author="mlx-community",
            downloads=1000,
            likes=50,
            last_modified="2023-01-01",
            is_mlx_community=True,
            tags=["llm"],
            size_bytes=None
        )

        assert large_result.size_gb == "14 GB"  # 15_000_000_000 / (1024^3) ≈ 13.97 GB → 14 GB
        assert small_result.size_gb == "3.3 GB"  # 3_500_000_000 / (1024^3) ≈ 3.26 GB → 3.3 GB
        assert none_result.size_gb == "-"
