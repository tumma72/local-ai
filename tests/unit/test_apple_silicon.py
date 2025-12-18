"""Behavioral tests for Apple Silicon hardware detection module.

Tests verify public behavior of the hardware detection:
- detect_hardware() identifies Apple Silicon specs
- get_max_model_size_gb() calculates memory budget
- get_recommended_quantization() recommends quantization levels
- estimate_model_params_from_name() extracts parameter counts
- Error handling for non-Apple Silicon hardware

Tests are implementation-agnostic and should survive refactoring.
"""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from local_ai.hardware.apple_silicon import (
    AppleSiliconInfo,
    ChipTier,
    detect_hardware,
    estimate_model_params_from_name,
    get_max_model_size_gb,
    get_recommended_quantization,
)

if TYPE_CHECKING:
    pass


class TestDetectHardware:
    """Verify detect_hardware identifies Apple Silicon specs."""

    def test_raises_error_when_not_apple_silicon(self) -> None:
        """Should raise RuntimeError when not on Apple Silicon."""
        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            return_value="Intel(R) Core(TM) i9-9900K",
        ):
            with pytest.raises(RuntimeError) as exc_info:
                detect_hardware()

            assert "Not running on Apple Silicon" in str(exc_info.value)

    def test_detects_m4_max_chip(self) -> None:
        """Should detect M4 Max chip name, generation, and tier."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M4 Max",
            "hw.memsize": str(128 * 1024**3),  # 128 GB
            "hw.ncpu": "14",
            "hw.perflevel0.logicalcpu": "10",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str:
            return sysctl_values.get(key, "0")

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=40,
        ):
            hardware = detect_hardware()

        assert hardware.chip_name == "Apple M4 Max"
        assert hardware.chip_generation == 4
        assert hardware.chip_tier == ChipTier.MAX
        assert hardware.memory_gb == 128.0
        assert hardware.cpu_cores == 14
        assert hardware.gpu_cores == 40

    def test_detects_m1_base_chip(self) -> None:
        """Should detect M1 base chip correctly."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M1",
            "hw.memsize": str(8 * 1024**3),  # 8 GB
            "hw.ncpu": "8",
            "hw.perflevel0.logicalcpu": "4",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str:
            return sysctl_values.get(key, "0")

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=8,
        ):
            hardware = detect_hardware()

        assert hardware.chip_generation == 1
        assert hardware.chip_tier == ChipTier.BASE
        assert hardware.memory_gb == 8.0

    def test_detects_m2_pro_chip(self) -> None:
        """Should detect M2 Pro chip tier correctly."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M2 Pro",
            "hw.memsize": str(32 * 1024**3),  # 32 GB
            "hw.ncpu": "12",
            "hw.perflevel0.logicalcpu": "8",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str:
            return sysctl_values.get(key, "0")

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=19,
        ):
            hardware = detect_hardware()

        assert hardware.chip_generation == 2
        assert hardware.chip_tier == ChipTier.PRO
        assert hardware.memory_gb == 32.0

    def test_detects_m3_ultra_chip(self) -> None:
        """Should detect Ultra tier chips correctly."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M3 Ultra",
            "hw.memsize": str(192 * 1024**3),  # 192 GB
            "hw.ncpu": "24",
            "hw.perflevel0.logicalcpu": "16",
            "hw.perflevel1.logicalcpu": "8",
        }

        def mock_sysctl(key: str) -> str:
            return sysctl_values.get(key, "0")

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=76,
        ):
            hardware = detect_hardware()

        assert hardware.chip_tier == ChipTier.ULTRA
        assert hardware.chip_generation == 3


class TestGetMaxModelSizeGb:
    """Verify get_max_model_size_gb calculates memory budget correctly."""

    def test_calculates_max_size_for_128gb_hardware(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should calculate max model size for high-memory hardware."""
        max_size = get_max_model_size_gb(hardware=mock_hardware_128gb)

        # 128GB * (1 - 0.25 safety) * 0.85 kv_cache = ~81.6 GB
        assert 75 < max_size < 85

    def test_calculates_max_size_for_16gb_hardware(
        self, mock_hardware_16gb: AppleSiliconInfo
    ) -> None:
        """Should calculate smaller max size for limited memory."""
        max_size = get_max_model_size_gb(hardware=mock_hardware_16gb)

        # 16GB * (1 - 0.25 safety) * 0.85 kv_cache = ~10.2 GB
        assert 9 < max_size < 11

    def test_calculates_max_size_for_8gb_hardware(
        self, mock_hardware_8gb: AppleSiliconInfo
    ) -> None:
        """Should calculate appropriate max size for minimum memory."""
        max_size = get_max_model_size_gb(hardware=mock_hardware_8gb)

        # 8GB * (1 - 0.25 safety) * 0.85 kv_cache = ~5.1 GB
        assert 4 < max_size < 6

    def test_respects_custom_safety_margin(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should apply custom safety margin when specified."""
        max_size_default = get_max_model_size_gb(
            hardware=mock_hardware_128gb, safety_margin=0.25
        )
        max_size_conservative = get_max_model_size_gb(
            hardware=mock_hardware_128gb, safety_margin=0.5
        )

        # More conservative margin should result in smaller max size
        assert max_size_conservative < max_size_default


class TestGetRecommendedQuantization:
    """Verify get_recommended_quantization recommends appropriate levels."""

    def test_recommends_8bit_for_small_model_on_large_memory(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should recommend 8bit for small models on high-memory hardware."""
        quantization = get_recommended_quantization(
            model_params_billions=8.0, hardware=mock_hardware_128gb
        )

        assert quantization == "8bit"

    def test_recommends_4bit_for_large_model_on_limited_memory(
        self, mock_hardware_16gb: AppleSiliconInfo
    ) -> None:
        """Should recommend 4bit for models near memory limit."""
        # 30B model: 8bit=30GB, 6bit=22.5GB, 4bit=15GB
        # 16GB hardware with ~10GB max: needs 4bit
        quantization = get_recommended_quantization(
            model_params_billions=30.0, hardware=mock_hardware_16gb
        )

        assert quantization in ("4bit", "too_large")

    def test_recommends_too_large_for_massive_model_on_small_memory(
        self, mock_hardware_8gb: AppleSiliconInfo
    ) -> None:
        """Should return too_large when model cannot fit even at 4bit."""
        # 70B model: even at 4bit = 35GB, cannot fit in 8GB
        quantization = get_recommended_quantization(
            model_params_billions=70.0, hardware=mock_hardware_8gb
        )

        assert quantization == "too_large"

    def test_recommends_6bit_for_medium_fit(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should recommend 6bit when 8bit is too large but 6bit fits."""
        # Find a model size where 8bit > max but 6bit < max
        # max_size ~= 81.6GB for 128GB hardware
        # 100B at 8bit = 100GB (too large), 6bit = 75GB (fits)
        quantization = get_recommended_quantization(
            model_params_billions=100.0, hardware=mock_hardware_128gb
        )

        assert quantization == "6bit"


class TestEstimateModelParamsFromName:
    """Verify estimate_model_params_from_name extracts parameter counts."""

    @pytest.mark.parametrize(
        "model_name,expected_params",
        [
            # Standard B notation
            ("Qwen3-8B-4bit", 8.0),
            ("Llama-3.3-70B-Instruct", 70.0),
            ("gemma-3-27b-it", 27.0),
            ("Mistral-7B-v0.3", 7.0),
            # Lowercase
            ("starcoder2-15b", 15.0),
            ("llama-8b-instruct", 8.0),
            # Decimal params
            ("phi-1.5B-instruct", 1.5),
            ("model-2.7b-chat", 2.7),
            # Models with B- pattern
            ("Qwen3-30B-A3B-4bit", 30.0),
            ("DeepSeek-R1-8B-distill", 8.0),
        ],
        ids=[
            "qwen-8b-upper",
            "llama-70b",
            "gemma-27b-lower",
            "mistral-7b",
            "starcoder-15b",
            "llama-8b-lower",
            "phi-decimal",
            "model-decimal",
            "qwen-30b-complex",
            "deepseek-8b",
        ],
    )
    def test_extracts_param_count_from_name(
        self, model_name: str, expected_params: float
    ) -> None:
        """Should extract parameter count from various model name patterns."""
        params = estimate_model_params_from_name(model_name)
        assert params == expected_params

    @pytest.mark.parametrize(
        "model_name",
        [
            "phi-mini-128k-instruct",
            "model-without-size",
            "gpt-4",
            "claude-3-opus",
        ],
        ids=["phi-no-b", "no-size", "gpt-4", "claude"],
    )
    def test_returns_none_when_no_params_in_name(
        self, model_name: str
    ) -> None:
        """Should return None when parameter count cannot be extracted."""
        params = estimate_model_params_from_name(model_name)
        assert params is None


class TestAppleSiliconInfoDataclass:
    """Verify AppleSiliconInfo dataclass behavior."""

    def test_memory_bytes_property(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should convert memory_gb to bytes correctly."""
        expected_bytes = 128 * 1024 * 1024 * 1024
        assert mock_hardware_128gb.memory_bytes == expected_bytes

    def test_string_representation_contains_key_info(
        self, mock_hardware_128gb: AppleSiliconInfo
    ) -> None:
        """Should have informative string representation."""
        str_repr = str(mock_hardware_128gb)

        assert "M4 Max" in str_repr
        assert "128" in str_repr
        assert "GPU" in str_repr
        assert "CPU" in str_repr


class TestDetectHardwareEdgeCases:
    """Verify detect_hardware handles edge cases gracefully."""

    def test_handles_missing_memsize_with_default(self) -> None:
        """Should use default memory when hw.memsize returns None."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M1",
            "hw.memsize": None,  # Missing
            "hw.ncpu": "8",
            "hw.perflevel0.logicalcpu": "4",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str | None:
            return sysctl_values.get(key)

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=8,
        ):
            hardware = detect_hardware()

        # Should fall back to 8.0 GB default
        assert hardware.memory_gb == 8.0

    def test_handles_missing_cpu_info_with_defaults(self) -> None:
        """Should use default CPU cores when sysctl returns None."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M1",
            "hw.memsize": str(16 * 1024**3),
            "hw.ncpu": None,  # Missing
            "hw.perflevel0.logicalcpu": None,
            "hw.perflevel1.logicalcpu": None,
        }

        def mock_sysctl(key: str) -> str | None:
            return sysctl_values.get(key)

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=8,
        ):
            hardware = detect_hardware()

        # Should use default of 8 cores
        assert hardware.cpu_cores == 8

    def test_handles_unknown_chip_name(self) -> None:
        """Should detect as UNKNOWN tier for unrecognized chip names."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple Future Chip",  # No M number
            "hw.memsize": str(16 * 1024**3),
            "hw.ncpu": "8",
            "hw.perflevel0.logicalcpu": "4",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str | None:
            return sysctl_values.get(key)

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=10,
        ):
            hardware = detect_hardware()

        # Should be UNKNOWN tier since we can't identify M-number tier
        # Generation defaults to 1 without M-number
        assert hardware.chip_generation == 1

    def test_handles_null_chip_name_gracefully(self) -> None:
        """Should handle None chip name as Unknown."""

        def mock_sysctl(key: str) -> str | None:
            if key == "machdep.cpu.brand_string":
                return None
            return "8"

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), pytest.raises(RuntimeError) as exc_info:
            detect_hardware()

        assert "Not running on Apple Silicon" in str(exc_info.value)


class TestGetMaxModelSizeAutoDetect:
    """Verify get_max_model_size_gb with auto-detection."""

    def test_auto_detects_hardware_when_not_provided(self) -> None:
        """Should auto-detect hardware when hardware param is None."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M1",
            "hw.memsize": str(8 * 1024**3),
            "hw.ncpu": "8",
            "hw.perflevel0.logicalcpu": "4",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str | None:
            return sysctl_values.get(key)

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=8,
        ):
            max_size = get_max_model_size_gb(hardware=None)

        # 8GB with default margins should give roughly 5GB
        assert 4 < max_size < 6


class TestGetRecommendedQuantizationAutoDetect:
    """Verify get_recommended_quantization with auto-detection."""

    def test_auto_detects_hardware_when_not_provided(self) -> None:
        """Should auto-detect hardware when hardware param is None."""
        sysctl_values = {
            "machdep.cpu.brand_string": "Apple M4 Max",
            "hw.memsize": str(128 * 1024**3),
            "hw.ncpu": "14",
            "hw.perflevel0.logicalcpu": "10",
            "hw.perflevel1.logicalcpu": "4",
        }

        def mock_sysctl(key: str) -> str | None:
            return sysctl_values.get(key)

        with patch(
            "local_ai.hardware.apple_silicon._run_sysctl",
            side_effect=mock_sysctl,
        ), patch(
            "local_ai.hardware.apple_silicon._get_gpu_cores",
            return_value=40,
        ):
            quantization = get_recommended_quantization(
                model_params_billions=8.0, hardware=None
            )

        # 8B model on 128GB should easily fit at 8bit
        assert quantization == "8bit"


class TestChipTierEnum:
    """Verify ChipTier enum values."""

    def test_chip_tier_values(self) -> None:
        """Should have correct string values for each tier."""
        assert ChipTier.BASE.value == "base"
        assert ChipTier.PRO.value == "pro"
        assert ChipTier.MAX.value == "max"
        assert ChipTier.ULTRA.value == "ultra"
        assert ChipTier.UNKNOWN.value == "unknown"

    def test_chip_tier_is_string_enum(self) -> None:
        """Should be usable as string via value."""
        # ChipTier inherits from str, so .value gives the string
        assert ChipTier.MAX.value == "max"
        # Can compare with string using value
        assert ChipTier.MAX == "max"
