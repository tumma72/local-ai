"""Hardware detection module for local-ai.

Detects Apple Silicon capabilities and provides recommendations for model sizing.
"""

from local_ai.hardware.apple_silicon import (
    AppleSiliconInfo,
    detect_hardware,
    estimate_model_params_from_name,
    get_max_model_size_gb,
    get_recommended_quantization,
)

__all__ = [
    "AppleSiliconInfo",
    "detect_hardware",
    "estimate_model_params_from_name",
    "get_max_model_size_gb",
    "get_recommended_quantization",
]
