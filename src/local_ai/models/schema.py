"""Data models for model discovery.

Defines Pydantic models for HuggingFace model search results.
"""

import re
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class Quantization(str, Enum):
    """Quantization level for models."""

    Q4 = "4bit"
    Q5 = "5bit"
    Q6 = "6bit"
    Q8 = "8bit"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    UNKNOWN = "-"


def extract_quantization(model_id: str) -> Quantization:
    """Extract quantization level from model ID or name.

    Args:
        model_id: Model identifier (e.g., "mlx-community/Qwen3-8B-4bit").

    Returns:
        Detected quantization level.
    """
    name_lower = model_id.lower()

    # Check for explicit bit patterns
    if "4bit" in name_lower or "-4b-" in name_lower or "-q4" in name_lower:
        return Quantization.Q4
    if "5bit" in name_lower or "-5b-" in name_lower or "-q5" in name_lower:
        return Quantization.Q5
    if "6bit" in name_lower or "-6b-" in name_lower or "-q6" in name_lower:
        return Quantization.Q6
    if "8bit" in name_lower or "-8b-" in name_lower or "-q8" in name_lower:
        return Quantization.Q8
    if "bf16" in name_lower:
        return Quantization.BF16
    if "fp16" in name_lower or "f16" in name_lower:
        return Quantization.FP16
    if "fp32" in name_lower or "f32" in name_lower:
        return Quantization.FP32

    # Check for DWQ (Dynamic Weight Quantization) patterns
    if re.search(r"\d+bit-dwq", name_lower):
        match = re.search(r"(\d+)bit-dwq", name_lower)
        if match:
            bits = match.group(1)
            if bits == "4":
                return Quantization.Q4
            if bits == "5":
                return Quantization.Q5
            if bits == "6":
                return Quantization.Q6
            if bits == "8":
                return Quantization.Q8

    return Quantization.UNKNOWN


class ModelSearchResult(BaseModel):
    """Result from a HuggingFace model search."""

    id: str = Field(description="Full model ID (e.g., mlx-community/Qwen3-8B-4bit)")
    author: str = Field(description="Model author/organization")
    downloads: int = Field(default=0, ge=0, description="Number of downloads")
    likes: int = Field(default=0, ge=0, description="Number of likes")
    last_modified: str = Field(default="", description="Last modification date")
    is_mlx_community: bool = Field(
        default=False, description="True if from mlx-community org"
    )
    tags: list[str] = Field(default_factory=list, description="Model tags")
    size_bytes: int | None = Field(default=None, description="Total model size in bytes")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """Extract model name from full ID."""
        return self.id.split("/")[-1] if "/" in self.id else self.id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quantization(self) -> Quantization:
        """Detect quantization level from model name."""
        return extract_quantization(self.id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_label(self) -> str:
        """Get display label for source."""
        return "★ MLX" if self.is_mlx_community else "mlx"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_gb(self) -> str:
        """Format size in GB for display."""
        if self.size_bytes is None:
            return "-"
        gb = self.size_bytes / (1024 * 1024 * 1024)
        if gb >= 10:
            return f"{gb:.0f} GB"
        return f"{gb:.1f} GB"
