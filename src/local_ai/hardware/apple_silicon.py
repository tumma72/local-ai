"""Apple Silicon hardware detection and model sizing recommendations.

Detects chip type, memory, cores, and provides recommendations for:
- Maximum model size that fits in memory
- Preferred quantization level based on available resources
"""

import re
import subprocess
from dataclasses import dataclass
from enum import Enum

from local_ai.logging import get_logger

_logger = get_logger("Hardware.apple_silicon")


class ChipTier(str, Enum):
    """Apple Silicon chip tier classification."""

    BASE = "base"  # M1, M2, M3, M4
    PRO = "pro"  # M1 Pro, M2 Pro, etc.
    MAX = "max"  # M1 Max, M2 Max, etc.
    ULTRA = "ultra"  # M1 Ultra, M2 Ultra, etc.
    UNKNOWN = "unknown"


@dataclass
class AppleSiliconInfo:
    """Information about Apple Silicon hardware."""

    chip_name: str  # e.g., "Apple M4 Max"
    chip_generation: int  # 1, 2, 3, 4
    chip_tier: ChipTier
    memory_gb: float  # Unified memory in GB
    cpu_cores: int  # Total CPU cores
    cpu_performance_cores: int  # Performance cores
    cpu_efficiency_cores: int  # Efficiency cores
    gpu_cores: int  # GPU cores
    neural_engine_cores: int  # Neural Engine cores (16 for all Apple Silicon)

    @property
    def memory_bytes(self) -> int:
        """Memory in bytes."""
        return int(self.memory_gb * 1024 * 1024 * 1024)

    def __str__(self) -> str:
        return (
            f"{self.chip_name} | {self.memory_gb:.0f}GB | "
            f"{self.cpu_cores} CPU ({self.cpu_performance_cores}P+{self.cpu_efficiency_cores}E) | "
            f"{self.gpu_cores} GPU"
        )


def _run_sysctl(key: str) -> str | None:
    """Run sysctl command and return value."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        _logger.debug("sysctl {} failed: {}", key, e)
    return None


def _get_gpu_cores() -> int:
    """Get GPU core count from system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1))
    except Exception as e:
        _logger.debug("system_profiler failed: {}", e)
    return 0


def _parse_chip_info(chip_name: str) -> tuple[int, ChipTier]:
    """Parse chip generation and tier from name.

    Args:
        chip_name: e.g., "Apple M4 Max"

    Returns:
        Tuple of (generation, tier)
    """
    name_lower = chip_name.lower()

    # Extract generation (M1=1, M2=2, M3=3, M4=4)
    generation = 1
    gen_match = re.search(r"m(\d)", name_lower)
    if gen_match:
        generation = int(gen_match.group(1))

    # Extract tier
    if "ultra" in name_lower:
        tier = ChipTier.ULTRA
    elif "max" in name_lower:
        tier = ChipTier.MAX
    elif "pro" in name_lower:
        tier = ChipTier.PRO
    elif re.search(r"m\d\s*$", name_lower) or "apple m" in name_lower:
        tier = ChipTier.BASE
    else:
        tier = ChipTier.UNKNOWN

    return generation, tier


def detect_hardware() -> AppleSiliconInfo:
    """Detect Apple Silicon hardware capabilities.

    Returns:
        AppleSiliconInfo with detected hardware specs.

    Raises:
        RuntimeError: If not running on Apple Silicon.
    """
    chip_name = _run_sysctl("machdep.cpu.brand_string") or "Unknown"

    if "apple" not in chip_name.lower():
        raise RuntimeError(f"Not running on Apple Silicon: {chip_name}")

    generation, tier = _parse_chip_info(chip_name)

    # Memory
    mem_bytes = _run_sysctl("hw.memsize")
    memory_gb = int(mem_bytes) / (1024**3) if mem_bytes else 8.0

    # CPU cores
    cpu_cores = int(_run_sysctl("hw.ncpu") or "8")
    perf_cores = int(_run_sysctl("hw.perflevel0.logicalcpu") or str(cpu_cores // 2))
    eff_cores = int(_run_sysctl("hw.perflevel1.logicalcpu") or str(cpu_cores - perf_cores))

    # GPU cores
    gpu_cores = _get_gpu_cores()

    # Neural Engine (16 cores on all Apple Silicon)
    neural_cores = 16

    info = AppleSiliconInfo(
        chip_name=chip_name,
        chip_generation=generation,
        chip_tier=tier,
        memory_gb=memory_gb,
        cpu_cores=cpu_cores,
        cpu_performance_cores=perf_cores,
        cpu_efficiency_cores=eff_cores,
        gpu_cores=gpu_cores,
        neural_engine_cores=neural_cores,
    )

    _logger.info("Detected hardware: {}", info)
    return info


def get_max_model_size_gb(
    hardware: AppleSiliconInfo | None = None,
    safety_margin: float = 0.25,
) -> float:
    """Calculate maximum model size that fits in memory.

    Reserves memory for:
    - System and other apps (~10-15% baseline)
    - KV cache during inference (~10-20% of model size)
    - Safety margin (configurable, default 25%)

    Args:
        hardware: Hardware info, or None to auto-detect.
        safety_margin: Fraction of memory to reserve (default 0.25 = 25%).

    Returns:
        Maximum model size in GB.
    """
    if hardware is None:
        hardware = detect_hardware()

    # Reserve memory for system + safety margin
    available_gb = hardware.memory_gb * (1 - safety_margin)

    # Further reduce for KV cache overhead during inference
    # Larger models need proportionally more KV cache
    max_model_gb = available_gb * 0.85

    _logger.debug(
        "Memory: {:.0f}GB, Available: {:.0f}GB, Max model: {:.0f}GB",
        hardware.memory_gb,
        available_gb,
        max_model_gb,
    )

    return max_model_gb


def get_recommended_quantization(
    model_params_billions: float,
    hardware: AppleSiliconInfo | None = None,
) -> str:
    """Recommend quantization level for a model based on hardware.

    Args:
        model_params_billions: Model size in billions of parameters.
        hardware: Hardware info, or None to auto-detect.

    Returns:
        Recommended quantization: "8bit", "6bit", "4bit", or "too_large".
    """
    if hardware is None:
        hardware = detect_hardware()

    max_size_gb = get_max_model_size_gb(hardware)

    # Estimate model size at different quantizations
    # Rule of thumb: ~1GB per billion params at 8bit, ~0.5GB at 4bit
    size_8bit = model_params_billions * 1.0
    size_6bit = model_params_billions * 0.75
    size_4bit = model_params_billions * 0.5

    _logger.debug(
        "Model {}B: 8bit={:.1f}GB, 6bit={:.1f}GB, 4bit={:.1f}GB, max={:.1f}GB",
        model_params_billions,
        size_8bit,
        size_6bit,
        size_4bit,
        max_size_gb,
    )

    if size_8bit <= max_size_gb:
        return "8bit"
    if size_6bit <= max_size_gb:
        return "6bit"
    if size_4bit <= max_size_gb:
        return "4bit"

    return "too_large"


def estimate_model_params_from_name(model_name: str) -> float | None:
    """Try to extract parameter count from model name.

    Args:
        model_name: Model name like "Qwen3-Coder-30B" or "gemma-3-27b-it".

    Returns:
        Estimated parameters in billions, or None if not detected.
    """
    name_lower = model_name.lower()

    # Look for patterns like "30B", "7b", "1.5B", "70b"
    match = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?(?:\s|$|-|_)", name_lower)
    if match:
        return float(match.group(1))

    # Look for "30b-" pattern
    match = re.search(r"(\d+(?:\.\d+)?)b-", name_lower)
    if match:
        return float(match.group(1))

    return None
