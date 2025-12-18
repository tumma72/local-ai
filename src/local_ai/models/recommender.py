"""Model recommendation engine for client settings.

Analyzes model metadata and hardware to recommend optimal client configuration:
- Temperature based on model type (reasoning, code, chat, etc.)
- max_tokens scaled to context length with floor/ceiling
- Memory fit analysis for hardware compatibility

Generation settings are client-side concerns; this module helps users
configure their clients (Zed, Claude Code, etc.) appropriately.
"""

import re
from dataclasses import dataclass

from local_ai.hardware.apple_silicon import AppleSiliconInfo, get_max_model_size_gb
from local_ai.logging import get_logger

_logger = get_logger("Models.recommender")

# Model type patterns for detection
_REASONING_PATTERNS = [
    r"-r1[-\s]",  # DeepSeek-R1
    r"[/_-]r1$",  # ends with R1
    r"qwq",  # QwQ reasoning model
    r"reasoning",
    r"think",
]

_CODE_PATTERNS = [
    r"coder",
    r"code[-_]",
    r"[-_]code",
    r"starcoder",
    r"codellama",
    r"devstral",  # Mistral's code model
    r"deepseek[-_]?coder",
]

_CHAT_PATTERNS = [
    r"instruct",
    r"[-_]it[-_\d]",  # -it- or -it3 (instruct-tuned)
    r"[-_]it$",  # ends with -it
    r"chat",
]

_CREATIVE_PATTERNS = [
    r"creative",
    r"story",
    r"writing",
]

# Temperature recommendations by model type
_TEMPERATURE_BY_TYPE: dict[str, float] = {
    "reasoning": 0.0,  # Deterministic for chain-of-thought
    "code": 0.2,  # Low temperature for accurate code
    "chat": 0.7,  # Balanced for conversation
    "creative": 1.0,  # High for creative output
    "general": 0.7,  # Default for unspecified
}

# max_tokens constraints
_MAX_TOKENS_FLOOR = 2048
_MAX_TOKENS_CEILING = 32768
_MAX_TOKENS_DEFAULT = 4096
_CONTEXT_TO_MAX_TOKENS_RATIO = 0.25


@dataclass
class ModelRecommendation:
    """Recommended settings for a model on specific hardware."""

    model_id: str
    model_type: str
    model_size_gb: float | None
    context_length: int | None

    # Hardware analysis
    fits_in_memory: bool
    memory_headroom_gb: float

    # Recommended settings
    temperature: float
    max_tokens: int
    top_p: float

    # Explanations
    temperature_reason: str
    max_tokens_reason: str


def detect_model_type(model_id: str) -> str:
    """Detect model type from name patterns.

    Args:
        model_id: Full model ID (e.g., mlx-community/DeepSeek-R1-Qwen3-8B-4bit).

    Returns:
        Model type: "reasoning", "code", "chat", "creative", or "general".
    """
    name_lower = model_id.lower()

    # Check reasoning first (highest priority)
    for pattern in _REASONING_PATTERNS:
        if re.search(pattern, name_lower):
            _logger.debug("Model {} detected as reasoning (pattern: {})", model_id, pattern)
            return "reasoning"

    # Check code models
    for pattern in _CODE_PATTERNS:
        if re.search(pattern, name_lower):
            _logger.debug("Model {} detected as code (pattern: {})", model_id, pattern)
            return "code"

    # Check chat/instruct models
    for pattern in _CHAT_PATTERNS:
        if re.search(pattern, name_lower):
            _logger.debug("Model {} detected as chat (pattern: {})", model_id, pattern)
            return "chat"

    # Check creative models
    for pattern in _CREATIVE_PATTERNS:
        if re.search(pattern, name_lower):
            _logger.debug("Model {} detected as creative (pattern: {})", model_id, pattern)
            return "creative"

    _logger.debug("Model {} detected as general (no specific patterns)", model_id)
    return "general"


def get_recommended_temperature(model_type: str) -> float:
    """Get recommended temperature for a model type.

    Args:
        model_type: One of "reasoning", "code", "chat", "creative", "general".

    Returns:
        Recommended temperature value (0.0 to 1.0+).
    """
    return _TEMPERATURE_BY_TYPE.get(model_type, _TEMPERATURE_BY_TYPE["general"])


def get_recommended_max_tokens(context_length: int | None) -> int:
    """Calculate recommended max_tokens from context length.

    Strategy: 25% of context length with floor and ceiling.
    - Floor: 2048 (minimum practical value)
    - Ceiling: 32768 (practical maximum)
    - Default: 4096 when context unknown

    Args:
        context_length: Model's maximum context length, or None if unknown.

    Returns:
        Recommended max_tokens value.
    """
    if context_length is None:
        return _MAX_TOKENS_DEFAULT

    # Calculate 25% of context
    scaled = int(context_length * _CONTEXT_TO_MAX_TOKENS_RATIO)

    # Apply floor and ceiling
    clamped = max(_MAX_TOKENS_FLOOR, min(scaled, _MAX_TOKENS_CEILING))

    _logger.debug(
        "max_tokens: context={}, scaled={}, clamped={}",
        context_length, scaled, clamped,
    )
    return clamped


def check_memory_fit(
    model_size_gb: float,
    hardware: AppleSiliconInfo,
) -> tuple[bool, float]:
    """Check if a model fits in hardware memory.

    Args:
        model_size_gb: Model size in GB.
        hardware: Hardware information.

    Returns:
        Tuple of (fits: bool, headroom_gb: float).
        Headroom is negative if model doesn't fit.
    """
    max_model_size = get_max_model_size_gb(hardware, safety_margin=0.25)
    headroom = max_model_size - model_size_gb
    fits = headroom >= 0

    _logger.debug(
        "Memory fit: model={:.1f}GB, max={:.1f}GB, headroom={:.1f}GB, fits={}",
        model_size_gb, max_model_size, headroom, fits,
    )
    return fits, headroom


def recommend_settings(
    model_id: str,
    model_size_gb: float | None = None,
    context_length: int | None = None,
    hardware: AppleSiliconInfo | None = None,
) -> ModelRecommendation:
    """Generate recommended client settings for a model.

    Args:
        model_id: Full model ID.
        model_size_gb: Model size in GB (optional).
        context_length: Model's context length (optional).
        hardware: Hardware info, or None to auto-detect.

    Returns:
        ModelRecommendation with settings and explanations.
    """
    from local_ai.hardware.apple_silicon import detect_hardware

    # Detect hardware if not provided
    if hardware is None:
        try:
            hardware = detect_hardware()
        except RuntimeError:
            # Not on Apple Silicon - use mock hardware for recommendations
            _logger.warning("Not on Apple Silicon, using default hardware assumptions")
            hardware = None

    # Detect model type
    model_type = detect_model_type(model_id)

    # Get temperature recommendation
    temperature = get_recommended_temperature(model_type)
    temperature_reason = _get_temperature_reason(model_type, temperature)

    # Get max_tokens recommendation
    max_tokens = get_recommended_max_tokens(context_length)
    max_tokens_reason = _get_max_tokens_reason(context_length, max_tokens)

    # Check memory fit
    fits_in_memory = True
    memory_headroom_gb = 0.0

    if model_size_gb is not None and hardware is not None:
        fits_in_memory, memory_headroom_gb = check_memory_fit(model_size_gb, hardware)

    return ModelRecommendation(
        model_id=model_id,
        model_type=model_type,
        model_size_gb=model_size_gb,
        context_length=context_length,
        fits_in_memory=fits_in_memory,
        memory_headroom_gb=memory_headroom_gb,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,  # Standard value
        temperature_reason=temperature_reason,
        max_tokens_reason=max_tokens_reason,
    )


def _get_temperature_reason(model_type: str, temperature: float) -> str:
    """Generate explanation for temperature recommendation."""
    reasons = {
        "reasoning": "Reasoning models work best at 0 for deterministic chain-of-thought",
        "code": "Low temperature ensures accurate, consistent code generation",
        "chat": "Balanced temperature for natural conversation",
        "creative": "High temperature enables diverse, creative output",
        "general": "Standard temperature for general-purpose use",
    }
    return reasons.get(model_type, f"Default temperature {temperature}")


def _get_max_tokens_reason(context_length: int | None, max_tokens: int) -> str:
    """Generate explanation for max_tokens recommendation."""
    if context_length is None:
        return "Default value (context length unknown)"

    if max_tokens == _MAX_TOKENS_CEILING:
        return f"Capped at {_MAX_TOKENS_CEILING:,} (model supports {context_length:,})"
    if max_tokens == _MAX_TOKENS_FLOOR:
        return f"Minimum practical value (model context: {context_length:,})"

    return f"25% of {context_length:,} context length"
