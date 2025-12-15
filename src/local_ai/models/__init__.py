"""Models module for local-ai.

Provides model discovery and management functionality.
"""

from local_ai.models.huggingface import search_models
from local_ai.models.schema import ModelSearchResult, Quantization

__all__ = [
    "ModelSearchResult",
    "Quantization",
    "search_models",
]
