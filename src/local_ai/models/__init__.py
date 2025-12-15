"""Models module for local-ai.

Provides model discovery and management functionality.
"""

from local_ai.models.huggingface import (
    SearchResults,
    search_models,
    search_models_enhanced,
)
from local_ai.models.schema import ModelSearchResult, Quantization

__all__ = [
    "ModelSearchResult",
    "Quantization",
    "SearchResults",
    "search_models",
    "search_models_enhanced",
]
