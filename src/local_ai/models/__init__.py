"""Models module for local-ai.

Provides model discovery and management functionality.
"""

from local_ai.models.huggingface import (
    SearchResults,
    search_models,
    search_models_enhanced,
)
from local_ai.models.recommender import (
    ModelRecommendation,
    detect_model_type,
    recommend_settings,
)
from local_ai.models.schema import ModelSearchResult, Quantization

__all__ = [
    "ModelRecommendation",
    "ModelSearchResult",
    "Quantization",
    "SearchResults",
    "detect_model_type",
    "recommend_settings",
    "search_models",
    "search_models_enhanced",
]
