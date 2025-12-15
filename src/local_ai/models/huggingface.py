"""HuggingFace Hub integration for model discovery.

Searches HuggingFace Hub for MLX-optimized models compatible with Apple Silicon.
"""

from typing import Literal

from huggingface_hub import HfApi, ModelInfo

from local_ai.logging import get_logger
from local_ai.models.schema import ModelSearchResult

_logger = get_logger("Models.huggingface")

# MLX Community org produces pre-optimized models
MLX_COMMUNITY_ORG = "mlx-community"

SortOption = Literal["downloads", "likes", "trending_score", "created_at", "last_modified"]


def _convert_model(model: ModelInfo, is_community: bool = False) -> ModelSearchResult:
    """Convert HuggingFace ModelInfo to ModelSearchResult.

    Args:
        model: HuggingFace Hub model info.
        is_community: Whether this is from mlx-community org.

    Returns:
        ModelSearchResult for display.
    """
    return ModelSearchResult(
        id=model.id or "",
        author=model.author or "",
        downloads=model.downloads or 0,
        likes=model.likes or 0,
        last_modified=str(model.last_modified) if model.last_modified else "",
        is_mlx_community=is_community or (model.author == MLX_COMMUNITY_ORG),
        tags=list(model.tags) if model.tags else [],
    )


def search_models(
    query: str,
    limit: int = 20,
    sort_by: SortOption = "downloads",
    include_all_mlx: bool = False,
) -> list[ModelSearchResult]:
    """Search HuggingFace Hub for MLX-optimized models.

    Uses a two-tier search strategy:
    1. First searches mlx-community org (pre-optimized models)
    2. Then searches all MLX-tagged models if needed

    Args:
        query: Search query (model name).
        limit: Maximum results to return.
        sort_by: Sort field (downloads, likes, trending_score, etc.).
        include_all_mlx: If True, include non-mlx-community MLX models.

    Returns:
        List of ModelSearchResult sorted by relevance.
    """
    api = HfApi()
    results: list[ModelSearchResult] = []
    seen_ids: set[str] = set()

    _logger.info("Searching HuggingFace for: {}", query)

    # Priority 1: Search mlx-community models (pre-optimized for Apple Silicon)
    try:
        community_models = api.list_models(
            search=query,
            author=MLX_COMMUNITY_ORG,
            sort=sort_by,
            direction=-1,  # Descending
            limit=limit,
            full=True,
        )

        for model in community_models:
            if model.id and model.id not in seen_ids and len(results) < limit:
                results.append(_convert_model(model, is_community=True))
                seen_ids.add(model.id)

        _logger.debug("Found {} mlx-community models", len(results))

    except Exception as e:
        _logger.warning("Failed to search mlx-community: {}", e)

    # Priority 2: Search all MLX-tagged models if we need more results
    if include_all_mlx and len(results) < limit:
        remaining = limit - len(results)

        try:
            all_mlx_models = api.list_models(
                search=query,
                filter="mlx",  # Filter by MLX library tag
                sort=sort_by,
                direction=-1,
                limit=remaining * 2,  # Fetch extra to account for duplicates
                full=True,
            )

            for model in all_mlx_models:
                if model.id and model.id not in seen_ids and len(results) < limit:
                    results.append(_convert_model(model, is_community=False))
                    seen_ids.add(model.id)

            _logger.debug("Found {} total models after MLX search", len(results))

        except Exception as e:
            _logger.warning("Failed to search MLX models: {}", e)

    _logger.info("Returning {} models", len(results))
    return results


def get_model_info(model_id: str) -> ModelSearchResult | None:
    """Get detailed information for a specific model.

    Args:
        model_id: Full model ID (e.g., mlx-community/Qwen3-8B-4bit).

    Returns:
        ModelSearchResult or None if not found.
    """
    api = HfApi()

    try:
        info = api.model_info(repo_id=model_id, files_metadata=False)
        return _convert_model(info, is_community=(info.author == MLX_COMMUNITY_ORG))

    except Exception as e:
        _logger.error("Failed to get model info for {}: {}", model_id, e)
        return None
