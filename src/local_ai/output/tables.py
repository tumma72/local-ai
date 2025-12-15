"""Table factory methods for local-ai CLI.

Provides consistent table styling across all commands.
See docs/UI_UX_GUIDELINES.md for design principles.
"""

from rich.table import Table

from local_ai.models.schema import ModelSearchResult
from local_ai.output.console import format_downloads


def create_model_table(
    title: str,
    *,
    show_quant: bool = False,
    show_downloads: bool = True,
    show_likes: bool = True,
    show_full_id: bool = True,
) -> Table:
    """Create a consistently styled model table.

    Args:
        title: Table title.
        show_quant: Whether to include quantization column.
        show_downloads: Whether to include downloads column.
        show_likes: Whether to include likes column.
        show_full_id: If True, single Model column with full ID; else Model + Author.

    Returns:
        A Rich Table with standard styling.
    """
    table = Table(title=title, expand=True)

    # Model column - either full ID or just name
    table.add_column("Model", style="cyan", no_wrap=True)

    # Author column only when not showing full ID
    if not show_full_id:
        table.add_column("Author", style="blue")

    if show_quant:
        table.add_column("Quant", style="yellow", justify="center")

    if show_downloads:
        table.add_column("Downloads", style="green", justify="right")

    if show_likes:
        table.add_column("Likes", style="magenta", justify="right")

    return table


def add_model_row(
    table: Table,
    model: ModelSearchResult,
    *,
    show_quant: bool = False,
    show_downloads: bool = True,
    show_likes: bool = True,
    show_full_id: bool = True,
) -> None:
    """Add a model row to a table with consistent formatting.

    Args:
        table: The table to add to.
        model: The model search result.
        show_quant: Whether quantization column is present.
        show_downloads: Whether downloads column is present.
        show_likes: Whether likes column is present.
        show_full_id: If True, show full ID; if False, show name + author columns.
    """
    if show_full_id:
        row: list[str] = [model.id]
    else:
        row = [model.name, model.author]

    if show_quant:
        row.append(model.quantization.value)

    if show_downloads:
        row.append(format_downloads(model.downloads))

    if show_likes:
        row.append(str(model.likes))

    table.add_row(*row)


def create_search_results_table(
    title: str,
    models: list[ModelSearchResult],
    *,
    show_quant: bool = False,
    show_full_id: bool = True,
) -> Table:
    """Create a complete search results table with models.

    Args:
        title: Table title.
        models: List of model search results.
        show_quant: Whether to show quantization column.
        show_full_id: If True, show full model ID; else show name + author.

    Returns:
        A Rich Table populated with model data.
    """
    table = create_model_table(title, show_quant=show_quant, show_full_id=show_full_id)

    for model in models:
        add_model_row(table, model, show_quant=show_quant, show_full_id=show_full_id)

    return table


def create_local_models_table(
    models: list[ModelSearchResult],
    title: str = "Local Models",
) -> Table:
    """Create a table for locally available models.

    Args:
        models: List of ModelSearchResult objects.
        title: Table title.

    Returns:
        A Rich Table with local model data.
    """
    table = Table(title=title, expand=True)
    table.add_column("Model", style="cyan", no_wrap=True, ratio=3)
    table.add_column("Quant", style="yellow", justify="center", width=6)
    table.add_column("Size", style="green", justify="right", width=8)

    for model in models:
        table.add_row(model.id, model.quantization.value, model.size_gb)

    return table
