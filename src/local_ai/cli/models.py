"""CLI models commands for local-ai.

Provides subcommands for model discovery and management:
- search: Search HuggingFace for MLX-optimized models
"""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from local_ai.logging import configure_logging, get_logger
from local_ai.models.huggingface import SortOption, search_models

console = Console()
_logger = get_logger("CLI.models")

models_app = typer.Typer(
    name="models",
    help="Search and manage local AI models",
    no_args_is_help=True,
)


def _format_downloads(count: int) -> str:
    """Format download count for display."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


@models_app.command()
def search(
    query: Annotated[str, typer.Argument(help="Model name to search for")],
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Maximum results to show")
    ] = 20,
    sort: Annotated[
        str, typer.Option("--sort", "-s", help="Sort by: downloads, likes, trending_score")
    ] = "downloads",
    all_sources: Annotated[
        bool, typer.Option("--all", "-a", help="Include non-mlx-community models")
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", "-l", help="Log level")
    ] = "INFO",
) -> None:
    """Search HuggingFace for MLX-optimized models.

    By default, searches only mlx-community models which are pre-optimized
    for Apple Silicon. Use --all to include other MLX-tagged models.

    Examples:
        local-ai models search qwen3-coder
        local-ai models search llama --limit 10 --sort likes
        local-ai models search devstral --all
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models search: query={}, limit={}, sort={}", query, limit, sort)

    # Validate sort option
    valid_sorts: list[SortOption] = ["downloads", "likes", "trending_score", "created_at", "last_modified"]
    if sort not in valid_sorts:
        console.print(f"[red]Invalid sort option: {sort}[/red]")
        console.print(f"Valid options: {', '.join(valid_sorts)}")
        raise typer.Exit(code=1)

    with console.status(f"[bold green]Searching for '{query}'...[/bold green]"):
        try:
            results = search_models(
                query=query,
                limit=limit,
                sort_by=sort,  # type: ignore[arg-type]
                include_all_mlx=all_sources,
            )
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            _logger.error("Search failed: {}", e)
            raise typer.Exit(code=1) from None

    if not results:
        console.print(f"[yellow]No models found for '{query}'[/yellow]")
        console.print("\nTips:")
        console.print("  - Try a broader search term")
        console.print("  - Use --all to include non-mlx-community models")
        return

    # Build results table
    table = Table(title=f"Model Search: \"{query}\"")
    table.add_column("Model", style="cyan", max_width=40)
    table.add_column("Quant", style="yellow", justify="center")
    table.add_column("Downloads", style="green", justify="right")
    table.add_column("Likes", style="magenta", justify="right")
    table.add_column("Source", style="blue", justify="center")

    for model in results:
        table.add_row(
            model.name,
            model.quantization.value,
            _format_downloads(model.downloads),
            str(model.likes),
            model.source_label,
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(results)} results. ★ MLX = mlx-community (pre-optimized for Apple Silicon)[/dim]")


@models_app.command()
def info(
    model_id: Annotated[str, typer.Argument(help="Full model ID (e.g., mlx-community/Qwen3-8B-4bit)")],
    log_level: Annotated[
        str, typer.Option("--log-level", "-l", help="Log level")
    ] = "INFO",
) -> None:
    """Get detailed information about a specific model."""
    from local_ai.models.huggingface import get_model_info

    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models info: model_id={}", model_id)

    with console.status(f"[bold green]Fetching info for '{model_id}'...[/bold green]"):
        model = get_model_info(model_id)

    if model is None:
        console.print(f"[red]Model not found: {model_id}[/red]")
        raise typer.Exit(code=1)

    # Display model info
    console.print(f"\n[bold cyan]{model.id}[/bold cyan]")
    console.print(f"  Author: {model.author}")
    console.print(f"  Quantization: {model.quantization.value}")
    console.print(f"  Downloads: {_format_downloads(model.downloads)}")
    console.print(f"  Likes: {model.likes}")
    console.print(f"  Last Modified: {model.last_modified}")
    console.print(f"  Source: {model.source_label}")

    if model.tags:
        console.print(f"  Tags: {', '.join(model.tags[:10])}")
