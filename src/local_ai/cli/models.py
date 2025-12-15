"""CLI models commands for local-ai.

Provides subcommands for model discovery and management:
- list: List locally available models
- search: Search HuggingFace for models
- info: Get detailed model information
"""

from typing import Annotated

import httpx
import typer

from local_ai.logging import configure_logging, get_logger
from local_ai.models.huggingface import SortOption, search_models_enhanced
from local_ai.output import (
    console,
    create_local_models_table,
    create_search_results_table,
    format_downloads,
)

_logger = get_logger("CLI.models")

models_app = typer.Typer(
    name="models",
    help="Search and manage local AI models",
    no_args_is_help=True,
)


@models_app.command("list")
def list_models(
    host: Annotated[str, typer.Option("--host", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 10240,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """List locally available models from the running server.

    Shows models currently loaded or available on the local MLX server.
    The server must be running for this command to work.

    Examples:
        local-ai models list
        local-ai models list --port 8080
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models list: host={}, port={}", host, port)

    url = f"http://{host}:{port}/v1/models"

    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    except httpx.ConnectError:
        console.print(f"[red]✗ Cannot connect to server at {host}:{port}[/red]")
        console.print("\nMake sure the server is running:")
        console.print("  local-ai server start")
        raise typer.Exit(code=1) from None
    except httpx.HTTPStatusError as e:
        console.print(f"[red]✗ Server error: {e.response.status_code}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]✗ Failed to list models: {e}[/red]")
        _logger.error("Failed to list models: {}", e)
        raise typer.Exit(code=1) from None

    models = data.get("data", [])

    if not models:
        console.print("[yellow]⚠ No models available on the server[/yellow]")
        return

    table = create_local_models_table(models)
    console.print(table)
    console.print(f"\n[dim]Server: {host}:{port}[/dim]")


@models_app.command()
def search(
    query: Annotated[str, typer.Argument(help="Model name to search for")],
    top: Annotated[
        int, typer.Option("--top", "-t", help="Number of top overall models to show")
    ] = 3,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of MLX-optimized models to show")
    ] = 10,
    sort: Annotated[
        str, typer.Option("--sort", "-s", help="Sort by: downloads, likes, trending_score")
    ] = "downloads",
    log_level: Annotated[
        str, typer.Option("--log-level", "-l", help="Log level")
    ] = "INFO",
) -> None:
    """Search HuggingFace for models.

    Shows two sections:
    1. Top models by downloads (from original creators like mistralai, Qwen)
    2. MLX-optimized versions for Apple Silicon

    Examples:
        local-ai models search devstral
        local-ai models search qwen3-coder --limit 5
        local-ai models search llama --sort likes
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models search: query={}, top={}, limit={}, sort={}", query, top, limit, sort)

    # Validate sort option
    valid_sorts: list[SortOption] = ["downloads", "likes", "trending_score", "created_at", "last_modified"]
    if sort not in valid_sorts:
        console.print(f"[red]✗ Invalid sort option: {sort}[/red]")
        console.print(f"Valid options: {', '.join(valid_sorts)}")
        raise typer.Exit(code=1)

    with console.status(f"[bold green]Searching for '{query}'...[/bold green]"):
        try:
            results = search_models_enhanced(
                query=query,
                top_limit=top,
                mlx_limit=limit,
                sort_by=sort,  # type: ignore[arg-type]
            )
        except Exception as e:
            console.print(f"[red]✗ Search failed: {e}[/red]")
            _logger.error("Search failed: {}", e)
            raise typer.Exit(code=1) from None

    if not results.top_models and not results.mlx_models:
        console.print(f"[yellow]⚠ No models found for '{query}'[/yellow]")
        console.print("\nTips:")
        console.print("  - Try a broader search term")
        console.print("  - Check spelling of model name")
        return

    # Section 1: Top overall models
    if results.top_models:
        table1 = create_search_results_table(
            f'Top Models: "{query}"',
            results.top_models,
            show_quant=False,
        )
        console.print(table1)
        console.print()

    # Section 2: MLX-optimized models
    if results.mlx_models:
        table2 = create_search_results_table(
            "MLX-Optimized for Apple Silicon",
            results.mlx_models,
            show_quant=True,
        )
        console.print(table2)

    total = len(results.top_models) + len(results.mlx_models)
    console.print(f"\n[dim]Showing {total} results ({len(results.top_models)} top + {len(results.mlx_models)} MLX-optimized)[/dim]")


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
        console.print(f"[red]✗ Model not found: {model_id}[/red]")
        raise typer.Exit(code=1)

    # Display model info
    console.print(f"\n[bold cyan]{model.id}[/bold cyan]")
    console.print(f"  Author: {model.author}")
    console.print(f"  Quantization: {model.quantization.value}")
    console.print(f"  Downloads: {format_downloads(model.downloads)}")
    console.print(f"  Likes: {model.likes}")
    console.print(f"  Last Modified: {model.last_modified}")
    console.print(f"  Source: {model.source_label}")

    if model.tags:
        console.print(f"  Tags: {', '.join(model.tags[:10])}")
