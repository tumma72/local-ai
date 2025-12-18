"""CLI models commands for local-ai.

Provides subcommands for model discovery and management:
- list: List locally available models
- search: Search HuggingFace for models
- info: Get detailed model information
- download: Download models from HuggingFace
"""

from pathlib import Path
from typing import Annotated

import httpx
import typer

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.hardware import (
    detect_hardware,
    estimate_model_params_from_name,
    get_max_model_size_gb,
    get_recommended_quantization,
)
from local_ai.logging import configure_logging, get_logger
from local_ai.models.huggingface import (
    SortOption,
    create_local_model_result,
    get_converted_model_info,
    get_converted_models,
    get_local_model_size,
    search_models_enhanced,
)
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
    host: Annotated[str, typer.Option("--host", help="Server host")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = DEFAULT_PORT,
    all_models: Annotated[
        bool, typer.Option("--all", "-a", help="Show all local models (cached + converted)")
    ] = False,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """List locally available models.

    By default, shows models from the running server.
    Use --all to show all locally available models (cached + converted).

    Examples:
        local-ai models list
        local-ai models list --all
        local-ai models list --port 8080
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models list: host={}, port={}, all={}", host, port, all_models)

    models = []

    # Get converted models (always include these)
    converted = get_converted_models()
    if converted:
        models.extend(converted)

    if all_models:
        # Show all local models without requiring server
        if models:
            table = create_local_models_table(models, title="Local Models")
            console.print(table)
            console.print(f"\n[dim]Showing {len(models)} locally converted models[/dim]")
        else:
            console.print("[yellow]⚠ No local models found[/yellow]")
            console.print("\nDownload models with:")
            console.print("  local-ai models download <model-id>")
        return

    # Get models from server
    url = f"http://{host}:{port}/v1/models"

    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    except httpx.ConnectError:
        console.print(f"[red]✗ Cannot connect to server at {host}:{port}[/red]")
        console.print("\nMake sure the server is running:")
        console.print("  local-ai server start")
        console.print("\nOr use --all to show locally available models.")
        raise typer.Exit(code=1) from None
    except httpx.HTTPStatusError as e:
        console.print(f"[red]✗ Server error: {e.response.status_code}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]✗ Failed to list models: {e}[/red]")
        _logger.error("Failed to list models: {}", e)
        raise typer.Exit(code=1) from None

    raw_models = data.get("data", [])

    # Convert server models to ModelSearchResult with local cache info
    server_models = [create_local_model_result(m.get("id", "")) for m in raw_models]

    if not server_models and not converted:
        console.print("[yellow]⚠ No models available[/yellow]")
        return

    # Show server models
    if server_models:
        table = create_local_models_table(server_models, title="Server Models")
        console.print(table)
        console.print(f"\n[dim]Server: {host}:{port}[/dim]")

    # Show converted models if any
    if converted:
        console.print()
        table2 = create_local_models_table(converted, title="Converted Models")
        console.print(table2)
        console.print("\n[dim]Location: ~/.local/share/local-ai/models/[/dim]")


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
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
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
    valid_sorts: list[SortOption] = [
        "downloads",
        "likes",
        "trending_score",
        "created_at",
        "last_modified",
    ]
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
    console.print(
        f"\n[dim]Showing {total} results ({len(results.top_models)} top + {len(results.mlx_models)} MLX-optimized)[/dim]"
    )


@models_app.command()
def info(
    model_id: Annotated[
        str,
        typer.Argument(
            help="Full model ID (e.g., mlx-community/Qwen3-8B-4bit or local/model-name)"
        ),
    ],
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Get detailed information about a specific model.

    Works with both HuggingFace models and locally converted models.

    Examples:
        local-ai models info mlx-community/Qwen3-8B-4bit
        local-ai models info local/mistralai_Devstral-Small-4bit-mlx
    """
    from local_ai.models.huggingface import get_model_info

    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models info: model_id={}", model_id)

    # Check if it's a local converted model
    if model_id.startswith("local/"):
        model = get_converted_model_info(model_id)
        if model is None:
            console.print(f"[red]✗ Local model not found: {model_id}[/red]")
            console.print("\nList available models with:")
            console.print("  local-ai models list --all")
            raise typer.Exit(code=1)
    else:
        with console.status(f"[bold green]Fetching info for '{model_id}'...[/bold green]"):
            model = get_model_info(model_id)

        if model is None:
            console.print(f"[red]✗ Model not found: {model_id}[/red]")
            raise typer.Exit(code=1)

    # Display model info
    console.print(f"\n[bold cyan]{model.id}[/bold cyan]")
    console.print(f"  Author: {model.author}")
    console.print(f"  Quantization: {model.quantization.value}")
    console.print(f"  Size: {model.size_gb}")

    # Only show HuggingFace-specific info for non-local models
    if not model_id.startswith("local/"):
        console.print(f"  Downloads: {format_downloads(model.downloads)}")
        console.print(f"  Likes: {model.likes}")
        console.print(f"  Last Modified: {model.last_modified}")
        console.print(f"  Source: {model.source_label}")

        if model.tags:
            console.print(f"  Tags: {', '.join(model.tags[:10])}")
    else:
        console.print(f"  Location: ~/.local/share/local-ai/models/{model_id[6:]}")


@models_app.command()
def recommend(
    model_id: Annotated[
        str,
        typer.Argument(help="Full model ID (e.g., mlx-community/DeepSeek-R1-Qwen3-8B-4bit)"),
    ],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text, json, zed"),
    ] = "text",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed analysis"),
    ] = False,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Get recommended client settings for a model.

    Analyzes the model and your hardware to recommend optimal generation settings
    for use in clients like Zed, Claude Code, or the API.

    Examples:
        local-ai models recommend mlx-community/DeepSeek-R1-Qwen3-8B-4bit
        local-ai models recommend mlx-community/Devstral-Small-4bit --format json
        local-ai models recommend mlx-community/Qwen3-8B-4bit --format zed
    """
    import json as json_module

    from rich.panel import Panel
    from rich.table import Table

    from local_ai.models.recommender import recommend_settings

    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI models recommend: model_id={}, format={}", model_id, output_format)

    # Validate format
    if output_format not in ("text", "json", "zed"):
        console.print(f"[red]✗ Invalid format: {output_format}[/red]")
        console.print("Valid formats: text, json, zed")
        raise typer.Exit(code=1)

    # Fetch model info
    with console.status("[bold green]Analyzing model...[/bold green]"):
        model_info = _fetch_model_analysis(model_id)
        if model_info is None:
            console.print(f"[red]✗ Model not found: {model_id}[/red]")
            raise typer.Exit(code=1)

        model_size_gb, context_length = model_info

        # Detect hardware
        try:
            hardware = detect_hardware()
        except RuntimeError:
            hardware = None
            if output_format == "text":
                console.print("[yellow]⚠ Not running on Apple Silicon[/yellow]")

        # Generate recommendation
        recommendation = recommend_settings(
            model_id=model_id,
            model_size_gb=model_size_gb,
            context_length=context_length,
            hardware=hardware,
        )

    # Output based on format
    if output_format == "json":
        output = {
            "model": {
                "id": recommendation.model_id,
                "type": recommendation.model_type,
                "size_gb": recommendation.model_size_gb,
                "context_length": recommendation.context_length,
            },
            "hardware": {
                "chip": hardware.chip_name if hardware else None,
                "memory_gb": hardware.memory_gb if hardware else None,
                "available_gb": get_max_model_size_gb(hardware) if hardware else None,
            } if hardware else None,
            "recommendation": {
                "temperature": recommendation.temperature,
                "max_tokens": recommendation.max_tokens,
                "top_p": recommendation.top_p,
                "fits": recommendation.fits_in_memory,
            },
        }
        console.print(json_module.dumps(output, indent=2))

    elif output_format == "zed":
        # Generate Zed settings.json snippet matching actual Zed config structure
        output = {
            "language_models": {
                "openai_compatible": {
                    "local-ai": {
                        "api_url": "http://localhost:8080/v1",
                        "available_models": [
                            {
                                "name": recommendation.model_id,
                                "display_name": _get_display_name(recommendation.model_id),
                                "max_tokens": recommendation.max_tokens,
                                "capabilities": {
                                    "tools": True,
                                    "images": False,
                                    "parallel_tool_calls": True,
                                    "prompt_cache_key": False,
                                },
                            }
                        ],
                    }
                }
            },
            "agent": {
                "model_parameters": [
                    {
                        "provider": "local-ai",
                        "model": recommendation.model_id,
                        "temperature": recommendation.temperature,
                    }
                ]
            },
        }
        console.print("[bold cyan]Add to your Zed settings.json:[/bold cyan]\n")
        console.print(json_module.dumps(output, indent=2))

    else:
        # Text format with rich formatting
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_column("Reason", style="dim")

        # Model info section
        table.add_row("[bold]Model[/bold]", "", "")
        table.add_row("  ID", recommendation.model_id, "")
        table.add_row("  Type", recommendation.model_type, "")
        if recommendation.model_size_gb:
            table.add_row("  Size", f"{recommendation.model_size_gb:.1f} GB", "")
        if recommendation.context_length:
            table.add_row("  Context", f"{recommendation.context_length:,} tokens", "")

        # Hardware section
        if hardware:
            table.add_row("", "", "")
            table.add_row("[bold]Hardware[/bold]", "", "")
            table.add_row("  Chip", hardware.chip_name, "")
            table.add_row("  Memory", f"{hardware.memory_gb:.0f} GB", "")
            max_size = get_max_model_size_gb(hardware)
            table.add_row("  Available", f"{max_size:.0f} GB", "(for models)")

            if recommendation.fits_in_memory:
                table.add_row(
                    "  Status",
                    f"[green]✓ Fits[/green] ({recommendation.memory_headroom_gb:.0f} GB headroom)",
                    "",
                )
            else:
                table.add_row(
                    "  Status",
                    f"[red]✗ Too large[/red] ({-recommendation.memory_headroom_gb:.0f} GB over)",
                    "",
                )

        # Recommended settings section
        table.add_row("", "", "")
        table.add_row("[bold]Recommended Settings[/bold]", "", "")
        table.add_row(
            "  temperature",
            str(recommendation.temperature),
            recommendation.temperature_reason,
        )
        table.add_row(
            "  max_tokens",
            str(recommendation.max_tokens),
            recommendation.max_tokens_reason,
        )
        table.add_row("  top_p", str(recommendation.top_p), "Standard value")

        panel = Panel(
            table,
            title="[bold cyan]Model Recommendation[/bold cyan]",
            border_style="cyan",
        )
        console.print()
        console.print(panel)

        # Usage tip
        console.print()
        console.print(f"[dim]For Zed config: local-ai models recommend {model_id} --format zed[/dim]")


def _fetch_model_analysis(model_id: str) -> tuple[float | None, int | None] | None:
    """Fetch model size and context length from HuggingFace.

    Returns:
        Tuple of (size_gb, context_length) or None if not found.
    """
    from local_ai.models.huggingface import get_model_info

    try:
        # Get model info for size
        model = get_model_info(model_id)
        if model is None:
            return None

        size_gb = model.size_bytes / (1024**3) if model.size_bytes else None

        # Fetch context length from config.json
        context_length = _fetch_context_length(model_id)

        return size_gb, context_length

    except Exception as e:
        _logger.error("Failed to fetch model analysis: {}", e)
        return None


def _fetch_context_length(model_id: str) -> int | None:
    """Fetch context length from model's config.json on HuggingFace."""
    import json as json_module

    from huggingface_hub import hf_hub_download

    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_files_only=False,
        )
        with open(config_path) as f:
            config = json_module.load(f)

        # Try common keys for context length
        context_length = (
            config.get("max_position_embeddings")
            or config.get("max_seq_len")
            or config.get("seq_length")
            or config.get("n_positions")
            or config.get("max_sequence_length")
        )
        return context_length

    except Exception as e:
        _logger.debug("Could not fetch context length: {}", e)
        return None


def _get_display_name(model_id: str) -> str:
    """Generate a display name for the model."""
    # Extract model name from ID
    name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Clean up common suffixes
    name = name.replace("-4bit", "")
    name = name.replace("-8bit", "")
    name = name.replace("-mlx", "")

    # Add "(local)" suffix
    return f"{name} (local)"


@models_app.command()
def download(
    model_id: Annotated[
        str, typer.Argument(help="Full model ID (e.g., mlx-community/Qwen3-8B-4bit)")
    ],
    convert: Annotated[
        bool, typer.Option("--convert", "-c", help="Convert to MLX format (for non-MLX models)")
    ] = False,
    quantize: Annotated[
        str | None,
        typer.Option("--quantize", "-q", help="Quantization level: 4bit, 6bit, 8bit, or 'auto'"),
    ] = None,
    output_dir: Annotated[
        str | None, typer.Option("--output", "-o", help="Output directory for converted models")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force re-download even if cached")
    ] = False,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Download a model from HuggingFace.

    For MLX-optimized models (from mlx-community), simply downloads to cache.
    For other models, use --convert to convert to MLX format.

    Examples:
        local-ai models download mlx-community/Qwen3-8B-4bit
        local-ai models download mistralai/Devstral-Small-2505 --convert --quantize auto
        local-ai models download Qwen/Qwen3-30B --convert --quantize 4bit
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info(
        "CLI models download: model_id={}, convert={}, quantize={}", model_id, convert, quantize
    )

    # Early format check for --convert to fail fast with helpful error
    if convert:
        _check_model_format(model_id)

    # Check if already downloaded
    existing_size = get_local_model_size(model_id)
    if existing_size and not force:
        size_gb = existing_size / (1024**3)
        console.print(f"[green]✓[/green] Model already downloaded: {model_id}")
        console.print(f"  Size: {size_gb:.1f} GB")
        console.print("\n[dim]Use --force to re-download[/dim]")
        return

    # If converting, determine quantization
    if convert:
        _download_and_convert(model_id, quantize, output_dir)
    else:
        _download_mlx_model(model_id, force)


def _download_mlx_model(model_id: str, force: bool) -> None:
    """Download an MLX-optimized model."""
    from huggingface_hub import snapshot_download

    console.print(f"[bold]Downloading {model_id}...[/bold]\n")

    try:
        cache_path = snapshot_download(
            repo_id=model_id,
            force_download=force,
        )
        console.print(f"\n[green]✓[/green] Downloaded to: {cache_path}")

        # Show size
        size = get_local_model_size(model_id)
        if size:
            size_gb = size / (1024**3)
            console.print(f"  Size: {size_gb:.1f} GB")

    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        _logger.error("Download failed: {}", e)
        raise typer.Exit(code=1) from None


def _check_model_format(model_id: str) -> None:
    """Check if model format is compatible with MLX conversion.

    Raises:
        typer.Exit: If model format is incompatible (GGUF, GGML, AWQ, etc.)
    """
    model_id_lower = model_id.lower()

    # GGUF/GGML models (llama.cpp format) - cannot be converted to MLX
    if "gguf" in model_id_lower or "ggml" in model_id_lower:
        console.print("[red]✗ Cannot convert GGUF/GGML models to MLX format[/red]")
        console.print(
            "\n[yellow]GGUF models are pre-quantized for llama.cpp and cannot be converted to MLX.[/yellow]"
        )
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Use an MLX-optimized version from mlx-community:")
        console.print("     [cyan]local-ai models search devstral[/cyan]")
        console.print("  2. Download the original safetensors model and convert it:")
        console.print(
            "     [cyan]local-ai models download mistralai/Devstral-Small-2505 --convert[/cyan]"
        )
        raise typer.Exit(code=1)

    # AWQ models (different quantization format)
    if "-awq" in model_id_lower:
        console.print("[red]✗ Cannot convert AWQ models to MLX format[/red]")
        console.print(
            "\n[yellow]AWQ models use a different quantization format incompatible with MLX.[/yellow]"
        )
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Use an MLX-optimized version from mlx-community")
        console.print("  2. Download the original safetensors model and convert it")
        raise typer.Exit(code=1)

    # GPTQ models (another quantization format)
    if "-gptq" in model_id_lower or "gptq" in model_id_lower:
        console.print("[red]✗ Cannot convert GPTQ models to MLX format[/red]")
        console.print(
            "\n[yellow]GPTQ models use a different quantization format incompatible with MLX.[/yellow]"
        )
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Use an MLX-optimized version from mlx-community")
        console.print("  2. Download the original safetensors model and convert it")
        raise typer.Exit(code=1)


def _download_and_convert(
    model_id: str,
    quantize: str | None,
    output_dir: str | None,
) -> None:
    """Download and convert a model to MLX format."""
    from mlx_lm import convert as mlx_convert

    # Determine quantization
    q_bits: int | None = None
    if quantize == "auto":
        # Auto-detect based on hardware
        try:
            hw = detect_hardware()
            params = estimate_model_params_from_name(model_id)
            if params:
                recommended = get_recommended_quantization(params, hw)
                if recommended == "too_large":
                    max_size = get_max_model_size_gb(hw)
                    console.print(
                        f"[red]✗ Model too large for available memory (~{max_size:.0f} GB max)[/red]"
                    )
                    console.print("\nTry a smaller model or lower quantization.")
                    raise typer.Exit(code=1)
                q_bits = int(recommended.replace("bit", ""))
                console.print(
                    f"[cyan]Auto-detected:[/cyan] {params:.0f}B params → {recommended} recommended"
                )
            else:
                console.print("[yellow]⚠ Could not detect model size, using 4bit default[/yellow]")
                q_bits = 4
        except RuntimeError:
            console.print("[yellow]⚠ Hardware detection failed, using 4bit default[/yellow]")
            q_bits = 4
    elif quantize:
        q_bits = int(quantize.replace("bit", ""))

    # Determine output directory
    if output_dir:
        mlx_path = Path(output_dir)
    else:
        # Default: ~/.local/share/local-ai/models/{model_name}
        model_name = model_id.replace("/", "_")
        if q_bits:
            model_name = f"{model_name}-{q_bits}bit-mlx"
        else:
            model_name = f"{model_name}-mlx"
        mlx_path = Path.home() / ".local" / "share" / "local-ai" / "models" / model_name

    console.print(f"[bold]Converting {model_id} to MLX format...[/bold]")
    if q_bits:
        console.print(f"  Quantization: {q_bits}bit")
    console.print(f"  Output: {mlx_path}\n")

    try:
        mlx_convert(
            hf_path=model_id,
            mlx_path=str(mlx_path),
            quantize=q_bits is not None,
            q_bits=q_bits or 4,
        )
        console.print(f"\n[green]✓[/green] Converted to: {mlx_path}")

        # Show size
        if mlx_path.exists():
            total_size = sum(f.stat().st_size for f in mlx_path.rglob("*") if f.is_file())
            size_gb = total_size / (1024**3)
            console.print(f"  Size: {size_gb:.1f} GB")

    except Exception as e:
        error_msg = str(e)

        # Handle specific error cases with user-friendly messages
        if "No safetensors found" in error_msg:
            console.print("[red]✗ Cannot convert: No safetensors weights found[/red]")
            console.print(
                "\n[yellow]This model doesn't contain safetensors weights required for MLX conversion.[/yellow]"
            )
            console.print("\nPossible reasons:")
            console.print("  • Model uses GGUF, GGML, or other incompatible format")
            console.print("  • Model only contains PyTorch .bin files (older format)")
            console.print("\n[bold]Options:[/bold]")
            console.print("  1. Search for an MLX-optimized version:")
            console.print("     [cyan]local-ai models search <model-name>[/cyan]")
            console.print("  2. Find the original model with safetensors and convert that")
        elif "model_type" in error_msg.lower() or "not supported" in error_msg.lower():
            console.print("[red]✗ Unsupported model architecture[/red]")
            console.print("\n[yellow]This model architecture is not yet supported by MLX.[/yellow]")
            console.print("\nSearch for compatible models:")
            console.print("  [cyan]local-ai models search <model-name>[/cyan]")
        else:
            console.print(f"[red]✗ Conversion failed: {e}[/red]")

        _logger.error("Conversion failed: {}", e)
        raise typer.Exit(code=1) from None
