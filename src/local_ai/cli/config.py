"""CLI config commands for local-ai.

Provides subcommands for configuration management:
- show: Display resolved configuration values
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from local_ai.config.loader import _discover_config_path, load_config
from local_ai.logging import configure_logging, get_logger

console = Console()
_logger = get_logger("CLI.config")

config_app = typer.Typer(
    name="config",
    help="Manage configuration",
    no_args_is_help=True,
)


@config_app.command()
def show(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Config file path"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Log level"),
    ] = "INFO",
) -> None:
    """Show resolved configuration values.

    Displays all configuration settings with their sources:
    - File: From TOML configuration file
    - Default: Built-in default values

    Examples:
        local-ai config show
        local-ai config show --config /path/to/config.toml
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI config show command")

    # Determine which config file was loaded
    config_file: Path | None = None
    config_source: str

    if config:
        config_file = config
        config_source = "explicit path"
    else:
        discovered = _discover_config_path()
        if discovered:
            config_file = discovered
            if discovered.name == "config.toml" and discovered.parent == Path.cwd():
                config_source = "current directory"
            else:
                config_source = "user config"
        else:
            config_source = "none (using defaults)"

    # Load config
    settings = load_config(config_path=config)

    # Display config file info
    console.print()
    if config_file:
        console.print(f"[cyan]Configuration file:[/cyan] {config_file}")
        console.print(f"[dim]Source: {config_source}[/dim]\n")
    else:
        console.print("[yellow]No configuration file found[/yellow]")
        console.print("[dim]Using default values[/dim]\n")

    # Create table for settings
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Source", style="dim")

    # Helper to determine source
    def get_source(has_file: bool) -> str:
        return "File" if has_file else "Default"

    has_config = config_file is not None

    # Server settings
    table.add_row("", "", "")  # Spacer
    table.add_row("[bold]Server[/bold]", "", "")
    table.add_row("  host", settings.server.host, get_source(has_config))
    table.add_row("  port", str(settings.server.port), get_source(has_config))
    table.add_row("  log_level", settings.server.log_level, get_source(has_config))

    # Model settings
    table.add_row("", "", "")  # Spacer
    table.add_row("[bold]Model[/bold]", "", "")
    model_path = settings.model.path or "[dim]none (dynamic loading)[/dim]"
    table.add_row("  path", model_path, get_source(has_config))

    if settings.model.adapter_path:
        table.add_row(
            "  adapter_path", str(settings.model.adapter_path), get_source(has_config)
        )

    table.add_row(
        "  trust_remote_code",
        str(settings.model.trust_remote_code).lower(),
        get_source(has_config),
    )

    # Note about generation settings
    table.add_row("", "", "")  # Spacer
    table.add_row(
        "[dim]Note[/dim]",
        "[dim]Generation settings (temperature, max_tokens) are client-side.[/dim]",
        "",
    )
    table.add_row(
        "",
        "[dim]Configure in your client (Zed, Claude Code) or per-request.[/dim]",
        "",
    )

    # Display
    panel = Panel(table, title="[bold cyan]Configuration[/bold cyan]", border_style="cyan")
    console.print(panel)
    console.print()

    # Show config file paths
    console.print("[dim]Config search paths:[/dim]")
    console.print("  [dim]1.[/dim] ./config.toml")
    console.print("  [dim]2.[/dim] ~/.config/local-ai/config.toml")
    console.print()
