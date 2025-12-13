"""CLI server commands for local-ai.

Provides subcommands for managing the MLX LM server:
- start: Start the server
- stop: Stop the server
- status: Show server status
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from local_ai.config.loader import load_config
from local_ai.logging import configure_logging, get_logger
from local_ai.server.manager import ServerManager

console = Console()
_logger = get_logger("CLI.server")

server_app = typer.Typer(
    name="server",
    help="Manage the local-ai server",
    no_args_is_help=True,
)


@server_app.command()
def start(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model path or name")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Server port")] = None,
    host: Annotated[str | None, typer.Option("--host", "-h", help="Server host")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Start the local-ai server."""
    configure_logging(log_level=log_level, console=False)
    _logger.info(
        "CLI start command: model={}, port={}, host={}, config={}",
        model, port, host, config,
    )

    settings = load_config(config_path=config, model=model, port=port, host=host)
    manager = ServerManager(settings)
    result = manager.start()

    if result.success:
        port = settings.server.port
        _logger.info("Server started successfully on port {}", port)
        console.print(f"[green]:heavy_check_mark: Server started on port {port}[/green]")
    else:
        _logger.error("Failed to start server: {}", result.error)
        console.print(f"[red]:heavy_multiplication_x: Error: {result.error}[/red]")
        raise typer.Exit(code=1)


@server_app.command()
def stop(
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Stop the local-ai server."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI stop command")

    settings = load_config(model="placeholder")
    manager = ServerManager(settings)
    result = manager.stop()

    if result.success:
        _logger.info("Server stopped successfully")
        console.print("[green]:heavy_check_mark: Server stopped[/green]")
    else:
        _logger.error("Failed to stop server: {}", result.error)
        console.print(f"[red]:heavy_multiplication_x: Error: {result.error}[/red]")
        raise typer.Exit(code=1)


@server_app.command()
def status(
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Show local-ai server status."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI status command")

    settings = load_config(model="placeholder")
    manager = ServerManager(settings)
    server_status = manager.status()

    if server_status.running:
        _logger.debug(
            "Server status: running=True, pid={}, host={}, port={}",
            server_status.pid, server_status.host, server_status.port,
        )
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Status", f"[green]running[/green] with PID {server_status.pid}")
        table.add_row("Host", str(server_status.host))
        table.add_row("Port", str(server_status.port))
        table.add_row("Model", str(server_status.model))
        table.add_row("Health", str(server_status.health))
        panel = Panel(table, title="[bold cyan]Server Status[/bold cyan]", border_style="cyan")
        console.print(panel)
    else:
        _logger.debug("Server status: running=False")
        panel = Panel(
            "[yellow]Server not running[/yellow]",
            title="[bold cyan]Server Status[/bold cyan]",
            border_style="cyan",
        )
        console.print(panel)
