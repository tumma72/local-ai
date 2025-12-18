"""CLI server commands for local-ai.

Provides subcommands for managing the MLX LM server:
- start: Start the server
- stop: Stop the server
- restart: Restart the server
- status: Show server status
- logs: View server logs
"""

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from local_ai.config.loader import load_config
from local_ai.logging import configure_logging, get_logger
from local_ai.server.manager import ServerManager

# Default state directory for logs
STATE_DIR = Path.home() / ".local" / "state" / "local-ai"

console = Console()
_logger = get_logger("CLI.server")


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds into human-readable string.

    Args:
        seconds: Uptime in seconds.

    Returns:
        Formatted string like "2h 30m 15s" or "5m 30s".
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

server_app = typer.Typer(
    name="server",
    help="Manage the local-ai server",
    no_args_is_help=True,
)


@server_app.command()
def start(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model path or name (optional - MLX Omni Server loads models dynamically)",
        ),
    ] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Server port")] = None,
    host: Annotated[str | None, typer.Option("--host", "-h", help="Server host")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    startup_timeout: Annotated[
        float, typer.Option("--timeout", "-t", help="Startup timeout in seconds")
    ] = 30.0,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Start the local-ai server.

    Note: Model is optional because MLX Omni Server loads models dynamically.
    Models are specified in API requests rather than at server startup.
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info(
        "CLI start command: model={}, port={}, host={}, config={}",
        model,
        port,
        host,
        config,
    )

    settings = load_config(config_path=config, model=model, port=port, host=host)
    manager = ServerManager(settings)

    model_info = (
        f"model: {settings.model.path}"
        if settings.model.path
        else "no specific model (dynamic loading)"
    )
    console.print(f"[cyan]Starting server with {model_info}...[/cyan]")

    result = manager.start(startup_timeout=startup_timeout)

    if result.success:
        port = settings.server.port
        _logger.info("Server started successfully on port {}", port)
        console.print(f"[green]:heavy_check_mark: Server started on port {port}[/green]")
    else:
        _logger.error("Failed to start server: {}", result.error)
        # Show error in a panel for better readability
        error_lines = (result.error or "Unknown error").split("\n")
        if len(error_lines) > 1:
            # Multi-line error with log output
            panel = Panel(
                result.error or "Unknown error",
                title="[bold red]Server Start Failed[/bold red]",
                border_style="red",
            )
            console.print(panel)
        else:
            console.print(f"[red]:heavy_multiplication_x: Error: {result.error}[/red]")
        raise typer.Exit(code=1)


@server_app.command()
def stop(
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Stop the local-ai server."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI stop command")

    manager = ServerManager()  # No settings needed for stop
    result = manager.stop()

    if result.success:
        _logger.info("Server stopped successfully")
        console.print("[green]:heavy_check_mark: Server stopped[/green]")
    else:
        _logger.error("Failed to stop server: {}", result.error)
        console.print(f"[red]:heavy_multiplication_x: Error: {result.error}[/red]")
        raise typer.Exit(code=1)


@server_app.command()
def restart(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model path or name (optional - MLX Omni Server loads models dynamically)",
        ),
    ] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Server port")] = None,
    host: Annotated[str | None, typer.Option("--host", "-h", help="Server host")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    startup_timeout: Annotated[
        float, typer.Option("--timeout", "-t", help="Startup timeout in seconds")
    ] = 30.0,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Restart the local-ai server.

    Stops the server if running, then starts it with the specified configuration.
    Equivalent to running 'local-ai server stop && local-ai server start'.

    Note: Model is optional because MLX Omni Server loads models dynamically.
    Models are specified in API requests rather than at server startup.
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info(
        "CLI restart command: model={}, port={}, host={}, config={}",
        model,
        port,
        host,
        config,
    )

    # First, stop the server if it's running
    stop_manager = ServerManager()
    stop_result = stop_manager.stop()

    if stop_result.success:
        _logger.info("Server stopped for restart")
        console.print("[cyan]Server stopped, restarting...[/cyan]")
    elif "not running" in (stop_result.error or "").lower():
        # Server wasn't running - that's fine for restart
        _logger.info("Server was not running, starting fresh")
        console.print("[cyan]Server was not running, starting...[/cyan]")
    else:
        # Actual error stopping
        _logger.error("Failed to stop server for restart: {}", stop_result.error)
        console.print(
            f"[red]:heavy_multiplication_x: Error stopping server: {stop_result.error}[/red]"
        )
        raise typer.Exit(code=1)

    # Now start the server
    settings = load_config(config_path=config, model=model, port=port, host=host)
    start_manager = ServerManager(settings)

    model_info = (
        f"model: {settings.model.path}"
        if settings.model.path
        else "no specific model (dynamic loading)"
    )
    console.print(f"[cyan]Starting server with {model_info}...[/cyan]")

    start_result = start_manager.start(startup_timeout=startup_timeout)

    if start_result.success:
        port_num = settings.server.port
        _logger.info("Server restarted successfully on port {}", port_num)
        console.print(f"[green]:heavy_check_mark: Server restarted on port {port_num}[/green]")
    else:
        _logger.error("Failed to start server during restart: {}", start_result.error)
        error_lines = (start_result.error or "Unknown error").split("\n")
        if len(error_lines) > 1:
            panel = Panel(
                start_result.error or "Unknown error",
                title="[bold red]Server Restart Failed[/bold red]",
                border_style="red",
            )
            console.print(panel)
        else:
            console.print(f"[red]:heavy_multiplication_x: Error: {start_result.error}[/red]")
        raise typer.Exit(code=1)


@server_app.command()
def status(
    port: Annotated[int | None, typer.Option("--port", "-p", help="Server port")] = None,
    host: Annotated[str | None, typer.Option("--host", "-h", help="Server host")] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Show local-ai server status."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI status command")

    manager = ServerManager(host=host, port=port)  # No settings needed for status
    server_status = manager.status()

    if server_status.running:
        _logger.debug(
            "Server status: running=True, pid={}, host={}, port={}",
            server_status.pid,
            server_status.host,
            server_status.port,
        )
        # Color health status appropriately
        health_display = server_status.health or "unknown"
        if health_display == "healthy":
            health_display = f"[green]{health_display}[/green]"
        elif health_display == "unhealthy":
            health_display = f"[red]{health_display}[/red]"
        else:
            health_display = f"[yellow]{health_display}[/yellow]"

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Status", f"[green]running[/green] with PID {server_status.pid}")
        table.add_row("Host", str(server_status.host))
        table.add_row("Port", str(server_status.port))

        # Format uptime
        if server_status.uptime_seconds is not None:
            uptime_str = _format_uptime(server_status.uptime_seconds)
            table.add_row("Uptime", uptime_str)

        table.add_row("Available Models", str(server_status.models))
        table.add_row("Health", health_display)
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


@server_app.command()
def logs(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output in real-time"),
    ] = False,
    lines: Annotated[
        int,
        typer.Option("--lines", "-n", help="Number of lines to show"),
    ] = 50,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Log level"),
    ] = "INFO",
) -> None:
    """View server logs.

    Shows logs from the server process. By default shows last 50 lines.
    Use --follow to stream logs in real-time.

    Examples:
        local-ai server logs
        local-ai server logs --follow
        local-ai server logs --lines 100
        local-ai server logs -f -n 20
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI logs command: follow={}, lines={}", follow, lines)

    log_file = STATE_DIR / "server.log"

    if not log_file.exists():
        console.print("[yellow]No log file found[/yellow]")
        console.print(f"\n[dim]Expected location: {log_file}[/dim]")
        console.print("\nThe server may not have been started yet, or logs may have been cleared.")
        console.print("\nStart the server with:")
        console.print("  local-ai server start")
        return

    try:
        if follow:
            # Stream logs in real-time using tail -f
            console.print(f"[cyan]Following logs from: {log_file}[/cyan]")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")

            try:
                subprocess.run(
                    ["tail", "-f", "-n", str(lines), str(log_file)],
                    check=False,
                )
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs[/dim]")
                return
        else:
            # Show last N lines
            console.print(f"[cyan]Last {lines} lines from: {log_file}[/cyan]\n")

            with open(log_file) as f:
                all_lines = f.readlines()
                display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

                for line in display_lines:
                    # Print without extra newline (lines already have \n)
                    console.print(line, end="", highlight=False)

            total = len(all_lines)
            shown = len(display_lines)
            console.print(f"\n[dim]Showing {shown} of {total} total lines[/dim]")
            console.print(f"[dim]Log file: {log_file}[/dim]")

    except PermissionError:
        console.print(f"[red]Permission denied reading log file: {log_file}[/red]")
        _logger.error("Permission denied: {}", log_file)
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Failed to read logs: {e}[/red]")
        _logger.error("Failed to read logs: {}", e)
        raise typer.Exit(code=1) from None
