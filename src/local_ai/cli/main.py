"""CLI entry point for local-ai."""

import typer

from local_ai.cli.benchmark import benchmark_app
from local_ai.cli.server import server_app

app = typer.Typer(
    name="local-ai",
    help="Offline AI coding assistant for Apple Silicon",
    no_args_is_help=True,
)

app.add_typer(server_app, name="server")
app.add_typer(benchmark_app, name="benchmark")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from local_ai import __version__

        typer.echo(f"local-ai version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """local-ai: Offline AI coding assistant for Apple Silicon."""
    pass
