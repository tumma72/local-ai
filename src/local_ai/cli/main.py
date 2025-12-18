"""CLI entry point for local-ai."""

import typer

from local_ai.cli.benchmark import benchmark_app
from local_ai.cli.config import config_app
from local_ai.cli.models import models_app
from local_ai.cli.server import server_app
from local_ai.cli.status import status_app

app = typer.Typer(
    name="local-ai",
    help="Offline AI coding assistant for Apple Silicon",
    no_args_is_help=True,
)

app.add_typer(server_app, name="server")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(models_app, name="models")
app.add_typer(status_app, name="status")
app.add_typer(config_app, name="config")


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
