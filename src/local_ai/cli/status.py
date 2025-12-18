"""CLI status command for local-ai.

Shows hardware info, configuration, and system status.
"""

from typing import Annotated

import typer

from local_ai.hardware import detect_hardware, get_max_model_size_gb
from local_ai.logging import configure_logging
from local_ai.output import console

status_app = typer.Typer(
    name="status",
    help="Show system status and hardware info",
)


@status_app.callback(invoke_without_command=True)
def status(
    ctx: typer.Context,
    log_level: Annotated[
        str, typer.Option("--log-level", "-l", help="Log level")
    ] = "WARNING",
) -> None:
    """Show hardware info and system status.

    Examples:
        local-ai status
    """
    if ctx.invoked_subcommand is not None:
        return

    configure_logging(log_level=log_level, console=False)

    try:
        hw = detect_hardware()
    except RuntimeError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1) from None

    max_model = get_max_model_size_gb(hw)

    console.print("\n[bold cyan]Hardware[/bold cyan]")
    console.print(f"  Chip: {hw.chip_name}")
    console.print(f"  Memory: {hw.memory_gb:.0f} GB (unified)")
    console.print(f"  CPU: {hw.cpu_cores} cores ({hw.cpu_performance_cores}P + {hw.cpu_efficiency_cores}E)")
    console.print(f"  GPU: {hw.gpu_cores} cores")
    console.print(f"  Neural Engine: {hw.neural_engine_cores} cores")

    console.print("\n[bold cyan]Model Limits[/bold cyan]")
    console.print(f"  Max model size: ~{max_model:.0f} GB")
    console.print("  Recommended format: MLX")

    # Show quantization recommendations
    console.print("\n[bold cyan]Quantization Guide[/bold cyan]")
    console.print(f"  [green]8bit[/green]: Models up to ~{int(max_model)}B params")
    console.print(f"  [yellow]4bit[/yellow]: Models up to ~{int(max_model * 2)}B params")

    console.print()
