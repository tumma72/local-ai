"""CLI benchmark commands for local-ai.

Provides subcommands for running and comparing model benchmarks:
- run: Execute benchmark on a model
- tasks: List available benchmark tasks
- compare: Compare benchmark results
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.benchmark.goose_runner import get_goose_output_dir, run_goose_command
from local_ai.benchmark.reporter import BenchmarkReporter
from local_ai.benchmark.runner import BenchmarkRunner
from local_ai.benchmark.tasks import get_builtin_tasks, get_task_by_id
from local_ai.logging import configure_logging, get_logger

console = Console()
_logger = get_logger("CLI.benchmark")

benchmark_app = typer.Typer(
    name="benchmark",
    help="Benchmark local LLM models for coding tasks",
    no_args_is_help=True,
)


@benchmark_app.command()
def tasks(
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """List available benchmark tasks."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI benchmark tasks command")

    builtin_tasks = get_builtin_tasks()

    if not builtin_tasks:
        console.print("[yellow]No benchmark tasks found[/yellow]")
        return

    table = Table(title="Available Benchmark Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Expected Tokens", style="green")
    table.add_column("Tags", style="magenta")

    for task in builtin_tasks:
        table.add_row(
            task.id,
            task.name,
            task.difficulty.value,
            str(task.expected_output_tokens),
            ", ".join(task.tags) if task.tags else "-",
        )

    console.print(table)


@benchmark_app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m", help="Model name or path")],
    task: Annotated[str, typer.Option("--task", "-t", help="Task ID to run")],
    requests: Annotated[int, typer.Option("--requests", "-n", help="Number of requests")] = 1,
    warmup: Annotated[int, typer.Option("--warmup", "-w", help="Warmup requests")] = 0,
    single: Annotated[  # noqa: ARG001
        bool, typer.Option("--single", "-1", help="Single run mode (default)")
    ] = False,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = DEFAULT_PORT,
    host: Annotated[str, typer.Option("--host", help="Server host")] = DEFAULT_HOST,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Output directory for results")
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds")] = 300.0,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Run benchmark on a model with specified task."""
    configure_logging(log_level=log_level, console=False)
    _logger.info(
        "CLI benchmark run: model={}, task={}, requests={}, warmup={}",
        model,
        task,
        requests,
        warmup,
    )

    # Validate task exists
    benchmark_task = get_task_by_id(task)
    if benchmark_task is None:
        console.print(f"[red]Task not found: {task}[/red]")
        console.print("Use 'local-ai benchmark tasks' to see available tasks.")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[cyan]Model:[/cyan] {model}\n"
            f"[cyan]Task:[/cyan] {benchmark_task.name}\n"
            f"[cyan]Server:[/cyan] {host}:{port}\n"
            f"[cyan]Requests:[/cyan] {requests} (+ {warmup} warmup)",
            title="[bold]Benchmark Configuration[/bold]",
        )
    )

    # Create runner and execute benchmark
    runner = BenchmarkRunner(host=host, port=port, model=model)

    with console.status("[bold green]Running benchmark...[/bold green]"):
        try:
            result = asyncio.run(
                runner.run(
                    task=benchmark_task,
                    num_requests=requests,
                    warmup_requests=warmup,
                    timeout=timeout,
                )
            )
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            _logger.error("Benchmark failed: {}", e)
            raise typer.Exit(code=1) from None

    # Display results
    console.print()
    results_table = Table(title="Benchmark Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green", justify="right")

    results_table.add_row("Average Tokens/sec", f"{result.avg_tokens_per_second:.1f}")
    results_table.add_row("Average TTFT", f"{result.avg_ttft_ms:.0f} ms")
    results_table.add_row("Success Rate", f"{result.success_rate:.0%}")
    results_table.add_row("Total Runs", str(len(result.runs)))

    console.print(results_table)

    # Save results
    reporter = BenchmarkReporter(output_dir=output_dir)
    path = reporter.save(result)
    console.print(f"\n[dim]Results saved to: {path}[/dim]")


@benchmark_app.command()
def compare(
    directory: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Directory containing benchmark results"),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Compare benchmark results across models."""
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI benchmark compare: directory={}", directory)

    reporter = BenchmarkReporter(output_dir=directory)
    reporter.print_comparison_table()


@benchmark_app.command()
def goose(
    model: Annotated[str, typer.Option("--model", "-m", help="Model name or path")],
    task: Annotated[str, typer.Option("--task", "-t", help="Task ID to run")],
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = DEFAULT_PORT,
    host: Annotated[str, typer.Option("--host", help="Server host")] = DEFAULT_HOST,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Base directory for generated code"),
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds")] = 300.0,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
) -> None:
    """Run task through Goose CLI for agentic comparison.

    Compares raw model output (from 'benchmark run') with Goose-enhanced
    agentic output using the same model.

    Generated code is saved to: benchmark_code/goose_<model>/<task>/
    """
    configure_logging(log_level=log_level, console=False)
    _logger.info("CLI benchmark goose: model={}, task={}", model, task)

    # Validate task exists
    benchmark_task = get_task_by_id(task)
    if benchmark_task is None:
        console.print(f"[red]Task not found: {task}[/red]")
        console.print("Use 'local-ai benchmark tasks' to see available tasks.")
        raise typer.Exit(code=1)

    # Get structured output directory
    working_dir = get_goose_output_dir(model, task, base_dir=output_dir)

    console.print(
        Panel(
            f"[cyan]Model:[/cyan] {model}\n"
            f"[cyan]Task:[/cyan] {benchmark_task.name}\n"
            f"[cyan]Server:[/cyan] {host}:{port}\n"
            f"[cyan]Output:[/cyan] {working_dir}\n"
            f"[cyan]Mode:[/cyan] Goose Agentic",
            title="[bold]Goose Benchmark[/bold]",
        )
    )

    # Construct prompt from task
    prompt = f"{benchmark_task.system_prompt}\n\n{benchmark_task.user_prompt}"

    with console.status("[bold green]Running Goose...[/bold green]"):
        result = run_goose_command(
            prompt=prompt,
            model=model,
            host=host,
            port=port,
            timeout=timeout,
            working_directory=working_dir,
        )

    if not result.success:
        console.print(f"[red]Goose run failed: {result.error}[/red]")
        raise typer.Exit(code=1)

    # Display results
    console.print()
    results_table = Table(title="Goose Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green", justify="right")

    results_table.add_row("Elapsed Time", f"{result.elapsed_ms:.0f} ms")
    results_table.add_row("Output Length", f"{len(result.output)} chars")
    results_table.add_row("Working Directory", str(result.working_directory))
    results_table.add_row("Status", "Success" if result.success else "Failed")

    console.print(results_table)

    # Show output preview
    console.print()
    output_preview = result.output[:500] + "..." if len(result.output) > 500 else result.output
    console.print(Panel(output_preview, title="[bold]Output Preview[/bold]"))
