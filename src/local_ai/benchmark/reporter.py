"""Benchmark result storage and comparison.

Saves benchmark results as JSON and provides comparison functionality.
"""

import json
import re
from pathlib import Path
from typing import Any

from local_ai.benchmark.schema import BenchmarkMode, BenchmarkResult
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.reporter")


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    # Convert to lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    # Remove leading/trailing hyphens
    return slug.strip("-")


class BenchmarkReporter:
    """Manages benchmark result storage and comparison."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize reporter with output directory.

        Args:
            output_dir: Directory to store results. Defaults to state dir.
        """
        if output_dir is None:
            # Default to user's state directory
            output_dir = Path.home() / ".local" / "state" / "local-ai" / "benchmarks"

        self._output_dir = output_dir

    def save(self, result: BenchmarkResult) -> Path:
        """Save benchmark result to JSON file.

        Args:
            result: The benchmark result to save.

        Returns:
            Path to the saved JSON file.
        """
        # Create output directory if needed
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: date_model-slug_task-id.json
        date_str = result.started_at.strftime("%Y%m%d_%H%M%S")
        model_slug = _slugify(result.model.split("/")[-1])
        task_id = result.task.id

        filename = f"{date_str}_{model_slug}_{task_id}.json"
        path = self._output_dir / filename

        # Serialize to JSON
        data = result.model_dump(mode="json")

        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        _logger.info("Saved benchmark result to {}", path)
        return path

    def load_all(self) -> list[BenchmarkResult]:
        """Load all benchmark results from output directory.

        Returns:
            List of BenchmarkResult objects.
        """
        if not self._output_dir.exists():
            return []

        results: list[BenchmarkResult] = []

        for json_file in sorted(self._output_dir.glob("*.json")):
            try:
                with json_file.open() as f:
                    data = json.load(f)

                result = BenchmarkResult.model_validate(data)
                results.append(result)
                _logger.debug("Loaded result from {}", json_file)

            except (json.JSONDecodeError, ValueError) as e:
                _logger.warning("Failed to load {}: {}", json_file, e)

        _logger.info("Loaded {} benchmark results", len(results))
        return results

    def compare(self) -> list[dict[str, Any]]:
        """Generate comparison data for all loaded results.

        Returns:
            List of comparison dictionaries with model metrics.
        """
        results = self.load_all()

        comparison: list[dict[str, Any]] = []

        for result in results:
            row = {
                "model": result.model,
                "task_id": result.task.id,
                "task_name": result.task.name,
                "num_runs": len(result.runs),
                "mode": result.mode.value if result.mode else BenchmarkMode.RAW.value,
                "avg_tokens_per_second": result.avg_tokens_per_second,
                "avg_ttft_ms": result.avg_ttft_ms,
                "success_rate": result.success_rate,
                "timestamp": result.started_at.isoformat(),
            }

            # Add test results if available
            if result.test_results:
                row["tests_passed"] = result.test_results.passed
                row["tests_failed"] = result.test_results.failed
                row["tests_total"] = result.test_results.total
                row["test_success_rate"] = result.test_results.success_rate
            else:
                row["tests_passed"] = None
                row["tests_failed"] = None
                row["tests_total"] = None
                row["test_success_rate"] = None

            comparison.append(row)

        # Sort by tokens per second (highest first)
        comparison.sort(key=lambda x: x["avg_tokens_per_second"], reverse=True)

        return comparison

    def print_comparison_table(self) -> None:
        """Print Rich comparison table to console."""
        from rich.console import Console
        from rich.table import Table

        comparison = self.compare()

        if not comparison:
            Console().print("[yellow]No benchmark results found[/yellow]")
            return

        # Check if any results have test data
        has_tests = any(row.get("tests_total") is not None for row in comparison)

        table = Table(title="Benchmark Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Task", style="white")
        table.add_column("Mode", style="blue")
        table.add_column("Tok/s", style="green", justify="right")
        table.add_column("TTFT", style="yellow", justify="right")
        table.add_column("Success", style="magenta", justify="right")

        if has_tests:
            table.add_column("Tests", style="cyan", justify="right")

        table.add_column("Runs", justify="right")

        for row in comparison:
            # Format mode: short version for display
            mode_display = "A" if row.get("mode") == "agentic" else "R"

            cells = [
                _truncate_model_name(row["model"]),
                row["task_id"],
                mode_display,
                f"{row['avg_tokens_per_second']:.1f}",
                f"{row['avg_ttft_ms']:.0f}ms",
                f"{row['success_rate']:.0%}",
            ]

            if has_tests:
                tests_total = row.get("tests_total")
                if tests_total is not None:
                    tests_passed = row.get("tests_passed", 0)
                    if tests_passed == tests_total:
                        cells.append(f"[green]{tests_passed}/{tests_total}[/green]")
                    elif tests_passed > 0:
                        cells.append(f"[yellow]{tests_passed}/{tests_total}[/yellow]")
                    else:
                        cells.append(f"[red]{tests_passed}/{tests_total}[/red]")
                else:
                    cells.append("-")

            cells.append(str(row["num_runs"]))

            table.add_row(*cells)

        Console().print(table)
        Console().print("\n[dim]Mode: R=Raw API, A=Agentic (Goose)[/dim]")


def _truncate_model_name(model: str, max_len: int = 30) -> str:
    """Truncate model name for display.

    Args:
        model: Full model identifier.
        max_len: Maximum display length.

    Returns:
        Truncated model name.
    """
    # Take only the model name part (after /)
    name = model.split("/")[-1]
    if len(name) > max_len:
        return name[: max_len - 3] + "..."
    return name
