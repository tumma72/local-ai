"""Benchmark result storage and comparison.

Saves benchmark results as JSON and provides comparison functionality.
"""

import json
import re
from pathlib import Path
from typing import Any

from local_ai.benchmark.schema import BenchmarkResult
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
            comparison.append({
                "model": result.model,
                "task_id": result.task.id,
                "task_name": result.task.name,
                "num_runs": len(result.runs),
                "avg_tokens_per_second": result.avg_tokens_per_second,
                "avg_ttft_ms": result.avg_ttft_ms,
                "success_rate": result.success_rate,
                "timestamp": result.started_at.isoformat(),
            })

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

        table = Table(title="Benchmark Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Task", style="white")
        table.add_column("Tok/s", style="green", justify="right")
        table.add_column("TTFT (ms)", style="yellow", justify="right")
        table.add_column("Success", style="magenta", justify="right")
        table.add_column("Runs", justify="right")

        for row in comparison:
            table.add_row(
                row["model"],
                row["task_id"],
                f"{row['avg_tokens_per_second']:.1f}",
                f"{row['avg_ttft_ms']:.0f}",
                f"{row['success_rate']:.0%}",
                str(row["num_runs"]),
            )

        Console().print(table)
