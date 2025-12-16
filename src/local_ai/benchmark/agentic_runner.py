"""Agentic benchmark runner using Goose recipes with test validation.

Executes multi-turn agentic workflows and validates results through tests.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.benchmark.goose_runner import get_goose_output_dir, get_recipe_path, run_goose_recipe
from local_ai.benchmark.reporter import BenchmarkReporter
from local_ai.benchmark.schema import (
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    ThroughputMetrics,
    TimingMetrics,
)
from local_ai.benchmark.test_validator import validate_tdd_output
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.agentic")


def run_agentic_benchmark(
    model: str,
    task: BenchmarkTask,
    recipe_name: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = 600.0,
    max_turns: int = 20,
    output_dir: Path | None = None,
    run_tests: bool = True,
) -> BenchmarkResult:
    """Run an agentic benchmark using a Goose recipe.

    Args:
        model: Model identifier.
        task: Benchmark task definition.
        recipe_name: Recipe name (without .yaml). Defaults to task ID.
        host: Server host.
        port: Server port.
        timeout: Max execution time in seconds.
        max_turns: Max turns for recipe execution.
        output_dir: Base directory for generated code.
        run_tests: Whether to run test validation.

    Returns:
        BenchmarkResult with agentic execution details and test results.
    """
    started_at = datetime.now(UTC)
    benchmark_id = f"agentic_{uuid.uuid4().hex[:8]}"

    # Determine recipe to use
    if recipe_name is None:
        recipe_name = task.id.replace("-", "_")

    recipe_path = get_recipe_path(recipe_name)

    if not recipe_path.exists():
        # Try alternate naming
        recipe_path = get_recipe_path(task.id)

    _logger.info("Starting agentic benchmark: {}", benchmark_id)
    _logger.info("Model: {}, Recipe: {}", model, recipe_path)

    # Get working directory
    working_dir = get_goose_output_dir(model, task.id, base_dir=output_dir)

    # Clean directory for fresh run
    if working_dir.exists():
        import shutil

        shutil.rmtree(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)

    # Run the recipe
    result = run_goose_recipe(
        recipe_path=recipe_path,
        model=model,
        working_directory=working_dir,
        host=host,
        port=port,
        timeout=timeout,
        max_turns=max_turns,
        recipe_params={"work_dir": str(working_dir)},
    )

    completed_at = datetime.now(UTC)
    total_time_ms = result.elapsed_ms

    # Create single run result (agentic runs are single runs)
    run_result = SingleRunResult(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        timestamp=started_at,
        timing=TimingMetrics(
            ttft_ms=total_time_ms / max(1, result.turns_taken),  # Estimate
            total_latency_ms=total_time_ms,
            generation_time_ms=total_time_ms,
        ),
        throughput=ThroughputMetrics(
            prompt_tokens=0,  # Not tracked in agentic mode
            completion_tokens=len(result.output) // 4,  # Rough estimate
            tokens_per_second=0,  # Not applicable to multi-turn
            prompt_processing_rate=0,
        ),
        memory=MemoryMetrics(
            baseline_memory_mb=0,
            peak_memory_mb=0,
        ),
        raw_output=result.output,
        error=result.error,
    )

    # Run test validation if requested
    test_results = None
    if run_tests and result.success:
        _logger.info("Running test validation in {}", working_dir)
        test_results = validate_tdd_output(working_dir)
        _logger.info(
            "Test results: {}/{} passed",
            test_results.passed,
            test_results.total,
        )

    benchmark_result = BenchmarkResult(
        benchmark_id=benchmark_id,
        model=model,
        task=task,
        started_at=started_at,
        completed_at=completed_at,
        runs=[run_result],
        mode=BenchmarkMode.AGENTIC,
        test_results=test_results,
        working_directory=str(working_dir),
    )

    return benchmark_result


def run_and_save_agentic_benchmark(
    model: str,
    task: BenchmarkTask,
    recipe_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 10240,
    timeout: float = 600.0,
    max_turns: int = 20,
    output_dir: Path | None = None,
    run_tests: bool = True,
) -> tuple[BenchmarkResult, Path]:
    """Run agentic benchmark and save results.

    Args:
        model: Model identifier.
        task: Benchmark task definition.
        recipe_name: Recipe name (without .yaml).
        host: Server host.
        port: Server port.
        timeout: Max execution time.
        max_turns: Max recipe turns.
        output_dir: Base directory for results.
        run_tests: Whether to run test validation.

    Returns:
        Tuple of (BenchmarkResult, path to saved JSON).
    """
    result = run_agentic_benchmark(
        model=model,
        task=task,
        recipe_name=recipe_name,
        host=host,
        port=port,
        timeout=timeout,
        max_turns=max_turns,
        output_dir=output_dir,
        run_tests=run_tests,
    )

    reporter = BenchmarkReporter()
    path = reporter.save(result)

    return result, path
