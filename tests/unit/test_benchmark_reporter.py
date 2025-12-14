"""Behavioral tests for benchmark reporter module.

Tests verify public behavior of result storage and comparison:
- BenchmarkReporter saves results to JSON
- BenchmarkReporter loads results from directory
- BenchmarkReporter generates comparison tables
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from local_ai.benchmark.reporter import BenchmarkReporter
from local_ai.benchmark.schema import (
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    TaskDifficulty,
    ThroughputMetrics,
    TimingMetrics,
)


@pytest.fixture
def sample_result() -> BenchmarkResult:
    """Create a sample benchmark result for tests."""
    task = BenchmarkTask(
        id="test-task",
        name="Test Task",
        system_prompt="System prompt",
        user_prompt="User prompt",
        difficulty=TaskDifficulty.SIMPLE,
    )

    timing = TimingMetrics(
        ttft_ms=100.0,
        total_latency_ms=2000.0,
        generation_time_ms=1900.0,
    )
    throughput = ThroughputMetrics(
        prompt_tokens=50,
        completion_tokens=100,
        tokens_per_second=52.6,
        prompt_processing_rate=500.0,
    )
    memory = MemoryMetrics(
        baseline_memory_mb=1000.0,
        peak_memory_mb=1500.0,
    )

    run = SingleRunResult(
        run_id="run-001",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        timing=timing,
        throughput=throughput,
        memory=memory,
        raw_output="def hello(): pass",
        error=None,
    )

    return BenchmarkResult(
        benchmark_id="bench-001",
        model="mlx-community/test-model",
        task=task,
        started_at=datetime(2024, 1, 15, 10, 30, 0),
        completed_at=datetime(2024, 1, 15, 10, 35, 0),
        runs=[run],
    )


class TestBenchmarkReporterSave:
    """Verify BenchmarkReporter saves results."""

    def test_saves_result_to_json_file(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should save benchmark result as JSON file."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        path = reporter.save(sample_result)

        assert path.exists()
        assert path.suffix == ".json"

    def test_creates_output_directory_if_missing(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "new_dir"
        reporter = BenchmarkReporter(output_dir=output_dir)

        path = reporter.save(sample_result)

        assert output_dir.exists()
        assert path.exists()

    def test_saved_json_contains_model_and_metrics(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should include model name and metrics in saved JSON."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        path = reporter.save(sample_result)

        with path.open() as f:
            data = json.load(f)

        assert data["model"] == "mlx-community/test-model"
        assert data["task"]["id"] == "test-task"
        assert len(data["runs"]) == 1
        assert "avg_tokens_per_second" in data

    def test_filename_includes_model_and_task(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should include model slug and task ID in filename."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        path = reporter.save(sample_result)

        filename = path.name
        assert "test-model" in filename
        assert "test-task" in filename


class TestBenchmarkReporterLoad:
    """Verify BenchmarkReporter loads results."""

    def test_loads_all_results_from_directory(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should load all JSON results from directory."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        # Save multiple results
        reporter.save(sample_result)

        # Modify and save another
        result2 = sample_result.model_copy()
        result2 = BenchmarkResult(
            benchmark_id="bench-002",
            model="mlx-community/other-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
        )
        reporter.save(result2)

        results = reporter.load_all()

        assert len(results) == 2

    def test_returns_empty_list_for_empty_directory(
        self, tmp_path: Path
    ) -> None:
        """Should return empty list if no results exist."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        results = reporter.load_all()

        assert results == []


class TestBenchmarkReporterCompare:
    """Verify BenchmarkReporter generates comparisons."""

    def test_generates_comparison_data(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should generate comparison data for loaded results."""
        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(sample_result)

        comparison = reporter.compare()

        assert len(comparison) >= 1
        assert "model" in comparison[0]
        assert "avg_tokens_per_second" in comparison[0]
        assert "avg_ttft_ms" in comparison[0]
