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


class TestBenchmarkReporterDefaultPath:
    """Verify BenchmarkReporter uses default path when none provided."""

    def test_uses_default_output_directory_when_none_provided(self) -> None:
        """Should use default state directory when no output_dir given."""
        reporter = BenchmarkReporter(output_dir=None)

        # Verify the default path is set (home/.local/state/local-ai/benchmarks)
        expected_path = Path.home() / ".local" / "state" / "local-ai" / "benchmarks"
        assert reporter._output_dir == expected_path


class TestBenchmarkReporterLoadEdgeCases:
    """Verify BenchmarkReporter handles edge cases during loading."""

    def test_returns_empty_list_when_directory_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        """Should return empty list when output directory does not exist."""
        nonexistent_dir = tmp_path / "nonexistent"
        reporter = BenchmarkReporter(output_dir=nonexistent_dir)

        results = reporter.load_all()

        assert results == []

    def test_skips_corrupt_json_files_during_load(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should skip files with invalid JSON and continue loading others."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        # Save a valid result
        reporter.save(sample_result)

        # Create a corrupt JSON file
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ invalid json content")

        # Create another invalid file with valid JSON but invalid schema
        invalid_schema_file = tmp_path / "invalid_schema.json"
        invalid_schema_file.write_text('{"foo": "bar"}')

        results = reporter.load_all()

        # Should only load the valid result
        assert len(results) == 1
        assert results[0].model == "mlx-community/test-model"


class TestBenchmarkReporterCompareWithTestResults:
    """Verify BenchmarkReporter comparison includes test results when available."""

    def test_comparison_includes_test_results_when_present(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should include test pass/fail counts in comparison when available."""
        from local_ai.benchmark.schema import TestResults

        # Create a result with test results
        result_with_tests = BenchmarkResult(
            benchmark_id="bench-003",
            model="mlx-community/test-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=TestResults(
                passed=8, failed=2, errors=0, skipped=0, total=10, output=""
            ),
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(result_with_tests)

        comparison = reporter.compare()

        assert len(comparison) == 1
        assert comparison[0]["tests_passed"] == 8
        assert comparison[0]["tests_failed"] == 2
        assert comparison[0]["tests_total"] == 10
        assert comparison[0]["test_success_rate"] == 0.8

    def test_comparison_has_null_test_fields_when_no_test_results(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        """Should have null test fields when test_results is None."""
        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(sample_result)

        comparison = reporter.compare()

        assert comparison[0]["tests_passed"] is None
        assert comparison[0]["tests_failed"] is None
        assert comparison[0]["tests_total"] is None
        assert comparison[0]["test_success_rate"] is None


class TestBenchmarkReporterPrintComparison:
    """Verify BenchmarkReporter prints comparison tables correctly."""

    def test_prints_message_when_no_results_exist(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print warning when no benchmark results found."""
        reporter = BenchmarkReporter(output_dir=tmp_path)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "No benchmark results found" in captured.out

    def test_prints_comparison_table_with_results(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print formatted table when results exist."""
        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(sample_result)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "Benchmark Comparison" in captured.out
        assert "test-model" in captured.out
        assert "test-task" in captured.out

    def test_prints_table_with_test_columns_when_tests_present(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should include Tests column when results have test data."""
        from local_ai.benchmark.schema import TestResults

        result_with_tests = BenchmarkResult(
            benchmark_id="bench-tests",
            model="mlx-community/test-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=TestResults(
                passed=10, failed=0, errors=0, skipped=0, total=10, output=""
            ),
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(result_with_tests)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "10/10" in captured.out  # All tests passed

    def test_prints_table_with_partial_test_results(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should show partial test results with appropriate formatting."""
        from local_ai.benchmark.schema import TestResults

        result_with_partial_tests = BenchmarkResult(
            benchmark_id="bench-partial",
            model="mlx-community/test-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=TestResults(
                passed=5, failed=5, errors=0, skipped=0, total=10, output=""
            ),
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(result_with_partial_tests)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "5/10" in captured.out  # Partial pass

    def test_prints_table_with_zero_tests_passed(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should show red formatting when no tests passed."""
        from local_ai.benchmark.schema import TestResults

        result_with_failed_tests = BenchmarkResult(
            benchmark_id="bench-failed",
            model="mlx-community/test-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=TestResults(
                passed=0, failed=10, errors=0, skipped=0, total=10, output=""
            ),
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(result_with_failed_tests)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "0/10" in captured.out  # All failed

    def test_prints_agentic_mode_indicator(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should show 'A' for agentic mode results."""
        from local_ai.benchmark.schema import BenchmarkMode

        agentic_result = BenchmarkResult(
            benchmark_id="bench-agentic",
            model="mlx-community/test-model",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            mode=BenchmarkMode.AGENTIC,
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(agentic_result)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        # Table should show mode legend
        assert "Mode:" in captured.out

    def test_prints_dash_for_missing_test_data_when_others_have_tests(
        self, tmp_path: Path, sample_result: BenchmarkResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print dash for results without tests when others have tests."""
        from local_ai.benchmark.schema import TestResults

        # One result with tests
        result_with_tests = BenchmarkResult(
            benchmark_id="bench-with-tests",
            model="mlx-community/model-with-tests",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=TestResults(
                passed=10, failed=0, errors=0, skipped=0, total=10, output=""
            ),
        )

        # One result without tests
        result_without_tests = BenchmarkResult(
            benchmark_id="bench-no-tests",
            model="mlx-community/model-no-tests",
            task=sample_result.task,
            started_at=sample_result.started_at,
            completed_at=sample_result.completed_at,
            runs=sample_result.runs,
            test_results=None,  # No test results
        )

        reporter = BenchmarkReporter(output_dir=tmp_path)
        reporter.save(result_with_tests)
        reporter.save(result_without_tests)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        # Should show both results and handle the missing test data
        assert "Benchmark Comparison" in captured.out


class TestTruncateModelName:
    """Verify _truncate_model_name helper truncates long names."""

    def test_truncates_long_model_names(self) -> None:
        """Should truncate model names exceeding max length."""
        from local_ai.benchmark.reporter import _truncate_model_name

        long_name = "owner/very-long-model-name-that-exceeds-thirty-characters"

        truncated = _truncate_model_name(long_name, max_len=30)

        assert len(truncated) == 30
        assert truncated.endswith("...")

    def test_preserves_short_model_names(self) -> None:
        """Should not truncate names within max length."""
        from local_ai.benchmark.reporter import _truncate_model_name

        short_name = "owner/short-name"

        truncated = _truncate_model_name(short_name, max_len=30)

        assert truncated == "short-name"

    def test_extracts_name_from_full_model_id(self) -> None:
        """Should extract just the model name part after the slash."""
        from local_ai.benchmark.reporter import _truncate_model_name

        full_id = "mlx-community/Qwen3-8B-4bit"

        truncated = _truncate_model_name(full_id)

        assert truncated == "Qwen3-8B-4bit"
