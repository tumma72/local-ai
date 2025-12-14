"""Behavioral tests for benchmark schema module.

Tests verify public behavior of benchmark data models:
- TimingMetrics captures TTFT, latency, and generation time
- ThroughputMetrics calculates tokens per second
- MemoryMetrics tracks baseline and peak memory
- BenchmarkTask defines coding task with prompts
- SingleRunResult captures one benchmark execution
- BenchmarkResult aggregates multiple runs with statistics
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from local_ai.benchmark.schema import (
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    TaskDifficulty,
    ThroughputMetrics,
    TimingMetrics,
)


class TestTimingMetrics:
    """Verify TimingMetrics captures timing measurements."""

    def test_creates_with_all_timing_values(self) -> None:
        """TimingMetrics should accept ttft, total latency, and generation time."""
        metrics = TimingMetrics(
            ttft_ms=150.5,
            total_latency_ms=2500.0,
            generation_time_ms=2349.5,
        )
        assert metrics.ttft_ms == 150.5
        assert metrics.total_latency_ms == 2500.0
        assert metrics.generation_time_ms == 2349.5

    def test_accepts_zero_ttft_for_immediate_response(self) -> None:
        """TTFT can be zero if first token arrives immediately."""
        metrics = TimingMetrics(ttft_ms=0.0, total_latency_ms=100.0, generation_time_ms=100.0)
        assert metrics.ttft_ms == 0.0

    def test_rejects_negative_timing_values(self) -> None:
        """Timing values must be non-negative."""
        with pytest.raises(ValidationError):
            TimingMetrics(ttft_ms=-1.0, total_latency_ms=100.0, generation_time_ms=100.0)


class TestThroughputMetrics:
    """Verify ThroughputMetrics captures token throughput."""

    def test_creates_with_token_counts_and_rates(self) -> None:
        """ThroughputMetrics should accept token counts and calculated rates."""
        metrics = ThroughputMetrics(
            prompt_tokens=500,
            completion_tokens=800,
            tokens_per_second=45.5,
            prompt_processing_rate=3333.3,
        )
        assert metrics.prompt_tokens == 500
        assert metrics.completion_tokens == 800
        assert metrics.tokens_per_second == 45.5
        assert metrics.prompt_processing_rate == 3333.3

    def test_accepts_zero_completion_tokens(self) -> None:
        """Completion tokens can be zero if model returns empty response."""
        metrics = ThroughputMetrics(
            prompt_tokens=100,
            completion_tokens=0,
            tokens_per_second=0.0,
            prompt_processing_rate=1000.0,
        )
        assert metrics.completion_tokens == 0

    def test_rejects_negative_token_counts(self) -> None:
        """Token counts must be non-negative."""
        with pytest.raises(ValidationError):
            ThroughputMetrics(
                prompt_tokens=-1,
                completion_tokens=100,
                tokens_per_second=50.0,
                prompt_processing_rate=1000.0,
            )


class TestMemoryMetrics:
    """Verify MemoryMetrics captures memory usage."""

    def test_creates_with_baseline_and_peak_memory(self) -> None:
        """MemoryMetrics should accept baseline and peak memory values."""
        metrics = MemoryMetrics(baseline_memory_mb=1024.0, peak_memory_mb=2048.0)
        assert metrics.baseline_memory_mb == 1024.0
        assert metrics.peak_memory_mb == 2048.0

    def test_peak_can_equal_baseline(self) -> None:
        """Peak memory can equal baseline if no additional memory used."""
        metrics = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1000.0)
        assert metrics.peak_memory_mb == metrics.baseline_memory_mb


class TestBenchmarkTask:
    """Verify BenchmarkTask defines coding tasks."""

    def test_creates_minimal_task_with_required_fields(self) -> None:
        """BenchmarkTask requires id, name, system_prompt, and user_prompt."""
        task = BenchmarkTask(
            id="test-task",
            name="Test Task",
            system_prompt="You are a helpful assistant.",
            user_prompt="Write a hello world function.",
        )
        assert task.id == "test-task"
        assert task.name == "Test Task"
        assert task.difficulty == TaskDifficulty.MODERATE  # default

    def test_creates_task_with_all_optional_fields(self) -> None:
        """BenchmarkTask accepts all optional configuration."""
        task = BenchmarkTask(
            id="complex-task",
            name="Complex Task",
            system_prompt="System prompt",
            user_prompt="User prompt",
            difficulty=TaskDifficulty.COMPLEX,
            expected_output_tokens=1500,
            language="python",
            tags=["api", "async"],
        )
        assert task.difficulty == TaskDifficulty.COMPLEX
        assert task.expected_output_tokens == 1500
        assert task.language == "python"
        assert "api" in task.tags

    def test_difficulty_enum_values(self) -> None:
        """TaskDifficulty should have simple, moderate, and complex levels."""
        assert TaskDifficulty.SIMPLE.value == "simple"
        assert TaskDifficulty.MODERATE.value == "moderate"
        assert TaskDifficulty.COMPLEX.value == "complex"


class TestSingleRunResult:
    """Verify SingleRunResult captures one benchmark execution."""

    def test_creates_successful_run_result(self) -> None:
        """SingleRunResult should capture successful benchmark run."""
        timing = TimingMetrics(ttft_ms=100.0, total_latency_ms=2000.0, generation_time_ms=1900.0)
        throughput = ThroughputMetrics(
            prompt_tokens=200,
            completion_tokens=500,
            tokens_per_second=26.3,
            prompt_processing_rate=2000.0,
        )
        memory = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1500.0)

        result = SingleRunResult(
            run_id="run-001",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            timing=timing,
            throughput=throughput,
            memory=memory,
            raw_output="def hello():\n    print('Hello')",
            error=None,
        )

        assert result.run_id == "run-001"
        assert result.timing.ttft_ms == 100.0
        assert result.throughput.tokens_per_second == 26.3
        assert result.error is None

    def test_creates_failed_run_result_with_error(self) -> None:
        """SingleRunResult should capture error message on failure."""
        timing = TimingMetrics(ttft_ms=0.0, total_latency_ms=5000.0, generation_time_ms=0.0)
        throughput = ThroughputMetrics(
            prompt_tokens=200,
            completion_tokens=0,
            tokens_per_second=0.0,
            prompt_processing_rate=0.0,
        )
        memory = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1000.0)

        result = SingleRunResult(
            run_id="run-002",
            timestamp=datetime(2024, 1, 15, 10, 35, 0),
            timing=timing,
            throughput=throughput,
            memory=memory,
            raw_output="",
            error="Connection timeout after 5000ms",
        )

        assert result.error == "Connection timeout after 5000ms"
        assert result.raw_output == ""


class TestBenchmarkResult:
    """Verify BenchmarkResult aggregates multiple runs."""

    @pytest.fixture
    def sample_task(self) -> BenchmarkTask:
        """Create sample benchmark task for tests."""
        return BenchmarkTask(
            id="sample-task",
            name="Sample Task",
            system_prompt="System",
            user_prompt="User",
        )

    @pytest.fixture
    def sample_runs(self) -> list[SingleRunResult]:
        """Create sample run results for aggregation tests."""
        runs = []
        for i, (ttft, tps) in enumerate([(100, 25.0), (120, 28.0), (90, 30.0)]):
            timing = TimingMetrics(
                ttft_ms=float(ttft),
                total_latency_ms=2000.0,
                generation_time_ms=1900.0,
            )
            throughput = ThroughputMetrics(
                prompt_tokens=200,
                completion_tokens=500,
                tokens_per_second=tps,
                prompt_processing_rate=2000.0,
            )
            memory = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1500.0)
            runs.append(
                SingleRunResult(
                    run_id=f"run-{i:03d}",
                    timestamp=datetime(2024, 1, 15, 10, 30 + i, 0),
                    timing=timing,
                    throughput=throughput,
                    memory=memory,
                    raw_output=f"output-{i}",
                    error=None,
                )
            )
        return runs

    def test_creates_with_aggregated_statistics(
        self, sample_task: BenchmarkTask, sample_runs: list[SingleRunResult]
    ) -> None:
        """BenchmarkResult should compute averages from runs."""
        result = BenchmarkResult(
            benchmark_id="bench-001",
            model="mlx-community/test-model",
            task=sample_task,
            started_at=datetime(2024, 1, 15, 10, 30, 0),
            completed_at=datetime(2024, 1, 15, 10, 35, 0),
            runs=sample_runs,
        )

        assert result.model == "mlx-community/test-model"
        assert result.task.id == "sample-task"
        assert len(result.runs) == 3

        # Check computed averages: (25 + 28 + 30) / 3 = 27.67
        assert result.avg_tokens_per_second == pytest.approx(27.67, rel=0.01)
        # TTFT: (100 + 120 + 90) / 3 = 103.33
        assert result.avg_ttft_ms == pytest.approx(103.33, rel=0.01)
        # All runs successful
        assert result.success_rate == 1.0

    def test_computes_success_rate_with_failed_runs(
        self, sample_task: BenchmarkTask
    ) -> None:
        """BenchmarkResult should compute success rate from error count."""
        timing = TimingMetrics(ttft_ms=100.0, total_latency_ms=2000.0, generation_time_ms=1900.0)
        throughput = ThroughputMetrics(
            prompt_tokens=200,
            completion_tokens=500,
            tokens_per_second=25.0,
            prompt_processing_rate=2000.0,
        )
        memory = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1500.0)

        runs = [
            SingleRunResult(
                run_id="run-001",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                timing=timing,
                throughput=throughput,
                memory=memory,
                raw_output="output",
                error=None,
            ),
            SingleRunResult(
                run_id="run-002",
                timestamp=datetime(2024, 1, 15, 10, 31, 0),
                timing=timing,
                throughput=throughput,
                memory=memory,
                raw_output="",
                error="Timeout",
            ),
        ]

        result = BenchmarkResult(
            benchmark_id="bench-002",
            model="test-model",
            task=sample_task,
            started_at=datetime(2024, 1, 15, 10, 30, 0),
            completed_at=datetime(2024, 1, 15, 10, 32, 0),
            runs=runs,
        )

        # 1 out of 2 successful = 50%
        assert result.success_rate == 0.5

    def test_handles_all_failed_runs(self, sample_task: BenchmarkTask) -> None:
        """BenchmarkResult should handle case where all runs failed."""
        timing = TimingMetrics(ttft_ms=0.0, total_latency_ms=5000.0, generation_time_ms=0.0)
        throughput = ThroughputMetrics(
            prompt_tokens=200,
            completion_tokens=0,
            tokens_per_second=0.0,
            prompt_processing_rate=0.0,
        )
        memory = MemoryMetrics(baseline_memory_mb=1000.0, peak_memory_mb=1000.0)

        runs = [
            SingleRunResult(
                run_id="run-001",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                timing=timing,
                throughput=throughput,
                memory=memory,
                raw_output="",
                error="Error 1",
            ),
        ]

        result = BenchmarkResult(
            benchmark_id="bench-003",
            model="test-model",
            task=sample_task,
            started_at=datetime(2024, 1, 15, 10, 30, 0),
            completed_at=datetime(2024, 1, 15, 10, 31, 0),
            runs=runs,
        )

        assert result.success_rate == 0.0
        assert result.avg_tokens_per_second == 0.0
