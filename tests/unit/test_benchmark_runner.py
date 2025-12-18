"""Behavioral tests for benchmark runner module.

Tests verify public behavior of benchmark execution:
- BenchmarkRunner executes benchmark runs
- BenchmarkRunner collects metrics from streaming requests
- BenchmarkRunner aggregates results across multiple runs
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from local_ai.benchmark.runner import BenchmarkRunner
from local_ai.benchmark.schema import (
    BenchmarkResult,
    BenchmarkTask,
    ThroughputMetrics,
    TimingMetrics,
)


@pytest.fixture
def mock_timing() -> TimingMetrics:
    """Create mock timing metrics."""
    return TimingMetrics(
        ttft_ms=100.0,
        total_latency_ms=2000.0,
        generation_time_ms=1900.0,
    )


@pytest.fixture
def mock_throughput() -> ThroughputMetrics:
    """Create mock throughput metrics."""
    return ThroughputMetrics(
        prompt_tokens=50,
        completion_tokens=100,
        tokens_per_second=52.6,
        prompt_processing_rate=500.0,
    )


class TestBenchmarkRunner:
    """Verify BenchmarkRunner executes benchmarks."""

    @pytest.mark.asyncio
    async def test_executes_benchmark_with_multiple_runs(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should execute specified number of benchmark runs."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.return_value = (mock_timing, mock_throughput, "def hello(): pass")

            result = await runner.run(
                task=sample_task,
                num_requests=3,
                warmup_requests=0,
            )

        assert isinstance(result, BenchmarkResult)
        assert len(result.runs) == 3
        assert mock_measure.call_count == 3

    @pytest.mark.asyncio
    async def test_performs_warmup_runs(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should perform warmup runs that are not included in results."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.return_value = (mock_timing, mock_throughput, "output")

            result = await runner.run(
                task=sample_task,
                num_requests=3,
                warmup_requests=2,
            )

        # 2 warmup + 3 actual = 5 total calls
        assert mock_measure.call_count == 5
        # But only 3 runs in result
        assert len(result.runs) == 3

    @pytest.mark.asyncio
    async def test_captures_timing_metrics(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should capture timing metrics in each run result."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.return_value = (mock_timing, mock_throughput, "output")

            result = await runner.run(
                task=sample_task,
                num_requests=1,
            )

        assert result.runs[0].timing.ttft_ms == 100.0
        assert result.runs[0].timing.total_latency_ms == 2000.0

    @pytest.mark.asyncio
    async def test_captures_throughput_metrics(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should capture throughput metrics in each run result."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.return_value = (mock_timing, mock_throughput, "output")

            result = await runner.run(
                task=sample_task,
                num_requests=1,
            )

        assert result.runs[0].throughput.tokens_per_second == 52.6
        assert result.runs[0].throughput.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_handles_request_errors(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should capture errors in run results."""
        import httpx

        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.side_effect = httpx.ConnectError("Connection refused")

            result = await runner.run(
                task=sample_task,
                num_requests=1,
            )

        assert len(result.runs) == 1
        assert result.runs[0].error is not None
        assert "Connection" in result.runs[0].error

    @pytest.mark.asyncio
    async def test_computes_aggregated_statistics(
        self,
        sample_task: BenchmarkTask,
    ) -> None:
        """Should compute average statistics across runs."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        # Return different metrics for each call
        timings = [
            TimingMetrics(ttft_ms=100.0, total_latency_ms=2000.0, generation_time_ms=1900.0),
            TimingMetrics(ttft_ms=120.0, total_latency_ms=2200.0, generation_time_ms=2080.0),
        ]
        throughputs = [
            ThroughputMetrics(
                prompt_tokens=50, completion_tokens=100,
                tokens_per_second=52.6, prompt_processing_rate=500.0,
            ),
            ThroughputMetrics(
                prompt_tokens=50, completion_tokens=100,
                tokens_per_second=48.0, prompt_processing_rate=450.0,
            ),
        ]

        call_count = 0

        async def mock_measure(*args, **kwargs):
            nonlocal call_count
            idx = min(call_count, len(timings) - 1)
            call_count += 1
            return timings[idx], throughputs[idx], "output"

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = mock_measure

            result = await runner.run(
                task=sample_task,
                num_requests=2,
                warmup_requests=0,
            )

        # Average: (52.6 + 48.0) / 2 = 50.3
        assert result.avg_tokens_per_second == pytest.approx(50.3, rel=0.01)
        # Average TTFT: (100 + 120) / 2 = 110
        assert result.avg_ttft_ms == pytest.approx(110.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_records_benchmark_timestamps(
        self,
        sample_task: BenchmarkTask,
        mock_timing: TimingMetrics,
        mock_throughput: ThroughputMetrics,
    ) -> None:
        """Should record start and completion timestamps."""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8080,
            model="test-model",
        )

        before = datetime.now()

        with patch(
            "local_ai.benchmark.runner.measure_streaming_request",
            new_callable=AsyncMock,
        ) as mock_measure:
            mock_measure.return_value = (mock_timing, mock_throughput, "output")

            result = await runner.run(
                task=sample_task,
                num_requests=1,
            )

        after = datetime.now()

        assert before <= result.started_at <= after
        assert before <= result.completed_at <= after
        assert result.started_at <= result.completed_at
