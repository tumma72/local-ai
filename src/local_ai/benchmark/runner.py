"""Benchmark execution orchestration.

Coordinates benchmark runs, collects metrics, and aggregates results.
"""

import uuid
from datetime import datetime

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.benchmark.metrics import MemoryTracker, measure_streaming_request
from local_ai.benchmark.schema import (
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    ThroughputMetrics,
    TimingMetrics,
)
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.runner")


class BenchmarkRunner:
    """Orchestrates benchmark execution against a model."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = "",
    ) -> None:
        """Initialize benchmark runner.

        Args:
            host: Server host address.
            port: Server port.
            model: Model identifier.
        """
        self._host = host
        self._port = port
        self._model = model

    async def run(
        self,
        task: BenchmarkTask,
        num_requests: int = 3,
        warmup_requests: int = 1,
        timeout: float = 300.0,
    ) -> BenchmarkResult:
        """Execute benchmark runs for a task.

        Args:
            task: The benchmark task to execute.
            num_requests: Number of benchmark requests to make.
            warmup_requests: Number of warmup requests (not counted).
            timeout: Request timeout in seconds.

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        benchmark_id = str(uuid.uuid4())[:8]
        started_at = datetime.now()

        _logger.info(
            "Starting benchmark '{}' with {} runs (+{} warmup)",
            task.id,
            num_requests,
            warmup_requests,
        )

        # Prepare messages for the task
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]

        # Warmup runs
        for i in range(warmup_requests):
            _logger.debug("Warmup run {}/{}", i + 1, warmup_requests)
            try:
                await measure_streaming_request(
                    host=self._host,
                    port=self._port,
                    model=self._model,
                    messages=messages,
                    timeout=timeout,
                )
            except Exception as e:
                _logger.warning("Warmup run {} failed: {}", i + 1, e)

        # Actual benchmark runs
        runs: list[SingleRunResult] = []
        memory_tracker = MemoryTracker()

        for i in range(num_requests):
            run_id = f"{benchmark_id}-{i:03d}"
            timestamp = datetime.now()

            _logger.info("Benchmark run {}/{}", i + 1, num_requests)
            memory_tracker.start()

            try:
                timing, throughput, output = await measure_streaming_request(
                    host=self._host,
                    port=self._port,
                    model=self._model,
                    messages=messages,
                    timeout=timeout,
                )

                memory_tracker.sample()
                memory = memory_tracker.get_metrics()

                run_result = SingleRunResult(
                    run_id=run_id,
                    timestamp=timestamp,
                    timing=timing,
                    throughput=throughput,
                    memory=memory,
                    raw_output=output,
                    error=None,
                )

                _logger.info(
                    "Run {} complete: {:.1f} tok/s, TTFT={:.0f}ms",
                    i + 1,
                    throughput.tokens_per_second,
                    timing.ttft_ms,
                )

            except Exception as e:
                _logger.error("Run {} failed: {}", i + 1, e)

                # Create a failed result with zero metrics
                run_result = SingleRunResult(
                    run_id=run_id,
                    timestamp=timestamp,
                    timing=TimingMetrics(
                        ttft_ms=0.0,
                        total_latency_ms=0.0,
                        generation_time_ms=0.0,
                    ),
                    throughput=ThroughputMetrics(
                        prompt_tokens=0,
                        completion_tokens=0,
                        tokens_per_second=0.0,
                        prompt_processing_rate=0.0,
                    ),
                    memory=MemoryMetrics(
                        baseline_memory_mb=0.0,
                        peak_memory_mb=0.0,
                    ),
                    raw_output="",
                    error=str(e),
                )

            runs.append(run_result)

        completed_at = datetime.now()

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            model=self._model,
            task=task,
            started_at=started_at,
            completed_at=completed_at,
            runs=runs,
        )

        _logger.info(
            "Benchmark complete: avg {:.1f} tok/s, {:.0f}ms TTFT, {:.0%} success",
            result.avg_tokens_per_second,
            result.avg_ttft_ms,
            result.success_rate,
        )

        return result
