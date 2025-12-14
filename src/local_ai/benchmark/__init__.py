"""Benchmark module for local-ai.

Provides tools for benchmarking LLM model performance.
"""

from local_ai.benchmark.goose_runner import (
    GooseResult,
    get_goose_output_dir,
    run_goose_command,
)
from local_ai.benchmark.metrics import MemoryTracker, measure_streaming_request
from local_ai.benchmark.reporter import BenchmarkReporter
from local_ai.benchmark.runner import BenchmarkRunner
from local_ai.benchmark.schema import (
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    TaskDifficulty,
    ThroughputMetrics,
    TimingMetrics,
)
from local_ai.benchmark.tasks import (
    get_builtin_tasks,
    get_task_by_id,
    list_tasks,
    load_task,
)

__all__ = [
    "BenchmarkReporter",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "GooseResult",
    "MemoryMetrics",
    "MemoryTracker",
    "SingleRunResult",
    "TaskDifficulty",
    "ThroughputMetrics",
    "TimingMetrics",
    "get_builtin_tasks",
    "get_goose_output_dir",
    "get_task_by_id",
    "list_tasks",
    "load_task",
    "measure_streaming_request",
    "run_goose_command",
]
