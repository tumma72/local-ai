"""Benchmark module for local-ai.

Provides tools for benchmarking LLM model performance.
"""

from local_ai.benchmark.agentic_runner import (
    run_agentic_benchmark,
    run_and_save_agentic_benchmark,
)
from local_ai.benchmark.goose_runner import (
    GooseResult,
    get_goose_output_dir,
    get_recipe_path,
    list_available_recipes,
    run_goose_command,
    run_goose_recipe,
)
from local_ai.benchmark.metrics import MemoryTracker, measure_streaming_request
from local_ai.benchmark.reporter import BenchmarkReporter
from local_ai.benchmark.runner import BenchmarkRunner
from local_ai.benchmark.schema import (
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkTask,
    MemoryMetrics,
    SingleRunResult,
    TaskDifficulty,
    TestResults,
    ThroughputMetrics,
    TimingMetrics,
)
from local_ai.benchmark.tasks import (
    get_builtin_tasks,
    get_task_by_id,
    list_tasks,
    load_task,
)
from local_ai.benchmark.test_validator import run_pytest, validate_tdd_output

__all__ = [
    "BenchmarkMode",
    "BenchmarkReporter",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "GooseResult",
    "MemoryMetrics",
    "MemoryTracker",
    "SingleRunResult",
    "TaskDifficulty",
    "TestResults",
    "ThroughputMetrics",
    "TimingMetrics",
    "get_builtin_tasks",
    "get_goose_output_dir",
    "get_recipe_path",
    "get_task_by_id",
    "list_available_recipes",
    "list_tasks",
    "load_task",
    "measure_streaming_request",
    "run_agentic_benchmark",
    "run_and_save_agentic_benchmark",
    "run_goose_command",
    "run_goose_recipe",
    "run_pytest",
    "validate_tdd_output",
]
