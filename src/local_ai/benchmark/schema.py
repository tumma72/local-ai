"""Benchmark data models for local-ai.

Defines Pydantic models for benchmark configuration, execution, and results.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class TaskDifficulty(str, Enum):
    """Difficulty level for benchmark tasks."""

    SIMPLE = "simple"  # ~200-500 output tokens
    MODERATE = "moderate"  # ~500-1000 output tokens
    COMPLEX = "complex"  # ~1000-2000 output tokens


class BenchmarkMode(str, Enum):
    """Execution mode for benchmarks."""

    RAW = "raw"  # Direct API call, single shot
    AGENTIC = "agentic"  # Multi-turn with Goose recipes


class TestResults(BaseModel):
    """Results from running tests on generated code."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    passed: int = Field(ge=0, description="Number of tests passed")
    failed: int = Field(ge=0, description="Number of tests failed")
    errors: int = Field(ge=0, description="Number of tests with errors")
    skipped: int = Field(ge=0, description="Number of tests skipped")
    total: int = Field(ge=0, description="Total number of tests")
    output: str = Field(default="", description="Raw pytest output")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Percentage of tests that passed."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


class TimingMetrics(BaseModel):
    """Timing measurements for a single benchmark run."""

    ttft_ms: float = Field(ge=0, description="Time to first token in milliseconds")
    total_latency_ms: float = Field(ge=0, description="Total request time in milliseconds")
    generation_time_ms: float = Field(ge=0, description="Time spent generating after first token")


class ThroughputMetrics(BaseModel):
    """Token throughput measurements."""

    prompt_tokens: int = Field(ge=0, description="Number of input tokens")
    completion_tokens: int = Field(ge=0, description="Number of output tokens")
    tokens_per_second: float = Field(ge=0, description="Output tokens per second (primary metric)")
    prompt_processing_rate: float = Field(ge=0, description="Input tokens per second")


class MemoryMetrics(BaseModel):
    """Memory usage measurements."""

    baseline_memory_mb: float = Field(ge=0, description="Memory before inference in MB")
    peak_memory_mb: float = Field(ge=0, description="Peak memory during inference in MB")


class BenchmarkTask(BaseModel):
    """Definition of a coding benchmark task."""

    id: str = Field(description="Unique task identifier")
    name: str = Field(description="Human-readable task name")
    system_prompt: str = Field(description="System context for the model")
    user_prompt: str = Field(description="The coding task prompt")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.MODERATE, description="Task complexity level"
    )
    expected_output_tokens: int = Field(
        default=800, ge=100, le=5000, description="Approximate expected output tokens"
    )
    language: str = Field(default="python", description="Target programming language")
    tags: list[str] = Field(default_factory=list, description="Task tags for filtering")


class SingleRunResult(BaseModel):
    """Result of a single benchmark execution."""

    run_id: str = Field(description="Unique identifier for this run")
    timestamp: datetime = Field(description="When the run was executed")
    timing: TimingMetrics = Field(description="Timing measurements")
    throughput: ThroughputMetrics = Field(description="Throughput measurements")
    memory: MemoryMetrics = Field(description="Memory measurements")
    raw_output: str = Field(description="Model's generated output")
    error: str | None = Field(default=None, description="Error message if run failed")


class BenchmarkResult(BaseModel):
    """Aggregated results for a benchmark session."""

    benchmark_id: str = Field(description="Unique identifier for this benchmark session")
    model: str = Field(description="Model identifier")
    task: BenchmarkTask = Field(description="Task that was benchmarked")
    started_at: datetime = Field(description="When benchmark started")
    completed_at: datetime = Field(description="When benchmark completed")
    runs: list[SingleRunResult] = Field(description="Individual run results")
    mode: BenchmarkMode = Field(default=BenchmarkMode.RAW, description="Execution mode")
    test_results: TestResults | None = Field(
        default=None, description="Test results if task uses test validation"
    )
    working_directory: str | None = Field(
        default=None, description="Directory containing generated code"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_tokens_per_second(self) -> float:
        """Average tokens per second across successful runs."""
        successful = [r for r in self.runs if r.error is None]
        if not successful:
            return 0.0
        return sum(r.throughput.tokens_per_second for r in successful) / len(successful)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_ttft_ms(self) -> float:
        """Average time to first token across successful runs."""
        successful = [r for r in self.runs if r.error is None]
        if not successful:
            return 0.0
        return sum(r.timing.ttft_ms for r in successful) / len(successful)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Percentage of runs that completed without error."""
        if not self.runs:
            return 0.0
        successful = sum(1 for r in self.runs if r.error is None)
        return successful / len(self.runs)
