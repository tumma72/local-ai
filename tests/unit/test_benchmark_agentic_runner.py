"""Behavioral tests for agentic benchmark runner.

Tests verify public behavior of agentic benchmark execution:
- run_agentic_benchmark() executes multi-turn workflows
- run_agentic_benchmark() validates results through tests
- run_and_save_agentic_benchmark() saves results automatically

Tests mock external dependencies (Goose, file system) for isolation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_ai.benchmark.agentic_runner import (
    run_agentic_benchmark,
    run_and_save_agentic_benchmark,
)
from local_ai.benchmark.goose_runner import GooseResult
from local_ai.benchmark.schema import (
    BenchmarkMode,
    BenchmarkTask,
    TaskDifficulty,
    TestResults,
)


class TestAgenticBenchmarkRunner:
    """Test agentic benchmark runner functionality."""

    @pytest.fixture
    def sample_task(self) -> BenchmarkTask:
        """Create a sample benchmark task."""
        return BenchmarkTask(
            id="test-agentic-task",
            name="Test Agentic Task",
            system_prompt="You are an AI assistant.",
            user_prompt="Create a Python function that calculates Fibonacci numbers.",
            difficulty=TaskDifficulty.MODERATE,
            expected_output_tokens=200,
        )

    @patch('local_ai.benchmark.agentic_runner.get_recipe_path')
    @patch('local_ai.benchmark.agentic_runner.get_goose_output_dir')
    @patch('local_ai.benchmark.agentic_runner.run_goose_recipe')
    @patch('local_ai.benchmark.agentic_runner.validate_tdd_output')
    def test_run_agentic_benchmark_success(
        self,
        mock_validate_tdd_output: MagicMock,
        mock_run_goose_recipe: MagicMock,
        mock_get_goose_output_dir: MagicMock,
        mock_get_recipe_path: MagicMock,
        sample_task: BenchmarkTask,
    ) -> None:
        """Test successful agentic benchmark execution."""
        # Mock recipe path
        mock_recipe_path = Path("/fake/recipes/test_agentic_task.yaml")
        mock_get_recipe_path.return_value = mock_recipe_path

        # Mock working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "working"
            mock_get_goose_output_dir.return_value = working_dir

            # Mock Goose recipe result
            mock_goose_result = GooseResult(
                output="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                elapsed_ms=15000.0,
                success=True,
                working_directory=working_dir,
                error=None,
                recipe_used="test_agentic_task",
                turns_taken=5,
                files_created=[],
            )
            mock_run_goose_recipe.return_value = mock_goose_result

            # Mock test validation
            mock_test_results = TestResults(
                passed=3,
                failed=0,
                errors=0,
                skipped=0,
                total=3,
                output="All tests passed",
            )
            mock_validate_tdd_output.return_value = mock_test_results

            # Run the benchmark
            result = run_agentic_benchmark(
                model="test-model",
                task=sample_task,
                host="127.0.0.1",
                port=8080,
                timeout=600.0,
                max_turns=20,
                output_dir=Path(temp_dir),
                run_tests=True,
            )

            # Verify result structure
            assert result.benchmark_id.startswith("agentic_")
            assert result.model == "test-model"
            assert result.task == sample_task
            assert result.mode == BenchmarkMode.AGENTIC
            assert len(result.runs) == 1

            # Verify run details
            run = result.runs[0]
            assert run.error is None  # Success is indicated by no error
            assert run.timing.total_latency_ms == 15000.0
            assert run.timing.generation_time_ms == 15000.0
            assert run.raw_output == mock_goose_result.output

            # Verify test results
            assert result.test_results == mock_test_results
            assert result.working_directory == str(working_dir)

            # Verify timestamps
            assert isinstance(result.started_at, datetime)
            assert isinstance(result.completed_at, datetime)

    @patch('local_ai.benchmark.agentic_runner.get_recipe_path')
    @patch('local_ai.benchmark.agentic_runner.get_goose_output_dir')
    @patch('local_ai.benchmark.agentic_runner.run_goose_recipe')
    def test_run_agentic_benchmark_failure(
        self,
        mock_run_goose_recipe: MagicMock,
        mock_get_goose_output_dir: MagicMock,
        mock_get_recipe_path: MagicMock,
        sample_task: BenchmarkTask,
    ) -> None:
        """Test agentic benchmark execution with recipe failure."""
        # Mock recipe path
        mock_recipe_path = Path("/fake/recipes/test_agentic_task.yaml")
        mock_get_recipe_path.return_value = mock_recipe_path

        # Mock working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "working"
            mock_get_goose_output_dir.return_value = working_dir

            # Mock Goose recipe failure
            mock_goose_result = GooseResult(
                success=False,
                output="",
                elapsed_ms=5000.0,
                turns_taken=2,
                error="Recipe execution failed: timeout",
            )
            mock_run_goose_recipe.return_value = mock_goose_result

            # Run the benchmark
            result = run_agentic_benchmark(
                model="test-model",
                task=sample_task,
                host="127.0.0.1",
                port=8080,
                timeout=600.0,
                max_turns=20,
                output_dir=Path(temp_dir),
                run_tests=True,
            )

            # Verify failure handling
            assert result.benchmark_id.startswith("agentic_")
            assert len(result.runs) == 1

            run = result.runs[0]
            assert run.error is not None
            assert run.error == "Recipe execution failed: timeout"
            assert run.raw_output == ""

            # Test validation should not run on failure
            assert result.test_results is None

    @patch('local_ai.benchmark.agentic_runner.get_recipe_path')
    @patch('local_ai.benchmark.agentic_runner.get_goose_output_dir')
    @patch('local_ai.benchmark.agentic_runner.run_goose_recipe')
    @patch('local_ai.benchmark.agentic_runner.validate_tdd_output')
    def test_run_agentic_benchmark_without_tests(
        self,
        mock_validate_tdd_output: MagicMock,
        mock_run_goose_recipe: MagicMock,
        mock_get_goose_output_dir: MagicMock,
        mock_get_recipe_path: MagicMock,
        sample_task: BenchmarkTask,
    ) -> None:
        """Test agentic benchmark execution without test validation."""
        # Mock recipe path
        mock_recipe_path = Path("/fake/recipes/test_agentic_task.yaml")
        mock_get_recipe_path.return_value = mock_recipe_path

        # Mock working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "working"
            mock_get_goose_output_dir.return_value = working_dir

            # Mock successful Goose execution
            mock_goose_result = GooseResult(
                success=True,
                output="def test_function(): pass",
                elapsed_ms=10000.0,
                turns_taken=3,
                error=None,
            )
            mock_run_goose_recipe.return_value = mock_goose_result

            # Run without tests
            result = run_agentic_benchmark(
                model="test-model",
                task=sample_task,
                host="127.0.0.1",
                port=8080,
                timeout=600.0,
                max_turns=20,
                output_dir=Path(temp_dir),
                run_tests=False,  # Disable test validation
            )

            # Verify test validation was not called
            mock_validate_tdd_output.assert_not_called()
            assert result.test_results is None

    @patch('local_ai.benchmark.agentic_runner.get_recipe_path')
    @patch('local_ai.benchmark.agentic_runner.get_goose_output_dir')
    @patch('local_ai.benchmark.agentic_runner.run_goose_recipe')
    @patch('local_ai.benchmark.agentic_runner.validate_tdd_output')
    @patch('local_ai.benchmark.agentic_runner.BenchmarkReporter')
    def test_run_and_save_agentic_benchmark(
        self,
        mock_reporter: MagicMock,
        mock_validate_tdd_output: MagicMock,
        mock_run_goose_recipe: MagicMock,
        mock_get_goose_output_dir: MagicMock,
        mock_get_recipe_path: MagicMock,
        sample_task: BenchmarkTask,
    ) -> None:
        """Test agentic benchmark execution with automatic saving."""
        # Mock recipe path
        mock_recipe_path = Path("/fake/recipes/test_agentic_task.yaml")
        mock_get_recipe_path.return_value = mock_recipe_path

        # Mock working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "working"
            mock_get_goose_output_dir.return_value = working_dir

            # Mock successful execution
            mock_goose_result = GooseResult(
                success=True,
                output="def solution(): pass",
                elapsed_ms=12000.0,
                turns_taken=4,
                error=None,
            )
            mock_run_goose_recipe.return_value = mock_goose_result

            # Mock test results
            mock_test_results = TestResults(
                passed=2,
                failed=1,
                errors=0,
                skipped=0,
                total=3,
                output="2/3 tests passed",
            )
            mock_validate_tdd_output.return_value = mock_test_results

            # Mock reporter save
            mock_save_path = Path(temp_dir) / "result.json"
            mock_reporter.return_value.save.return_value = mock_save_path

            # Run and save
            result, save_path = run_and_save_agentic_benchmark(
                model="test-model",
                task=sample_task,
                host="127.0.0.1",
                port=8080,
                timeout=600.0,
                max_turns=20,
                output_dir=Path(temp_dir),
                run_tests=True,
            )

            # Verify result
            assert result.benchmark_id.startswith("agentic_")
            assert result.model == "test-model"
            assert result.test_results == mock_test_results

            # Verify saving
            assert save_path == mock_save_path
            mock_reporter.return_value.save.assert_called_once()

    def test_recipe_name_fallback_logic(self) -> None:
        """Test recipe name fallback when primary recipe not found."""
        with patch('local_ai.benchmark.agentic_runner.get_recipe_path') as mock_get_recipe_path:
            # Mock primary recipe not found, but alternate exists
            mock_get_recipe_path.side_effect = [
                Path("/fake/recipes/nonexistent.yaml"),  # First call (task.id with -)
                Path("/fake/recipes/test_agentic_task.yaml"),  # Second call (task.id)
            ]

            task = BenchmarkTask(
                id="test-agentic-task",  # Contains hyphens
                name="Test",
                system_prompt="System",
                user_prompt="Prompt",
                difficulty=TaskDifficulty.SIMPLE,
                expected_output_tokens=100,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('local_ai.benchmark.agentic_runner.get_goose_output_dir') as mock_dir:
                    with patch('local_ai.benchmark.agentic_runner.run_goose_recipe') as mock_run:
                        with patch('local_ai.benchmark.agentic_runner.validate_tdd_output') as mock_validate:
                            mock_dir.return_value = Path(temp_dir)
                            mock_run.return_value = GooseResult(
                                success=True,
                                output="result",
                                elapsed_ms=1000.0,
                                turns_taken=1,
                                error=None,
                            )
                            mock_validate.return_value = TestResults(
                                passed=1, failed=0, errors=0, skipped=0, total=1, output="ok"
                            )

                            result = run_agentic_benchmark(
                                model="test-model",
                                task=task,
                                output_dir=Path(temp_dir),
                            )

                            # Should have tried both recipe paths
                            assert mock_get_recipe_path.call_count == 2
                            # Should have used the second (alternate) path
                            assert result.runs[0].error is None  # Success indicated by no error
