"""Comprehensive behavioral tests for CLI benchmark commands.

Tests verify public CLI interface behavior for benchmarking:
- `local-ai benchmark tasks` - List available benchmark tasks
- `local-ai benchmark run` - Execute benchmark on a model
- `local-ai benchmark compare` - Compare benchmark results

Tests mock external dependencies for isolation and focus on CLI behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_ai.benchmark.schema import BenchmarkTask, TaskDifficulty
from local_ai.cli.main import app


class TestBenchmarkCLI:
    """Test benchmark CLI functionality."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Provide a Typer CLI test runner."""
        return CliRunner()

    def test_benchmark_tasks_help_shows_usage(self, cli_runner: CliRunner) -> None:
        """Test benchmark tasks help command shows usage information."""
        result = cli_runner.invoke(app, ["benchmark", "tasks", "--help"])

        assert result.exit_code == 0
        assert "List available benchmark tasks" in result.output
        assert "--log-level" in result.output

    @patch('local_ai.cli.benchmark.get_builtin_tasks')
    def test_benchmark_tasks_with_no_tasks(
        self,
        mock_get_builtin_tasks: MagicMock,
        cli_runner: CliRunner
    ) -> None:
        """Test benchmark tasks command when no tasks are available."""
        mock_get_builtin_tasks.return_value = []

        result = cli_runner.invoke(app, ["benchmark", "tasks"])

        assert result.exit_code == 0
        assert "No benchmark tasks found" in result.output

    @patch('local_ai.cli.benchmark.get_builtin_tasks')
    def test_benchmark_tasks_with_multiple_tasks(
        self,
        mock_get_builtin_tasks: MagicMock,
        cli_runner: CliRunner
    ) -> None:
        """Test benchmark tasks command displays available tasks."""
        mock_tasks = [
            BenchmarkTask(
                id="task-1",
                name="Task One",
                system_prompt="System",
                user_prompt="Prompt 1",
                difficulty=TaskDifficulty.SIMPLE,
                expected_output_tokens=100,
            ),
            BenchmarkTask(
                id="task-2",
                name="Task Two",
                system_prompt="System",
                user_prompt="Prompt 2",
                difficulty=TaskDifficulty.MODERATE,
                expected_output_tokens=200,
            ),
        ]
        mock_get_builtin_tasks.return_value = mock_tasks

        result = cli_runner.invoke(app, ["benchmark", "tasks"])

        assert result.exit_code == 0
        assert "Available Benchmark Tasks" in result.output
        assert "task-1" in result.output
        assert "task-2" in result.output

    def test_benchmark_run_help_shows_usage(self, cli_runner: CliRunner) -> None:
        """Test benchmark run help command shows usage information."""
        result = cli_runner.invoke(app, ["benchmark", "run", "--help"])

        assert result.exit_code == 0
        assert "Run benchmark on a model with specified task" in result.output
        assert "--model" in result.output
        assert "--task" in result.output

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.BenchmarkRunner')
    @patch('local_ai.cli.benchmark.BenchmarkReporter')
    def test_benchmark_run_success(
        self,
        mock_reporter: MagicMock,
        mock_runner: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark run command executes successfully."""
        # Mock task retrieval
        mock_task = BenchmarkTask(
            id="test-task",
            name="Test Task",
            system_prompt="System",
            user_prompt="Prompt",
            difficulty=TaskDifficulty.SIMPLE,
            expected_output_tokens=100,
        )
        mock_get_task_by_id.return_value = mock_task

        # Mock benchmark execution - needs to return a coroutine since runner.run is async
        mock_result = MagicMock()
        mock_result.model = "test-model"
        mock_result.task = mock_task
        mock_result.runs = [MagicMock()]
        mock_result.avg_tokens_per_second = 25.0
        mock_result.avg_ttft_ms = 150.0
        mock_result.success_rate = 1.0

        async def mock_run(*args, **kwargs):
            return mock_result

        # Create a MagicMock for the run method that returns our coroutine
        mock_run_method = MagicMock(side_effect=mock_run)
        mock_runner.return_value.run = mock_run_method

        # Mock reporter
        mock_save_path = Path("/fake/result.json")
        mock_reporter.return_value.save.return_value = mock_save_path

        result = cli_runner.invoke(app, [
            "benchmark", "run",
            "--model", "test-model",
            "--task", "test-task",
            "--requests", "3",
        ])

        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        assert result.exit_code == 0
        assert "Benchmark Results" in result.output
        assert "test-model" in result.output

        # Verify task was retrieved
        mock_get_task_by_id.assert_called_once_with("test-task")

        # Verify benchmark was executed
        mock_runner.return_value.run.assert_called_once()

        # Verify result was saved
        mock_reporter.return_value.save.assert_called_once()

    @patch('local_ai.cli.benchmark.get_task_by_id')
    def test_benchmark_run_with_invalid_task(
        self,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark run command handles invalid task."""
        mock_get_task_by_id.return_value = None

        result = cli_runner.invoke(app, [
            "benchmark", "run",
            "--model", "test-model",
            "--task", "nonexistent-task",
        ])

        assert result.exit_code == 1
        assert "Task not found" in result.output

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.BenchmarkRunner')
    def test_benchmark_run_with_execution_failure(
        self,
        mock_runner: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark run command handles execution failure."""
        # Mock task retrieval
        mock_task = BenchmarkTask(
            id="test-task",
            name="Test Task",
            system_prompt="System",
            user_prompt="Prompt",
            difficulty=TaskDifficulty.SIMPLE,
            expected_output_tokens=100,
        )
        mock_get_task_by_id.return_value = mock_task

        # Mock benchmark execution failure
        mock_runner.return_value.execute.side_effect = RuntimeError("Benchmark failed")

        result = cli_runner.invoke(app, [
            "benchmark", "run",
            "--model", "test-model",
            "--task", "test-task",
        ])

        assert result.exit_code == 1
        assert "Benchmark failed" in result.output

    def test_benchmark_compare_help_shows_usage(self, cli_runner: CliRunner) -> None:
        """Test benchmark compare help command shows usage information."""
        result = cli_runner.invoke(app, ["benchmark", "compare", "--help"])

        assert result.exit_code == 0
        assert "Compare benchmark results" in result.output

    @patch('local_ai.cli.benchmark.BenchmarkReporter')
    def test_benchmark_compare_with_no_results(
        self,
        mock_reporter: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark compare command when no results are available."""
        # Mock the print_comparison_table method to print the expected message
        mock_reporter.return_value.print_comparison_table = MagicMock()

        result = cli_runner.invoke(app, ["benchmark", "compare"])

        assert result.exit_code == 0
        # The compare command should work without errors
        mock_reporter.return_value.print_comparison_table.assert_called_once()

    @patch('local_ai.cli.benchmark.BenchmarkReporter')
    def test_benchmark_compare_with_results(
        self,
        mock_reporter: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark compare command displays comparison table."""
        # Mock the print_comparison_table method
        mock_reporter.return_value.print_comparison_table = MagicMock()

        result = cli_runner.invoke(app, ["benchmark", "compare"])

        assert result.exit_code == 0
        # The compare command should work without errors
        mock_reporter.return_value.print_comparison_table.assert_called_once()

    def test_benchmark_help_shows_subcommands(self, cli_runner: CliRunner) -> None:
        """Test benchmark help shows available subcommands."""
        result = cli_runner.invoke(app, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "Benchmark local LLM models" in result.output
        assert "tasks" in result.output
        assert "run" in result.output
        assert "compare" in result.output

    @patch('local_ai.cli.benchmark.get_builtin_tasks')
    def test_benchmark_tasks_with_log_level_option(
        self,
        mock_get_builtin_tasks: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark tasks command with log level option."""
        mock_get_builtin_tasks.return_value = []

        result = cli_runner.invoke(app, ["benchmark", "tasks", "--log-level", "DEBUG"])

        assert result.exit_code == 0
        # Should work without errors with different log level

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.BenchmarkRunner')
    @patch('local_ai.cli.benchmark.BenchmarkReporter')
    def test_benchmark_run_with_multiple_runs(
        self,
        mock_reporter: MagicMock,
        mock_runner: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Test benchmark run command with multiple runs."""
        # Mock task retrieval
        mock_task = BenchmarkTask(
            id="test-task",
            name="Test Task",
            system_prompt="System",
            user_prompt="Prompt",
            difficulty=TaskDifficulty.SIMPLE,
            expected_output_tokens=100,
        )
        mock_get_task_by_id.return_value = mock_task

        # Mock benchmark execution - needs to return a coroutine since runner.run is async
        mock_result = MagicMock()
        mock_result.model = "test-model"
        mock_result.task = mock_task
        mock_result.runs = [MagicMock() for _ in range(5)]  # 5 runs
        mock_result.avg_tokens_per_second = 30.0
        mock_result.avg_ttft_ms = 180.0
        mock_result.success_rate = 0.8

        async def mock_run(*args, **kwargs):
            return mock_result

        # Create a MagicMock for the run method that returns our coroutine
        mock_run_method = MagicMock(side_effect=mock_run)
        mock_runner.return_value.run = mock_run_method

        # Mock reporter
        mock_save_path = Path("/fake/result.json")
        mock_reporter.return_value.save.return_value = mock_save_path

        result = cli_runner.invoke(app, [
            "benchmark", "run",
            "--model", "test-model",
            "--task", "test-task",
            "--requests", "5",
        ])

        assert result.exit_code == 0
        assert "Total Runs" in result.output
        assert "5" in result.output  # Should show 5 total runs


class TestBenchmarkGooseCommand:
    """Tests for benchmark goose command success path (lines 196-242)."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Provide a Typer CLI test runner."""
        return CliRunner()

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.get_goose_output_dir')
    @patch('local_ai.cli.benchmark.run_goose_command')
    def test_goose_command_success_path(
        self,
        mock_run_goose: MagicMock,
        mock_get_output_dir: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test goose command executes successfully with valid model and task.

        Covers lines 196-242: the success path including output directory setup,
        panel display, goose execution, and results table display.
        """
        # Mock task retrieval
        mock_task = BenchmarkTask(
            id="todo-api",
            name="TODO REST API",
            system_prompt="You are a Python expert.",
            user_prompt="Create a REST API for todo management.",
            difficulty=TaskDifficulty.MODERATE,
            expected_output_tokens=500,
        )
        mock_get_task_by_id.return_value = mock_task

        # Mock output directory
        working_dir = tmp_path / "goose_test-model" / "todo-api"
        mock_get_output_dir.return_value = working_dir

        # Mock successful goose execution
        from local_ai.benchmark.goose_runner import GooseResult
        mock_goose_result = GooseResult(
            output="def create_todo():\n    pass\n# Created todo API",
            elapsed_ms=5000.0,
            success=True,
            working_directory=working_dir,
        )
        mock_run_goose.return_value = mock_goose_result

        result = cli_runner.invoke(app, [
            "benchmark", "goose",
            "--model", "test-model",
            "--task", "todo-api",
        ])

        assert result.exit_code == 0
        # Verify goose was called with correct parameters
        mock_run_goose.assert_called_once()
        call_kwargs = mock_run_goose.call_args
        assert call_kwargs.kwargs['model'] == 'test-model'
        assert 'prompt' in call_kwargs.kwargs
        # Output should contain results table
        assert "Goose Results" in result.output
        assert "Elapsed Time" in result.output
        assert "5000" in result.output  # elapsed time in ms
        assert "Success" in result.output

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.get_goose_output_dir')
    @patch('local_ai.cli.benchmark.run_goose_command')
    def test_goose_command_failure_path(
        self,
        mock_run_goose: MagicMock,
        mock_get_output_dir: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test goose command handles execution failure gracefully.

        Covers line 222-224: error handling when goose run fails.
        """
        # Mock task retrieval
        mock_task = BenchmarkTask(
            id="todo-api",
            name="TODO REST API",
            system_prompt="System",
            user_prompt="User",
            difficulty=TaskDifficulty.MODERATE,
            expected_output_tokens=500,
        )
        mock_get_task_by_id.return_value = mock_task

        # Mock output directory
        mock_get_output_dir.return_value = tmp_path / "goose_test" / "task"

        # Mock failed goose execution
        from local_ai.benchmark.goose_runner import GooseResult
        mock_goose_result = GooseResult(
            output="",
            elapsed_ms=1000.0,
            success=False,
            error="Goose CLI timed out",
        )
        mock_run_goose.return_value = mock_goose_result

        result = cli_runner.invoke(app, [
            "benchmark", "goose",
            "--model", "test-model",
            "--task", "todo-api",
        ])

        assert result.exit_code == 1
        assert "Goose run failed" in result.output
        assert "timed out" in result.output

    @patch('local_ai.cli.benchmark.get_task_by_id')
    @patch('local_ai.cli.benchmark.get_goose_output_dir')
    @patch('local_ai.cli.benchmark.run_goose_command')
    def test_goose_command_with_long_output_preview(
        self,
        mock_run_goose: MagicMock,
        mock_get_output_dir: MagicMock,
        mock_get_task_by_id: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test goose command truncates long output in preview.

        Covers lines 239-242: output preview truncation.
        """
        mock_task = BenchmarkTask(
            id="todo-api",
            name="TODO REST API",
            system_prompt="System",
            user_prompt="User",
            difficulty=TaskDifficulty.MODERATE,
            expected_output_tokens=500,
        )
        mock_get_task_by_id.return_value = mock_task
        mock_get_output_dir.return_value = tmp_path / "goose_test" / "task"

        # Mock goose with long output (>500 chars)
        from local_ai.benchmark.goose_runner import GooseResult
        long_output = "x" * 1000  # 1000 character output
        mock_goose_result = GooseResult(
            output=long_output,
            elapsed_ms=3000.0,
            success=True,
            working_directory=tmp_path,
        )
        mock_run_goose.return_value = mock_goose_result

        result = cli_runner.invoke(app, [
            "benchmark", "goose",
            "--model", "test-model",
            "--task", "todo-api",
        ])

        assert result.exit_code == 0
        # Preview should be truncated with ...
        assert "..." in result.output
        # Should show "Output Preview" panel
        assert "Output Preview" in result.output
