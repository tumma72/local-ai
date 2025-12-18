"""Behavioral tests for benchmark CLI commands.

Tests verify public behavior of benchmark CLI:
- benchmark tasks lists available tasks
- benchmark run executes benchmark on a task
- benchmark compare shows comparison table
"""

from typer.testing import CliRunner

from local_ai.cli.main import app

runner = CliRunner()


class TestBenchmarkTasks:
    """Verify benchmark tasks command."""

    def test_lists_available_tasks(self) -> None:
        """Should list all available benchmark tasks."""
        result = runner.invoke(app, ["benchmark", "tasks"])

        assert result.exit_code == 0
        assert "todo-api" in result.stdout

    def test_shows_task_details(self) -> None:
        """Should show task names and difficulties."""
        result = runner.invoke(app, ["benchmark", "tasks"])

        assert result.exit_code == 0
        # Should show task name
        assert "REST API" in result.stdout or "Todo" in result.stdout


class TestBenchmarkRun:
    """Verify benchmark run command."""

    def test_requires_model_option(self) -> None:
        """Should require --model option."""
        result = runner.invoke(app, ["benchmark", "run", "--task", "todo-api"])

        # Should fail without model
        assert result.exit_code != 0

    def test_requires_task_option(self) -> None:
        """Should require --task option."""
        result = runner.invoke(app, ["benchmark", "run", "--model", "test-model"])

        # Should fail without task
        assert result.exit_code != 0

    def test_shows_error_for_unknown_task(self) -> None:
        """Should show error for invalid task ID."""
        result = runner.invoke(
            app, ["benchmark", "run", "--model", "test", "--task", "nonexistent-task"]
        )

        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "unknown" in result.stdout.lower()


class TestBenchmarkCompare:
    """Verify benchmark compare command."""

    def test_compare_command_exists(self) -> None:
        """Should have compare subcommand."""
        result = runner.invoke(app, ["benchmark", "compare", "--help"])

        assert result.exit_code == 0
        assert "compare" in result.stdout.lower() or "benchmark" in result.stdout.lower()


class TestBenchmarkGoose:
    """Verify benchmark goose command."""

    def test_goose_requires_model_option(self) -> None:
        """Should require --model option."""
        result = runner.invoke(app, ["benchmark", "goose", "--task", "todo-api"])

        # Should fail without model
        assert result.exit_code != 0

    def test_goose_requires_task_option(self) -> None:
        """Should require --task option."""
        result = runner.invoke(app, ["benchmark", "goose", "--model", "test-model"])

        # Should fail without task
        assert result.exit_code != 0

    def test_goose_shows_error_for_unknown_task(self) -> None:
        """Should show error for invalid task ID."""
        result = runner.invoke(
            app, ["benchmark", "goose", "--model", "test", "--task", "nonexistent-task"]
        )

        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()

    def test_goose_command_help_shows_description(self) -> None:
        """Should show command description in help."""
        result = runner.invoke(app, ["benchmark", "goose", "--help"])

        assert result.exit_code == 0
        assert "goose" in result.stdout.lower() or "agentic" in result.stdout.lower()
