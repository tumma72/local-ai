"""Behavioral tests for benchmark task system.

Tests verify public behavior of task loading:
- load_task loads task from TOML file
- load_task validates required fields
- get_builtin_tasks returns built-in task definitions
- list_tasks returns all available task IDs
"""

from pathlib import Path
from textwrap import dedent

import pytest

from local_ai.benchmark.schema import BenchmarkTask, TaskDifficulty
from local_ai.benchmark.tasks import get_builtin_tasks, list_tasks, load_task


class TestLoadTask:
    """Verify task loading from TOML files."""

    def test_loads_task_from_toml_file(self, tmp_path: Path) -> None:
        """Should parse TOML file and return BenchmarkTask."""
        task_file = tmp_path / "test_task.toml"
        task_file.write_text(dedent("""
            [task]
            id = "test-task"
            name = "Test Task"
            difficulty = "simple"
            expected_output_tokens = 500
            language = "python"
            tags = ["api", "rest"]

            [prompts]
            system = "You are a Python expert."
            user = "Write a hello world function."
        """).strip())

        task = load_task(task_file)

        assert isinstance(task, BenchmarkTask)
        assert task.id == "test-task"
        assert task.name == "Test Task"
        assert task.difficulty == TaskDifficulty.SIMPLE
        assert task.expected_output_tokens == 500
        assert task.language == "python"
        assert task.tags == ["api", "rest"]
        assert task.system_prompt == "You are a Python expert."
        assert task.user_prompt == "Write a hello world function."

    def test_loads_task_with_minimal_fields(self, tmp_path: Path) -> None:
        """Should accept TOML with only required fields."""
        task_file = tmp_path / "minimal.toml"
        task_file.write_text(dedent("""
            [task]
            id = "minimal-task"
            name = "Minimal Task"

            [prompts]
            system = "System prompt"
            user = "User prompt"
        """).strip())

        task = load_task(task_file)

        assert task.id == "minimal-task"
        assert task.difficulty == TaskDifficulty.MODERATE  # default
        assert task.expected_output_tokens == 800  # default

    def test_raises_on_missing_task_section(self, tmp_path: Path) -> None:
        """Should raise error if [task] section missing."""
        task_file = tmp_path / "invalid.toml"
        task_file.write_text(dedent("""
            [prompts]
            system = "System"
            user = "User"
        """).strip())

        with pytest.raises(ValueError, match="Missing.*task.*section"):
            load_task(task_file)

    def test_raises_on_missing_prompts_section(self, tmp_path: Path) -> None:
        """Should raise error if [prompts] section missing."""
        task_file = tmp_path / "invalid.toml"
        task_file.write_text(dedent("""
            [task]
            id = "task-1"
            name = "Task 1"
        """).strip())

        with pytest.raises(ValueError, match="Missing.*prompts.*section"):
            load_task(task_file)

    def test_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        """Should raise error if required fields missing."""
        task_file = tmp_path / "invalid.toml"
        task_file.write_text(dedent("""
            [task]
            name = "Task without ID"

            [prompts]
            system = "System"
            user = "User"
        """).strip())

        with pytest.raises(ValueError, match="id"):
            load_task(task_file)

    def test_raises_on_nonexistent_file(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_task(Path("/nonexistent/task.toml"))


class TestGetBuiltinTasks:
    """Verify built-in task retrieval."""

    def test_returns_list_of_benchmark_tasks(self) -> None:
        """Should return list of BenchmarkTask objects."""
        tasks = get_builtin_tasks()

        assert isinstance(tasks, list)
        # Should have at least one built-in task
        assert len(tasks) >= 1
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_includes_todo_api_task(self) -> None:
        """Should include the todo-api benchmark task."""
        tasks = get_builtin_tasks()
        task_ids = [t.id for t in tasks]

        assert "todo-api" in task_ids


class TestListTasks:
    """Verify task listing functionality."""

    def test_returns_task_ids(self) -> None:
        """Should return list of available task IDs."""
        task_ids = list_tasks()

        assert isinstance(task_ids, list)
        assert len(task_ids) >= 1
        assert all(isinstance(tid, str) for tid in task_ids)

    def test_includes_builtin_task_ids(self) -> None:
        """Should include IDs from built-in tasks."""
        task_ids = list_tasks()

        assert "todo-api" in task_ids
