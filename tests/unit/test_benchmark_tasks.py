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


class TestLoadTaskMissingPromptFields:
    """Verify load_task handles missing prompt fields (lines 59-60)."""

    def test_raises_on_missing_system_prompt(self, tmp_path: Path) -> None:
        """Should raise error if system prompt is missing.

        Covers lines 59-60: missing 'system' field in [prompts] section.
        """
        task_file = tmp_path / "missing_system.toml"
        task_file.write_text(dedent("""
            [task]
            id = "test-task"
            name = "Test Task"

            [prompts]
            user = "User prompt only"
        """).strip())

        with pytest.raises(ValueError, match="system"):
            load_task(task_file)

    def test_raises_on_missing_user_prompt(self, tmp_path: Path) -> None:
        """Should raise error if user prompt is missing.

        Covers lines 59-60: missing 'user' field in [prompts] section.
        """
        task_file = tmp_path / "missing_user.toml"
        task_file.write_text(dedent("""
            [task]
            id = "test-task"
            name = "Test Task"

            [prompts]
            system = "System prompt only"
        """).strip())

        with pytest.raises(ValueError, match="user"):
            load_task(task_file)


class TestGetBuiltinTasksEdgeCases:
    """Verify get_builtin_tasks handles edge cases (lines 91-92, 98-99)."""

    def test_returns_empty_list_when_tasks_dir_missing(self, tmp_path: Path) -> None:
        """Should return empty list when built-in tasks directory does not exist.

        Covers lines 91-92: BUILTIN_TASKS_DIR.exists() returns False.
        """
        from local_ai.benchmark import tasks as tasks_module

        # Temporarily replace BUILTIN_TASKS_DIR with non-existent path
        original_dir = tasks_module.BUILTIN_TASKS_DIR
        try:
            tasks_module.BUILTIN_TASKS_DIR = tmp_path / "nonexistent"

            result = tasks_module.get_builtin_tasks()

            assert result == []
        finally:
            tasks_module.BUILTIN_TASKS_DIR = original_dir

    def test_handles_invalid_toml_file_gracefully(self, tmp_path: Path) -> None:
        """Should skip invalid TOML files and continue loading others.

        Covers lines 98-99: exception handling during task loading.
        """
        from local_ai.benchmark import tasks as tasks_module

        # Create a directory with one valid and one invalid TOML file
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        # Invalid TOML (missing required fields)
        invalid_file = tasks_dir / "01_invalid.toml"
        invalid_file.write_text(dedent("""
            [task]
            name = "Missing ID"

            [prompts]
            system = "System"
            user = "User"
        """).strip())

        # Valid TOML
        valid_file = tasks_dir / "02_valid.toml"
        valid_file.write_text(dedent("""
            [task]
            id = "valid-task"
            name = "Valid Task"

            [prompts]
            system = "System prompt"
            user = "User prompt"
        """).strip())

        original_dir = tasks_module.BUILTIN_TASKS_DIR
        try:
            tasks_module.BUILTIN_TASKS_DIR = tasks_dir

            result = tasks_module.get_builtin_tasks()

            # Should have loaded only the valid task
            assert len(result) == 1
            assert result[0].id == "valid-task"
        finally:
            tasks_module.BUILTIN_TASKS_DIR = original_dir


class TestGetTaskById:
    """Verify get_task_by_id function behavior (line 126)."""

    def test_returns_none_for_nonexistent_task(self) -> None:
        """Should return None when task ID is not found.

        Covers line 126: return None when no matching task found.
        """
        from local_ai.benchmark.tasks import get_task_by_id

        result = get_task_by_id("nonexistent-task-id-12345")

        assert result is None

    def test_returns_task_for_valid_id(self) -> None:
        """Should return BenchmarkTask when task ID exists."""
        from local_ai.benchmark.tasks import get_task_by_id

        result = get_task_by_id("todo-api")

        assert result is not None
        assert result.id == "todo-api"
