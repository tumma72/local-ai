"""Benchmark task loading and management.

Provides functions to load benchmark tasks from TOML files and list available tasks.
"""

import tomllib
from pathlib import Path

from local_ai.benchmark.schema import BenchmarkTask, TaskDifficulty
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.tasks")

# Directory containing built-in task definitions
BUILTIN_TASKS_DIR = Path(__file__).parent / "tasks"


def load_task(path: Path) -> BenchmarkTask:
    """Load a benchmark task from a TOML file.

    Args:
        path: Path to the TOML task file.

    Returns:
        BenchmarkTask parsed from the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required sections or fields are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    # Validate required sections
    if "task" not in data:
        msg = f"Missing [task] section in {path}"
        raise ValueError(msg)

    if "prompts" not in data:
        msg = f"Missing [prompts] section in {path}"
        raise ValueError(msg)

    task_data = data["task"]
    prompts = data["prompts"]

    # Validate required fields
    required_task_fields = ["id", "name"]
    for field in required_task_fields:
        if field not in task_data:
            msg = f"Missing required field '{field}' in [task] section of {path}"
            raise ValueError(msg)

    required_prompt_fields = ["system", "user"]
    for field in required_prompt_fields:
        if field not in prompts:
            msg = f"Missing required field '{field}' in [prompts] section of {path}"
            raise ValueError(msg)

    # Parse difficulty if present
    difficulty = TaskDifficulty.MODERATE
    if "difficulty" in task_data:
        difficulty = TaskDifficulty(task_data["difficulty"])

    task = BenchmarkTask(
        id=task_data["id"],
        name=task_data["name"],
        system_prompt=prompts["system"],
        user_prompt=prompts["user"],
        difficulty=difficulty,
        expected_output_tokens=task_data.get("expected_output_tokens", 800),
        language=task_data.get("language", "python"),
        tags=task_data.get("tags", []),
    )

    _logger.debug("Loaded task '{}' from {}", task.id, path)
    return task


def get_builtin_tasks() -> list[BenchmarkTask]:
    """Get all built-in benchmark tasks.

    Returns:
        List of BenchmarkTask objects from the built-in tasks directory.
    """
    tasks: list[BenchmarkTask] = []

    if not BUILTIN_TASKS_DIR.exists():
        _logger.warning("Built-in tasks directory not found: {}", BUILTIN_TASKS_DIR)
        return tasks

    for task_file in sorted(BUILTIN_TASKS_DIR.glob("*.toml")):
        try:
            task = load_task(task_file)
            tasks.append(task)
        except (ValueError, FileNotFoundError) as e:
            _logger.error("Failed to load task from {}: {}", task_file, e)

    _logger.debug("Loaded {} built-in tasks", len(tasks))
    return tasks


def list_tasks() -> list[str]:
    """List all available task IDs.

    Returns:
        List of task ID strings.
    """
    tasks = get_builtin_tasks()
    return [t.id for t in tasks]


def get_task_by_id(task_id: str) -> BenchmarkTask | None:
    """Get a specific task by its ID.

    Args:
        task_id: The task identifier to look up.

    Returns:
        BenchmarkTask if found, None otherwise.
    """
    for task in get_builtin_tasks():
        if task.id == task_id:
            return task
    return None
