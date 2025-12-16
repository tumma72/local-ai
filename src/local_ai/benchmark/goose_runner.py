"""Goose CLI runner for agentic benchmark comparison.

Enables comparison between raw model output and Goose-enhanced agentic output
using the same underlying model via mlx-omni-server.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from local_ai import DEFAULT_HOST, DEFAULT_PORT
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.goose")

# Default path for recipes
DEFAULT_RECIPE_DIR = Path(__file__).parent / "recipes"


def _slugify_model_name(model: str) -> str:
    """Convert model name to a safe directory name.

    Args:
        model: Full model identifier (e.g., mlx-community/Qwen3-Coder-30B).

    Returns:
        Safe directory name (e.g., qwen3-coder-30b).
    """
    # Take the last part after /
    name = model.split("/")[-1]
    # Convert to lowercase and replace unsafe chars
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name)  # Collapse multiple dashes
    return name.strip("-")


@dataclass
class GooseResult:
    """Result from a Goose CLI run."""

    output: str
    elapsed_ms: float
    success: bool
    working_directory: Path | None = None
    error: str | None = None
    recipe_used: str | None = None
    turns_taken: int = 0
    files_created: list[str] = field(default_factory=list)


def run_goose_command(
    prompt: str,
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = 300.0,
    working_directory: Path | None = None,
) -> GooseResult:
    """Execute a prompt through Goose CLI using a local model.

    Args:
        prompt: The prompt/instruction to send to Goose.
        model: Model identifier (e.g., mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit).
        host: Server host.
        port: Server port.
        timeout: Command timeout in seconds.
        working_directory: Directory where Goose will create files. If None,
            uses current directory.

    Returns:
        GooseResult with output, timing, and status.
    """
    env = os.environ.copy()
    env.update(
        {
            "GOOSE_PROVIDER": "openai",
            "OPENAI_HOST": f"http://{host}:{port}/",
            "OPENAI_API_KEY": "not-needed",
            "GOOSE_MODEL": model,
            "GOOSE_LEAD_MODEL": model,
            "GOOSE_ENABLE_ROUTER": "false",
        }
    )

    # Determine working directory
    cwd = working_directory or Path.cwd()
    cwd = Path(cwd)

    # Create directory if it doesn't exist
    cwd.mkdir(parents=True, exist_ok=True)

    cmd = ["goose", "run", "--text", prompt, "--no-session"]

    _logger.info("Running Goose with model {} on {}:{}", model, host, port)
    _logger.info("Working directory: {}", cwd)
    _logger.debug("Prompt: {}", prompt[:100])

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
            _logger.error("Goose command failed: {}", error_msg)
            return GooseResult(
                output="",
                elapsed_ms=elapsed_ms,
                success=False,
                working_directory=cwd,
                error=error_msg,
            )

        # Extract actual output (skip session info lines)
        lines = result.stdout.strip().split("\n")
        output_lines = []
        skip_header = True

        for line in lines:
            if skip_header:
                # Skip session header lines
                if line.startswith("starting session") or line.strip().startswith("session id:"):
                    continue
                if line.strip().startswith("working directory:"):
                    skip_header = False
                    continue
            else:
                output_lines.append(line)

        output = "\n".join(output_lines).strip()

        _logger.info("Goose completed in {:.0f}ms", elapsed_ms)
        return GooseResult(
            output=output,
            elapsed_ms=elapsed_ms,
            success=True,
            working_directory=cwd,
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.error("Goose command timed out after {}s", timeout)
        return GooseResult(
            output="",
            elapsed_ms=elapsed_ms,
            success=False,
            working_directory=cwd,
            error=f"Timeout after {timeout}s",
        )
    except FileNotFoundError:
        _logger.error("Goose CLI not found in PATH")
        return GooseResult(
            output="",
            elapsed_ms=0,
            success=False,
            working_directory=cwd,
            error="Goose CLI not found",
        )


def get_goose_output_dir(
    model: str,
    task_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Get the output directory for Goose benchmark results.

    Args:
        model: Model identifier.
        task_id: Benchmark task ID.
        base_dir: Base directory for benchmark outputs. Defaults to
            ~/.local/state/local-ai/benchmark_code/

    Returns:
        Path to the model/task specific output directory.
    """
    if base_dir is None:
        base_dir = Path.home() / ".local" / "state" / "local-ai" / "benchmark_code"

    model_slug = _slugify_model_name(model)
    return base_dir / f"goose_{model_slug}" / task_id


def run_goose_recipe(
    recipe_path: Path,
    model: str,
    working_directory: Path,
    host: str = "127.0.0.1",
    port: int = 10240,
    timeout: float = 600.0,
    max_turns: int = 20,
    recipe_params: dict[str, str] | None = None,
) -> GooseResult:
    """Execute a Goose recipe for agentic benchmark execution.

    Uses `goose recipe run` with proper environment for local models.
    This enables multi-turn agentic workflows with tool use.

    Args:
        recipe_path: Path to the YAML recipe file.
        model: Model identifier (e.g., mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit).
        working_directory: Directory where Goose will create files.
        host: Server host.
        port: Server port.
        timeout: Command timeout in seconds.
        max_turns: Maximum turns for the recipe (-1 for unlimited).
        recipe_params: Recipe parameters (e.g., {"work_dir": "/path/to/dir"}).

    Returns:
        GooseResult with output, timing, status, and created files.
    """
    if not recipe_path.exists():
        _logger.error("Recipe file not found: {}", recipe_path)
        return GooseResult(
            output="",
            elapsed_ms=0,
            success=False,
            working_directory=working_directory,
            error=f"Recipe not found: {recipe_path}",
        )

    env = os.environ.copy()
    env.update(
        {
            "GOOSE_PROVIDER": "openai",
            "OPENAI_HOST": f"http://{host}:{port}/",
            "OPENAI_API_KEY": "not-needed",
            "GOOSE_MODEL": model,
            "GOOSE_LEAD_MODEL": model,
            "GOOSE_ENABLE_ROUTER": "false",
        }
    )

    # Ensure working directory exists
    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "goose",
        "recipe",
        "run",
        str(recipe_path),
        "--max-turns",
        str(max_turns),
    ]

    # Add recipe parameters
    if recipe_params:
        for key, value in recipe_params.items():
            cmd.extend(["--param", f"{key}={value}"])

    _logger.info("Running Goose recipe: {}", recipe_path.name)
    _logger.info("Model: {} on {}:{}", model, host, port)
    _logger.info("Working directory: {}", working_directory)
    _logger.info("Max turns: {}", max_turns)

    # Record files before execution
    files_before = set(working_directory.glob("**/*"))

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(working_directory),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Find created files
        files_after = set(working_directory.glob("**/*"))
        created_files = [
            str(f.relative_to(working_directory))
            for f in (files_after - files_before)
            if f.is_file()
        ]

        output = result.stdout + result.stderr

        # Count turns from output (look for turn indicators)
        turns = _count_turns_from_output(output)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
            _logger.error("Goose recipe failed: {}", error_msg[:200])
            return GooseResult(
                output=output,
                elapsed_ms=elapsed_ms,
                success=False,
                working_directory=working_directory,
                error=error_msg,
                recipe_used=recipe_path.name,
                turns_taken=turns,
                files_created=created_files,
            )

        _logger.info(
            "Goose recipe completed in {:.0f}ms ({} turns, {} files created)",
            elapsed_ms,
            turns,
            len(created_files),
        )

        return GooseResult(
            output=output,
            elapsed_ms=elapsed_ms,
            success=True,
            working_directory=working_directory,
            recipe_used=recipe_path.name,
            turns_taken=turns,
            files_created=created_files,
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.error("Goose recipe timed out after {}s", timeout)
        return GooseResult(
            output="",
            elapsed_ms=elapsed_ms,
            success=False,
            working_directory=working_directory,
            error=f"Timeout after {timeout}s",
            recipe_used=recipe_path.name,
        )
    except FileNotFoundError:
        _logger.error("Goose CLI not found in PATH")
        return GooseResult(
            output="",
            elapsed_ms=0,
            success=False,
            working_directory=working_directory,
            error="Goose CLI not found",
            recipe_used=recipe_path.name,
        )


def _count_turns_from_output(output: str) -> int:
    """Count the number of turns from Goose output.

    Args:
        output: Raw Goose stdout/stderr.

    Returns:
        Estimated number of turns.
    """
    # Goose outputs turn markers like "Turn X" or numbered messages
    # Count assistant response blocks as turns
    turn_patterns = [
        r"─{3,}",  # Turn separator lines
        r"^\s*\d+\.\s+",  # Numbered steps
    ]

    turns = 0
    for pattern in turn_patterns:
        matches = re.findall(pattern, output, re.MULTILINE)
        if matches:
            # Use the count from first matching pattern
            turns = max(turns, len(matches) // 2)  # Divide by 2 for request/response

    # Minimum 1 turn if there's any output
    if turns == 0 and output.strip():
        turns = 1

    return turns


def get_recipe_path(recipe_name: str) -> Path:
    """Get the path to a recipe file.

    Args:
        recipe_name: Recipe name (without .yaml extension) or full path.

    Returns:
        Path to the recipe file.
    """
    # If it's already a path, return it
    if "/" in recipe_name or recipe_name.endswith(".yaml"):
        return Path(recipe_name)

    # Look in the default recipes directory
    recipe_path = DEFAULT_RECIPE_DIR / f"{recipe_name}.yaml"
    return recipe_path


def list_available_recipes() -> list[dict[str, str]]:
    """List available recipes in the recipes directory.

    Returns:
        List of dicts with 'name' and 'path' keys.
    """
    recipes = []

    if DEFAULT_RECIPE_DIR.exists():
        for yaml_file in DEFAULT_RECIPE_DIR.glob("*.yaml"):
            recipes.append(
                {
                    "name": yaml_file.stem,
                    "path": str(yaml_file),
                }
            )

    return recipes
