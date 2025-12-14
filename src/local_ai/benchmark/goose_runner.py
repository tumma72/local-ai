"""Goose CLI runner for agentic benchmark comparison.

Enables comparison between raw model output and Goose-enhanced agentic output
using the same underlying model via mlx-omni-server.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from local_ai.logging import get_logger

_logger = get_logger("Benchmark.goose")


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


def run_goose_command(
    prompt: str,
    model: str,
    host: str = "127.0.0.1",
    port: int = 8080,
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
    env.update({
        "GOOSE_PROVIDER": "openai",
        "OPENAI_HOST": f"http://{host}:{port}/",
        "OPENAI_API_KEY": "not-needed",
        "GOOSE_MODEL": model,
        "GOOSE_LEAD_MODEL": model,
        "GOOSE_ENABLE_ROUTER": "false",
    })

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
