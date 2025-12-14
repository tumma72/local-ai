"""Goose CLI runner for agentic benchmark comparison.

Enables comparison between raw model output and Goose-enhanced agentic output
using the same underlying model via mlx-omni-server.
"""

import os
import subprocess
import time
from dataclasses import dataclass

from local_ai.logging import get_logger

_logger = get_logger("Benchmark.goose")


@dataclass
class GooseResult:
    """Result from a Goose CLI run."""

    output: str
    elapsed_ms: float
    success: bool
    error: str | None = None


def run_goose_command(
    prompt: str,
    model: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    timeout: float = 300.0,
) -> GooseResult:
    """Execute a prompt through Goose CLI using a local model.

    Args:
        prompt: The prompt/instruction to send to Goose.
        model: Model identifier (e.g., mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit).
        host: Server host.
        port: Server port.
        timeout: Command timeout in seconds.

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

    cmd = ["goose", "run", "--text", prompt, "--no-session"]

    _logger.info("Running Goose with model {} on {}:{}", model, host, port)
    _logger.debug("Prompt: {}", prompt[:100])

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
            _logger.error("Goose command failed: {}", error_msg)
            return GooseResult(
                output="",
                elapsed_ms=elapsed_ms,
                success=False,
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
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.error("Goose command timed out after {}s", timeout)
        return GooseResult(
            output="",
            elapsed_ms=elapsed_ms,
            success=False,
            error=f"Timeout after {timeout}s",
        )
    except FileNotFoundError:
        _logger.error("Goose CLI not found in PATH")
        return GooseResult(
            output="",
            elapsed_ms=0,
            success=False,
            error="Goose CLI not found",
        )
