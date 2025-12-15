"""Test validation for TDD benchmark tasks.

Runs pytest on generated code and captures results for benchmark comparison.
"""

import re
import subprocess
from pathlib import Path

from local_ai.benchmark.schema import TestResults
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.test_validator")


def run_pytest(working_dir: Path, timeout: float = 120.0) -> TestResults:
    """Run pytest in a directory and capture results.

    Args:
        working_dir: Directory containing test files.
        timeout: Maximum time to run tests in seconds.

    Returns:
        TestResults with pass/fail counts and output.
    """
    if not working_dir.exists():
        _logger.error("Working directory does not exist: {}", working_dir)
        return TestResults(
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total=0,
            output=f"Directory not found: {working_dir}",
        )

    # Check for test files
    test_files = list(working_dir.glob("test_*.py")) + list(working_dir.glob("*_test.py"))
    if not test_files:
        _logger.warning("No test files found in {}", working_dir)
        return TestResults(
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total=0,
            output="No test files found",
        )

    _logger.info("Running pytest in {} ({} test files)", working_dir, len(test_files))

    cmd = ["python", "-m", "pytest", "-v", "--tb=short", str(working_dir)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(working_dir),
        )

        output = result.stdout + result.stderr

        # Parse pytest output to extract counts
        test_results = _parse_pytest_output(output)
        test_results.output = output

        _logger.info(
            "Tests completed: {} passed, {} failed, {} errors",
            test_results.passed,
            test_results.failed,
            test_results.errors,
        )

        return test_results

    except subprocess.TimeoutExpired:
        _logger.error("Pytest timed out after {}s", timeout)
        return TestResults(
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total=0,
            output=f"Pytest timed out after {timeout}s",
        )
    except FileNotFoundError:
        _logger.error("Python/pytest not found")
        return TestResults(
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total=0,
            output="Python or pytest not found",
        )


def _parse_pytest_output(output: str) -> TestResults:
    """Parse pytest output to extract test counts.

    Args:
        output: Raw pytest stdout/stderr.

    Returns:
        TestResults with parsed counts (output field not set).
    """
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    # Look for the summary line: "X passed, Y failed, Z errors, W skipped"
    # Patterns like: "12 passed", "3 failed, 9 passed", etc.
    summary_pattern = r"(\d+)\s+(passed|failed|error|errors|skipped)"
    matches = re.findall(summary_pattern, output.lower())

    for count_str, status in matches:
        count = int(count_str)
        if status == "passed":
            passed = count
        elif status == "failed":
            failed = count
        elif status in ("error", "errors"):
            errors = count
        elif status == "skipped":
            skipped = count

    total = passed + failed + errors + skipped

    # If no summary found, try counting test result lines
    if total == 0:
        # Count PASSED/FAILED lines like "test_foo PASSED"
        passed = len(re.findall(r"^\s*\w+.*PASSED", output, re.MULTILINE))
        failed = len(re.findall(r"^\s*\w+.*FAILED", output, re.MULTILINE))
        errors = len(re.findall(r"^\s*\w+.*ERROR", output, re.MULTILINE))
        skipped = len(re.findall(r"^\s*\w+.*SKIPPED", output, re.MULTILINE))
        total = passed + failed + errors + skipped

    return TestResults(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        output="",  # Will be set by caller
    )


def validate_tdd_output(working_dir: Path, timeout: float = 120.0) -> TestResults:
    """Validate TDD benchmark output by running tests.

    This is the main entry point for test validation. It:
    1. Checks for required files (main.py, test_main.py)
    2. Runs pytest
    3. Returns structured results

    Args:
        working_dir: Directory containing generated code and tests.
        timeout: Maximum time for test execution.

    Returns:
        TestResults with validation outcome.
    """
    main_file = working_dir / "main.py"
    test_file = working_dir / "test_main.py"

    missing_files = []
    if not main_file.exists():
        missing_files.append("main.py")
    if not test_file.exists():
        missing_files.append("test_main.py")

    if missing_files:
        _logger.warning("Missing required files: {}", missing_files)
        return TestResults(
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total=0,
            output=f"Missing required files: {', '.join(missing_files)}",
        )

    return run_pytest(working_dir, timeout=timeout)
