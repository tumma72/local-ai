"""Tests for benchmark test validator module."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_ai.benchmark.test_validator import (
    TestResults,
    _parse_pytest_output,
    run_pytest,
    validate_tdd_output,
)


class TestBenchmarkTestValidator:
    """Test benchmark test validator functionality."""

    def test_parse_pytest_output_with_summary_line(self) -> None:
        """Test parsing pytest output with summary line."""
        output = """
        test_foo.py .                                                                   [100%]
        test_bar.py F                                                                   [100%]
        
        =============================== test session starts ===============================
        collected 2 items
        
        test_foo.py::test_pass PASSED                                                  [ 50%]
        test_bar.py::test_fail FAILED                                                  [100%]
        
        ============================= short test summary info =============================
        FAILED test_bar.py::test_fail - AssertionError: test failure
        ======================= 1 passed, 1 failed in 0.01s ========================
        """

        results = _parse_pytest_output(output)

        assert results.passed == 1
        assert results.failed == 1
        assert results.errors == 0
        assert results.skipped == 0
        assert results.total == 2

    def test_parse_pytest_output_with_individual_results(self) -> None:
        """Test parsing pytest output by counting individual test results."""
        output = """
        test_foo.py::test_one PASSED
        test_foo.py::test_two FAILED
        test_foo.py::test_three ERROR
        test_foo.py::test_four SKIPPED
        """

        results = _parse_pytest_output(output)

        assert results.passed == 1
        assert results.failed == 1
        assert results.errors == 1
        assert results.skipped == 1
        assert results.total == 4

    def test_parse_pytest_output_no_tests(self) -> None:
        """Test parsing pytest output with no tests found."""
        output = "No tests found"

        results = _parse_pytest_output(output)

        assert results.passed == 0
        assert results.failed == 0
        assert results.errors == 0
        assert results.skipped == 0
        assert results.total == 0

    @patch('local_ai.benchmark.test_validator.subprocess.run')
    def test_run_pytest_success(self, mock_subprocess_run: MagicMock) -> None:
        """Test running pytest successfully."""
        mock_result = MagicMock()
        mock_result.stdout = "1 passed, 0 failed"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            # Create a dummy test file
            (working_dir / "test_dummy.py").write_text("def test_pass(): pass")

            results = run_pytest(working_dir)

            assert results.passed == 1
            assert results.failed == 0
            assert "1 passed, 0 failed" in results.output

    @patch('local_ai.benchmark.test_validator.subprocess.run')
    def test_run_pytest_timeout(self, mock_subprocess_run: MagicMock) -> None:
        """Test handling pytest timeout."""
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("pytest", 1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            # Create a test file so it gets past the file check
            (working_dir / "test_dummy.py").write_text("def test_pass(): pass")

            results = run_pytest(working_dir, timeout=1.0)

            assert results.passed == 0
            assert "timed out" in results.output

    @patch('local_ai.benchmark.test_validator.subprocess.run')
    def test_run_pytest_python_not_found(self, mock_subprocess_run: MagicMock) -> None:
        """Test handling when Python/pytest not found."""
        mock_subprocess_run.side_effect = FileNotFoundError("Python not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            # Create a test file so it gets past the file check
            (working_dir / "test_dummy.py").write_text("def test_pass(): pass")

            results = run_pytest(working_dir)

            assert results.passed == 0
            assert "Python or pytest not found" in results.output

    def test_run_pytest_directory_not_found(self) -> None:
        """Test handling when working directory doesn't exist."""
        non_existent_dir = Path("/non/existent/dir")

        results = run_pytest(non_existent_dir)

        assert results.passed == 0
        assert "Directory not found" in results.output

    def test_run_pytest_no_test_files(self) -> None:
        """Test handling when no test files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            # Create a non-test file
            (working_dir / "main.py").write_text("print('hello')")

            results = run_pytest(working_dir)

            assert results.passed == 0
            assert "No test files found" in results.output

    def test_validate_tdd_output_missing_files(self) -> None:
        """Test validation when required files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)

            results = validate_tdd_output(working_dir)

            assert results.passed == 0
            assert "Missing required files" in results.output

    @patch('local_ai.benchmark.test_validator.run_pytest')
    def test_validate_tdd_output_success(self, mock_run_pytest: MagicMock) -> None:
        """Test successful TDD validation."""
        mock_run_pytest.return_value = TestResults(
            passed=2,
            failed=0,
            errors=0,
            skipped=0,
            total=2,
            output="All tests passed"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            # Create required files
            (working_dir / "main.py").write_text("def main(): pass")
            (working_dir / "test_main.py").write_text("def test_main(): pass")

            results = validate_tdd_output(working_dir)

            assert results.passed == 2
            assert results.failed == 0
            mock_run_pytest.assert_called_once()


class TestPytestOutputParsingEdgeCases:
    """Verify pytest output parsing handles all status types."""

    def test_parse_pytest_output_with_errors_status(self) -> None:
        """Test parsing pytest output that includes 'errors' in summary."""
        output = """
        =============================== test session starts ===============================
        collected 5 items

        test_foo.py::test_one PASSED                                                  [ 20%]
        test_foo.py::test_two FAILED                                                  [ 40%]
        test_foo.py::test_three ERROR                                                 [ 60%]
        test_foo.py::test_four ERROR                                                  [ 80%]
        test_foo.py::test_five PASSED                                                 [100%]

        ======================= 2 passed, 1 failed, 2 errors in 0.05s ===================
        """

        results = _parse_pytest_output(output)

        assert results.passed == 2
        assert results.failed == 1
        assert results.errors == 2
        assert results.total == 5

    def test_parse_pytest_output_with_single_error(self) -> None:
        """Test parsing pytest output with singular 'error' word."""
        output = """
        ======================= 3 passed, 1 error in 0.02s ===================
        """

        results = _parse_pytest_output(output)

        assert results.passed == 3
        assert results.errors == 1
        assert results.total == 4

    def test_parse_pytest_output_with_skipped_tests(self) -> None:
        """Test parsing pytest output that includes skipped tests."""
        output = """
        =============================== test session starts ===============================
        collected 6 items

        test_foo.py::test_one PASSED                                                  [ 17%]
        test_foo.py::test_two SKIPPED                                                 [ 33%]
        test_foo.py::test_three PASSED                                                [ 50%]
        test_foo.py::test_four SKIPPED                                                [ 67%]
        test_foo.py::test_five PASSED                                                 [ 83%]
        test_foo.py::test_six SKIPPED                                                 [100%]

        ======================= 3 passed, 3 skipped in 0.03s =======================
        """

        results = _parse_pytest_output(output)

        assert results.passed == 3
        assert results.skipped == 3
        assert results.total == 6

    def test_parse_pytest_output_with_all_status_types(self) -> None:
        """Test parsing pytest output with all possible status types."""
        output = """
        ======================= 5 passed, 2 failed, 1 error, 3 skipped in 0.10s =======
        """

        results = _parse_pytest_output(output)

        assert results.passed == 5
        assert results.failed == 2
        assert results.errors == 1
        assert results.skipped == 3
        assert results.total == 11
