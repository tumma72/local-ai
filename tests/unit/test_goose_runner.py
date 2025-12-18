"""Behavioral tests for Goose CLI runner module.

Tests verify public behavior of the Goose integration:
- run_goose_recipe executes recipes via subprocess
- Output directory generation from model/task identifiers
- Turn counting from Goose output
- Error handling for missing CLI, timeouts, and failed recipes

Tests are implementation-agnostic and should survive refactoring.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from local_ai.benchmark.goose_runner import (
    GooseResult,
    get_goose_output_dir,
    get_recipe_path,
    list_available_recipes,
    run_goose_command,
    run_goose_recipe,
)


class TestRunGooseCommand:
    """Verify run_goose_command executes prompts through Goose CLI."""

    def test_returns_success_result_when_goose_succeeds(
        self, temp_dir: Path
    ) -> None:
        """Should return GooseResult with success=True when command succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "starting session\nsession id: abc123\nworking directory: /tmp\nHello, world!"
        )
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        assert isinstance(result, GooseResult)
        assert result.success is True
        assert result.error is None
        assert result.elapsed_ms > 0

    def test_returns_failure_when_goose_returns_nonzero(
        self, temp_dir: Path
    ) -> None:
        """Should return GooseResult with success=False on non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: model not found"

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is False
        assert result.error is not None
        assert "model not found" in result.error

    def test_returns_failure_when_goose_cli_not_found(
        self, temp_dir: Path
    ) -> None:
        """Should handle missing Goose CLI gracefully."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is False
        assert result.error == "Goose CLI not found"
        assert result.elapsed_ms == 0

    def test_returns_failure_on_timeout(self, temp_dir: Path) -> None:
        """Should handle command timeout gracefully."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="goose", timeout=300),
        ):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                timeout=300.0,
                working_directory=temp_dir,
            )

        assert result.success is False
        assert result.error is not None
        assert "Timeout" in result.error

    def test_creates_working_directory_if_not_exists(
        self, temp_dir: Path
    ) -> None:
        """Should create working directory if it does not exist."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "working directory: test\nOutput"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=new_dir,
            )

        assert new_dir.exists()
        assert result.working_directory == new_dir


class TestRunGooseRecipe:
    """Verify run_goose_recipe executes YAML recipes through Goose CLI."""

    def test_returns_failure_when_recipe_not_found(
        self, temp_dir: Path
    ) -> None:
        """Should return failure result when recipe file does not exist."""
        nonexistent_recipe = temp_dir / "nonexistent.yaml"

        result = run_goose_recipe(
            recipe_path=nonexistent_recipe,
            model="test-model",
            working_directory=temp_dir,
        )

        assert result.success is False
        assert result.error is not None
        assert "Recipe not found" in result.error

    def test_returns_success_when_recipe_executes(
        self, temp_dir: Path
    ) -> None:
        """Should return success result when recipe completes successfully."""
        # Create a mock recipe file
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Recipe completed successfully"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is True
        assert result.recipe_used == "test_recipe.yaml"

    def test_tracks_files_created_during_execution(
        self, temp_dir: Path
    ) -> None:
        """Should track files created by Goose during recipe execution."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: create files")

        def create_files_side_effect(*args, **kwargs):
            # Simulate Goose creating files
            (temp_dir / "output.py").write_text("print('hello')")
            (temp_dir / "test_output.py").write_text("def test(): pass")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Created files"
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=create_files_side_effect):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is True
        assert "output.py" in result.files_created
        assert "test_output.py" in result.files_created

    def test_returns_failure_when_goose_cli_not_found(
        self, temp_dir: Path
    ) -> None:
        """Should handle missing Goose CLI gracefully."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is False
        assert result.error == "Goose CLI not found"
        assert result.recipe_used == "test_recipe.yaml"

    def test_returns_failure_on_timeout(self, temp_dir: Path) -> None:
        """Should handle recipe timeout gracefully."""
        import subprocess

        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: long task")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="goose", timeout=600),
        ):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
                timeout=600.0,
            )

        assert result.success is False
        assert "Timeout" in result.error


class TestGetGooseOutputDir:
    """Verify output directory path generation."""

    def test_generates_slugified_path_from_model_name(self) -> None:
        """Should generate a safe directory path from model identifier."""
        output_dir = get_goose_output_dir(
            model="mlx-community/Qwen3-Coder-30B",
            task_id="tdd-fizzbuzz",
        )

        # Path should contain slugified model name
        assert "qwen3-coder-30b" in str(output_dir)
        assert "tdd-fizzbuzz" in str(output_dir)
        assert "goose_" in str(output_dir)

    def test_uses_custom_base_directory(self, temp_dir: Path) -> None:
        """Should use custom base directory when provided."""
        output_dir = get_goose_output_dir(
            model="test-model",
            task_id="test-task",
            base_dir=temp_dir,
        )

        assert str(output_dir).startswith(str(temp_dir))

    def test_handles_special_characters_in_model_name(self) -> None:
        """Should handle special characters in model names."""
        output_dir = get_goose_output_dir(
            model="org/Model_Name.v2@latest",
            task_id="task-1",
        )

        # Should not contain unsafe characters
        parent_name = output_dir.parent.name
        assert "/" not in parent_name
        assert "@" not in parent_name


class TestGetRecipePath:
    """Verify recipe path resolution."""

    def test_returns_path_for_recipe_name(self) -> None:
        """Should return path in default recipes directory for recipe name."""
        path = get_recipe_path("tdd-coding")

        assert path.suffix == ".yaml"
        assert "tdd-coding" in str(path)

    def test_returns_path_unchanged_for_full_path(self) -> None:
        """Should return path unchanged when given a full path."""
        full_path = "/custom/path/to/recipe.yaml"
        path = get_recipe_path(full_path)

        assert str(path) == full_path

    def test_returns_path_unchanged_for_relative_path_with_slash(self) -> None:
        """Should return path unchanged when given a relative path with slash."""
        relative_path = "custom/recipe.yaml"
        path = get_recipe_path(relative_path)

        assert str(path) == relative_path


class TestListAvailableRecipes:
    """Verify recipe listing functionality."""

    def test_returns_list_of_recipe_dicts(self) -> None:
        """Should return list of dicts with name and path keys."""
        recipes = list_available_recipes()

        assert isinstance(recipes, list)
        # Each recipe should have name and path
        for recipe in recipes:
            assert "name" in recipe
            assert "path" in recipe


class TestTurnCountingFromOutput:
    """Verify turn counting from Goose CLI output."""

    def test_counts_turns_from_separator_lines(self, temp_dir: Path) -> None:
        """Should count turns from separator patterns in output."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        # Output with separator lines indicating turns
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "Turn 1\n"
            "-------------------\n"
            "Response 1\n"
            "-------------------\n"
            "Turn 2\n"
            "-------------------\n"
            "Response 2\n"
            "-------------------\n"
        )
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is True
        assert result.turns_taken >= 1  # At least one turn detected

    def test_returns_one_turn_for_minimal_output(self, temp_dir: Path) -> None:
        """Should return at least 1 turn when there is any output."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Simple output without separators"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.turns_taken == 1


class TestRunGooseRecipeWithParams:
    """Verify run_goose_recipe handles recipe parameters."""

    def test_passes_recipe_params_to_command(self, temp_dir: Path) -> None:
        """Should pass recipe parameters to goose command."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Output"
        mock_result.stderr = ""

        captured_cmd = None

        def capture_cmd(*args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = args[0]
            return mock_result

        with patch("subprocess.run", side_effect=capture_cmd):
            run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
                recipe_params={"work_dir": "/some/path", "task": "fizzbuzz"},
            )

        # Verify params were passed
        assert "--param" in captured_cmd
        assert "work_dir=/some/path" in captured_cmd
        assert "task=fizzbuzz" in captured_cmd

    def test_returns_failure_with_nonzero_exit_code(
        self, temp_dir: Path
    ) -> None:
        """Should return failure when recipe exits with non-zero code."""
        recipe_path = temp_dir / "test_recipe.yaml"
        recipe_path.write_text("name: test\nprompt: hello")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Recipe execution failed"

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_recipe(
                recipe_path=recipe_path,
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is False
        assert result.error is not None
        assert "Recipe execution failed" in result.error


class TestOutputParsing:
    """Verify Goose output parsing behavior."""

    def test_extracts_output_skipping_session_header(
        self, temp_dir: Path
    ) -> None:
        """Should extract output while skipping session header lines."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "starting session\n"
            "session id: abc-123-def\n"
            "working directory: /tmp/test\n"
            "Actual output line 1\n"
            "Actual output line 2"
        )
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        # Should have extracted output, skipping headers
        assert result.success is True
        assert "Actual output line 1" in result.output
        assert "starting session" not in result.output

    def test_handles_empty_output_gracefully(self, temp_dir: Path) -> None:
        """Should handle empty output without errors."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is True
        assert result.output == ""

    def test_uses_exit_code_for_error_when_no_stderr(
        self, temp_dir: Path
    ) -> None:
        """Should use exit code in error message when stderr is empty."""
        mock_result = MagicMock()
        mock_result.returncode = 127
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_goose_command(
                prompt="Say hello",
                model="test-model",
                working_directory=temp_dir,
            )

        assert result.success is False
        assert "127" in result.error
