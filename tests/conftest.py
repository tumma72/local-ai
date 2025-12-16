"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from local_ai.cli.main import app
from local_ai.config.schema import LocalAISettings, ModelConfig, ServerConfig
from local_ai.benchmark.schema import BenchmarkTask, TaskDifficulty
from local_ai.server.welcome import WelcomeApp


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Provide a temporary directory for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_toml(temp_dir: Path) -> Path:
    """Create a sample config.toml for testing."""
    config_path = temp_dir / "config.toml"
    config_path.write_text("""
[server]
host = "127.0.0.1"
port = 8080
log_level = "INFO"

[model]
path = "mlx-community/test-model"

[generation]
max_tokens = 4096
temperature = 0.0
""")
    return config_path


@pytest.fixture
def minimal_config_toml(temp_dir: Path) -> Path:
    """Create a minimal config.toml with only required fields."""
    config_path = temp_dir / "config.toml"
    config_path.write_text("""
[model]
path = "mlx-community/test-model"
""")
    return config_path


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def welcome_app() -> WelcomeApp:
    """Create a welcome app for testing."""
    settings = LocalAISettings()
    return WelcomeApp(settings)


@pytest.fixture
def settings() -> LocalAISettings:
    """Create minimal valid settings for ServerManager."""
    return LocalAISettings(
        server=ServerConfig(host="127.0.0.1", port=8080),
        model=ModelConfig(path="mlx-community/test-model"),
    )


@pytest.fixture
def sample_task() -> BenchmarkTask:
    """Create a sample benchmark task for tests."""
    return BenchmarkTask(
        id="test-task",
        name="Test Task",
        system_prompt="You are a test assistant.",
        user_prompt="Write a hello world function.",
        difficulty=TaskDifficulty.SIMPLE,
        expected_output_tokens=100,
    )
