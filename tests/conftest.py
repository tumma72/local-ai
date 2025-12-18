"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from local_ai.benchmark.schema import BenchmarkTask, TaskDifficulty
from local_ai.config.schema import LocalAISettings, ModelConfig, ServerConfig
from local_ai.server.welcome import WelcomeApp

if TYPE_CHECKING:
    from local_ai.hardware.apple_silicon import AppleSiliconInfo


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


# Hardware fixtures for testing hardware-dependent functionality


@pytest.fixture
def mock_hardware_128gb() -> "AppleSiliconInfo":
    """Create mock hardware info for M4 Max with 128GB."""
    from local_ai.hardware.apple_silicon import AppleSiliconInfo, ChipTier

    return AppleSiliconInfo(
        chip_name="Apple M4 Max",
        chip_generation=4,
        chip_tier=ChipTier.MAX,
        memory_gb=128.0,
        cpu_cores=14,
        cpu_performance_cores=10,
        cpu_efficiency_cores=4,
        gpu_cores=40,
        neural_engine_cores=16,
    )


@pytest.fixture
def mock_hardware_16gb() -> "AppleSiliconInfo":
    """Create mock hardware info for M2 with 16GB."""
    from local_ai.hardware.apple_silicon import AppleSiliconInfo, ChipTier

    return AppleSiliconInfo(
        chip_name="Apple M2",
        chip_generation=2,
        chip_tier=ChipTier.BASE,
        memory_gb=16.0,
        cpu_cores=8,
        cpu_performance_cores=4,
        cpu_efficiency_cores=4,
        gpu_cores=10,
        neural_engine_cores=16,
    )


@pytest.fixture
def mock_hardware_8gb() -> "AppleSiliconInfo":
    """Create mock hardware info for M1 with 8GB."""
    from local_ai.hardware.apple_silicon import AppleSiliconInfo, ChipTier

    return AppleSiliconInfo(
        chip_name="Apple M1",
        chip_generation=1,
        chip_tier=ChipTier.BASE,
        memory_gb=8.0,
        cpu_cores=8,
        cpu_performance_cores=4,
        cpu_efficiency_cores=4,
        gpu_cores=8,
        neural_engine_cores=16,
    )
