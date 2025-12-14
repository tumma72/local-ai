"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


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
