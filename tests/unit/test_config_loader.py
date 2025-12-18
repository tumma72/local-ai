"""Behavioral tests for configuration loader module.

Tests verify public behavior of the load_config() function:
- TOML file loading and parsing
- CLI override precedence
- Error handling for missing required fields
- Graceful handling of minimal configurations

Tests are implementation-agnostic and should survive refactoring.
"""

from pathlib import Path

import pytest

from local_ai.config.loader import (
    ConfigError,
    _resolve_model_path,
    load_config,
)
from local_ai.config.schema import LocalAISettings


class TestLoadConfigFromTomlFile:
    """Verify load_config correctly loads configuration from TOML files."""

    def test_returns_valid_settings_from_complete_toml(
        self, sample_config_toml: Path
    ) -> None:
        """load_config should return LocalAISettings with values from valid TOML file."""
        settings = load_config(config_path=sample_config_toml)

        assert isinstance(settings, LocalAISettings)
        assert settings.model.path == "mlx-community/test-model"
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080

    def test_returns_valid_settings_from_minimal_toml(
        self, minimal_config_toml: Path
    ) -> None:
        """load_config should work with minimal TOML containing only required model section."""
        settings = load_config(config_path=minimal_config_toml)

        assert isinstance(settings, LocalAISettings)
        assert settings.model.path == "mlx-community/test-model"
        # Should use defaults for unspecified sections
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080


class TestCliOverrides:
    """Verify CLI arguments take precedence over TOML file values."""

    def test_cli_model_overrides_toml_model(self, sample_config_toml: Path) -> None:
        """load_config should use CLI model argument over TOML file value."""
        cli_model = "mlx-community/cli-override-model"

        settings = load_config(config_path=sample_config_toml, model=cli_model)

        assert settings.model.path == cli_model

    def test_cli_port_overrides_toml_port(self, sample_config_toml: Path) -> None:
        """load_config should use CLI port argument over TOML file value."""
        cli_port = 9999

        settings = load_config(config_path=sample_config_toml, port=cli_port)

        assert settings.server.port == cli_port

    def test_cli_host_overrides_toml_host(self, sample_config_toml: Path) -> None:
        """load_config should use CLI host argument over TOML file value."""
        cli_host = "0.0.0.0"

        settings = load_config(config_path=sample_config_toml, host=cli_host)

        assert settings.server.host == cli_host


class TestMissingModelError:
    """Verify load_config raises helpful error when model is not specified."""

    def test_works_without_model_optional(self, temp_dir: Path) -> None:
        """load_config should work when model is not specified (dynamic loading)."""
        # Create config without model section
        config_without_model = temp_dir / "no_model_config.toml"
        config_without_model.write_text("""
[server]
port = 8080
""")

        # Should not raise an error - model is now optional
        settings = load_config(config_path=config_without_model)

        # Should have default model config with no path
        assert settings.model.path is None
        assert settings.server.port == 8080


class TestNoConfigFile:
    """Verify load_config works without any config file when model is provided via CLI."""

    def test_works_with_cli_model_only(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_config should succeed with only CLI model and no config file."""
        # Change to temp_dir to avoid finding any existing config.toml
        monkeypatch.chdir(temp_dir)
        cli_model = "mlx-community/cli-only-model"

        settings = load_config(model=cli_model)

        assert isinstance(settings, LocalAISettings)
        assert settings.model.path == cli_model
        # Should use defaults for everything else
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080


class TestConfigFileErrors:
    """Verify load_config handles file errors appropriately."""

    def test_raises_config_error_when_explicit_file_not_found(
        self, temp_dir: Path
    ) -> None:
        """load_config should raise ConfigError when explicit config path does not exist."""
        nonexistent_path = temp_dir / "nonexistent.toml"

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path=nonexistent_path)

        error_message = str(exc_info.value)
        assert "not found" in error_message.lower()
        assert str(nonexistent_path) in error_message

    def test_raises_config_error_when_toml_is_invalid(
        self, temp_dir: Path
    ) -> None:
        """load_config should raise ConfigError when TOML file is malformed."""
        invalid_toml = temp_dir / "invalid.toml"
        invalid_toml.write_text("""
[server
port = "not a valid toml - missing bracket"
""")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path=invalid_toml)

        error_message = str(exc_info.value)
        assert "failed to parse" in error_message.lower() or "parse" in error_message.lower()


class TestConfigDiscovery:
    """Verify load_config discovers config files in expected locations."""

    def test_discovers_config_in_current_directory(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_config should discover config.toml in current directory."""
        # Create config.toml in temp_dir
        config_path = temp_dir / "config.toml"
        config_path.write_text("""
[model]
path = "discovered-model"

[server]
port = 9999
""")
        # Change to temp_dir
        monkeypatch.chdir(temp_dir)

        settings = load_config()

        assert settings.model.path == "discovered-model"
        assert settings.server.port == 9999

    def test_discovers_config_in_user_config_directory(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_config should discover config.toml in ~/.config/local-ai/."""
        # Create mock user config directory
        user_config_dir = temp_dir / ".config" / "local-ai"
        user_config_dir.mkdir(parents=True)
        config_path = user_config_dir / "config.toml"
        config_path.write_text("""
[model]
path = "user-config-model"
""")

        # Create empty working directory (no config.toml)
        working_dir = temp_dir / "workdir"
        working_dir.mkdir()

        # Patch Path.home() to return temp_dir and chdir to working_dir
        monkeypatch.chdir(working_dir)
        monkeypatch.setattr(Path, "home", lambda: temp_dir)

        settings = load_config()

        assert settings.model.path == "user-config-model"


class TestLocalModelPathResolution:
    """Verify local model path resolution with local/ prefix."""

    def test_huggingface_model_id_passes_through_unchanged(self) -> None:
        """_resolve_model_path should return HuggingFace model IDs unchanged."""
        model_id = "mlx-community/Qwen3-8B-4bit"

        resolved = _resolve_model_path(model_id)

        assert resolved == model_id

    def test_local_prefix_resolves_to_converted_models_dir(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_resolve_model_path should resolve local/ prefix to CONVERTED_MODELS_DIR."""
        # Create mock converted models directory with a model
        converted_dir = temp_dir / "models"
        converted_dir.mkdir()
        model_dir = converted_dir / "mistralai_Devstral-Small-4bit-mlx"
        model_dir.mkdir()

        # Patch CONVERTED_MODELS_DIR
        monkeypatch.setattr(
            "local_ai.config.loader.CONVERTED_MODELS_DIR", converted_dir
        )

        resolved = _resolve_model_path("local/mistralai_Devstral-Small-4bit-mlx")

        assert resolved == str(model_dir)

    def test_local_prefix_raises_error_when_model_not_found(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_resolve_model_path should raise ConfigError when local model doesn't exist."""
        # Create empty converted models directory
        converted_dir = temp_dir / "models"
        converted_dir.mkdir()

        # Patch CONVERTED_MODELS_DIR
        monkeypatch.setattr(
            "local_ai.config.loader.CONVERTED_MODELS_DIR", converted_dir
        )

        with pytest.raises(ConfigError) as exc_info:
            _resolve_model_path("local/nonexistent-model")

        error_message = str(exc_info.value)
        assert "local model not found" in error_message.lower()
        assert "nonexistent-model" in error_message

    def test_load_config_resolves_local_model_path(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_config should resolve local/ model paths when specified via CLI."""
        # Create mock converted models directory with a model
        converted_dir = temp_dir / "models"
        converted_dir.mkdir()
        model_dir = converted_dir / "test-model-4bit"
        model_dir.mkdir()

        # Patch CONVERTED_MODELS_DIR and chdir to avoid finding config.toml
        monkeypatch.setattr(
            "local_ai.config.loader.CONVERTED_MODELS_DIR", converted_dir
        )
        monkeypatch.chdir(temp_dir)

        settings = load_config(model="local/test-model-4bit")

        assert settings.model.path == str(model_dir)
