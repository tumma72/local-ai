"""Behavioral tests for configuration schema validation.

Tests verify public behavior of Pydantic configuration models:
- Default value application
- Required field enforcement
- Value constraint validation
- Valid configuration acceptance

Tests are implementation-agnostic and should survive refactoring.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from local_ai.config.schema import (
    LocalAISettings,
    ModelConfig,
    ServerConfig,
)


class TestServerConfigDefaults:
    """Verify ServerConfig applies correct default values."""

    def test_creates_with_all_defaults(self) -> None:
        """ServerConfig should initialize with sensible defaults when no values provided."""
        config = ServerConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.log_level == "INFO"

    def test_accepts_override_of_defaults(self) -> None:
        """ServerConfig should allow overriding default values."""
        config = ServerConfig(host="0.0.0.0", port=9000, log_level="DEBUG")

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.log_level == "DEBUG"


class TestRequiredFields:
    """Verify required fields are enforced across all configuration models."""

    def test_model_config_path_is_optional(self) -> None:
        """ModelConfig should allow instantiation without 'path' field (dynamic loading)."""
        # Should not raise an error - path is now optional
        config = ModelConfig()
        assert config.path is None
        assert config.adapter_path is None
        assert config.trust_remote_code is False

    def test_model_config_accepts_path(self) -> None:
        """ModelConfig should accept and store 'path' field when provided."""
        config = ModelConfig(path="test-model")
        assert config.path == "test-model"
        assert config.adapter_path is None
        assert config.trust_remote_code is False

    def test_local_ai_settings_model_is_optional(self) -> None:
        """LocalAISettings should allow instantiation without model (dynamic loading)."""
        # Should not raise an error - model now has default
        settings = LocalAISettings()
        assert settings.model.path is None  # Default ModelConfig with no path
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080

    def test_local_ai_settings_accepts_model(self) -> None:
        """LocalAISettings should accept and store model when provided."""
        settings = LocalAISettings(model=ModelConfig(path="test-model"))
        assert settings.model.path == "test-model"

    def test_model_config_accepts_only_required_path(self) -> None:
        """ModelConfig should initialize successfully with only 'path' provided."""
        config = ModelConfig(path="mlx-community/test-model")

        assert config.path == "mlx-community/test-model"
        assert config.adapter_path is None
        assert config.trust_remote_code is False


class TestValueConstraints:
    """Verify value constraints are enforced for bounded parameters."""

    @pytest.mark.parametrize(
        "invalid_port",
        [0, -1, 65536, 100000],
        ids=["zero", "negative", "above_max", "far_above_max"],
    )
    def test_server_config_rejects_invalid_ports(self, invalid_port: int) -> None:
        """ServerConfig must reject port values outside valid range [1, 65535]."""
        with pytest.raises(ValidationError) as exc_info:
            ServerConfig(port=invalid_port)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("port",) for error in errors)

    @pytest.mark.parametrize("valid_port", [1, 80, 8080, 65535])
    def test_server_config_accepts_valid_ports(self, valid_port: int) -> None:
        """ServerConfig should accept port values within valid range [1, 65535]."""
        config = ServerConfig(port=valid_port)
        assert config.port == valid_port

    @pytest.mark.parametrize(
        "invalid_log_level",
        ["info", "debug", "TRACE", "VERBOSE", ""],
        ids=["lowercase_info", "lowercase_debug", "trace", "verbose", "empty"],
    )
    def test_server_config_rejects_invalid_log_levels(self, invalid_log_level: str) -> None:
        """ServerConfig must reject log levels not in allowed set."""
        with pytest.raises(ValidationError) as exc_info:
            ServerConfig(log_level=invalid_log_level)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("log_level",) for error in errors)

    @pytest.mark.parametrize(
        "valid_log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_server_config_accepts_valid_log_levels(self, valid_log_level: str) -> None:
        """ServerConfig should accept standard Python logging levels."""
        config = ServerConfig(log_level=valid_log_level)
        assert config.log_level == valid_log_level


class TestCompleteConfiguration:
    """Verify LocalAISettings accepts valid complete configurations."""

    def test_minimal_valid_configuration(self) -> None:
        """LocalAISettings should accept configuration with only required fields."""
        settings = LocalAISettings(
            model=ModelConfig(path="mlx-community/test-model")
        )

        # Required field present
        assert settings.model.path == "mlx-community/test-model"

        # Sub-configs use defaults
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080

    def test_complete_valid_configuration(self) -> None:
        """LocalAISettings should accept fully specified valid configuration."""
        settings = LocalAISettings(
            server=ServerConfig(host="0.0.0.0", port=9000, log_level="DEBUG"),
            model=ModelConfig(
                path="mlx-community/test-model",
                adapter_path=Path("/path/to/adapter"),
                trust_remote_code=True,
            ),
        )

        # Server config
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 9000
        assert settings.server.log_level == "DEBUG"

        # Model config
        assert settings.model.path == "mlx-community/test-model"
        assert settings.model.adapter_path == Path("/path/to/adapter")
        assert settings.model.trust_remote_code is True

    def test_settings_does_not_have_generation_config(self) -> None:
        """LocalAISettings should not have generation config (client-side concern)."""
        settings = LocalAISettings()

        # Verify generation is not an attribute
        assert not hasattr(settings, "generation")
