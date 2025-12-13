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
    GenerationConfig,
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

    def test_model_config_requires_path(self) -> None:
        """ModelConfig must reject instantiation without required 'path' field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("path",) for error in errors)
        assert any(error["type"] == "missing" for error in errors)

    def test_local_ai_settings_requires_model(self) -> None:
        """LocalAISettings must reject instantiation without required 'model' field."""
        with pytest.raises(ValidationError) as exc_info:
            LocalAISettings()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("model",) for error in errors)
        assert any(error["type"] == "missing" for error in errors)

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

    @pytest.mark.parametrize(
        "invalid_max_tokens",
        [0, -1, 32769, 100000],
        ids=["zero", "negative", "above_max", "far_above_max"],
    )
    def test_generation_config_rejects_invalid_max_tokens(
        self, invalid_max_tokens: int
    ) -> None:
        """GenerationConfig must reject max_tokens outside valid range [1, 32768]."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(max_tokens=invalid_max_tokens)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_tokens",) for error in errors)

    @pytest.mark.parametrize("valid_max_tokens", [1, 4096, 32768])
    def test_generation_config_accepts_valid_max_tokens(self, valid_max_tokens: int) -> None:
        """GenerationConfig should accept max_tokens within valid range [1, 32768]."""
        config = GenerationConfig(max_tokens=valid_max_tokens)
        assert config.max_tokens == valid_max_tokens

    @pytest.mark.parametrize(
        "invalid_temperature",
        [-0.1, -1.0, 2.1, 3.0],
        ids=["slightly_negative", "negative", "above_max", "far_above_max"],
    )
    def test_generation_config_rejects_invalid_temperatures(
        self, invalid_temperature: float
    ) -> None:
        """GenerationConfig must reject temperature outside valid range [0.0, 2.0]."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(temperature=invalid_temperature)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("temperature",) for error in errors)

    @pytest.mark.parametrize("valid_temperature", [0.0, 0.7, 1.0, 2.0])
    def test_generation_config_accepts_valid_temperatures(
        self, valid_temperature: float
    ) -> None:
        """GenerationConfig should accept temperature within valid range [0.0, 2.0]."""
        config = GenerationConfig(temperature=valid_temperature)
        assert config.temperature == valid_temperature

    @pytest.mark.parametrize(
        "invalid_top_p",
        [-0.1, -1.0, 1.1, 2.0],
        ids=["slightly_negative", "negative", "above_max", "far_above_max"],
    )
    def test_generation_config_rejects_invalid_top_p(self, invalid_top_p: float) -> None:
        """GenerationConfig must reject top_p outside valid range [0.0, 1.0]."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(top_p=invalid_top_p)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("top_p",) for error in errors)

    @pytest.mark.parametrize("valid_top_p", [0.0, 0.9, 1.0])
    def test_generation_config_accepts_valid_top_p(self, valid_top_p: float) -> None:
        """GenerationConfig should accept top_p within valid range [0.0, 1.0]."""
        config = GenerationConfig(top_p=valid_top_p)
        assert config.top_p == valid_top_p

    @pytest.mark.parametrize(
        "invalid_min_p",
        [-0.1, -1.0, 1.1, 2.0],
        ids=["slightly_negative", "negative", "above_max", "far_above_max"],
    )
    def test_generation_config_rejects_invalid_min_p(self, invalid_min_p: float) -> None:
        """GenerationConfig must reject min_p outside valid range [0.0, 1.0]."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(min_p=invalid_min_p)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("min_p",) for error in errors)

    @pytest.mark.parametrize("valid_min_p", [0.0, 0.1, 1.0])
    def test_generation_config_accepts_valid_min_p(self, valid_min_p: float) -> None:
        """GenerationConfig should accept min_p within valid range [0.0, 1.0]."""
        config = GenerationConfig(min_p=valid_min_p)
        assert config.min_p == valid_min_p

    def test_generation_config_rejects_negative_top_k(self) -> None:
        """GenerationConfig must reject negative top_k values."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(top_k=-1)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("top_k",) for error in errors)

    @pytest.mark.parametrize("valid_top_k", [0, 10, 50, 100])
    def test_generation_config_accepts_valid_top_k(self, valid_top_k: int) -> None:
        """GenerationConfig should accept non-negative top_k values."""
        config = GenerationConfig(top_k=valid_top_k)
        assert config.top_k == valid_top_k


class TestGenerationConfigDefaults:
    """Verify GenerationConfig applies correct default values."""

    def test_creates_with_all_defaults(self) -> None:
        """GenerationConfig should initialize with sensible defaults when no values provided."""
        config = GenerationConfig()

        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.top_k == 0
        assert config.min_p == 0.0


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
        assert settings.generation.max_tokens == 4096

    def test_complete_valid_configuration(self) -> None:
        """LocalAISettings should accept fully specified valid configuration."""
        settings = LocalAISettings(
            server=ServerConfig(host="0.0.0.0", port=9000, log_level="DEBUG"),
            model=ModelConfig(
                path="mlx-community/test-model",
                adapter_path=Path("/path/to/adapter"),
                trust_remote_code=True,
            ),
            generation=GenerationConfig(
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.05,
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

        # Generation config
        assert settings.generation.max_tokens == 2048
        assert settings.generation.temperature == 0.7
        assert settings.generation.top_p == 0.9
        assert settings.generation.top_k == 50
        assert settings.generation.min_p == 0.05
