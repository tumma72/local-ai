"""Configuration loader for LocalAI settings.

Handles TOML file loading, path discovery, and resolution priority:
1. CLI arguments (highest priority)
2. TOML file (discovered or explicit)
3. Environment variables
4. Default values (lowest priority)
"""

import tomllib
from pathlib import Path
from typing import Any

from local_ai.config.schema import (
    GenerationConfig,
    LocalAISettings,
    ModelConfig,
    ServerConfig,
)
from local_ai.logging import get_logger

_logger = get_logger("Config")


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def load_config(
    config_path: Path | None = None,
    model: str | None = None,
    port: int | None = None,
    host: str | None = None,
) -> LocalAISettings:
    """Load and resolve LocalAI configuration.

    Resolution priority (highest to lowest):
    1. CLI arguments (model, port, host)
    2. TOML file (config_path or discovered)
    3. Default values

    Path discovery (when config_path is None):
    1. ./config.toml (current directory)
    2. ~/.config/local-ai/config.toml (user config)
    3. Use defaults if neither exists

    Args:
        config_path: Explicit path to config TOML file
        model: Model path/name (CLI override)
        port: Server port (CLI override)
        host: Server host (CLI override)

    Returns:
        LocalAISettings: Resolved configuration

    Raises:
        ConfigError: If model is not specified anywhere
        ConfigError: If TOML file cannot be parsed
    """
    _logger.debug(
        "Loading config: config_path={}, model={}, port={}, host={}",
        config_path, model, port, host,
    )

    # Load TOML config if available
    toml_config: dict[str, Any] = {}

    if config_path:
        # Explicit config path provided
        _logger.debug("Loading explicit config file: {}", config_path)
        toml_config = _load_toml_file(config_path)
    else:
        # Discover config path
        discovered_path = _discover_config_path()
        if discovered_path:
            _logger.debug("Discovered config file: {}", discovered_path)
            toml_config = _load_toml_file(discovered_path)
        else:
            _logger.debug("No config file found, using defaults")

    # Merge configuration with priority: CLI > TOML > defaults
    model_path = model or _get_nested(toml_config, ["model", "path"])
    port_value = port if port is not None else _get_nested(toml_config, ["server", "port"])
    host_value = host or _get_nested(toml_config, ["server", "host"])

    # Validate that model is specified
    if not model_path:
        _logger.error("Model is required but not specified")
        raise ConfigError(
            "Model is required but not specified. Provide it via:\n"
            "  1. --model CLI argument\n"
            "  2. [model] section in config.toml\n"
            "  3. LOCAL_AI_MODEL environment variable"
        )

    # Build server config from TOML or CLI overrides
    server_dict = toml_config.get("server", {})
    if host_value is not None:
        server_dict["host"] = host_value
    if port_value is not None:
        server_dict["port"] = port_value

    server_config = ServerConfig(**server_dict) if server_dict else ServerConfig()

    # Build model config from TOML
    model_dict = toml_config.get("model", {})
    model_dict["path"] = model_path
    model_config = ModelConfig(**model_dict)

    # Build generation config from TOML
    generation_dict = toml_config.get("generation", {})
    generation_config = (
        GenerationConfig(**generation_dict) if generation_dict else GenerationConfig()
    )

    settings = LocalAISettings(
        server=server_config,
        model=model_config,
        generation=generation_config,
    )
    _logger.info(
        "Config loaded: model={}, host={}, port={}",
        model_path, server_config.host, server_config.port,
    )
    return settings


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file.

    Args:
        path: Path to TOML file

    Returns:
        dict: Parsed TOML content

    Raises:
        ConfigError: If file cannot be read or parsed
    """
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
            _logger.debug("Successfully loaded TOML file: {}", path)
            return config
    except FileNotFoundError as e:
        _logger.error("Config file not found: {}", path)
        raise ConfigError(f"Config file not found: {path}") from e
    except Exception as e:
        _logger.error("Failed to parse config file {}: {}", path, e)
        raise ConfigError(f"Failed to parse config file {path}: {e}") from e


def _discover_config_path() -> Path | None:
    """Discover config file path following resolution order.

    Priority:
    1. ./config.toml (current directory)
    2. ~/.config/local-ai/config.toml (user config)

    Returns:
        Path: Path to existing config file, or None if not found
    """
    # Check current directory
    current_dir_config = Path("config.toml")
    if current_dir_config.exists():
        return current_dir_config

    # Check user config directory
    user_config = Path.home() / ".config" / "local-ai" / "config.toml"
    if user_config.exists():
        return user_config

    return None


def _get_nested(data: dict[str, Any], keys: list[str]) -> Any:
    """Get value from nested dictionary safely.

    Args:
        data: Dictionary to search
        keys: List of keys to traverse

    Returns:
        Value at nested key, or None if not found
    """
    result: Any = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return None
    return result
