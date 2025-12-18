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
    LocalAISettings,
    ModelConfig,
    ServerConfig,
    _default_server_config,
)
from local_ai.logging import get_logger

_logger = get_logger("Config")

# Directory for locally converted MLX models
CONVERTED_MODELS_DIR = Path.home() / ".local" / "share" / "local-ai" / "models"


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def _resolve_model_path(model_path: str) -> str:
    """Resolve model path, handling local/ prefix for converted models.

    Args:
        model_path: Model identifier. Can be:
            - local/<model-name>: Resolved to ~/.local/share/local-ai/models/<model-name>
            - HuggingFace model ID (e.g., mlx-community/Qwen3-8B-4bit)
            - Local file path

    Returns:
        Resolved model path (full path for local models, unchanged for others)

    Raises:
        ConfigError: If local model path doesn't exist
    """
    if model_path.startswith("local/"):
        model_name = model_path[6:]  # Strip "local/" prefix
        resolved_path = CONVERTED_MODELS_DIR / model_name

        if not resolved_path.exists():
            _logger.error("Local model not found: {}", model_path)
            raise ConfigError(
                f"Local model not found: {model_path}\n"
                f"Expected at: {resolved_path}\n\n"
                "List available models with:\n"
                "  local-ai models list --all"
            )

        _logger.debug("Resolved local model path: {} -> {}", model_path, resolved_path)
        return str(resolved_path)

    return model_path


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
    raw_model_path = model or _get_nested(toml_config, ["model", "path"])
    port_value = port if port is not None else _get_nested(toml_config, ["server", "port"])
    host_value = host or _get_nested(toml_config, ["server", "host"])

    # Resolve model path if specified (handles local/ prefix for converted models)
    model_path = None
    if raw_model_path:
        model_path = _resolve_model_path(raw_model_path)

    # Build server config from TOML or CLI overrides
    server_dict = toml_config.get("server", {})
    if host_value is not None:
        server_dict["host"] = host_value
    if port_value is not None:
        server_dict["port"] = port_value

    server_config = ServerConfig(**server_dict) if server_dict else _default_server_config()

    # Build model config from TOML
    model_dict = toml_config.get("model", {})
    if model_path is not None:
        model_dict["path"] = model_path
    model_config = ModelConfig(**model_dict)

    settings = LocalAISettings(
        server=server_config,
        model=model_config,
    )
    model_info = model_path if model_path else "no specific model (dynamic loading)"
    _logger.info(
        "Config loaded: model={}, host={}, port={}",
        model_info, server_config.host, server_config.port,
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
