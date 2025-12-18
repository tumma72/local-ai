"""Pydantic configuration schema models for LocalAI settings.

Defines configuration models for server and model settings.
Generation parameters are client-side concerns (configured in Zed, Claude Code, etc.).
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class ModelConfig(BaseModel):
    """Model configuration settings.

    Note: path is optional because MLX Omni Server loads models dynamically.
    Models are specified in API requests rather than at server startup.
    """

    path: str | None = None
    adapter_path: Path | None = None
    trust_remote_code: bool = False


def _default_server_config() -> ServerConfig:
    """Create default ServerConfig instance."""
    return ServerConfig(host="127.0.0.1", port=8080, log_level="INFO")


class LocalAISettings(BaseModel):
    """Complete LocalAI configuration settings.

    Note: Generation parameters (temperature, max_tokens, etc.) are not included
    here as they are client-side concerns. Configure them in your client
    (Zed, Claude Code, etc.) or per-request via the API.
    """

    server: ServerConfig = Field(default_factory=_default_server_config)
    model: ModelConfig = Field(default_factory=ModelConfig)
