"""Pydantic configuration schema models for LocalAI settings.

Defines configuration models for server, model, generation parameters,
and complete LocalAI settings with validation constraints.
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


class GenerationConfig(BaseModel):
    """Generation configuration settings for model inference."""

    max_tokens: int = Field(4096, ge=1, le=32768)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)


def _default_server_config() -> ServerConfig:
    """Create default ServerConfig instance."""
    return ServerConfig(host="127.0.0.1", port=8080, log_level="INFO")


def _default_generation_config() -> GenerationConfig:
    """Create default GenerationConfig instance."""
    return GenerationConfig(
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
    )


class LocalAISettings(BaseModel):
    """Complete LocalAI configuration settings."""

    server: ServerConfig = Field(default_factory=_default_server_config)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=_default_generation_config)
