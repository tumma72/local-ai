"""Custom exception hierarchy for local-ai."""


class LocalAIError(Exception):
    """Base exception for all local-ai errors."""

    pass


class ConfigError(LocalAIError):
    """Configuration-related errors."""

    pass


class ServerError(LocalAIError):
    """Server lifecycle errors."""

    pass


class HealthCheckError(ServerError):
    """Health check failures."""

    pass
