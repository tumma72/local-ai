"""Behavioral tests for custom exception hierarchy.

Tests verify the public behavior of exception classes:
- Exception inheritance hierarchy allows proper catching
- Exception messages are preserved and accessible
- Exceptions can be raised and caught at appropriate levels

Tests focus on WHAT the exceptions do (behavior), not HOW they are implemented.
"""

import pytest

from local_ai.exceptions import (
    ConfigError,
    HealthCheckError,
    LocalAIError,
    ServerError,
)


class TestExceptionHierarchy:
    """Verify exception inheritance allows appropriate error handling patterns."""

    def test_local_ai_error_inherits_from_exception(self) -> None:
        """LocalAIError should inherit from Exception."""
        assert issubclass(LocalAIError, Exception)
        error = LocalAIError("test error")
        assert isinstance(error, Exception)

    def test_config_error_is_catchable_as_local_ai_error(self) -> None:
        """ConfigError should be catchable as LocalAIError for broad error handling."""
        with pytest.raises(LocalAIError):
            raise ConfigError("configuration problem")

    def test_server_error_is_catchable_as_local_ai_error(self) -> None:
        """ServerError should be catchable as LocalAIError for broad error handling."""
        with pytest.raises(LocalAIError):
            raise ServerError("server problem")

    def test_health_check_error_is_catchable_as_server_error(self) -> None:
        """HealthCheckError should be catchable as ServerError for server-level handling."""
        with pytest.raises(ServerError):
            raise HealthCheckError("health check failed")

    def test_health_check_error_is_catchable_as_local_ai_error(self) -> None:
        """HealthCheckError should be catchable as LocalAIError for broad handling."""
        with pytest.raises(LocalAIError):
            raise HealthCheckError("health check failed")


class TestExceptionMessages:
    """Verify exception messages are preserved and accessible."""

    def test_local_ai_error_preserves_message(self) -> None:
        """LocalAIError should preserve the error message for display."""
        error_message = "Something went wrong in local-ai"
        error = LocalAIError(error_message)

        assert str(error) == error_message

    def test_config_error_preserves_message(self) -> None:
        """ConfigError should preserve configuration error details."""
        error_message = "Invalid configuration: port must be between 1 and 65535"
        error = ConfigError(error_message)

        assert str(error) == error_message

    def test_server_error_preserves_message(self) -> None:
        """ServerError should preserve server lifecycle error details."""
        error_message = "Failed to start server: port 8080 already in use"
        error = ServerError(error_message)

        assert str(error) == error_message

    def test_health_check_error_preserves_message(self) -> None:
        """HealthCheckError should preserve health check failure details."""
        error_message = "Server did not become healthy within 30 seconds"
        error = HealthCheckError(error_message)

        assert str(error) == error_message


class TestExceptionRaisingPatterns:
    """Verify exceptions work correctly in typical usage patterns."""

    def test_can_catch_specific_error_from_mixed_exceptions(self) -> None:
        """Should be able to catch specific exception types selectively."""
        caught_type = None

        try:
            raise ConfigError("config issue")
        except ConfigError:
            caught_type = "config"
        except ServerError:
            caught_type = "server"
        except LocalAIError:
            caught_type = "generic"

        assert caught_type == "config"

    def test_server_error_does_not_catch_config_error(self) -> None:
        """ServerError handler should not catch ConfigError."""
        with pytest.raises(ConfigError):
            try:
                raise ConfigError("config issue")
            except ServerError:
                pass  # Should not catch ConfigError

    def test_health_check_error_caught_before_generic_server_error(self) -> None:
        """More specific HealthCheckError should be catchable separately from ServerError."""
        caught_specific = False

        try:
            raise HealthCheckError("health failed")
        except HealthCheckError:
            caught_specific = True
        except ServerError:
            caught_specific = False

        assert caught_specific is True
