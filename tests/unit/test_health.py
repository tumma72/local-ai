"""Behavioral tests for health checking module.

Tests verify public behavior of health check functions:
- check_health() queries /v1/models and returns health status
- wait_for_health() polls until healthy or timeout

Tests mock httpx for isolation from actual HTTP requests.
Tests are implementation-agnostic and should survive refactoring.
"""

from unittest.mock import MagicMock, patch

import httpx

from local_ai.server.health import check_health, wait_for_health


class TestCheckHealth:
    """Verify check_health() behavior for querying server health."""

    def test_check_health_returns_healthy_when_models_endpoint_returns_200(self) -> None:
        """check_health() should return 'healthy' when /v1/models responds with 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.get", return_value=mock_response):
            result = check_health(host="127.0.0.1", port=8080)

        assert result == "healthy"

    def test_check_health_returns_unhealthy_when_models_endpoint_returns_non_200(
        self,
    ) -> None:
        """check_health() should return 'unhealthy' when /v1/models responds with non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 503  # Service Unavailable

        with patch("httpx.get", return_value=mock_response):
            result = check_health(host="127.0.0.1", port=8080)

        assert result == "unhealthy"

    def test_check_health_returns_unknown_when_connection_fails(self) -> None:
        """check_health() should return 'unknown' when connection to server fails."""
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = check_health(host="127.0.0.1", port=8080)

        assert result == "unknown"


class TestWaitForHealth:
    """Verify wait_for_health() behavior for polling until server is healthy."""

    def test_wait_for_health_returns_true_when_server_becomes_healthy_within_timeout(
        self,
    ) -> None:
        """wait_for_health() should return True when server becomes healthy before timeout."""
        # Simulate server becoming healthy on third check
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            # First two calls fail, third succeeds
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            response.status_code = 200
            return response

        with patch("httpx.get", side_effect=mock_get), \
             patch("time.sleep"):  # Skip actual sleep for fast tests
            result = wait_for_health(
                host="127.0.0.1", port=8080, timeout=10.0, interval=1.0
            )

        assert result is True

    def test_wait_for_health_returns_false_when_timeout_expires(self) -> None:
        """wait_for_health() should return False when server never becomes healthy."""
        # Simulate server never becoming healthy
        connect_error = httpx.ConnectError("Connection refused")
        with patch("httpx.get", side_effect=connect_error), \
             patch("time.sleep"), \
             patch("time.monotonic") as mock_time:
            # Simulate time passing beyond timeout
            mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 61.0]  # Last call > timeout
            result = wait_for_health(
                host="127.0.0.1", port=8080, timeout=60.0, interval=1.0
            )

        assert result is False


class TestGetModels:
    """Verify get_models() behavior for querying server models (lines 32-36)."""

    def test_get_models_returns_empty_list_on_non_200_status(self) -> None:
        """Should return empty list when server returns non-200 status.

        Covers lines 32-33: non-200 status code handling.
        """
        from local_ai.server.health import get_models

        mock_response = MagicMock()
        mock_response.status_code = 500  # Internal Server Error

        with patch("httpx.get", return_value=mock_response):
            result = get_models(host="127.0.0.1", port=8080)

        assert result == []

    def test_get_models_returns_empty_list_on_connection_error(self) -> None:
        """Should return empty list when connection fails.

        Covers lines 34-36: RequestError exception handling.
        """
        from local_ai.server.health import get_models

        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = get_models(host="127.0.0.1", port=8080)

        assert result == []

    def test_get_models_returns_model_ids_on_success(self) -> None:
        """Should return list of model IDs when server responds successfully."""
        from local_ai.server.health import get_models

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1", "object": "model"},
                {"id": "model-2", "object": "model"},
            ]
        }

        with patch("httpx.get", return_value=mock_response):
            result = get_models(host="127.0.0.1", port=8080)

        assert result == ["model-1", "model-2"]

    def test_get_models_handles_missing_id_field(self) -> None:
        """Should use 'unknown' when model entry lacks 'id' field."""
        from local_ai.server.health import get_models

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"object": "model"},  # Missing 'id'
                {"id": "model-with-id", "object": "model"},
            ]
        }

        with patch("httpx.get", return_value=mock_response):
            result = get_models(host="127.0.0.1", port=8080)

        assert result == ["unknown", "model-with-id"]

    def test_get_models_handles_timeout_error(self) -> None:
        """Should return empty list when request times out.

        Covers line 35: httpx.TimeoutException is a RequestError subclass.
        """
        from local_ai.server.health import get_models

        with patch("httpx.get", side_effect=httpx.TimeoutException("Request timed out")):
            result = get_models(host="127.0.0.1", port=8080)

        assert result == []
