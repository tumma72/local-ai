"""Health checking module for local AI server."""

import time

import httpx

from local_ai.logging import get_logger

_logger = get_logger("Health")


def get_models(host: str, port: int, timeout: float = 5.0) -> list[str]:
    """Query server for loaded models.

    Args:
        host: Server host
        port: Server port
        timeout: Request timeout in seconds

    Returns:
        List of model IDs, or empty list if query fails
    """
    url = f"http://{host}:{port}/v1/models"
    _logger.debug("Querying models at {}", url)
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("id", "unknown") for m in data.get("data", [])]
            _logger.debug("Found {} models: {}", len(models), models)
            return models
        _logger.warning("Failed to query models: status={}", response.status_code)
        return []
    except httpx.RequestError as e:
        _logger.debug("Failed to query models: {} ({})", e, type(e).__name__)
        return []


def check_health(host: str, port: int, timeout: float = 5.0) -> str:
    """Check server health via /v1/models endpoint.

    Args:
        host: Server host
        port: Server port
        timeout: Request timeout in seconds

    Returns:
        "healthy" if /v1/models returns 200
        "unhealthy" if non-200 response
        "unknown" if connection fails
    """
    url = f"http://{host}:{port}/v1/models"
    _logger.debug("Checking health at {}", url)
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            _logger.debug("Health check passed (status=200)")
            return "healthy"
        _logger.warning("Health check failed: status={}", response.status_code)
        return "unhealthy"
    except httpx.RequestError as e:
        _logger.debug("Health check failed: connection error ({})", type(e).__name__)
        return "unknown"


def wait_for_health(host: str, port: int, timeout: float = 60.0, interval: float = 1.0) -> bool:
    """Wait for server to become healthy.

    Args:
        host: Server host
        port: Server port
        timeout: Total time to wait in seconds
        interval: Time between checks in seconds

    Returns:
        True if healthy within timeout, False otherwise
    """
    _logger.info("Waiting for server to become healthy (timeout={}s)", timeout)
    start = time.monotonic()
    check_count = 0
    while True:
        check_count += 1
        status = check_health(host, port)
        if status == "healthy":
            elapsed = time.monotonic() - start
            _logger.info(
                "Server became healthy after {:.1f}s ({} checks)",
                elapsed,
                check_count,
            )
            return True
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            _logger.warning(
                "Server did not become healthy within {}s ({} checks)",
                timeout,
                check_count,
            )
            return False
        time.sleep(interval)
