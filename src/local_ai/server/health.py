"""Health checking module for local AI server."""

import time

import httpx


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
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            return "healthy"
        return "unhealthy"
    except httpx.RequestError:
        return "unknown"


def wait_for_health(
    host: str, port: int, timeout: float = 60.0, interval: float = 1.0
) -> bool:
    """Wait for server to become healthy.

    Args:
        host: Server host
        port: Server port
        timeout: Total time to wait in seconds
        interval: Time between checks in seconds

    Returns:
        True if healthy within timeout, False otherwise
    """
    start = time.monotonic()
    while True:
        status = check_health(host, port)
        if status == "healthy":
            return True
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            return False
        time.sleep(interval)
