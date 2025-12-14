"""Streaming metrics collection for benchmark runs.

Provides precise TTFT and tokens/second measurement using OpenAI SDK.
"""

import time
import uuid

import psutil
from openai import OpenAI

from local_ai.benchmark.schema import MemoryMetrics, ThroughputMetrics, TimingMetrics
from local_ai.logging import get_logger

_logger = get_logger("Benchmark.metrics")


def get_memory_mb(pid: int | None = None) -> float:
    """Get current memory usage in MB.

    Args:
        pid: Process ID to measure. If None, measures current process.

    Returns:
        Memory usage in megabytes.
    """
    if pid:
        try:
            process = psutil.Process(pid)
            return process.memory_info().rss / (1024 * 1024)
        except psutil.NoSuchProcess:
            pass
    # Fallback to current process
    return psutil.Process().memory_info().rss / (1024 * 1024)


class MemoryTracker:
    """Track memory usage during benchmark execution."""

    def __init__(self, server_pid: int | None = None) -> None:
        """Initialize memory tracker.

        Args:
            server_pid: Optional server process ID to track.
        """
        self._server_pid = server_pid
        self.baseline: float = 0.0
        self.peak: float = 0.0

    def start(self) -> None:
        """Record baseline memory at start of benchmark."""
        self.baseline = get_memory_mb(self._server_pid)
        self.peak = self.baseline
        _logger.debug("Memory baseline: {:.1f} MB", self.baseline)

    def sample(self) -> None:
        """Sample current memory and update peak if higher."""
        current = get_memory_mb(self._server_pid)
        if current > self.peak:
            self.peak = current
            _logger.debug("New peak memory: {:.1f} MB", self.peak)

    def get_metrics(self) -> MemoryMetrics:
        """Get memory metrics.

        Returns:
            MemoryMetrics with baseline and peak values.
        """
        return MemoryMetrics(
            baseline_memory_mb=self.baseline,
            peak_memory_mb=self.peak,
        )


def _inject_request_id(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Inject unique request ID into system message to bypass prompt cache.

    Args:
        messages: Original messages list.

    Returns:
        Messages with unique request ID in system prompt.
    """
    request_id = str(uuid.uuid4())[:8]
    result = []

    for msg in messages:
        if msg.get("role") == "system":
            # Prepend request ID to system message
            content = msg.get("content", "")
            result.append({
                "role": "system",
                "content": f"[rid:{request_id}] {content}",
            })
        else:
            result.append(msg.copy())

    # If no system message, add one with just the request ID
    if not any(m.get("role") == "system" for m in messages):
        result.insert(0, {"role": "system", "content": f"[rid:{request_id}]"})

    return result


async def measure_streaming_request(
    host: str,
    port: int,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 300.0,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    bypass_cache: bool = True,
) -> tuple[TimingMetrics, ThroughputMetrics, str]:
    """Execute streaming request and measure timing metrics.

    Uses OpenAI SDK with streaming to measure precise TTFT (time to first token)
    and tokens per second throughput.

    Args:
        host: Server host.
        port: Server port.
        model: Model identifier.
        messages: Chat messages to send.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
        bypass_cache: If True, inject unique ID to bypass prompt cache.

    Returns:
        Tuple of (TimingMetrics, ThroughputMetrics, output_string).

    Raises:
        openai.APIError: On API errors.
    """
    # Prepare messages with optional cache bypass
    request_messages = _inject_request_id(messages) if bypass_cache else messages

    # Create OpenAI client pointing to local server
    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="not-needed",
        timeout=timeout,
    )

    _logger.debug("Starting streaming request to {}:{} for model {}", host, port, model)

    start_time = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    # Use streaming to capture TTFT
    # Note: type ignore needed because we use simple dict messages instead of typed params
    stream = client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=request_messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        stream_options={"include_usage": True},
    )
    with stream:
        for chunk in stream:
            # Check for content
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                # Record TTFT on first content
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000
                    _logger.debug("TTFT: {:.2f} ms", ttft)

                chunks.append(content)

            # Extract usage from final chunk
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

    end_time = time.perf_counter()
    total_latency_ms = (end_time - start_time) * 1000
    generation_time_ms = total_latency_ms - (ttft or 0)

    output = "".join(chunks)

    # Estimate tokens if not provided in response
    if completion_tokens == 0 and output:
        # Rough estimate: ~4 characters per token
        completion_tokens = len(output) // 4
        _logger.debug("Estimated completion tokens: {}", completion_tokens)

    # Calculate tokens per second
    tokens_per_second = 0.0
    if generation_time_ms > 0:
        tokens_per_second = completion_tokens / (generation_time_ms / 1000)

    # Calculate prompt processing rate
    prompt_processing_rate = 0.0
    if ttft and ttft > 0 and prompt_tokens > 0:
        prompt_processing_rate = prompt_tokens / (ttft / 1000)

    _logger.info(
        "Completed: {:.2f} tok/s, TTFT={:.0f}ms, total={:.0f}ms",
        tokens_per_second, ttft or 0, total_latency_ms,
    )

    timing = TimingMetrics(
        ttft_ms=ttft or 0,
        total_latency_ms=total_latency_ms,
        generation_time_ms=generation_time_ms,
    )

    throughput = ThroughputMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=tokens_per_second,
        prompt_processing_rate=prompt_processing_rate,
    )

    return timing, throughput, output
