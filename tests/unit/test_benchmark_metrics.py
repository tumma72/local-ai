"""Behavioral tests for benchmark metrics module.

Tests verify public behavior of streaming metrics collection:
- measure_streaming_request captures TTFT on first content chunk
- measure_streaming_request calculates tokens per second
- measure_streaming_request handles errors gracefully
- MemoryTracker captures baseline and peak memory
"""

from unittest.mock import MagicMock, patch

import pytest

from local_ai.benchmark.metrics import MemoryTracker, measure_streaming_request
from local_ai.benchmark.schema import ThroughputMetrics, TimingMetrics


class MockStreamChunk:
    """Mock OpenAI stream chunk."""

    def __init__(
        self,
        content: str | None = None,
        usage: dict | None = None,
    ):
        self.choices = []
        if content is not None:
            delta = MagicMock()
            delta.content = content
            choice = MagicMock()
            choice.delta = delta
            self.choices = [choice]

        self.usage = None
        if usage:
            self.usage = MagicMock()
            self.usage.prompt_tokens = usage.get("prompt_tokens", 0)
            self.usage.completion_tokens = usage.get("completion_tokens", 0)


class MockStream:
    """Mock OpenAI streaming response."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = chunks
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class TestMeasureStreamingRequest:
    """Verify streaming request measurement behavior."""

    @pytest.fixture
    def mock_stream_chunks(self) -> list[MockStreamChunk]:
        """Create mock stream chunks for chat completion."""
        return [
            MockStreamChunk(content=None),  # Role chunk
            MockStreamChunk(content="def "),
            MockStreamChunk(content="hello"),
            MockStreamChunk(content="():\n"),
            MockStreamChunk(content="    print"),
            MockStreamChunk(content="('Hello')"),
            MockStreamChunk(
                content=None,
                usage={"prompt_tokens": 50, "completion_tokens": 10},
            ),
        ]

    @pytest.mark.asyncio
    async def test_captures_ttft_on_first_content_chunk(
        self, mock_stream_chunks: list[MockStreamChunk]
    ) -> None:
        """Should record TTFT when first content chunk arrives."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(mock_stream_chunks)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            timing, throughput, output = await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # TTFT should be positive (first content chunk)
        assert isinstance(timing, TimingMetrics)
        assert timing.ttft_ms >= 0
        assert timing.total_latency_ms >= timing.ttft_ms

    @pytest.mark.asyncio
    async def test_calculates_tokens_per_second(
        self, mock_stream_chunks: list[MockStreamChunk]
    ) -> None:
        """Should compute throughput as completion_tokens / generation_time."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(mock_stream_chunks)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            timing, throughput, output = await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(throughput, ThroughputMetrics)
        assert throughput.completion_tokens >= 0
        # tokens_per_second should be non-negative
        assert throughput.tokens_per_second >= 0

    @pytest.mark.asyncio
    async def test_concatenates_output_from_chunks(
        self, mock_stream_chunks: list[MockStreamChunk]
    ) -> None:
        """Should concatenate all content chunks into output string."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(mock_stream_chunks)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            timing, throughput, output = await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert "def hello" in output
        assert "print" in output

    @pytest.mark.asyncio
    async def test_raises_on_connection_error(self) -> None:
        """Should raise exception on connection failure."""
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client), \
                pytest.raises(APIConnectionError):
            await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_injects_request_id_for_cache_bypass(
        self, mock_stream_chunks: list[MockStreamChunk]
    ) -> None:
        """Should inject unique request ID when bypass_cache=True."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(mock_stream_chunks)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                bypass_cache=True,
            )

        # Check that messages were modified with request ID
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]
        assert "[rid:" in system_msg["content"]


class TestMemoryTracker:
    """Verify MemoryTracker captures memory usage."""

    def test_captures_baseline_memory_on_start(self) -> None:
        """Should record baseline memory when start() is called."""
        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=1024.0):
            tracker = MemoryTracker()
            tracker.start()

        assert tracker.baseline == 1024.0

    def test_tracks_peak_memory_on_sample(self) -> None:
        """Should update peak memory when sample() is called with higher value."""
        tracker = MemoryTracker()

        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=1000.0):
            tracker.start()

        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=1500.0):
            tracker.sample()

        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=1200.0):
            tracker.sample()  # Lower, should not update peak

        assert tracker.baseline == 1000.0
        assert tracker.peak == 1500.0

    def test_get_metrics_returns_memory_metrics(self) -> None:
        """Should return MemoryMetrics with baseline and peak values."""
        tracker = MemoryTracker()

        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=1000.0):
            tracker.start()

        with patch("local_ai.benchmark.metrics.get_memory_mb", return_value=2000.0):
            tracker.sample()

        metrics = tracker.get_metrics()

        assert metrics.baseline_memory_mb == 1000.0
        assert metrics.peak_memory_mb == 2000.0


class TestGetMemoryMb:
    """Verify get_memory_mb function behavior (lines 28-32)."""

    def test_get_memory_with_valid_pid(self) -> None:
        """Should return memory for specified process when PID is valid.

        Covers lines 28-30: pid provided and process exists.
        """
        from local_ai.benchmark.metrics import get_memory_mb
        import os

        # Use current process PID which definitely exists
        current_pid = os.getpid()
        result = get_memory_mb(pid=current_pid)

        # Should return positive memory value
        assert result > 0
        assert isinstance(result, float)

    def test_get_memory_with_invalid_pid_falls_back_to_current_process(self) -> None:
        """Should fall back to current process when PID does not exist.

        Covers lines 31-32: NoSuchProcess exception caught, fallback triggered.
        """
        from local_ai.benchmark.metrics import get_memory_mb

        # Use an invalid PID that definitely doesn't exist
        invalid_pid = 999999999
        result = get_memory_mb(pid=invalid_pid)

        # Should still return a positive value (from current process fallback)
        assert result > 0
        assert isinstance(result, float)

    def test_get_memory_without_pid(self) -> None:
        """Should return current process memory when no PID provided."""
        from local_ai.benchmark.metrics import get_memory_mb

        result = get_memory_mb()

        assert result > 0
        assert isinstance(result, float)


class TestMeasureStreamingRequestEdgeCases:
    """Verify edge cases in measure_streaming_request (lines 191-192)."""

    @pytest.mark.asyncio
    async def test_estimates_tokens_when_usage_not_provided(self) -> None:
        """Should estimate completion tokens when server doesn't provide usage.

        Covers lines 191-192: completion_tokens == 0 and output exists.
        """
        # Create stream chunks without usage information
        chunks_without_usage = [
            MockStreamChunk(content="def "),
            MockStreamChunk(content="hello"),
            MockStreamChunk(content="():\n"),
            MockStreamChunk(content="    print"),
            MockStreamChunk(content="('Hello World!')"),
            # Final chunk WITHOUT usage
            MockStreamChunk(content=None, usage=None),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks_without_usage)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            timing, throughput, output = await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Output should be concatenated
        assert "def hello" in output
        # Tokens should be estimated (roughly len(output) // 4)
        assert throughput.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_handles_no_system_message_in_cache_bypass(self) -> None:
        """Should add system message with request ID when none exists.

        Tests the _inject_request_id function path when no system message present.
        """
        chunks = [
            MockStreamChunk(content="response"),
            MockStreamChunk(content=None, usage={"prompt_tokens": 10, "completion_tokens": 5}),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        with patch("local_ai.benchmark.metrics.OpenAI", return_value=mock_client):
            await measure_streaming_request(
                host="127.0.0.1",
                port=8080,
                model="test-model",
                # Only user message, no system message
                messages=[{"role": "user", "content": "Hello"}],
                bypass_cache=True,
            )

        # Check that a system message with request ID was injected
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # First message should now be a system message with rid
        assert messages[0]["role"] == "system"
        assert "[rid:" in messages[0]["content"]
