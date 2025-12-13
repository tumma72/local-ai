# local-ai

Offline AI coding assistant for Apple Silicon. Serves local LLMs via MLX with an OpenAI-compatible API.

## Features

- **Server Management**: Start, stop, and monitor a local LLM server
- **OpenAI-Compatible API**: Works with any client that supports OpenAI's API format
- **Apple Silicon Optimized**: Uses MLX for Metal acceleration on M-series chips
- **TOML Configuration**: Simple configuration with CLI overrides
- **Rich CLI Output**: Beautiful terminal output with status panels and colors

## Installation

```bash
# Requires Python 3.14+ and uv
uv pip install local-ai
```

## Quick Start

```bash
# Start server with a model
local-ai server start --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Check server status
local-ai server status

# Stop server
local-ai server stop
```

## Configuration

Create a `config.toml` file:

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "INFO"

[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[generation]
max_tokens = 4096
temperature = 0.0
```

Use with CLI:

```bash
local-ai server start --config config.toml
```

## Development

```bash
# Install development dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=local_ai

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## License

MIT
