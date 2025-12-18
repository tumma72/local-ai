# local-ai

[![Tests](https://github.com/tumma72/local-ai/actions/workflows/test.yml/badge.svg)](https://github.com/tumma72/local-ai/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/tumma72/local-ai)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![Release](https://img.shields.io/github/v/release/tumma72/local-ai?include_prereleases)](https://github.com/tumma72/local-ai/releases)

**Run AI models locally on your Mac with zero cloud dependencies.**

local-ai brings the power of large language models to your Apple Silicon Mac, completely offline. No API keys, no usage limits, no data leaving your machine.

## Why local-ai?

### The Problem

- **Privacy concerns**: Cloud AI services see all your code, prompts, and data
- **API costs**: Pay-per-token pricing adds up quickly for heavy usage
- **Rate limits**: Cloud providers throttle requests during peak times
- **Internet dependency**: No connection = no AI assistance
- **Latency**: Round-trip to cloud servers adds delay to every interaction

### The Solution

local-ai runs models directly on your Mac's GPU using Apple's MLX framework:

- **100% Private**: Your data never leaves your machine
- **Zero Cost**: No API fees, subscriptions, or usage limits
- **Always Available**: Works offline, on planes, in secure environments
- **Low Latency**: Direct GPU inference, no network round-trips
- **OpenAI Compatible**: Works with existing tools that support OpenAI's API

## Features

- **One-Command Server**: Start a local LLM server with `local-ai server start`
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI clients
- **Model Browser**: Discover and download optimized MLX models from Hugging Face
- **Hardware Detection**: Automatically detects your Mac's capabilities
- **Smart Recommendations**: Suggests models that fit your available memory
- **Web Interface**: Built-in chat UI for testing models at `http://localhost:8080`
- **Tool Calling**: Function calling support for agentic workflows
- **Rich CLI**: Beautiful terminal output with progress bars and status panels

## Quick Start

### Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/tumma72/local-ai.git
cd local-ai

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Start the server (models load dynamically)
local-ai server start

# Open http://localhost:8080 in your browser for the web UI

# Or use with any OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Discover Models

```bash
# See recommended models for your hardware
local-ai models recommend

# Search for specific models
local-ai models search "llama 8b"

# Get detailed model info
local-ai models info mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Server Management

```bash
# Check server status
local-ai server status

# View server logs
local-ai server logs --follow

# Restart with new settings
local-ai server restart --port 9000

# Stop the server
local-ai server stop
```

## Configuration

Create a `config.toml` file for persistent settings:

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "INFO"

[model]
# Default model (optional - models load dynamically)
path = "mlx-community/Qwen3-0.6B-4bit"
```

Use with CLI:

```bash
local-ai server start --config config.toml
```

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11+**
- **8GB+ RAM** recommended (16GB+ for larger models)

## Use Cases

### IDE Integration

local-ai works with any IDE that supports OpenAI-compatible endpoints:

- **VS Code** with Continue extension
- **Cursor** (set custom API endpoint)
- **Zed** editor (configure assistant)
- **JetBrains IDEs** with AI plugins

### Claude Code / Aider / Other Tools

```bash
# Set environment variables
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=not-needed

# Use your favorite AI coding tool
aider --model mlx-community/Qwen3-0.6B-4bit
```

### Python Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mlx-community/Qwen3-0.6B-4bit",
    messages=[{"role": "user", "content": "Explain Python decorators"}]
)
print(response.choices[0].message.content)
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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-omni-server](https://github.com/AlenaSamodurova/mlx-omni-server) - OpenAI-compatible server (with local patches)
- [Hugging Face](https://huggingface.co) - Model hosting and community
