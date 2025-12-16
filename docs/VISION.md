# Vision: local-ai

## Purpose

**local-ai** enables offline AI-assisted coding by serving high-performance local LLMs optimized for Apple Silicon. It provides a unified CLI to manage, serve, and interact with local models while maintaining compatibility with popular development tools.

## Problem Statement

Developers using Apple Silicon Macs need reliable offline access to AI coding assistants. Current solutions either:
- Require constant internet connectivity
- Are not optimized for Apple Silicon (Metal, unified memory)
- Lack unified tooling for model management and serving
- Have complex configuration spread across multiple tools

## Vision Statement

A single, well-crafted CLI that makes local LLM serving on Apple Silicon as simple as:
```bash
local-ai serve --model qwen3-coder-30b
```

With the confidence that:
- The model is optimally quantized for your hardware
- The server exposes the right API endpoints for your tools
- Performance is maximized for your specific Mac configuration

## Goals

### G1: Offline Coding with Zed
**"I want to code with Zed editor even when I am offline using AI assistants served locally"**

Success Criteria:
- Server starts and serves an OpenAI-compatible API endpoint
- Zed connects and provides code completions/chat without internet
- Systemctl-like interface: `local-ai server start|stop|restart|status`

### G2: Model Discovery
**"I want to search for a model by name on HuggingFace, and find the most appropriate version"**

Success Criteria:
- Search HuggingFace by model name
- Filter/rank results by: MLX optimization, quantization level, Apple Silicon compatibility
- Clear output showing model variants with relevant metadata
- Single command: `local-ai models search <query>`

### G3: Model Conversion
**"If an optimized version isn't available, I want to convert GGUF to MLX format"**

Success Criteria:
- Download standard model formats from HuggingFace
- Convert to MLX-optimized format with appropriate quantization
- Store converted models in organized local cache
- Single command: `local-ai models convert <model-id> --quantize <level>`

### G4: Unified CLI Experience
**"I want to do all of the above using a single CLI with nice output and solid error handling"**

Success Criteria:
- Consistent command structure across all operations
- Rich, informative console output with progress indicators
- Clear error messages with actionable troubleshooting steps
- All errors point to specific files, configurations, or commands to check

### G5: Interactive Model Testing Interface
"""I want a simple web interface to quickly test all available models and validate they load correctly before integrating with clients"""

Success Criteria:
- Root endpoint (`/`) displays welcome page with server status information
- Interactive chat window with dropdown menu of available models
- Real-time model loading and response display
- Error handling and debugging information for failed model loads
- Lightweight implementation (no full frontend framework required)

## Target Models

Primary focus for coding assistance:
- **Qwen3-Coder-30B**: Strong coding performance, good size for 128GB
- **Devstral2-24B**: Coding-focused, efficient
- **DeepSeek-3.2**: Latest DeepSeek with coding capabilities
- **GLM-4.6 (Air)**: Evaluate for balance of performance/size

Secondary focus for general assistance:
- **Coordinator-8B**: For orchestration and non-coding tasks
- **MiniMax-M2 / Kimi-K2**: Evaluate for general capabilities

## Performance Priorities

Primary metric: **Tokens per second (tok/s)**

Full benchmarking includes:
1. **Throughput**: Tokens per second during generation
2. **TTFT**: Time to first token (interactive responsiveness)
3. **Memory**: Peak and sustained memory usage
4. **Concurrency**: Behavior under multiple simultaneous requests

Trade-off principle: Optimize for tokens/sec first, then balance other metrics.

## Integration Targets

| Tool | Integration Method |
|------|-------------------|
| Zed Editor | OpenAI-compatible API endpoint (`/v1/*`) |
| Black Goose AI CLI | OpenAI-compatible API endpoint (`/v1/*`) |
| Claude Code | Anthropic-compatible API endpoint (`/anthropic/v1/*`) |

**Note**: MLX Omni Server provides both OpenAI and Anthropic API endpoints natively, eliminating the need for a LiteLLM proxy.

## Technical Constraints

- **Python**: 3.14.2 (latest stable)
- **ML Backend**: MLX Omni Server (wraps mlx-lm for Metal-optimized inference)
- **API**: OpenAI-compatible endpoints
- **Configuration**: TOML format with Pydantic v2 validation
- **CLI Framework**: Typer with Rich output
- **Async**: Where I/O benefits (downloads, API calls)

## Non-Goals

To keep scope minimal, we explicitly will NOT:
- Build a general-purpose model hosting platform
- Support non-Apple-Silicon hardware
- Create a web UI
- Implement user authentication or multi-tenancy
- Build model training or fine-tuning capabilities
- Support every possible model format (focus on MLX and GGUF)

## Success Definition

The project succeeds when a developer can:

1. Install with `uv pip install local-ai`
2. Search and download a model with `local-ai models add qwen3-coder-30b`
3. Start serving with `local-ai server start`
4. Configure Zed to use `http://localhost:8080/v1` as the AI endpoint
5. Code offline with responsive AI assistance

All while seeing clear progress, helpful errors, and achieving >30 tok/s on an M4 Max.

---

## Related Documents

- **[PROCESS.md](./PROCESS.md)**: How we work together to achieve these goals
- **[DESIGN.md](./DESIGN.md)**: Technical architecture (created during Phase 3)
