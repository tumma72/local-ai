# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0a0] - 2024-12-18

### Added
- **Welcome Page Redesign**: Two-column layout with configuration panel (left) and chat window (right)
- **Model Parameters Panel**: Collapsible panel with temperature, top_p, max_tokens, and seed controls
- **System Prompt Presets**: Built-in presets (Coding Assistant, Creative Writer, Data Analyst, Concise Helper) plus custom prompt support
- **Markdown Rendering**: Chat messages now render markdown with syntax highlighting for code blocks
- **Tool Calling Support**: Built-in tools (calculator, get_time, get_weather) for testing function calling
- **Export Chat**: Copy conversation transcript to clipboard in markdown format
- **Tool Status Indicator**: Visual feedback when tools are enabled but not called by the model
- **Server Restart Command**: `local-ai server restart` for quick server restarts
- **Apache 2.0 License**: Changed from MIT to Apache 2.0
- **Contributing Guidelines**: Added CONTRIBUTING.md with development setup and PR process
- **GitHub Workflows**: Added test.yml for CI and release-and-publish.yml for automated releases

### Changed
- **Python Version**: Switched to Python 3.11-3.13 (from 3.14) due to Rust dependency compatibility (outlines-core via PyO3)
- **README**: Complete rewrite with badges, problem/solution framing, feature list, and use cases
- Removed server status panel from welcome page (redundant - if page loads, server is running)

### Fixed
- Fixed `from __future__ import annotations` for deferred annotation evaluation in Python 3.13
- Fixed test using `pytest.Mock` → `MagicMock`

## [0.1.0] - 2024-12-14

### Added
- Initial release
- Server management commands: start, stop, status, logs
- OpenAI-compatible API endpoint
- Model discovery and search from Hugging Face Hub
- Hardware detection for Apple Silicon
- Smart model recommendations based on available memory
- TOML configuration support
- Rich CLI output with progress bars and status panels
- Welcome page with model tester
- Thinking tag parsing for reasoning models
