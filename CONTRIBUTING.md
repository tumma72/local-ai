# Contributing to local-ai

Thank you for your interest in contributing to local-ai! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## How to Contribute

### Reporting Issues

- Check if the issue already exists in [GitHub Issues](https://github.com/tumma72/local-ai/issues)
- Use a clear, descriptive title
- Include steps to reproduce the issue
- Include your environment details (macOS version, Python version, Mac model)
- Include relevant logs or error messages

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the use case and why it would be valuable
- Consider if it aligns with the project's goal of simple, local AI

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**:
   ```bash
   uv sync
   ```
3. **Make your changes** with clear, focused commits
4. **Add tests** for new functionality
5. **Ensure all tests pass**:
   ```bash
   uv run pytest
   ```
6. **Check code quality**:
   ```bash
   uv run ruff check src/
   uv run mypy src/
   ```
7. **Update documentation** if needed
8. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Getting Started

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/local-ai.git
cd local-ai

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest

# Start development server
uv run local-ai server start
```

### Project Structure

```
local-ai/
├── src/local_ai/
│   ├── cli/           # Command-line interface
│   ├── config/        # Configuration handling
│   ├── hardware/      # Apple Silicon detection
│   ├── models/        # Model management
│   ├── output/        # Rich terminal output
│   └── server/        # Server management & web UI
├── tests/             # Test suite
├── lib/               # Local dependencies (mlx-omni-server)
└── docs/              # Documentation
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=local_ai --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run with verbose output
uv run pytest -v
```

### Code Style

We use:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **Black-compatible** formatting via Ruff

```bash
# Check linting
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Type checking
uv run mypy src/
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Reference issues when applicable: "Fix #123: Handle missing config file"

Examples:
- `Add model search command`
- `Fix server restart not waiting for shutdown`
- `Update README with installation instructions`

## Testing Guidelines

- Write tests for new functionality
- Maintain or improve code coverage (currently 95%+)
- Use pytest fixtures for common setup
- Mock external services (Hugging Face API, etc.)
- Test both success and error cases

## Documentation

- Update README.md for user-facing changes
- Add docstrings to public functions and classes
- Include type hints for all function signatures

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG if present
3. Create a GitHub release with tag
4. Build and publish to PyPI

## Questions?

- Open a [GitHub Discussion](https://github.com/tumma72/local-ai/discussions)
- Check existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
