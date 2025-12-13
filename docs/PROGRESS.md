# Progress: MLX Omni Server Migration

## Current Status

**Phase**: Phase 3 - Design-Driven Implementation
**Goal**: Migrate from mlx-lm server to MLX Omni Server

## Research Findings (Complete)

MLX Omni Server tested and validated:
- `/v1/models` response: **2ms** (vs ~5000ms with mlx-lm server)
- `/anthropic/v1/models` response: **5ms**
- Dual API support eliminates LiteLLM proxy requirement
- FastAPI-based, production-ready

## Code Changes Required

### 1. Update Dependencies (`pyproject.toml`)

**Before:**
```toml
dependencies = [
    ...
    "mlx-lm>=0.28.4",
    ...
]
```

**After:**
```toml
dependencies = [
    ...
    "mlx-omni-server>=0.0.7",
    "mlx-lm<0.28.3",  # Required by mlx-omni-server (API change in 0.28.3+)
    ...
]
```

### 2. Update Server Command (`src/local_ai/server/manager.py`)

**Before (lines 143-155):**
```python
# Use newer mlx_lm command syntax (python -m mlx_lm server instead of mlx_lm.server)
cmd = [
    "python",
    "-m",
    "mlx_lm",
    "server",
    "--host",
    self._settings.server.host,
    "--port",
    str(self._settings.server.port),
    "--model",
    self._settings.model.path,
]
```

**After:**
```python
# Use MLX Omni Server for dual OpenAI/Anthropic API support
# Note: mlx-omni-server loads models dynamically per request
cmd = [
    "mlx-omni-server",
    "--host",
    self._settings.server.host,
    "--port",
    str(self._settings.server.port),
]
```

**Key difference:** mlx-omni-server doesn't take a `--model` argument - it loads models dynamically when requests include the model name in the request body.

### 3. Update Docstrings (`src/local_ai/server/manager.py`)

- Line 3: "MLX LM server" → "MLX Omni Server"
- Line 50: "MLX LM server" → "MLX Omni Server"

### 4. Reinstall Dependencies

```bash
uv pip uninstall mlx-lm
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv pip install mlx-omni-server
uv pip install "mlx-lm<0.28.3"
```

## Test Plan

1. Run unit tests (should pass - mock subprocess.Popen is backend-agnostic)
2. Run integration test: `local-ai server start` → `status` → `stop`
3. Verify OpenAI API: `curl http://localhost:8080/v1/models`
4. Verify Anthropic API: `curl http://localhost:8080/anthropic/v1/models`

## Files Changed

| File | Change Type |
|------|-------------|
| `docs/VISION.md` | Updated integration targets, ML backend |
| `docs/DESIGN.md` | Complete rewrite for MLX Omni Server |
| `docs/PROGRESS.md` | Created (this file) |
| `pyproject.toml` | Updated dependencies |
| `src/local_ai/server/manager.py` | Updated command and docstrings |
| `tests/integration/test_server_lifecycle.py` | Removed --model assertion |

## Status: COMPLETE

All tests pass (108/108). Migration verified with manual integration test:
- `local-ai server start --model mlx-community/Orchestrator-8B-8bit` - Success
- `/v1/models` - Returns model list in 2ms
- `/anthropic/v1/models` - Returns model list in Anthropic format
- `local-ai server stop` - Success
