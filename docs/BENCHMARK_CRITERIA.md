# Benchmark Evaluation Criteria

This document defines the criteria for evaluating LLM-generated code during benchmarks. These criteria enable consistent, reproducible quality assessment across different models.

## Overview

The benchmark system measures two types of metrics:

1. **Quantitative Metrics** - Automatically collected during benchmark runs
2. **Qualitative Metrics** - Evaluated by reviewing generated code artifacts

## Quantitative Metrics

These metrics are automatically collected by the benchmark runner:

| Metric | Description | Unit |
|--------|-------------|------|
| **Tokens per Second** | Output generation rate (primary performance metric) | tok/s |
| **TTFT** | Time to First Token - latency before generation starts | ms |
| **Total Latency** | Total request time including all tokens | ms |
| **Success Rate** | Percentage of requests that complete without error | % |
| **Memory Usage** | Peak memory during inference | MB |

## Qualitative Evaluation Criteria

When evaluating generated code quality, use these criteria with scores from 0-10:

### 1. Correctness (0-10)

Does the code solve the stated problem?

- **10**: Perfect solution, handles all requirements correctly
- **8-9**: Mostly correct with minor issues
- **5-7**: Partially correct, some requirements not met
- **2-4**: Significant errors or missing functionality
- **0-1**: Does not address the problem

**Checkpoints:**
- [ ] Code compiles/parses without syntax errors
- [ ] Core functionality works as specified
- [ ] Edge cases handled appropriately
- [ ] No logical errors in control flow

### 2. Completeness (0-10)

Are all requirements addressed in the solution?

- **10**: All requirements fully implemented
- **8-9**: Minor features missing or partially implemented
- **5-7**: Some requirements not addressed
- **2-4**: Many requirements missing
- **0-1**: Most requirements not addressed

**Checkpoints:**
- [ ] All specified features present
- [ ] Required endpoints/functions implemented
- [ ] Data models match specification
- [ ] Input validation present where required

### 3. Code Quality (0-10)

Is the code idiomatic, readable, and maintainable?

- **10**: Exemplary code following best practices
- **8-9**: Clean code with minor style issues
- **5-7**: Functional but could be cleaner
- **2-4**: Hard to read or maintain
- **0-1**: Poor quality, unmaintainable

**Checkpoints:**
- [ ] Follows language idioms and conventions
- [ ] Appropriate naming for variables/functions
- [ ] Proper code organization and structure
- [ ] Reasonable function/method lengths
- [ ] Type hints present (for Python)

### 4. Error Handling (0-10)

Does the code handle errors gracefully?

- **10**: Comprehensive error handling with informative messages
- **8-9**: Good error handling, minor gaps
- **5-7**: Basic error handling present
- **2-4**: Minimal or incorrect error handling
- **0-1**: No error handling, crashes on invalid input

**Checkpoints:**
- [ ] Invalid inputs handled gracefully
- [ ] External errors (network, file) caught
- [ ] Error messages are informative
- [ ] HTTP status codes appropriate (for APIs)

## Evaluation Process

### Using Claude Code for Evaluation

When evaluating generated code with Claude Code:

1. **Read the generated output file**
   ```bash
   # Results are stored in ~/.local/state/local-ai/benchmarks/
   cat ~/.local/state/local-ai/benchmarks/<result-file>.json | jq '.runs[0].raw_output'
   ```

2. **Apply evaluation criteria consistently**
   - Review against each criterion above
   - Assign scores based on the 0-10 scale
   - Document specific issues found

3. **Record evaluation results**
   - Note the model, task, and scores
   - Include specific observations
   - Compare across multiple models using same criteria

### Example Evaluation

```
Model: mlx-community/Qwen3-Coder-8B
Task: todo-api
Run: 2024-01-15

Correctness: 9/10
- FastAPI app correctly structured
- All CRUD endpoints implemented
- Minor: Missing input validation on PUT

Completeness: 8/10
- All required endpoints present
- Pydantic models correct
- Missing: created_at field not implemented

Code Quality: 9/10
- Clean, idiomatic Python
- Good function naming
- Type hints present

Error Handling: 7/10
- 404 errors handled
- Missing validation error handling
- No try/except for edge cases

Overall: 33/40 (82.5%)
```

## Comparison Guidelines

When comparing models:

1. **Run identical tasks** - Use the same benchmark task for all models
2. **Multiple runs** - Use at least 3 runs to account for variance
3. **Same conditions** - Test under similar system load conditions
4. **Document context** - Note hardware, model quantization, etc.

### Comparison Table Format

| Model | Task | Tok/s | TTFT | Correctness | Completeness | Quality | Errors | Total |
|-------|------|-------|------|-------------|--------------|---------|--------|-------|
| Model A | todo-api | 45.2 | 120ms | 9 | 8 | 9 | 7 | 33/40 |
| Model B | todo-api | 38.1 | 150ms | 8 | 9 | 8 | 8 | 33/40 |

## Task-Specific Criteria

### todo-api Task

For the `todo-api` benchmark task, verify:

- [ ] FastAPI app structure present
- [ ] Pydantic model for Todo items
- [ ] GET /todos endpoint (list all)
- [ ] POST /todos endpoint (create)
- [ ] GET /todos/{id} endpoint (get one)
- [ ] PUT /todos/{id} endpoint (update)
- [ ] DELETE /todos/{id} endpoint (delete)
- [ ] Proper HTTP status codes (201 for create, 404 for not found)
- [ ] In-memory storage implementation

## Adding New Tasks

When creating new benchmark tasks:

1. Define clear, testable requirements
2. Specify expected output format
3. Add task-specific evaluation criteria to this document
4. Include example correct solutions for reference
