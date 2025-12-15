# CLI UI/UX Guidelines

Visual language and interaction patterns for local-ai CLI. All commands MUST follow these guidelines for a consistent user experience.

**Status**: v0.1.0

---

## 1. Core Principles

### 1.1 Consistency
Same visual patterns across all commands. Users should recognize patterns from one command to another.

### 1.2 Clarity
Users always know what's happening and what to do next. Progress is visible, errors are actionable.

### 1.3 Full Information
Never truncate information that users need to copy/paste or act upon. Model IDs, paths, and commands must be complete.

### 1.4 Accessibility
Colors have text fallbacks. Symbols accompany colors. No meaning conveyed through color alone.

---

## 2. Visual Language

### 2.1 Symbols & Semantics

| Symbol | Color | Meaning | Usage |
|--------|-------|---------|-------|
| `✓` | green | Success | Completed operations, passed checks |
| `✗` | red | Failure | Failed operations, errors |
| `⚠` | yellow | Warning | Non-fatal issues, caution needed |
| `ℹ` | blue | Info | Informational messages |
| `○` | dim | Pending | Not yet started |
| `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` | cyan | Running | Animated spinner |

### 2.2 Color Semantics

| Color | Rich Markup | Semantic Meaning |
|-------|-------------|------------------|
| Green | `[green]` | Success, completion, passed |
| Red | `[red]` | Errors, failures, critical |
| Yellow | `[yellow]` | Warnings, caution, quantization info |
| Cyan | `[cyan]` | Primary identifiers, model names |
| Blue | `[blue]` | Secondary identifiers, authors |
| Magenta | `[magenta]` | Metrics (likes, stats) |
| Dim | `[dim]` | Secondary info, metadata, hints |

### 2.3 Typography

| Style | Rich Markup | Usage |
|-------|-------------|-------|
| Bold | `[bold]` | Emphasis, headers, important values |
| Dim | `[dim]` | Secondary info, timestamps, hints |
| Normal | (none) | Primary content |

---

## 3. Tables

### 3.1 Table Principles

1. **Full width**: Tables expand to use available terminal width
2. **No truncation**: Model IDs and paths are never truncated
3. **Consistent columns**: Same data type = same column name across commands
4. **Right-align numbers**: Downloads, likes, counts are right-aligned

### 3.2 Standard Column Names

| Data Type | Column Name | Style | Alignment |
|-----------|-------------|-------|-----------|
| Model identifier | "Model" | cyan | left |
| Author/organization | "Author" | blue | left |
| Download count | "Downloads" | green | right |
| Like count | "Likes" | magenta | right |
| Quantization level | "Quant" | yellow | center |

**Note**: Always use "Author" (not "Owner") for consistency.

### 3.3 Table Width

Tables should use `expand=True` to fill available terminal width:

```python
table = Table(title="...", expand=True)
```

### 3.4 Multiple Related Tables

When displaying multiple tables for the same query:
1. All tables use the same width (full terminal width)
2. Tables have consistent column styling
3. Add a blank line between tables

---

## 4. Model Display

### 4.1 Model Identifiers

Always show the **full model ID** (e.g., `mlx-community/Qwen3-8B-4bit`). Never truncate.

For the "Model" column, show only the model name (after the `/`) since Author is a separate column:
- Full ID: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit`
- Model column: `Qwen3-Coder-30B-A3B-Instruct-8bit`
- Author column: `mlx-community`

### 4.2 Download Formatting

Format large numbers for readability:
- `1,234` → `1.2K`
- `1,234,567` → `1.2M`

---

## 5. Error Handling

### 5.1 User-Friendly Errors

**DO:**
```
[red]Cannot connect to server at 127.0.0.1:10240[/red]

Make sure the server is running:
  local-ai server start
```

**DON'T:**
```
httpx.ConnectError: All connection attempts failed
Traceback (most recent call last):
  ...
```

### 5.2 Error Categories

| Category | Display | Include Suggestion |
|----------|---------|-------------------|
| Connection | "Cannot connect to {service}" | Yes |
| Not Found | "Model not found: {id}" | Yes |
| Invalid Input | "Invalid {thing}: {detail}" | Optional |

---

## 6. Progress Indicators

### 6.1 Spinner for Network Operations

Use Rich status spinner for operations that may take time:

```python
with console.status("[bold green]Searching for 'query'...[/bold green]"):
    results = search_models(query)
```

---

## 7. Summary Lines

After displaying results, show a summary in dim text:

```
[dim]Showing 13 results (3 top + 10 MLX-optimized)[/dim]
[dim]Server: 127.0.0.1:10240[/dim]
```

---

## 8. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Operation failed or error |

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2024-12-15 | 0.1.0 | Initial version for local-ai |
