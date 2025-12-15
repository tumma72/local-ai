"""Shared console and output helpers for local-ai CLI.

Provides a single Console instance and formatting utilities.
See docs/UI_UX_GUIDELINES.md for design principles.
"""

from rich.console import Console

# Shared console instance - use this everywhere for consistent output
console = Console()


def format_downloads(count: int) -> str:
    """Format download count for display.

    Args:
        count: Number of downloads.

    Returns:
        Formatted string (e.g., "1.2M", "45.3K", "892").
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def print_success(message: str) -> None:
    """Print a success message with green checkmark.

    Args:
        message: The message to display.
    """
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str, suggestion: str | None = None) -> None:
    """Print an error message with red X and optional suggestion.

    Args:
        message: The error message.
        suggestion: Optional suggestion for how to fix the error.
    """
    console.print(f"[red]✗ {message}[/red]")
    if suggestion:
        console.print(f"\n{suggestion}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow symbol.

    Args:
        message: The warning message.
    """
    console.print(f"[yellow]⚠ {message}[/yellow]")


def print_info(message: str) -> None:
    """Print an informational message.

    Args:
        message: The info message.
    """
    console.print(f"[blue]ℹ[/blue] {message}")
