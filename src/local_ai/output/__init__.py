"""Output module for local-ai CLI.

Provides consistent formatting, tables, and display helpers.
See docs/UI_UX_GUIDELINES.md for design principles.
"""

from local_ai.output.console import console, format_downloads, print_error, print_success
from local_ai.output.tables import (
    create_local_models_table,
    create_model_table,
    create_search_results_table,
)

__all__ = [
    "console",
    "create_local_models_table",
    "create_model_table",
    "create_search_results_table",
    "format_downloads",
    "print_error",
    "print_success",
]
