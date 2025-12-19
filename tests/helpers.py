"""Test helper utilities."""

import re


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text.

    Rich/Typer may emit ANSI codes even with NO_COLOR=1 in some CI environments.
    This helper ensures consistent text matching regardless of formatting.
    """
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)
