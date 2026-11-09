"""Small filesystem helpers used by downstream runtime utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path


def windows_safe_path(path: Path) -> str:
    """Return a path string that can address long Windows filenames."""

    resolved_path = str(path.resolve())

    if os.name == "nt" and not resolved_path.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved_path

    return resolved_path


def safe_is_file(path: Path) -> bool:
    """Return whether a path is a file, including long Windows paths."""

    return os.path.isfile(windows_safe_path(path))


def safe_filename_stem(stem: str, max_length: int = 120) -> str:
    """Return a report/output-safe filename stem."""

    cleaned_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")

    if not cleaned_stem:
        cleaned_stem = "image"

    return cleaned_stem[:max_length]
