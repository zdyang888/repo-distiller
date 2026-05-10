"""NotebookTool: construct and parse nbformat 4.5 notebooks."""

import json
import re
from pathlib import Path


def _to_source_list(text: str) -> list[str]:
    """Convert a string to nbformat source list (lines with embedded newlines)."""
    lines = text.split("\n")
    if len(lines) == 1:
        return lines
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def make_markdown_cell(source: str) -> dict:
    """Create an nbformat 4 markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_source_list(source),
    }


def make_code_cell(source: str) -> dict:
    """Create an nbformat 4 code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source_list(source),
    }


def build_notebook(cells: list[dict], title: str = "") -> dict:
    """Wrap cells in an nbformat 4.5 notebook with Python 3 kernelspec."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def save_notebook(nb: dict, path: str) -> None:
    """Write notebook JSON to disk, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def load_notebook(path: str) -> dict:
    """Read and parse notebook JSON from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DELIMITER_RE = re.compile(
    r"===\s*(MARKDOWN|CODE)\s*===", re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_END_RE = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove leading ```python or ``` and trailing ``` from code blocks."""
    text = _CODE_FENCE_RE.sub("", text, count=1)
    text = _CODE_FENCE_END_RE.sub("", text)
    return text


def parse_llm_response(response: str) -> list[dict]:
    """
    Parse ===MARKDOWN=== / ===CODE=== delimited text into notebook cells.

    - Case-insensitive delimiters
    - Strips ```python / ``` fences from code cells
    - No delimiters → single markdown cell
    - Empty cells are skipped
    """
    if not _DELIMITER_RE.search(response):
        # No delimiters — wrap whole response in a markdown cell
        stripped = response.strip()
        if not stripped:
            return []
        return [make_markdown_cell(stripped)]

    # Split on delimiters, keeping the type tag
    parts = _DELIMITER_RE.split(response)
    # parts[0] is text before first delimiter (ignored)
    # parts[1], parts[2], parts[3], parts[4], ... are: type, content, type, content...
    cells: list[dict] = []
    i = 1
    while i + 1 < len(parts):
        cell_type = parts[i].strip().upper()
        content = parts[i + 1].strip()
        i += 2
        if not content:
            continue
        if cell_type == "CODE":
            content = _strip_code_fences(content).strip()
            if content:
                cells.append(make_code_cell(content))
        else:
            cells.append(make_markdown_cell(content))

    return cells


def notebook_from_llm_response(response: str, title: str = "") -> dict:
    """Parse LLM response and wrap in a notebook."""
    cells = parse_llm_response(response)
    return build_notebook(cells, title=title)


def validate_notebook(nb: dict) -> list[str]:
    """
    Validate notebook structure. Returns list of error strings (empty if valid).

    Checks:
    - Correct nbformat (4) and nbformat_minor (≥ 5)
    - Has metadata.kernelspec
    - All cells have valid cell_type
    - Code cells have outputs (list) and execution_count
    - All cells have source as a list
    - At least one cell
    """
    errors: list[str] = []

    if nb.get("nbformat") != 4:
        errors.append(f"nbformat must be 4, got {nb.get('nbformat')!r}")
    if nb.get("nbformat_minor", 0) < 5:
        errors.append(f"nbformat_minor must be ≥ 5, got {nb.get('nbformat_minor')!r}")

    meta = nb.get("metadata", {})
    if "kernelspec" not in meta:
        errors.append("metadata.kernelspec is missing")

    cells = nb.get("cells", [])
    if not cells:
        errors.append("notebook has no cells")

    valid_types = {"markdown", "code", "raw"}
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        if ctype not in valid_types:
            errors.append(f"cell[{i}] has invalid cell_type: {ctype!r}")
        if not isinstance(cell.get("source"), list):
            errors.append(f"cell[{i}] source must be a list, got {type(cell.get('source')).__name__}")
        if ctype == "code":
            if not isinstance(cell.get("outputs"), list):
                errors.append(f"cell[{i}] (code) outputs must be a list")
            if "execution_count" not in cell:
                errors.append(f"cell[{i}] (code) missing execution_count")

    return errors
