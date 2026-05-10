"""Tests for tools/notebook.py — pure function unit tests, no I/O."""

import json
import os
import tempfile

import pytest

from tools.notebook import (
    build_notebook,
    load_notebook,
    make_code_cell,
    make_markdown_cell,
    notebook_from_llm_response,
    parse_llm_response,
    save_notebook,
    validate_notebook,
)


def test_parse_basic_alternation():
    """4 cells alternating markdown and code."""
    response = (
        "===MARKDOWN===\n# Title\nSome text\n"
        "===CODE===\nimport torch\n"
        "===MARKDOWN===\nMore text\n"
        "===CODE===\nx = 1\n"
    )
    cells = parse_llm_response(response)
    assert len(cells) == 4
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert cells[2]["cell_type"] == "markdown"
    assert cells[3]["cell_type"] == "code"


def test_parse_no_delimiters_fallback():
    """No delimiters → single markdown cell."""
    response = "This is just some plain text without any delimiters."
    cells = parse_llm_response(response)
    assert len(cells) == 1
    assert cells[0]["cell_type"] == "markdown"
    assert "plain text" in "".join(cells[0]["source"])


def test_parse_strips_python_fences():
    """Code cells wrapped in ```python ... ``` have fences stripped."""
    response = "===CODE===\n```python\nimport os\nprint('hi')\n```\n"
    cells = parse_llm_response(response)
    assert len(cells) == 1
    source = "".join(cells[0]["source"])
    assert "```" not in source
    assert "import os" in source


def test_parse_strips_bare_fences():
    """Code cells wrapped in bare ``` ... ``` have fences stripped."""
    response = "===CODE===\n```\nx = 42\n```\n"
    cells = parse_llm_response(response)
    assert len(cells) == 1
    source = "".join(cells[0]["source"])
    assert "```" not in source
    assert "x = 42" in source


def test_parse_skips_empty_cells():
    """Empty cell content after stripping is not added."""
    response = "===MARKDOWN===\nHello\n===CODE===\n\n===MARKDOWN===\nWorld\n"
    cells = parse_llm_response(response)
    assert len(cells) == 2
    assert all(c["cell_type"] == "markdown" for c in cells)


def test_parse_handles_lowercase_delimiters():
    """===markdown=== and ===code=== are recognized case-insensitively."""
    response = "===markdown===\nContent\n===code===\nx = 1\n"
    cells = parse_llm_response(response)
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"


def test_save_load_roundtrip(tmp_path):
    """save_notebook + load_notebook preserves content exactly."""
    nb = build_notebook(
        [make_markdown_cell("Hello"), make_code_cell("x = 1")],
        title="Test",
    )
    path = str(tmp_path / "test.ipynb")
    save_notebook(nb, path)
    loaded = load_notebook(path)
    assert loaded["nbformat"] == 4
    assert len(loaded["cells"]) == 2
    assert "".join(loaded["cells"][0]["source"]) == "Hello"
    assert "".join(loaded["cells"][1]["source"]) == "x = 1"


def test_validate_catches_missing_kernelspec():
    """validate_notebook reports missing kernelspec."""
    nb = build_notebook([make_code_cell("x = 1")])
    del nb["metadata"]["kernelspec"]
    errors = validate_notebook(nb)
    assert any("kernelspec" in e for e in errors)


def test_validate_catches_invalid_cell_type():
    """validate_notebook reports invalid cell_type."""
    nb = build_notebook([make_code_cell("x = 1")])
    nb["cells"][0]["cell_type"] = "invalid"
    errors = validate_notebook(nb)
    assert any("invalid" in e for e in errors)


def test_validate_passes_for_well_formed_notebook():
    """No errors on a correctly-built notebook."""
    nb = build_notebook(
        [make_markdown_cell("## Hello"), make_code_cell("print('hi')")],
        title="Good Notebook",
    )
    errors = validate_notebook(nb)
    assert errors == []
