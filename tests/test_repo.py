"""Tests for tools/repo.py — uses fixture directory, no git clone needed."""

import os
from pathlib import Path

import pytest

from tools.repo import RepoTool


@pytest.fixture
def repo(sample_repo_path, tmp_path) -> RepoTool:
    """RepoTool pointed at the sample_repo fixture, no clone needed."""
    rt = RepoTool("https://fake/url", work_dir=str(tmp_path))
    rt.repo_path = str(sample_repo_path)
    return rt


def test_clone_reuses_existing(tmp_path):
    """Second clone() call reuses the existing directory."""
    # Simulate an already-cloned repo by creating .git dir
    repo_dir = tmp_path / "myrepo"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# Existing")

    rt = RepoTool("https://github.com/user/myrepo", work_dir=str(tmp_path))
    # Manually trigger clone — it should detect existing .git and skip
    result = rt.clone()
    assert result == str(repo_dir)
    assert rt.repo_path == str(repo_dir)


def test_list_dir_skips_hidden_dirs(repo, sample_repo_path):
    """__pycache__ and other SKIP_DIRS are not listed."""
    # Create a __pycache__ dir in the fixture root temporarily
    pycache = sample_repo_path / "__pycache__"
    pycache.mkdir(exist_ok=True)
    try:
        output = repo.list_dir()
        assert "__pycache__" not in output
    finally:
        pycache.rmdir()


def test_list_dir_skips_binary_extensions(repo, sample_repo_path, tmp_path):
    """Binary extension files (.png) are not listed."""
    rt = RepoTool("https://fake/url", work_dir=str(tmp_path))
    # Point at a temp dir with a .png file
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "image.png").write_bytes(b"\x89PNG")
    (test_dir / "script.py").write_text("x = 1")
    rt.repo_path = str(tmp_path)

    output = rt.list_dir("test_dir")
    assert "image.png" not in output
    assert "script.py" in output


def test_read_file_truncates_long_files(repo, sample_repo_path, tmp_path):
    """read_file truncates content and shows truncation notice."""
    rt = RepoTool("https://fake/url", work_dir=str(tmp_path))
    test_dir = tmp_path / "bigfile_repo"
    test_dir.mkdir()
    big_content = "x" * 10_000
    (test_dir / "big.py").write_text(big_content)
    rt.repo_path = str(test_dir)

    output = rt.read_file("big.py", max_chars=100)
    assert "Truncated" in output
    assert "10,000" in output or "10000" in output


def test_read_file_returns_error_for_missing(repo):
    """read_file returns 'File not found' for nonexistent paths."""
    output = repo.read_file("nonexistent/file.py")
    assert "File not found" in output


def test_search_code_finds_keyword(repo):
    """search_code finds a keyword case-insensitively."""
    output = repo.search_code("coreclass")
    assert "core.py" in output.lower()
    assert "CoreClass" in output


def test_search_code_respects_max_results(repo):
    """search_code caps results at max_results."""
    # 'def' appears in multiple files
    output = repo.search_code("def", max_results=2)
    lines = [l for l in output.splitlines() if ".py:" in l]
    assert len(lines) <= 2


def test_get_priority_files_orders_correctly(repo):
    """README.md must be first in priority files."""
    files = repo.get_priority_files()
    assert len(files) >= 1
    assert files[0] == "README.md"


def test_extract_symbols_class(repo):
    """Extracts full class definition."""
    output = repo.extract_symbols("src/core.py", ["CoreClass"])
    assert "class CoreClass" in output
    assert "def greet" in output
    assert "def farewell" in output


def test_extract_symbols_method(repo):
    """Extracts a single method from a class."""
    output = repo.extract_symbols("src/core.py", ["CoreClass.greet"])
    assert "def greet" in output
    # Should NOT contain farewell (it's a different method)
    assert "def farewell" not in output


def test_extract_symbols_function(repo):
    """Extracts a top-level function."""
    output = repo.extract_symbols("src/utils.py", ["helper"])
    assert "def helper" in output
    assert "value * 2" in output


def test_extract_symbols_missing_falls_back(repo):
    """Non-existent symbol produces a 'not found' comment."""
    output = repo.extract_symbols("src/core.py", ["NonExistentClass"])
    assert "not found" in output.lower()


def test_extract_symbols_non_python_falls_back(repo):
    """Non-.py file falls back to read_file."""
    output = repo.extract_symbols("README.md", ["SomeSymbol"])
    # Falls back to read_file which returns file content
    assert "Sample Repo" in output or "README.md" in output
