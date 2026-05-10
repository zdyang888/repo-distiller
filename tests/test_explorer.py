"""Tests for ExplorerAgent."""

import json

import pytest

from agents.explorer import ExplorerAgent
from tests.conftest import MockLLM
from tools.repo import RepoTool

_VALID_FINDINGS = {
    "title": "Fake",
    "one_liner": "test",
    "mental_model": "test",
    "domain": "other",
    "concepts": [],
    "dependency_order": [],
    "key_files": [],
    "skip_files": [],
}

_DONE_RESPONSE = json.dumps({"action": "done", "findings": _VALID_FINDINGS})


def _make_repo(tmp_path):
    fake_repo_dir = tmp_path / "fake_repo"
    fake_repo_dir.mkdir()
    (fake_repo_dir / "README.md").write_text("# Fake Repo")
    repo = RepoTool("https://fake", work_dir=str(tmp_path))
    repo.repo_path = str(fake_repo_dir)
    return repo


def test_explorer_completes_when_llm_says_done(tmp_path):
    """LLM does one tool call then returns done."""
    llm = MockLLM(responses=[
        json.dumps({"action": "tool", "tool": "read_file", "path": "README.md"}),
        _DONE_RESPONSE,
    ])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Fake"
    assert len(llm.calls) == 2


def test_explorer_recovers_from_malformed_json(tmp_path):
    """If LLM returns garbage, agent sends correction and continues."""
    llm = MockLLM(responses=[
        "not json at all",
        _DONE_RESPONSE,
    ])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Fake"
    assert len(llm.calls) == 2


def test_explorer_force_finishes_at_max_steps(tmp_path):
    """If max_steps hit, agent forces a finalize and accepts the result."""
    # Keep returning tool calls to exhaust steps, then return done on finalize
    tool_call = json.dumps({"action": "tool", "tool": "list_dir", "path": ""})
    # max_explore_steps is 20 in MockLLM.gen, but we override via a small MockLLM
    # We'll use a subclass with max_steps=2 by patching gen
    class SmallStepMockLLM(MockLLM):
        @property
        def gen(self) -> dict:
            return {"max_explore_steps": 2, "max_notebooks": 8}

    llm = SmallStepMockLLM(responses=[
        tool_call,
        tool_call,
        _DONE_RESPONSE,  # returned on the forced finalize call
    ])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Fake"
    # 2 tool calls + 1 forced finalize = 3 llm.calls
    assert len(llm.calls) == 3


def test_explorer_calls_fallback_when_finalize_fails(tmp_path):
    """If even forced-finalize returns garbage, fallback returns defaults."""
    class SmallStepMockLLM(MockLLM):
        @property
        def gen(self) -> dict:
            return {"max_explore_steps": 1, "max_notebooks": 8}

    llm = SmallStepMockLLM(responses=[
        json.dumps({"action": "tool", "tool": "list_dir", "path": ""}),
        "still garbage",  # forced finalize returns bad JSON
    ])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    # Fallback returns default title
    assert findings["title"] == "Unknown Project"


def test_explorer_fills_defaults_for_missing_findings_fields(tmp_path):
    """Partial findings dict gets filled with defaults."""
    partial_done = json.dumps({"action": "done", "findings": {"title": "Partial"}})
    llm = MockLLM(responses=[partial_done])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Partial"
    assert findings["concepts"] == []
    assert findings["domain"] == "other"


def test_explorer_logs_usage(tmp_path):
    """usage_log is populated after exploration."""
    llm = MockLLM(responses=[_DONE_RESPONSE])
    repo = _make_repo(tmp_path)
    agent = ExplorerAgent(llm, repo)
    agent.explore()
    assert len(agent.usage_log) == 1
    assert "step" in agent.usage_log[0]


def test_explorer_search_code_tool(tmp_path):
    """search_code tool call is dispatched correctly."""
    llm = MockLLM(responses=[
        json.dumps({"action": "tool", "tool": "search_code", "keyword": "class"}),
        _DONE_RESPONSE,
    ])
    repo = _make_repo(tmp_path)
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Fake"
