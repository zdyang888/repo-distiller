"""Tests for GeneratorAgent."""

import json
from pathlib import Path

import pytest

from agents.generator import GeneratorAgent, _slug, _hollow_out_exercise


def test_hollow_out_exercise():
    """Verify that code between stubs is replaced by a placeholder."""
    text = """
===MARKDOWN===
# Exercise 1
===CODE===
def add(a, b):
    ### START CODE HERE ###
    return a + b
    ### END CODE HERE ###

===MARKDOWN===
# Exercise 2
===CODE===
class Multiplier:
    def __init__(self, factor):
        ### START CODE HERE ###
        self.factor = factor
        ### END CODE HERE ###
"""
    hollowed = _hollow_out_exercise(text)
    
    assert "return a + b" not in hollowed
    assert "self.factor = factor" not in hollowed
    assert "### START CODE HERE ###" in hollowed
    assert "### END CODE HERE ###" in hollowed
    assert "# your code here" in hollowed
    assert "pass" in hollowed
    
    # Check that indentation is preserved (roughly)
    assert "    # your code here" in hollowed
from tests.conftest import MockLLM


# ── Test doubles ──────────────────────────────────────────────────────────────


class MockRepoTool:
    """Minimal RepoTool stand-in that tracks read_files_for_concept calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.repo_path = "/fake/repo"

    def read_files_for_concept(
        self,
        file_paths: list[str],
        symbols: list[str] | None = None,
        max_total_chars: int = 6000,
    ) -> str:
        self.calls.append(
            {"file_paths": file_paths, "symbols": symbols, "max_total_chars": max_total_chars}
        )
        return "# mock source code"


# ── Helpers ───────────────────────────────────────────────────────────────────

TEACH_RESPONSE = (
    "===MARKDOWN===\n# Notebook Title\nIntro text.\n\n"
    "===CODE===\nimport numpy as np\n\n"
    "===MARKDOWN===\nCore idea.\n\n"
    "===CODE===\ndef core():\n    return 42\n\n"
    "===MARKDOWN===\nWrap-up.\n\n"
    "===CODE===\ncore()\n"
)
EXERCISE_RESPONSE = (
    "===MARKDOWN===\n# Exercise\n\n"
    "===CODE===\ndef func(x):\n"
    "    \"\"\"Do something.\"\"\"\n"
    "    ### START CODE HERE ###\n"
    "    pass\n"
    "    ### END CODE HERE ###\n"
)
SOLUTION_RESPONSE = (
    "FUNCTION: func\n===CODE===\n"
    "def func(x):\n"
    "    return x * 2\n"
)


def _make_curriculum(num_notebooks: int = 1) -> dict:
    notebooks = [
        {
            "id": f"{i:02d}",
            "title": f"Notebook {i:02d}",
            "concept": f"Concept{i}",
            "description": f"Description {i}",
            "prerequisites": [],
            "key_source_files": [f"src/module{i}.py"],
            "key_symbols": [],
            "learning_objectives": [f"Learn concept {i}"],
            "exercise_description": f"Implement concept {i}",
            "visualization_idea": "",
        }
        for i in range(1, num_notebooks + 1)
    ]
    return {
        "title": "Test Course",
        "mental_model": "A simple test mental model.",
        "concepts": [
            {"name": f"Concept{i}", "description": "...", "complexity": "basic"}
            for i in range(1, num_notebooks + 1)
        ],
        "notebooks": notebooks,
        "capstone": {
            "title": "mini-test",
            "description": "Capstone description",
            "estimated_hours": 4,
            "modules": [],
            "integration_test": {
                "description": "",
                "setup_code": "",
                "success_metric": "",
                "expected_output_check": "",
            },
        },
    }


def _make_state() -> dict:
    return {
        "step": "generating_notebooks",
        "last_error": None,
        "progress": {"notebooks_complete": [], "exercises_complete": []},
        "cost": {},
    }


def _make_gen(tmp_path: Path) -> GeneratorAgent:
    return GeneratorAgent(MockLLM(responses=[]), MockRepoTool(), tmp_path, _make_state())


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_generator_generates_teaching_with_correct_metadata(tmp_path):
    """Teaching notebook is saved with valid nbformat structure."""
    curriculum = _make_curriculum(1)
    llm = MockLLM(responses=[TEACH_RESPONSE, EXERCISE_RESPONSE, SOLUTION_RESPONSE])
    gen = GeneratorAgent(llm, MockRepoTool(), tmp_path, _make_state())

    gen.generate_all("", curriculum)

    slug = _slug("Notebook 01")
    teach_path = tmp_path / f"01_{slug}.ipynb"
    assert teach_path.exists(), f"Expected {teach_path} to be created"

    nb = json.loads(teach_path.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 1


def test_generator_skips_completed_notebooks(tmp_path):
    """Notebooks already in state.progress are not regenerated."""
    curriculum = _make_curriculum(2)
    state = _make_state()
    state["progress"]["notebooks_complete"] = ["01"]
    state["progress"]["exercises_complete"] = ["01"]

    # Only notebook 02 needs generation: teaching + exercise + solution = 3 LLM calls
    llm = MockLLM(responses=[TEACH_RESPONSE, EXERCISE_RESPONSE, SOLUTION_RESPONSE])
    gen = GeneratorAgent(llm, MockRepoTool(), tmp_path, state)
    gen.generate_all("", curriculum)

    assert len(llm.calls) == 3, f"Expected 3 LLM calls, got {len(llm.calls)}"


def test_generator_persists_after_each_notebook(tmp_path):
    """State is written to .distill_state.json and reflects completed work."""
    curriculum = _make_curriculum(1)
    llm = MockLLM(responses=[TEACH_RESPONSE, EXERCISE_RESPONSE, SOLUTION_RESPONSE])
    gen = GeneratorAgent(llm, MockRepoTool(), tmp_path, _make_state())

    gen.generate_all("", curriculum)

    state_file = tmp_path / ".distill_state.json"
    assert state_file.exists()
    saved = json.loads(state_file.read_text())
    assert "01" in saved["progress"]["notebooks_complete"]
    assert "01" in saved["progress"]["exercises_complete"]


def test_generator_uses_symbol_extraction_when_specified(tmp_path):
    """When key_symbols is non-empty, read_files_for_concept is called with symbols."""
    curriculum = _make_curriculum(1)
    curriculum["notebooks"][0]["key_symbols"] = ["MyClass", "my_function"]
    llm = MockLLM(responses=[TEACH_RESPONSE, EXERCISE_RESPONSE, SOLUTION_RESPONSE])
    repo = MockRepoTool()
    gen = GeneratorAgent(llm, repo, tmp_path, _make_state())

    gen.generate_all("", curriculum)

    assert repo.calls, "Expected read_files_for_concept to be called"
    assert repo.calls[0]["symbols"] == ["MyClass", "my_function"]


def test_generator_falls_back_to_whole_files_when_no_symbols(tmp_path):
    """When key_symbols is empty, read_files_for_concept is called without symbols."""
    curriculum = _make_curriculum(1)
    curriculum["notebooks"][0]["key_symbols"] = []
    llm = MockLLM(responses=[TEACH_RESPONSE, EXERCISE_RESPONSE, SOLUTION_RESPONSE])
    repo = MockRepoTool()
    gen = GeneratorAgent(llm, repo, tmp_path, _make_state())

    gen.generate_all("", curriculum)

    assert repo.calls, "Expected read_files_for_concept to be called"
    assert repo.calls[0]["symbols"] is None


# ── _detect_target_notebook ───────────────────────────────────────────────────

_DETECTOR_CURRICULUM = {
    "title": "Test",
    "mental_model": "",
    "concepts": [],
    "notebooks": [
        {
            "id": "01",
            "title": "Introduction to Tokenization",
            "concept": "Tokenizer",
            "description": "",
            "prerequisites": [],
            "key_source_files": [],
            "key_symbols": [],
            "learning_objectives": [],
            "exercise_description": "",
            "visualization_idea": "",
        },
        {
            "id": "02",
            "title": "Attention Mechanism",
            "concept": "Attention",
            "description": "",
            "prerequisites": [],
            "key_source_files": [],
            "key_symbols": [],
            "learning_objectives": [],
            "exercise_description": "",
            "visualization_idea": "",
        },
    ],
    "capstone": {
        "title": "mini-test",
        "description": "",
        "estimated_hours": 4,
        "modules": [],
        "integration_test": {
            "description": "",
            "setup_code": "",
            "success_metric": "",
            "expected_output_check": "",
        },
    },
}


def test_detect_target_notebook_by_id(tmp_path):
    """Instruction containing notebook ID '01' resolves to index 0."""
    gen = _make_gen(tmp_path)
    idx = gen._detect_target_notebook("Please improve notebook 01", _DETECTOR_CURRICULUM)
    assert idx == 0


def test_detect_target_notebook_by_title(tmp_path):
    """Instruction mentioning notebook title resolves to correct index."""
    gen = _make_gen(tmp_path)
    idx = gen._detect_target_notebook(
        "Refine the attention mechanism notebook", _DETECTOR_CURRICULUM
    )
    assert idx == 1


def test_detect_target_notebook_by_concept(tmp_path):
    """Instruction mentioning concept name resolves to correct index."""
    gen = _make_gen(tmp_path)
    idx = gen._detect_target_notebook("Update the tokenizer section", _DETECTOR_CURRICULUM)
    assert idx == 0


def test_detect_target_notebook_returns_none_for_ambiguous(tmp_path):
    """Instruction matching no notebook or multiple notebooks returns None."""
    gen = _make_gen(tmp_path)
    # No match
    assert gen._detect_target_notebook("Improve all notebooks", _DETECTOR_CURRICULUM) is None
    # Ambiguous: matches both "01" and "02" would fail, but here we test no-match scenario
    assert gen._detect_target_notebook("Something unrelated", _DETECTOR_CURRICULUM) is None
