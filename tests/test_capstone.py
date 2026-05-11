"""Tests for CapstoneAgent and the validation loop."""

from pathlib import Path

import pytest

from agents.capstone import CapstoneAgent
from tests.conftest import MockLLM

# ── Shared test fixtures ───────────────────────────────────────────────────────

_INTERFACES = """\
from abc import ABC, abstractmethod

class Adder(ABC):
    \"\"\"Adds two integers.\"\"\"

    @abstractmethod
    def add(self, a: int, b: int) -> int:
        raise NotImplementedError
"""

_TESTS_GOOD = """\
from implementation import MyAdder

def test_add_positive():
    \"\"\"add(2, 3) should return 5.\"\"\"
    # STUDENT_HINT: Check your add() implementation
    assert MyAdder().add(2, 3) == 5, f"Expected 5, got {MyAdder().add(2, 3)}"

def test_add_zeros():
    assert MyAdder().add(0, 0) == 0
"""

_TESTS_BAD = """\
from implementation import MyAdder

def test_add_positive():
    \"\"\"add(2, 3) should return 99 (intentionally wrong).\"\"\"
    assert MyAdder().add(2, 3) == 99
"""

_REFERENCE_IMPL = """\
from interfaces import Adder

class MyAdder(Adder):
    def add(self, a: int, b: int) -> int:
        return a + b
"""

_README = "# mini-adder\n\nBuild a number adder."

_CURRICULUM = {
    "capstone": {
        "title": "mini-adder",
        "description": "Build a simple integer adder.",
        "estimated_hours": 2,
        "modules": [
            {
                "name": "Adder",
                "description": "Adds two integers",
                "depends_on": [],
                "interface_sketch": "class Adder:\n    def add(self, a: int, b: int) -> int: ...",
                "test_behaviors": ["add(2, 3) == 5", "add(0, 0) == 0"],
            }
        ],
        "integration_test": {
            "description": "Chain two adder calls",
            "setup_code": "adder = MyAdder()",
            "success_metric": "adder.add(1, adder.add(2, 3)) == 6",
            "expected_output_check": "6",
        },
    }
}


def _make_state() -> dict:
    return {"step": "capstone", "last_error": None, "progress": {}, "cost": {}}


# ── Validation runner tests (no LLM) ──────────────────────────────────────────


def test_validation_passes_for_correct_impl(tmp_path):
    """A correct reference implementation makes all tests pass."""
    agent = CapstoneAgent(MockLLM([]), tmp_path, _make_state())
    passed, output = agent._run_validation(_INTERFACES, _TESTS_GOOD, _REFERENCE_IMPL)
    assert passed, f"Expected validation to pass, got:\n{output}"


def test_validation_fails_for_wrong_impl(tmp_path):
    """A wrong expected value in a test causes validation to fail."""
    agent = CapstoneAgent(MockLLM([]), tmp_path, _make_state())
    passed, output = agent._run_validation(_INTERFACES, _TESTS_BAD, _REFERENCE_IMPL)
    assert not passed
    # pytest output should mention the failing assertion values
    assert "99" in output or "assert" in output.lower()


def test_validation_handles_syntax_errors(tmp_path):
    """Broken Python code in interfaces causes validation to fail cleanly."""
    bad_interfaces = "this is not valid python!!!"
    agent = CapstoneAgent(MockLLM([]), tmp_path, _make_state())
    passed, output = agent._run_validation(bad_interfaces, _TESTS_GOOD, _REFERENCE_IMPL)
    assert not passed


def test_validation_times_out_on_infinite_loop(tmp_path):
    """A reference impl with an infinite loop is killed and reported as failure."""
    looping_impl = """\
from interfaces import Adder

class MyAdder(Adder):
    def add(self, a: int, b: int) -> int:
        while True:
            pass
"""
    agent = CapstoneAgent(MockLLM([]), tmp_path, _make_state())
    passed, output = agent._run_validation(
        _INTERFACES, _TESTS_GOOD, looping_impl, timeout=4
    )
    assert not passed
    assert "timed out" in output.lower()


# ── Full loop test (with MockLLM) ──────────────────────────────────────────────


def test_full_loop_with_mock_llm(tmp_path):
    """Full generate_with_validation: first attempt fails, second passes.

    LLM call sequence:
      1. _generate_interfaces    → valid interfaces
      2. _generate_tests         → bad tests (wrong expected value)
      3. _generate_reference_impl (attempt 1) → correct impl (fails bad tests)
      [validation fails]
      4. _regenerate_tests       → corrected tests
      5. _generate_reference_impl (attempt 2) → correct impl (passes good tests)
      [validation passes]
      6. _generate_readme        → README text
    """
    llm = MockLLM(
        responses=[
            _INTERFACES,        # call 1: interfaces
            _TESTS_BAD,         # call 2: initial tests (will fail validation)
            _REFERENCE_IMPL,    # call 3: reference impl, attempt 1
            _TESTS_GOOD,        # call 4: regenerated tests
            _REFERENCE_IMPL,    # call 5: reference impl, attempt 2
            _README,            # call 6: README
        ]
    )

    state = _make_state()
    agent = CapstoneAgent(llm, tmp_path, state)
    agent.generate_with_validation(_CURRICULUM, max_attempts=3)

    capstone_dir = tmp_path / "capstone"

    # All four expected files are present
    assert (capstone_dir / "interfaces.py").exists()
    assert (capstone_dir / "test_capstone.py").exists()
    assert (capstone_dir / "implementation.py").exists()
    assert (capstone_dir / "README.md").exists()

    # test_capstone.py should contain the GOOD (regenerated) tests
    test_content = (capstone_dir / "test_capstone.py").read_text()
    assert "== 5" in test_content, "Final tests should have the correct expected value"

    # implementation.py should be the student starter, not the reference impl
    impl_content = (capstone_dir / "implementation.py").read_text()
    assert "TODO" in impl_content, "implementation.py should be the student starter"
    assert "return a + b" not in impl_content, "Reference impl must not be shipped"

    # State reflects successful validation
    assert state["progress"]["capstone_validated"] is True
    assert state["progress"]["capstone_complete"] is True
    assert state["progress"]["validation_attempts"] == 2

    # All 6 LLM calls were made
    assert len(llm.calls) == 6, f"Expected 6 LLM calls, got {len(llm.calls)}"
