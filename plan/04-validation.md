# 04 — Capstone Validation Loop

> **This is the most important quality mechanism in the project.**
>
> The user said: *"test cases 要足够 solid 不要让学生跑了半天发现是错的"*
> ("Test cases must be solid — don't let students debug for hours only to discover the
> tests are broken.")
>
> Without this validation loop, generated tests are unverified. With it, every test in
> the capstone has been proven to pass against a correct implementation before shipping.

---

## The Problem

LLMs frequently generate test code that is internally inconsistent:

- The test asserts behavior the interface doesn't actually require
- The test uses wrong expected values
- The test calls methods with wrong signatures
- The integration test requires setup that nothing produces

A student implementing the interfaces correctly would still see these tests fail — and
they have no way to know whether the failure is their bug or the test's bug.

## The Solution

After generating `interfaces.py` and `test_capstone.py`, we ALSO ask the LLM to generate
a complete reference implementation. We run pytest. If tests fail on the reference, the
tests are wrong — regenerate them with the failures as feedback.

The reference implementation is **not shipped** to the student. It exists solely to
validate the test suite.

---

## The Loop

```
┌─────────────────────────────────────────────────────┐
│  CapstoneAgent.generate_with_validation(curriculum) │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  1. Generate interfaces.py │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  2. Generate test_capstone.py │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌────────────────────────────────────────────┐
              │  3. Generate reference_impl.py (HIDDEN)    │
              │     Must satisfy interfaces + pass tests   │
              └────────────┬───────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────────────────────────┐
              │  4. Run: pytest test_capstone.py           │
              │     against reference_impl.py              │
              │     in an isolated subprocess              │
              └────────────┬───────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐        ┌────────────────────────┐
     │  All pass ✓    │        │  Failures ✗            │
     └────────┬───────┘        └────────────┬───────────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐        ┌────────────────────────────┐
     │  Generate      │        │  attempt < max_attempts?   │
     │  README.md     │        └────┬───────────────────┬───┘
     │  Generate      │         yes │              no   │
     │  implementation│             ▼                   ▼
     │  .py starter   │     ┌──────────────┐   ┌────────────────────┐
     │  (sans         │     │ Regenerate   │   │ SHIP WITH WARNING  │
     │  reference)    │     │ tests with   │   │ in capstone/       │
     │                │     │ failure ctx, │   │ README.md:         │
     │  Delete        │     │ then re-gen  │   │ "⚠️ Validation:    │
     │  reference_impl│     │ ref impl,    │   │  N tests may have  │
     │  .py           │     │ goto step 4  │   │  issues. See..."   │
     └────────────────┘     └──────────────┘   └────────────────────┘
```

---

## Implementation

```python
# agents/capstone.py

import json
import subprocess
import tempfile
from pathlib import Path

from llm.client import LLMClient
from prompts.capstone import (
    CAPSTONE_README_USER, INTERFACES_USER, TESTS_USER,
    REFERENCE_IMPL_USER, TESTS_REGEN_USER,
)
from infra.logging import get_logger

logger = get_logger(__name__)


class CapstoneAgent:
    def __init__(self, llm_client: LLMClient, output_dir: Path, state: dict):
        self.llm = llm_client
        self.output_dir = output_dir
        self.state = state
        self.capstone_dir = output_dir / "capstone"
        self.capstone_dir.mkdir(exist_ok=True)
    
    def generate_with_validation(self, curriculum: dict, max_attempts: int = 3) -> None:
        """Generate capstone with the full validation loop."""
        cap = curriculum["capstone"]
        
        # Step 1: interfaces (generated once, never re-generated in the loop)
        interfaces_content = self._generate_interfaces(cap)
        (self.capstone_dir / "interfaces.py").write_text(interfaces_content)
        logger.info("Generated interfaces.py")
        
        # Steps 2-4: validation loop
        tests_content = self._generate_tests(cap, interfaces_content)
        validation_passed = False
        validation_warnings = []
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Validation attempt {attempt}/{max_attempts}")
            
            # Step 3: reference implementation
            reference_impl = self._generate_reference_impl(
                cap, interfaces_content, tests_content
            )
            
            # Step 4: run tests
            passed, output = self._run_validation(
                interfaces_content, tests_content, reference_impl
            )
            
            if passed:
                logger.info("Validation passed!")
                validation_passed = True
                break
            else:
                logger.warning(f"Validation failed:\n{output[:1000]}")
                validation_warnings.append(f"Attempt {attempt}: {output[:500]}")
                
                if attempt < max_attempts:
                    # Regenerate tests with failure context
                    tests_content = self._regenerate_tests(
                        original_tests=tests_content,
                        reference_impl=reference_impl,
                        pytest_output=output,
                    )
        
        # Persist final tests (validated or not)
        (self.capstone_dir / "test_capstone.py").write_text(tests_content)
        
        # Step 5: README (with warning if validation failed)
        warning_section = ""
        if not validation_passed:
            warning_section = (
                "\n\n## ⚠️ Validation Notice\n\n"
                "These tests could not be fully validated against a reference implementation.\n"
                "Some tests may have issues. If you believe a failing test is wrong "
                "(not your implementation), please file an issue.\n\n"
                "Validation history:\n"
                + "\n".join(f"- {w}" for w in validation_warnings[:3])
            )
        
        readme_content = self._generate_readme(cap, warning_section)
        (self.capstone_dir / "README.md").write_text(readme_content)
        
        # Step 6: student starter file (NOT the reference impl)
        self._write_implementation_starter(cap)
        
        # Update state
        self.state["progress"]["capstone_complete"] = True
        self.state["progress"]["capstone_validated"] = validation_passed
        self.state["progress"]["validation_attempts"] = attempt
    
    # ── Private generation methods ──
    
    def _generate_interfaces(self, cap: dict) -> str:
        prompt = INTERFACES_USER.format(modules_json=json.dumps(cap["modules"], indent=2))
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=self.llm.gen.get("capstone_max_tokens", 4000),
        )
        return _strip_code_fences(response)
    
    def _generate_tests(self, cap: dict, interfaces_content: str) -> str:
        prompt = TESTS_USER.format(
            interfaces_content=interfaces_content,
            modules_json=json.dumps(cap["modules"], indent=2),
            integration_test_json=json.dumps(cap.get("integration_test", {}), indent=2),
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=self.llm.gen.get("capstone_max_tokens", 4000),
        )
        return _strip_code_fences(response)
    
    def _generate_reference_impl(
        self, cap: dict, interfaces_content: str, tests_content: str
    ) -> str:
        prompt = REFERENCE_IMPL_USER.format(
            interfaces_content=interfaces_content,
            modules_json=json.dumps(cap["modules"], indent=2),
            tests_content=tests_content,
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=self.llm.gen.get("capstone_max_tokens", 4000) + 2000,
        )
        return _strip_code_fences(response)
    
    def _regenerate_tests(
        self, original_tests: str, reference_impl: str, pytest_output: str,
    ) -> str:
        prompt = TESTS_REGEN_USER.format(
            original_tests=original_tests,
            reference_impl=reference_impl,
            pytest_output=pytest_output[:3000],  # truncate to fit context
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=self.llm.gen.get("capstone_max_tokens", 4000),
        )
        return _strip_code_fences(response)
    
    def _generate_readme(self, cap: dict, warning_section: str) -> str:
        prompt = CAPSTONE_README_USER.format(
            capstone_title=cap.get("title", "Capstone"),
            capstone_description=cap.get("description", ""),
            estimated_hours=cap.get("estimated_hours", 4),
            modules_json=json.dumps(cap["modules"], indent=2),
            integration_test_json=json.dumps(cap.get("integration_test", {}), indent=2),
            warning_section=warning_section,
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=self.llm.gen.get("capstone_max_tokens", 4000),
        )
    
    def _write_implementation_starter(self, cap: dict) -> None:
        modules_list = "\n".join(
            f"#   {i}. {m['name']}" for i, m in enumerate(cap.get("modules", []), 1)
        )
        starter = f'''"""Capstone implementation — fill in each class.

Run tests with: pytest test_capstone.py -v
"""

from interfaces import *

# TODO: Implement each abstract class from interfaces.py here.
# Suggested implementation order:
{modules_list}

# Example structure:
#
# class YourTokenizer(Tokenizer):
#     def __init__(self, corpus: str):
#         # TODO: build vocabulary from corpus
#         pass
#     
#     @property
#     def vocab_size(self) -> int:
#         # TODO
#         pass
#     
#     def encode(self, text: str) -> list[int]:
#         # TODO
#         pass
#     
#     def decode(self, ids: list[int]) -> str:
#         # TODO
#         pass
'''
        (self.capstone_dir / "implementation.py").write_text(starter)
    
    # ── Validation runner ──
    
    def _run_validation(
        self, interfaces_content: str, tests_content: str, reference_impl: str,
    ) -> tuple[bool, str]:
        """
        Run pytest in an isolated tempdir.
        Returns (passed: bool, output: str).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "interfaces.py").write_text(interfaces_content)
            (tmp / "test_capstone.py").write_text(tests_content)
            (tmp / "implementation.py").write_text(reference_impl)
            (tmp / "conftest.py").write_text(
                # Make pytest discover the implementation module
                "import sys; sys.path.insert(0, '.')"
            )
            
            # Run pytest with timeout to prevent hangs
            result = subprocess.run(
                ["python", "-m", "pytest", "test_capstone.py", "-v", "--tb=short",
                 "--no-header", "-x"],  # -x: stop after first failure
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes max
            )
            
            output = result.stdout + "\n" + result.stderr
            return result.returncode == 0, output
    
    def refine(self, instruction: str, curriculum: dict) -> None:
        """Re-run generation with extra requirement injected into capstone description."""
        curriculum = dict(curriculum)
        curriculum["capstone"] = dict(curriculum["capstone"])
        curriculum["capstone"]["description"] += f"\n\nAdditional requirement: {instruction}"
        self.generate_with_validation(curriculum)


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
    if code.endswith("```"):
        code = code[: code.rfind("```")]
    return code.strip()
```

---

## Testing the validator (`tests/test_capstone_validation.py`)

The validation loop itself must be tested. Strategy: use synthetic test/impl content
(no LLM needed for these tests).

```python
def test_validation_passes_for_correct_impl(tmp_path):
    """If reference impl satisfies tests, validation passes."""
    interfaces = "from abc import ABC, abstractmethod\nclass Foo(ABC):\n    @abstractmethod\n    def bar(self) -> int: ...\n"
    tests = "from implementation import MyFoo\ndef test_bar():\n    assert MyFoo().bar() == 42\n"
    impl = "from interfaces import Foo\nclass MyFoo(Foo):\n    def bar(self): return 42\n"
    
    agent = CapstoneAgent(MockLLM([]), tmp_path, {"progress": {}})
    passed, output = agent._run_validation(interfaces, tests, impl)
    assert passed, f"Expected pass, got:\n{output}"


def test_validation_fails_for_wrong_impl(tmp_path):
    """If reference impl doesn't satisfy tests, validation fails."""
    interfaces = "from abc import ABC, abstractmethod\nclass Foo(ABC):\n    @abstractmethod\n    def bar(self) -> int: ...\n"
    tests = "from implementation import MyFoo\ndef test_bar():\n    assert MyFoo().bar() == 42\n"
    impl = "from interfaces import Foo\nclass MyFoo(Foo):\n    def bar(self): return 999\n"
    
    agent = CapstoneAgent(MockLLM([]), tmp_path, {"progress": {}})
    passed, output = agent._run_validation(interfaces, tests, impl)
    assert not passed
    assert "999" in output or "42" in output  # failure shown


def test_validation_handles_syntax_errors(tmp_path):
    """If generated code has Python syntax errors, validation fails cleanly."""
    interfaces = "this is not python"
    tests = "def test_x(): pass"
    impl = "def x(): pass"
    
    agent = CapstoneAgent(MockLLM([]), tmp_path, {"progress": {}})
    passed, output = agent._run_validation(interfaces, tests, impl)
    assert not passed


def test_validation_times_out_on_infinite_loop(tmp_path):
    """If reference impl has infinite loop, validation kills it."""
    # ... use a test that triggers an infinite loop in impl
    # subprocess.TimeoutExpired should be caught and treated as failure


def test_full_loop_with_mock_llm(tmp_path):
    """Test the full generate_with_validation flow with canned LLM responses."""
    # MockLLM returns:
    #   1. interfaces (valid)
    #   2. tests (with intentional bug)
    #   3. reference impl
    #   [validation fails]
    #   4. corrected tests
    #   5. new reference impl
    #   [validation passes]
    #   6. README
    # Verify all files written correctly
    # Verify state.progress.capstone_validated == True
```
