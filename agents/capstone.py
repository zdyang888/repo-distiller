"""Capstone agent: generate interfaces, tests, and starter with a validation loop."""

import json
import logging
from pathlib import Path

from llm.client import LLMClient
from prompts.capstone import (
    CAPSTONE_README_USER,
    INTERFACES_USER,
    REFERENCE_IMPL_USER,
    TESTS_REGEN_USER,
    TESTS_USER,
)
from validation.runner import run_pytest

logger = logging.getLogger(__name__)


class CapstoneAgent:
    """Generate a validated capstone project from a Curriculum.

    The validation loop ensures every test in test_capstone.py passes against a
    correct reference implementation before the capstone is shipped to students.
    The reference implementation itself is never written to the output directory.
    """

    def __init__(self, llm_client: LLMClient, output_dir: Path, state: dict) -> None:
        self.llm = llm_client
        self.output_dir = output_dir
        self.state = state
        self.capstone_dir = output_dir / "capstone"
        self.capstone_dir.mkdir(parents=True, exist_ok=True)

    def generate_with_validation(self, curriculum: dict, max_attempts: int = 3) -> None:
        """Generate interfaces, tests, and starter with a pytest validation loop.

        Algorithm:
        1. Generate interfaces.py (once — not re-generated in the loop).
        2. Generate initial test_capstone.py.
        3. Loop up to max_attempts:
           a. Generate a reference implementation.
           b. Run pytest against it in an isolated subprocess.
           c. If all pass: write final files and return.
           d. If failures and attempts remain: regenerate tests with failure context.
        4. If loop ends without passing: ship with a warning in README.

        Args:
            curriculum: Curriculum dict with a "capstone" key.
            max_attempts: Maximum validation attempts before shipping with warning.
        """
        cap = curriculum["capstone"]
        max_tokens = self.llm.gen.get("capstone_max_tokens", 4000)
        validation_timeout = self.llm.gen.get("capstone_validation_timeout", 120)

        # Step 1: interfaces (generated once)
        interfaces_content = self._generate_interfaces(cap, max_tokens)
        (self.capstone_dir / "interfaces.py").write_text(interfaces_content, encoding="utf-8")
        logger.info("Generated interfaces.py")

        # Step 2: initial tests
        tests_content = self._generate_tests(cap, interfaces_content, max_tokens)

        validation_passed = False
        validation_warnings: list[str] = []
        attempt = 0

        # Steps 3–4: validation loop
        for attempt in range(1, max_attempts + 1):
            logger.info("Validation attempt %d/%d", attempt, max_attempts)

            reference_impl = self._generate_reference_impl(
                cap, interfaces_content, tests_content, max_tokens
            )
            passed, output = self._run_validation(
                interfaces_content, tests_content, reference_impl,
                timeout=validation_timeout,
            )

            if passed:
                logger.info("Validation passed on attempt %d", attempt)
                validation_passed = True
                break

            logger.warning("Validation failed (attempt %d):\n%s", attempt, output[:800])
            validation_warnings.append(f"Attempt {attempt}: {output[:500]}")

            if attempt < max_attempts:
                tests_content = self._regenerate_tests(
                    original_tests=tests_content,
                    reference_impl=reference_impl,
                    pytest_output=output,
                    max_tokens=max_tokens,
                )

        # Persist final tests (validated or not)
        (self.capstone_dir / "test_capstone.py").write_text(tests_content, encoding="utf-8")

        # Step 5: README with optional warning
        warning_section = ""
        if not validation_passed:
            warning_section = (
                "\n\n## ⚠️ Validation Notice\n\n"
                "These tests could not be fully validated against a reference implementation.\n"
                "Some tests may have issues. If a failing test appears to be wrong "
                "(not your implementation), please file an issue.\n\n"
                "Validation history:\n"
                + "\n".join(f"- {w}" for w in validation_warnings[:3])
            )

        readme_content = self._generate_readme(cap, warning_section, max_tokens)
        (self.capstone_dir / "README.md").write_text(readme_content, encoding="utf-8")

        # Step 6: student starter (no reference impl shipped)
        self._write_implementation_starter(cap)

        # Update state
        progress = self.state.setdefault("progress", {})
        progress["capstone_complete"] = True
        progress["capstone_validated"] = validation_passed
        progress["validation_attempts"] = attempt

    def refine(self, instruction: str, curriculum: dict) -> None:
        """Re-run generation with an extra requirement injected into the description.

        Args:
            instruction: User refinement instruction.
            curriculum: Curriculum dict.
        """
        curriculum = dict(curriculum)
        curriculum["capstone"] = dict(curriculum["capstone"])
        curriculum["capstone"]["description"] += f"\n\nAdditional requirement: {instruction}"
        self.generate_with_validation(curriculum)

    # ── Private generation methods ──────────────────────────────────────────

    def _generate_interfaces(self, cap: dict, max_tokens: int) -> str:
        """Generate interfaces.py content."""
        prompt = INTERFACES_USER.format(
            modules_json=json.dumps(cap.get("modules", []), indent=2)
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=max_tokens,
        )
        return _strip_code_fences(response)

    def _generate_tests(self, cap: dict, interfaces_content: str, max_tokens: int) -> str:
        """Generate initial test_capstone.py content."""
        prompt = TESTS_USER.format(
            interfaces_content=interfaces_content,
            modules_json=json.dumps(cap.get("modules", []), indent=2),
            integration_test_json=json.dumps(cap.get("integration_test", {}), indent=2),
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=max_tokens,
        )
        return _strip_code_fences(response)

    def _generate_reference_impl(
        self, cap: dict, interfaces_content: str, tests_content: str, max_tokens: int
    ) -> str:
        """Generate a reference implementation that should pass all tests."""
        prompt = REFERENCE_IMPL_USER.format(
            interfaces_content=interfaces_content,
            modules_json=json.dumps(cap.get("modules", []), indent=2),
            tests_content=tests_content,
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=max_tokens + 2000,
        )
        return _strip_code_fences(response)

    def _regenerate_tests(
        self,
        original_tests: str,
        reference_impl: str,
        pytest_output: str,
        max_tokens: int,
    ) -> str:
        """Regenerate tests given pytest failure output and the reference impl."""
        prompt = TESTS_REGEN_USER.format(
            original_tests=original_tests,
            reference_impl=reference_impl,
            pytest_output=pytest_output[:3000],
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=max_tokens,
        )
        return _strip_code_fences(response)

    def _generate_readme(self, cap: dict, warning_section: str, max_tokens: int) -> str:
        """Generate the capstone README.md."""
        prompt = CAPSTONE_README_USER.format(
            capstone_title=cap.get("title", "Capstone"),
            capstone_description=cap.get("description", ""),
            estimated_hours=cap.get("estimated_hours", 4),
            modules_json=json.dumps(cap.get("modules", []), indent=2),
            integration_test_json=json.dumps(cap.get("integration_test", {}), indent=2),
            warning_section=warning_section,
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("capstone"),
            max_tokens=max_tokens,
        )

    def _write_implementation_starter(self, cap: dict) -> None:
        """Write implementation.py — the student starter file (no reference logic)."""
        modules_list = "\n".join(
            f"#   {i}. {m['name']}"
            for i, m in enumerate(cap.get("modules", []), 1)
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
# class YourClass(AbstractBase):
#     def __init__(self):
#         # TODO: initialise
#         pass
#
#     def key_method(self, arg):
#         # TODO: implement
#         pass
'''
        (self.capstone_dir / "implementation.py").write_text(starter, encoding="utf-8")

    # ── Validation runner ────────────────────────────────────────────────────

    def _run_validation(
        self,
        interfaces_content: str,
        tests_content: str,
        reference_impl: str,
        timeout: int = 120,
    ) -> tuple[bool, str]:
        """Delegate to validation.runner.run_pytest.

        Args:
            interfaces_content: Python source for interfaces.py.
            tests_content: Python source for test_capstone.py.
            reference_impl: Python source for the reference implementation.
            timeout: Subprocess timeout in seconds.

        Returns:
            (passed, output)
        """
        return run_pytest(interfaces_content, tests_content, reference_impl, timeout=timeout)


def _strip_code_fences(code: str) -> str:
    """Strip leading ``` / ```python and trailing ``` fences from LLM output."""
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
    if code.endswith("```"):
        code = code[: code.rfind("```")]
    return code.strip()
