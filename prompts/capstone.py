"""Prompt templates for capstone project generation and validation."""

CAPSTONE_README_USER = """Write a detailed capstone project README for students.

── Project info ──
Title: {capstone_title}
Description: {capstone_description}
Estimated time: {estimated_hours} hours

── Modules to implement ──
{modules_json}

── Integration test ──
{integration_test_json}

Write a README.md that:
1. Opens with motivation: "By completing this project, you will have built..."
2. Describes the system architecture with an ASCII diagram
3. For each module: describes its responsibility, interface, and expected behavior
4. Explains how modules connect to each other
5. Lists the success criteria (pytest tests pass)
6. Provides a suggested implementation order
7. Includes a "Getting Started" section:
    cd capstone/
    # Implement in implementation.py
    pytest test_capstone.py -v

Be specific and concrete. This is a guide, not documentation.

{warning_section}
"""
# warning_section is "" normally, or a "⚠️ Note: Validation issues" block
# if the validation loop ended without all tests passing.

INTERFACES_USER = """Generate a Python interfaces.py file defining abstract base classes for a capstone project.

Modules:
{modules_json}

Rules:
- Use Python ABC (from abc import ABC, abstractmethod)
- Comprehensive docstrings explaining the CONTRACT (what each method must do)
- Type annotations for all parameters and return types (use modern syntax: list[int], etc.)
- NotImplementedError with helpful message in each abstract method body
- Module-level docstring explaining the overall architecture
- Do NOT implement any logic — only define the interface
- Add `# TODO: Implement this` comments as hints

Output ONLY valid Python code, no markdown fences.
"""

TESTS_USER = """Generate a pytest test file (test_capstone.py) for a capstone project.

── Interfaces ──
{interfaces_content}

── Module behaviors ──
{modules_json}

── Integration spec ──
{integration_test_json}

Generate test_capstone.py with:

1. A fixture for each module (instantiate it with minimal args)
2. Unit tests: 3-5 per module (def test_<ModuleName>_<behavior>)
3. An integration test (@pytest.mark.integration) wiring all modules together

Rules:
- Check BEHAVIOR, not implementation details
- Concrete, deterministic test inputs and expected outputs
- Each test has a docstring explaining what it verifies and why
- Tests are independent (each runs in isolation)
- Failure messages are descriptive: assert result == expected, f"Expected X because Y, got {{result}}"
- Each test has a # STUDENT_HINT comment explaining where to look if it fails
- Import student code from `implementation` (i.e., `from implementation import *`)

CRITICAL: Tests must fail if the interface is implemented incorrectly. Use realistic
test data that exercises edge cases.

Output ONLY valid Python code.
"""

# Used by the validation loop (see plan/04-validation.md) to generate a reference
# implementation that should make all tests pass.
REFERENCE_IMPL_USER = """You are writing a complete REFERENCE implementation of the
following capstone project. This implementation will be used to validate the test suite —
all tests should pass against your implementation.

── Interfaces ──
{interfaces_content}

── Module behaviors ──
{modules_json}

── Test file (your implementation must pass all of these) ──
{tests_content}

Write a complete, correct implementation.py file that:
- Imports all needed abstract bases from interfaces
- Defines a concrete subclass of each abstract base
- Implements ALL abstract methods
- Passes ALL tests in the test file above

Output ONLY valid Python code, no markdown fences. Start with imports.
"""

# Used when validation fails: ask LLM to fix the test file given the failure output
TESTS_REGEN_USER = """Your previous test file failed when run against a reference implementation.
Fix the tests so they pass.

── Original test file ──
{original_tests}

── Reference implementation ──
{reference_impl}

── pytest failure output ──
{pytest_output}

Common causes:
- Test assumes specific implementation detail not in the interface
- Test uses wrong expected value
- Test calls method with wrong signature
- Integration test requires setup that the reference impl doesn't satisfy

Fix the tests. Keep them rigorous (must catch real implementation bugs) but make them
pass against this correct reference. Output ONLY the corrected test_capstone.py file.
"""
