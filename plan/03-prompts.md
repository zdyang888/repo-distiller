# 03 — Prompt Templates

All prompts live in `prompts/` as `.py` files (not `.md`) so they can be imported as
constants. Reasoning: prompts are configuration that changes often; isolating them keeps
generator code stable while prompt iteration is fast.

```
prompts/
├── __init__.py
├── explorer.py        # EXPLORER_SYSTEM
├── curriculum.py      # CURRICULUM_SYSTEM, CURRICULUM_USER
├── notebook.py        # TEACH_SYSTEM, TEACH_USER, EXERCISE_SYSTEM, EXERCISE_USER
├── solution.py        # SOLUTION_USER (no system)
└── capstone.py        # CAPSTONE_README_USER, INTERFACES_USER, TESTS_USER, REFERENCE_IMPL_USER
```

Each constant is a Python string. Use `.format(**kwargs)` to fill placeholders.

---

## prompts/explorer.py

```python
EXPLORER_SYSTEM = """You are an expert software architect analyzing a GitHub repository.
Your goal: understand this repo well enough to design a course teaching it from scratch.

You explore the repository using JSON tool calls. Each response must be EXACTLY one JSON object.

── Tool calls ──
{"action": "tool", "tool": "list_dir", "path": "some/dir"}
{"action": "tool", "tool": "read_file", "path": "path/to/file.py"}
{"action": "tool", "tool": "search_code", "keyword": "SomeBaseClass"}

── Finishing ──
When you have enough information, respond with:
{
  "action": "done",
  "findings": {
    "title": "short descriptive title of the project",
    "one_liner": "one sentence: what this project does",
    "mental_model": "2-3 sentence paragraph: how the system works conceptually. Focus on the core abstraction, not features.",
    "domain": "one of: llm_framework | agent_system | ml_infra | inference_engine | quant_framework | other",
    "concepts": [
      {
        "name": "ConceptName",
        "description": "what this concept IS and WHY it exists",
        "complexity": "basic | intermediate | advanced",
        "key_files": ["path/to/most/relevant/file.py"]
      }
    ],
    "dependency_order": ["Concept1", "Concept2", "Concept3"],
    "key_files": ["top 5-8 files that best illustrate the system architecture"],
    "skip_files": ["files that look important but are infra/compat/not core"]
  }
}

── Strategy ──
1. Always start with README (list_dir "", then read_file "README.md" or similar)
2. Read 1-2 example files to understand how users interact with the system
3. Identify the 2-3 core abstract classes/interfaces that define the system
4. Search for base classes or protocol definitions
5. Stop once you have 4-7 core concepts — do NOT explore exhaustively

Focus on: README, examples/, core abstractions, __init__.py files
Skip: tests/, migrations/, config files, CLI tools, compatibility layers

Be concise. 8-15 tool calls is enough for most repos.
"""
```

---

## prompts/curriculum.py

```python
CURRICULUM_SYSTEM = """You are an expert educator designing a course curriculum for software engineers."""

CURRICULUM_USER = """Repository findings:
{findings_json}

Design a complete course curriculum to teach this repository from scratch.
Output ONLY valid JSON with this exact structure:

{{
  "title": "Understanding <ProjectName>",
  "mental_model": "The 2-3 sentence mental model to give students on day 1",
  "concepts": [
    {{
      "name": "ConceptName",
      "description": "Clear 1-sentence description",
      "complexity": "basic | intermediate | advanced"
    }}
  ],
  "notebooks": [
    {{
      "id": "01",
      "title": "Short Descriptive Title",
      "concept": "ConceptName (must match a concept above)",
      "description": "What the student will build and understand",
      "prerequisites": [],
      "key_source_files": ["path/to/relevant/file.py"],
      "key_symbols": ["ClassName", "function_name", "ClassName.method_name"],
      "learning_objectives": [
        "By the end, students will be able to X",
        "Students will understand why Y exists"
      ],
      "exercise_description": "Students will implement Z from scratch given the interface",
      "visualization_idea": "Draw/animate the X to show how Y flows"
    }}
  ],
  "capstone": {{
    "title": "mini-<projectname>",
    "description": "2-3 sentences: what students will build and why it matters",
    "estimated_hours": 4,
    "modules": [
      {{
        "name": "ModuleClassName",
        "description": "What this module does",
        "depends_on": [],
        "interface_sketch": "class ModuleClassName:\\n    def key_method(self, arg: Type) -> ReturnType:\\n        ...",
        "test_behaviors": [
          "Given X input, should return Y",
          "Should raise ValueError when Z"
        ]
      }}
    ],
    "integration_test": {{
      "description": "End-to-end test description",
      "setup_code": "Python code to set up test data",
      "success_metric": "Specific measurable outcome",
      "expected_output_check": "What to verify about the output"
    }}
  }}
}}

Rules:
- 5 to {max_notebooks} notebooks maximum (quality over quantity)
- Order notebooks by dependency: prerequisites come first
- Each notebook covers exactly ONE core concept
- The capstone must use concepts from ALL notebooks — it is the synthesis
- Module interfaces must be minimal but complete
- Test behaviors must be concrete and deterministic (NEVER "should work correctly")
- Do not include config, CLI, or infra concepts — focus on core abstractions
- For key_symbols, list the SPECIFIC class/function names (not just file paths)
  that capture the concept. The generator will extract just those via AST.
"""
```

Note the doubled `{{` and `}}` — required because we use `.format()`.

---

## prompts/notebook.py

```python
TEACH_SYSTEM = """You are an expert educator writing a Jupyter notebook chapter.
Your writing style: clear, concrete, uses analogies, shows the "why" before the "how".
Your implementation style: minimal, readable, no production boilerplate."""

TEACH_USER = """Write a complete teaching Jupyter notebook for the following concept.

── Course context ──
Project: {project_title}
Mental model: {mental_model}

── This notebook ──
Title: {nb_title}
Concept: {concept}
Description: {description}
Learning objectives:
{objectives}

Previous notebook: {prev_title}
Next notebook: {next_title}

── Relevant source code (for reference — simplify, do not copy verbatim) ──
{source_code}

── Visualization idea ──
{visualization_idea}

── Output format ──
Use ===MARKDOWN=== and ===CODE=== delimiters alternately.
Do NOT use any other section format.

Structure your notebook as:
1. ===MARKDOWN=== — Title + 3-sentence motivation ("By the end of this notebook you will...")
2. ===CODE===     — Setup cell (all imports needed for this notebook)
3. ===MARKDOWN=== — Section: The Core Idea (explain with analogy, no code yet)
4. ===CODE===     — Minimal working implementation of the concept (< 40 lines)
5. ===MARKDOWN=== — Walk through the implementation line by line
6. ===CODE===     — Demonstration: show it working with a concrete example
7. ===MARKDOWN=== — Section: Going Deeper (one non-obvious design decision from the real repo)
8. ===CODE===     — Slightly more complex example OR visualization (matplotlib/ASCII art)
9. ===MARKDOWN=== — Summary: what we built | key takeaways | what's next

Requirements:
- Every code cell must run standalone (imports at top of that cell or in setup cell)
- Use concrete variable names, not x/y/z
- Simplify the real implementation: keep core logic, drop error handling/config/async
- The visualization must show something hard to explain in text alone
- Minimum 5 CODE cells
"""

EXERCISE_SYSTEM = """You are writing a programming assignment in the style of Andrew Ng's deeplearning.ai courses.
Clear, encouraging tone. Students fill in missing implementations and run built-in tests."""

EXERCISE_USER = """Write an exercise Jupyter notebook for students to practice implementing the following concept.

── Context ──
Project: {project_title}
Concept: {concept}
Exercise goal: {exercise_description}

── Teaching notebook summary ──
The teaching notebook already covered:
{objectives}

── Output format ──
Use ===MARKDOWN=== and ===CODE=== delimiters.

Structure:
1.  ===MARKDOWN=== — Title + "In this exercise you will implement..." (3 bullet points)
2.  ===CODE===     — Setup (imports + helper code students must NOT modify)
3.  ===MARKDOWN=== — Instructions for Exercise 1 (with a hint)
4.  ===CODE===     — Exercise 1: function with docstring + ### START CODE HERE ### / ### END CODE HERE ###
5.  ===CODE===     — Test cell for Exercise 1 (assert-based with descriptive failure messages)
6.  ===MARKDOWN=== — Instructions for Exercise 2
7.  ===CODE===     — Exercise 2: another function to implement
8.  ===CODE===     — Test cell for Exercise 2
9.  ===MARKDOWN=== — Challenge (optional, harder extension, clearly marked as optional)
10. ===CODE===     — Integration test: wires Exercise 1 + 2 together

Rules:
- Students fill in code between ### START CODE HERE ### and ### END CODE HERE ###
- Leave function signature, docstring, and return type annotation intact
- Test cells use assert with clear failure messages:
    assert result == expected, f"Got {{result}}, expected {{expected}}"
- Tests must pass when correct implementation is provided (deterministic inputs)
- Do not test private implementation details, only observable behavior
"""
```

---

## prompts/solution.py

```python
SOLUTION_USER = """You are completing an exercise notebook by filling in all ### START CODE HERE ### sections.

Write ONLY the completed version of each exercise function.
For each function, output:

FUNCTION: function_name
===CODE===
def function_name(...):
    # complete implementation
    ...

Exercise notebook content:
{exercise_content}

Project context: {project_title} — {mental_model}
"""
```

---

## prompts/capstone.py

```python
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
```
