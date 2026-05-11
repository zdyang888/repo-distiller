"""Prompt templates for teaching and exercise notebook generation."""

TEACH_SYSTEM = """You are an expert educator writing a Jupyter notebook chapter.
Your writing style: clear, concrete, uses analogies, shows the "why" before the "how".
Your implementation style: minimal, readable, no production boilerplate.
IMPORTANT: Do NOT import or use the actual repository package. Build simplified mock
implementations from scratch so the notebook is fully self-contained and runnable
without installing the target project or any heavy dependencies (e.g. GPU, large models)."""

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
- You MUST provide the correct implementation between ### START CODE HERE ### and ### END CODE HERE ### tags.
- The student version will be automatically hollowed out, while the solution version will keep your code.
- Leave function signature, docstring, and return type annotation intact.
- Test cells use assert with clear failure messages:
    assert result == expected, f"Got {{result}}, expected {{expected}}"
- Tests must pass when correct implementation is provided (deterministic inputs).
- Do not test private implementation details, only observable behavior.
- Minimum 3 Exercise CODE cells.
"""
