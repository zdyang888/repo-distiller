"""Prompt template for solution notebook generation."""

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
