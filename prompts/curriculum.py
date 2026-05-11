"""Curriculum agent prompts."""

CURRICULUM_SYSTEM = """You are an expert educator designing a course curriculum for software engineers."""

REVISE_SYSTEM = """You are an expert educator revising a course curriculum based on feedback."""

REVISE_USER = """Here is the current curriculum for a software engineering course:

{curriculum_md}

The user has provided the following feedback:
{feedback}

Revise the curriculum to address this feedback. Keep everything that works well — only change what the feedback asks for.
Output ONLY valid JSON using the exact same structure as the embedded JSON in the curriculum above.
Do not add any explanation, markdown, or text outside the JSON object."""

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
      "visualization_idea": "matplotlib or graphviz diagram: e.g. architecture block diagram (graphviz), attention heatmap (matplotlib), data pipeline flow (graphviz), memory layout (matplotlib), or training loss curve (matplotlib)"
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
