# 01 — Data Models & Schemas

All data passed between modules is plain Python dicts (matching JSON schema). Defined here
in one place. No dataclasses, no Pydantic — keep it simple, validate explicitly where it
matters.

## ExplorerFindings

Output of `ExplorerAgent.explore()`. Persisted as `state["findings"]` in `.distill_state.json`.

```python
{
  "title": str,           # Short project title, e.g. "nanoGPT"
  "one_liner": str,       # One sentence: what this project does
  "mental_model": str,    # 2–3 sentence paragraph: how the system works conceptually
  "domain": str,          # One of:
                          #   "llm_framework" | "agent_system" | "ml_infra"
                          #   "inference_engine" | "quant_framework" | "other"
  "concepts": [
    {
      "name": str,        # Short name in PascalCase, e.g. "CausalSelfAttention"
      "description": str, # One sentence: what it is and why it exists
      "complexity": str,  # "basic" | "intermediate" | "advanced"
      "key_files": [str]  # Relative paths in the cloned repo, 1-3 items
    }
  ],
  "dependency_order": [str],  # Concept names in learning order (must reference concepts above)
  "key_files": [str],         # Top 5-8 files for understanding the architecture
  "skip_files": [str]         # Files that look important but are infra/compat (informational)
}
```

**Validation rules:**
- `concepts` must be non-empty (≥ 3, ≤ 10)
- Every name in `dependency_order` must appear in `concepts`
- All paths in `key_files` must be relative (no leading `/`)

If validation fails: log warning, proceed anyway. The downstream agents are forgiving.

## Curriculum

Output of `CurriculumAgent.generate()`. Persisted as `state["curriculum"]`. Also serialized
to human-readable `CURRICULUM.md` for review.

```python
{
  "title": str,                  # Full course title, e.g. "Understanding nanoGPT"
  "mental_model": str,           # Inherited from findings or refined
  "concepts": [                  # Subset/reordering of findings.concepts
    {
      "name": str,
      "description": str,
      "complexity": str          # "basic" | "intermediate" | "advanced"
    }
  ],
  "notebooks": [
    {
      "id": str,                 # Zero-padded sequential: "01", "02", ...
      "title": str,              # Short, descriptive notebook title
      "concept": str,            # Must match a concept name above
      "description": str,        # What the student builds and learns
      "prerequisites": [str],    # List of concept names (may be empty)
      "key_source_files": [str], # Relative paths in cloned repo
      "key_symbols": [str],      # NEW: specific class/function names to extract via AST
                                 # e.g. ["CausalSelfAttention", "Block.forward"]
                                 # If empty, generator falls back to whole-file truncation
      "learning_objectives": [str],  # 2-4 "By the end..." bullets
      "exercise_description": str,   # What students implement in the exercise
      "visualization_idea": str      # What to visualize (may be empty)
    }
  ],
  "capstone": {
    "title": str,                  # e.g. "mini-langchain"
    "description": str,            # 2-3 sentences
    "estimated_hours": int,        # 2-8
    "modules": [
      {
        "name": str,               # Python class name, PascalCase
        "description": str,        # What this module does
        "depends_on": [str],       # Other module names this depends on
        "interface_sketch": str,   # Python pseudocode showing method signatures
        "test_behaviors": [str]    # Concrete behavior descriptions for test generation
                                   # Each must be specific & verifiable, e.g.:
                                   # "encode(decode(ids)) == ids for any valid id list"
                                   # NOT: "should work correctly"
      }
    ],
    "integration_test": {          # NEW: structured integration test spec
      "description": str,          # What the test does
      "setup_code": str,           # Python code to set up (data, config)
      "success_metric": str,       # E.g. "loss decreases by at least 50% after 500 steps"
      "expected_output_check": str # E.g. "generated string contains spaces and is printable ASCII"
    }
  }
}
```

**Validation rules:**
- 3 ≤ `notebooks` ≤ 10
- Every `notebook.concept` must appear in `concepts`
- Every name in `notebook.prerequisites` must be a concept name
- Notebook IDs are sequential starting from "01"
- Every `capstone.modules[*].depends_on` name must reference another module name

## DistillState

The complete `.distill_state.json` file. Internal state, never shown to users directly.

```python
{
  "schema_version": 1,                # Bump if schema changes
  "repo_url": str,                    # Original GitHub URL
  "repo_path": str,                   # Absolute path to cloned repo on disk
  "output_dir": str,                  # Absolute path to output directory
  "findings": dict,                   # ExplorerFindings (see above)
  "curriculum": dict,                 # Curriculum (see above)
  "step": str,                        # "explored" | "generating" | "generated" | "failed"
  "progress": {                       # NEW: resumable generation
    "notebooks_complete": [str],      # IDs of completed notebooks (e.g. ["01", "02"])
    "exercises_complete": [str],      # Same for exercises
    "capstone_complete": bool,
    "last_error": str | None          # If step == "failed", what went wrong
  },
  "cost": {                           # NEW: token tracking
    "explore_input_tokens": int,
    "explore_output_tokens": int,
    "curriculum_input_tokens": int,
    "curriculum_output_tokens": int,
    "notebooks_input_tokens": int,
    "notebooks_output_tokens": int,
    "capstone_input_tokens": int,
    "capstone_output_tokens": int,
    "total_estimated_usd": float      # Computed from per-model rates in llm/cost.py
  },
  "config_snapshot": {                # NEW: which model/config was used
    "default_model": str,
    "step_models": dict
  }
}
```

**State transitions:**
- After `--repo`: `step = "explored"`, `progress.*` initialized empty
- During `--continue`: `step = "generating"`, `progress.*` updated incrementally
- After `--continue` success: `step = "generated"`
- On any failure: `step = "failed"`, `progress.last_error` set

## CURRICULUM.md (human-editable serialization of Curriculum)

Format spec — generator MUST produce this format exactly. Parser MUST handle this format
robustly even if the user edits it.

```markdown
# {curriculum.title}

> {curriculum.mental_model}

**Source:** {repo_url}

<!--
HOW TO EDIT THIS FILE:
- Toggle `enabled: false` in any notebook block to skip it
- Edit `description` text to change focus
- Add `extra_requirements` list to add specific asks
- Reorder notebook blocks to change order (IDs auto-renumber)
- Lines outside notebook blocks are ignored
-->

## Concepts

- **{name}** ({complexity}): {description}
- ...

## Notebooks

<!-- nb-block-start id=01 -->
### Notebook 01: {title}

```yaml
enabled: true
concept: {concept_name}
prerequisites: [list of concept names]
key_source_files:
  - path/to/file.py
key_symbols:
  - ClassName
  - function_name
extra_requirements: []
```

**Description:** {description}

**Learning objectives:**
- {obj 1}
- {obj 2}

**Exercise:** {exercise_description}

**Visualization:** {visualization_idea}
<!-- nb-block-end id=01 -->

<!-- nb-block-start id=02 -->
### Notebook 02: ...
<!-- nb-block-end id=02 -->

## Capstone

{render capstone struct similarly with YAML block}
```

**Why this format:**

- The `<!-- nb-block-start ... -->` HTML comments are invisible in rendered Markdown but
  give the parser unambiguous block boundaries.
- The YAML code block is the **structured data** — parser reads YAML, not free text.
- The free-text descriptions below YAML are also parsed (description, objectives, exercise,
  viz) but the YAML is canonical for non-text fields.

**Parsing approach (`tools/curriculum_md.py`):**

```python
def parse_curriculum_md(md_text: str, original: dict) -> dict:
    """
    Parse user-edited CURRICULUM.md back into a Curriculum dict.

    Strategy:
    1. Find all <!-- nb-block-start id=NN --> ... <!-- nb-block-end id=NN --> blocks
    2. For each block:
       a. Extract YAML between ```yaml ... ``` — this is the canonical data
       b. Extract title from `### Notebook NN: TITLE` line
       c. Extract description, objectives, etc. from the markdown body
    3. Apply yaml.enabled = false → skip notebook
    4. Apply yaml.extra_requirements → append to description
    5. Renumber notebook IDs sequentially (01, 02, 03...) regardless of original IDs
    6. For any field not present in user edit, fall back to `original`
    """
```

This is much more robust than the v1 approach of regex-matching `[SKIP]` markers.

## MockLLM (for tests)

Used in unit tests to avoid real API calls. Lives in `tests/conftest.py`.

```python
class MockLLM:
    """Test double for LLMClient. Returns canned responses in order."""

    def __init__(self, responses: list[str]):
        """responses: list of strings, returned one per chat() call."""
        self.responses = responses
        self.calls = []  # List of (messages, system, model) for assertions

    @property
    def model(self) -> str:
        return "mock-model"

    @property
    def gen(self) -> dict:
        return {"max_explore_steps": 20, "max_notebooks": 8}

    def model_for_step(self, step: str) -> str:
        return "mock-model"

    def chat(self, messages, system=None, model=None, max_tokens=8192, retries=3) -> str:
        self.calls.append({"messages": messages, "system": system, "model": model})
        if not self.responses:
            raise RuntimeError("MockLLM ran out of canned responses")
        return self.responses.pop(0)
```
