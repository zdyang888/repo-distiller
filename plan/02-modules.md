# 02 — Module Specifications

Detailed spec for each module. Implement in the order in CLAUDE.md.

---

## 6.1 LLM Client (`llm/client.py`)

### Purpose

Single entry point for LLM calls. Routes to Claude / OpenAI / Gemini based on model name.
Handles retries, errors, and token tracking.

### Public API

```python
class LLMClient:
    def __init__(self, config_path: str = "config.yaml") -> None
    
    @property
    def model(self) -> str: ...
    
    @model.setter
    def model(self, value: str) -> None: ...
    
    @property
    def gen(self) -> dict: 
        """Return the generation: section of config."""
    
    def model_for_step(self, step: str) -> str:
        """
        Return step-specific model from config.step_models, or default_model.
        step is one of: "explore", "curriculum", "notebooks", "capstone".
        """
    
    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        """Returns assistant text. Tracks tokens via self.last_usage."""
    
    @property
    def last_usage(self) -> dict:
        """
        Returns: {"input_tokens": int, "output_tokens": int, "model": str}
        Populated after each chat() call. Used by cost tracking.
        """
```

### Provider routing

| Model name pattern | Provider | SDK | Env var |
|---|---|---|---|
| contains `"claude"` | Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| contains `"gpt-"`, `"o1"`, `"o3"`, `"o4"` | OpenAI | `openai` | `OPENAI_API_KEY` |
| contains `"gemini"` | Google (via OpenAI-compat) | `openai` | `GOOGLE_API_KEY` |

### Implementation notes

- **Anthropic**: `system` is a top-level kwarg, NOT in `messages`.
- **OpenAI/Gemini**: `system` is the first message with `role: "system"`.
- **Gemini base URL**: `https://generativelanguage.googleapis.com/v1beta/openai`
- **Retries**: exponential backoff `5 * 2^attempt` seconds. Catch all exceptions, log, retry.
- **last_usage**: extract from `response.usage` (Anthropic) or `response.usage` (OpenAI).
  Store the model name too — needed for cost calc.

### Errors

- Missing API key: raise `EnvironmentError("ANTHROPIC_API_KEY not set")`.
- Missing SDK: raise `ImportError("Install with: pip install anthropic")`.
- All retries exhausted: raise the last underlying exception.

### Test (`tests/test_llm_client.py`)

- `test_model_for_step_returns_default_when_no_override`
- `test_model_for_step_returns_override_when_configured`
- `test_routing_claude_calls_anthropic` (mock the SDK)
- `test_routing_gpt_calls_openai` (mock)
- `test_routing_gemini_uses_compat_endpoint` (mock, verify base_url)
- `test_chat_retries_on_transient_error` (mock raises 2x then succeeds)
- `test_missing_api_key_raises_envvar_error`

---

## 6.2 RepoTool (`tools/repo.py`)

### Purpose

Clone a repo, then provide controlled file access to the explorer agent.

### Public API

```python
SKIP_DIRS = {".git", ".github", ".gitlab", "__pycache__", "node_modules",
             "migrations", "alembic", ".pytest_cache", "dist", "build",
             ".eggs", ".tox", "venv", ".venv", "env", ".mypy_cache",
             ".ruff_cache", "site-packages"}

SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
                   ".zip", ".tar", ".gz", ".lock", ".bin", ".whl", ".pyc",
                   ".pyo", ".so", ".dylib", ".dll", ".exe", ".db", ".sqlite",
                   ".DS_Store", ".parquet", ".npy", ".npz", ".h5", ".ckpt",
                   ".safetensors", ".pt", ".pth"}


class RepoTool:
    def __init__(self, url: str, work_dir: str | None = None): ...
    
    self.url: str
    self.work_dir: str
    self.repo_path: str | None  # set after clone()
    
    def clone(self) -> str:
        """Shallow clone (--depth 1). If already exists, reuse. Returns abs path."""
    
    # ── Tools called by ExplorerAgent ──
    
    def list_dir(self, rel_path: str = "") -> str:
        """
        Returns formatted string for LLM consumption:
            Contents of /agents:
              [DIR]  __pycache__/         <-- SKIPPED in actual output
              [FILE] explorer.py  (4,521 B)
              [FILE] curriculum.py  (3 KB)
        Skips SKIP_DIRS, SKIP_EXTENSIONS.
        Returns "Path not found: ..." or "Not a directory: ..." for errors.
        """
    
    def read_file(self, rel_path: str, max_chars: int = 8000) -> str:
        """
        Returns: '=== {rel_path} ===\n{content}'
        Truncates at max_chars with visible notice showing total chars.
        Returns 'File not found: ...' or 'Binary file skipped: ...' for errors.
        """
    
    def search_code(self, keyword: str, file_pattern: str = "*.py", max_results: int = 20) -> str:
        """
        Recursive case-insensitive search. Returns:
            Search 'BaseRetriever' (3 results):
            langchain/retrievers/base.py:42  class BaseRetriever(ABC):
            ...
        """
    
    # ── Helper methods for generators ──
    
    def get_priority_files(self) -> list[str]:
        """
        Returns up to 12 high-value files in this priority order:
        1. README.{md,rst,txt}
        2. ARCHITECTURE.md, DESIGN.md, OVERVIEW.md
        3. examples/, example/, tutorials/, cookbook/, demo/ (.py and .ipynb, max 3 each)
        4. Top-level __init__.py files (max 6, sorted by depth)
        """
    
    def extract_symbols(self, rel_path: str, symbols: list[str]) -> str:
        """
        NEW: AST-based extraction of specific class/function definitions.
        
        Args:
            rel_path: file path
            symbols: list of names. Supports:
                - "ClassName"        — full class
                - "ClassName.method" — single method
                - "function_name"    — top-level function
        
        Returns formatted string:
            === {rel_path} ===
            
            # === ClassName ===
            class ClassName:
                ...full class source...
            
            # === function_name ===
            def function_name(...):
                ...full function source...
        
        Behavior:
        - If file is not Python or AST parse fails: fall back to read_file()
        - If a symbol isn't found: include "# {symbol} not found" in output
        - Includes ONLY the requested symbols (not surrounding context)
        - Preserves docstrings and decorators
        
        Implementation: use `ast` module, walk the tree, extract source via
        `ast.get_source_segment()` (Python 3.8+).
        """
    
    def read_files_for_concept(
        self,
        file_paths: list[str],
        symbols: list[str] | None = None,
        max_total_chars: int = 6000,
    ) -> str:
        """
        Read multiple files with shared char budget.
        
        If symbols is provided AND non-empty: use extract_symbols on each file.
        Else: use read_file() on each file.
        
        Concatenates results, stops when budget exhausted.
        """
```

### Test (`tests/test_repo.py`)

Use a fixture directory at `tests/fixtures/sample_repo/` with known contents:

```
tests/fixtures/sample_repo/
├── README.md
├── src/
│   ├── __init__.py
│   ├── core.py        # contains class CoreClass with method greet()
│   └── utils.py       # contains function helper()
└── tests/
    └── test_core.py   # should be skipped by priority logic
```

Tests:
- `test_clone_reuses_existing` — second clone() doesn't actually clone
- `test_list_dir_skips_hidden_dirs` — `__pycache__` not in output
- `test_list_dir_skips_binary_extensions` — `.png` not in output
- `test_read_file_truncates_long_files` — truncation notice present
- `test_read_file_returns_error_for_missing` — "File not found" in output
- `test_search_code_finds_keyword` — searches case-insensitively
- `test_search_code_respects_max_results` — caps at limit
- `test_get_priority_files_orders_correctly` — README first
- `test_extract_symbols_class` — extracts `CoreClass` correctly
- `test_extract_symbols_method` — extracts `CoreClass.greet` correctly
- `test_extract_symbols_function` — extracts `helper` correctly
- `test_extract_symbols_missing_falls_back` — non-existent symbol → "not found" comment
- `test_extract_symbols_non_python_falls_back` — .md file → falls back to read_file

---

## 6.3 NotebookTool (`tools/notebook.py`)

### Purpose

Construct valid `nbformat 4.5` notebook JSON. Parse LLM responses with cell delimiters.

### Cell delimiter format

LLM responses use `===MARKDOWN===` and `===CODE===` between cells:

```
===MARKDOWN===
# Title
Content

===CODE===
import torch

===MARKDOWN===
More content
```

### Public API

```python
def make_markdown_cell(source: str) -> dict: ...
def make_code_cell(source: str) -> dict: ...

def build_notebook(cells: list[dict], title: str = "") -> dict:
    """Wrap cells in nbformat 4.5 notebook. Adds standard Python 3 kernelspec."""

def save_notebook(nb: dict, path: str) -> None:
    """Write notebook JSON. Creates parent dirs. ensure_ascii=False."""

def load_notebook(path: str) -> dict:
    """Read and parse notebook JSON."""

def parse_llm_response(response: str) -> list[dict]:
    """
    Parse ===MARKDOWN=== / ===CODE=== delimited response into cells.
    
    Handles:
    - Mixed case delimiters (===markdown===, ===Code===)
    - Code cells wrapped in ```python ... ``` (strips fences)
    - No delimiters present → wrap entire response in one markdown cell
    - Empty cells → skipped
    """

def notebook_from_llm_response(response: str, title: str = "") -> dict:
    """parse + build in one step."""

def validate_notebook(nb: dict) -> list[str]:
    """
    NEW: Validate notebook structure. Returns list of error strings (empty if valid).
    
    Checks:
    - Has correct nbformat (4) and nbformat_minor (≥ 5)
    - Has metadata.kernelspec
    - All cells have valid cell_type
    - All code cells have outputs (can be empty list) and execution_count (can be None)
    - All cells have a source list of strings
    - At least one cell present
    """
```

### Internal: source list format

nbformat requires `source` as a list of strings with newlines:
```python
def _to_source_list(text: str) -> list[str]:
    lines = text.split("\n")
    if len(lines) == 1:
        return lines  # no newlines
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
```

### Test (`tests/test_notebook.py`)

- `test_parse_basic_alternation` — 4 cells alternating
- `test_parse_no_delimiters_fallback` — single markdown cell
- `test_parse_strips_python_fences` — ```python``` stripped
- `test_parse_strips_bare_fences` — ```...``` stripped
- `test_parse_skips_empty_cells`
- `test_parse_handles_lowercase_delimiters` — `===markdown===`
- `test_save_load_roundtrip`
- `test_validate_catches_missing_kernelspec`
- `test_validate_catches_invalid_cell_type`
- `test_validate_passes_for_well_formed_notebook`

---

## 6.4 Explorer Agent (`agents/explorer.py`)

### Purpose

Navigate a repo with an LLM-driven tool-call loop until it has enough context to produce
ExplorerFindings.

### Protocol

LLM responds with EXACTLY one JSON object per turn:

```json
{"action": "tool", "tool": "list_dir", "path": "some/dir"}
{"action": "tool", "tool": "read_file", "path": "some/file.py"}
{"action": "tool", "tool": "search_code", "keyword": "BaseClass"}
{"action": "done", "findings": { ... ExplorerFindings ... }}
```

### Public API

```python
class ExplorerAgent:
    def __init__(self, llm_client: LLMClient, repo_tool: RepoTool):
        self.llm = llm_client
        self.repo = repo_tool
        self.max_steps = llm_client.gen.get("max_explore_steps", 20)
        self.usage_log = []  # NEW: list of {"step": N, "input_tokens": X, "output_tokens": Y}
    
    def explore(self) -> dict:
        """
        Run the loop. Returns ExplorerFindings dict.
        
        Algorithm:
        1. Seed: list_dir result + priority files hint
        2. Loop up to max_steps:
           a. LLM response → parse JSON
           b. If invalid JSON: send correction message, count as a step
           c. If action == "done": return findings
           d. If action == "tool": execute, append result, continue
        3. If max_steps reached: send "finalize NOW" message, parse last response
        4. If still no valid findings: call _extract_findings_fallback
        
        Logs each step's token usage to self.usage_log.
        Logs every tool call to logger (DEBUG level) — see plan/06-edge-cases.md.
        """
    
    def _execute_tool(self, call: dict) -> str: ...
    def _parse_json(self, text: str) -> dict | None: ...
    def _extract_findings_fallback(self, text: str) -> dict: ...
```

### Test (`tests/test_explorer.py`)

Use `MockLLM` (defined in `tests/conftest.py`).

```python
def test_explorer_completes_when_llm_says_done(tmp_path):
    """LLM does one tool call then returns done."""
    fake_repo_dir = tmp_path / "fake_repo"
    fake_repo_dir.mkdir()
    (fake_repo_dir / "README.md").write_text("# Fake Repo")
    
    llm = MockLLM(responses=[
        '{"action": "tool", "tool": "read_file", "path": "README.md"}',
        '{"action": "done", "findings": {"title": "Fake", "one_liner": "test", '
            '"mental_model": "test", "domain": "other", "concepts": [], '
            '"dependency_order": [], "key_files": [], "skip_files": []}}',
    ])
    
    repo = RepoTool("https://fake", work_dir=str(tmp_path))
    repo.repo_path = str(fake_repo_dir)
    
    findings = ExplorerAgent(llm, repo).explore()
    assert findings["title"] == "Fake"
    assert len(llm.calls) == 2

def test_explorer_recovers_from_malformed_json():
    """If LLM returns garbage, agent sends correction and continues."""
    # ... LLM returns "not json" then valid done

def test_explorer_force_finishes_at_max_steps():
    """If max_steps hit, agent forces a finalize and accepts the result."""
    # ... LLM keeps doing tool calls until force-finish triggers

def test_explorer_calls_fallback_when_finalize_fails():
    """If even forced-finalize returns garbage, fallback is called."""
```

---

## 6.5 Curriculum Agent (`agents/curriculum.py`)

### Purpose

Single LLM call. Input: ExplorerFindings. Output: validated Curriculum.

### Public API

```python
class CurriculumAgent:
    def __init__(self, llm_client: LLMClient): ...
    
    def generate(self, findings: dict) -> dict:
        """
        Generate curriculum.
        
        1. Format prompt with findings JSON and max_notebooks from config
        2. Single LLM call
        3. Parse JSON (strip ```json fences if present)
        4. Validate (see _validate)
        5. Return Curriculum dict
        
        Raises ValueError if response is not parseable JSON.
        """
    
    def _validate(self, curriculum: dict, findings: dict) -> dict:
        """
        Fill missing fields with defaults. Lenient — never raises.
        
        - title defaults to "Understanding {findings.title}"
        - mental_model defaults to findings.mental_model
        - notebook IDs are forced to sequential "01", "02", ...
        - prerequisites/key_source_files/key_symbols/learning_objectives/
          exercise_description/visualization_idea all default to sensible empties
        - capstone gets defaults: title from project name, modules empty list, etc.
        """
    
    def _parse_json(self, text: str) -> dict | None:
        """Strip ```json fences, then json.loads. Return None on failure."""
```

### Test (`tests/test_curriculum.py`)

```python
def test_curriculum_generates_valid_structure():
    """LLM returns valid curriculum, agent passes through."""

def test_curriculum_strips_json_fences():
    """LLM wraps response in ```json ... ```, agent extracts."""

def test_curriculum_fills_missing_notebook_fields():
    """LLM omits learning_objectives, agent fills with []."""

def test_curriculum_renumbers_notebook_ids():
    """LLM uses ids ['1', '3', '5'], agent renumbers to ['01', '02', '03']."""

def test_curriculum_raises_on_invalid_json():
    """LLM returns 'not json at all', agent raises ValueError."""
```

---

## 6.6 Generator Agent (`agents/generator.py`)

### Purpose

Generate teaching + exercise + solution notebooks from a Curriculum. Resumable — skips
already-completed notebooks based on `state.progress`.

### Public API

```python
class GeneratorAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        repo_tool: RepoTool,
        output_dir: Path,
        state: dict,  # NEW: passed state for resumability
    ): ...
    
    def generate_all(self, user_curriculum_md: str, curriculum: dict) -> None:
        """
        Main pipeline.
        
        For each notebook:
          1. Skip if id in state.progress.notebooks_complete (resumability!)
          2. Extract source code (uses key_symbols if present, else falls back)
          3. Generate teaching .ipynb → save → update state
          4. Skip exercise if id in state.progress.exercises_complete
          5. Generate exercise .ipynb + solution → save → update state
        
        Then write README.md.
        
        Each save also calls _persist_state() to make progress durable.
        On any LLM exception: set state.step = "failed", state.last_error,
                              persist, re-raise.
        """
    
    def generate_one(self, nb_id: str, curriculum: dict) -> None:
        """
        NEW: Generate (or regenerate) a single notebook by ID. Used by:
        - --preview-notebook
        - --refine when target notebook is identified
        Always overwrites without backing up — caller's responsibility to back up.
        """
    
    def refine(self, instruction: str, curriculum: dict) -> None:
        """
        Refine specific content based on user instruction.
        
        1. _detect_target_notebook(instruction) → id or None
        2. If id found:
           - Confirm with user (interactive prompt) — see plan/05-cli.md
           - Backup existing files to .bak
           - Inject instruction into notebook spec's description
           - Regenerate that notebook only
        3. If "capstone" in instruction.lower():
           - Capstone refinement is handled by CapstoneAgent.refine() — call it
        4. Else: print warning, suggest being more specific
        """
    
    # ── Internal ──
    
    def _generate_teaching_notebook(self, nb_spec, curriculum, source_code, 
                                    prev_title, next_title, out_path): ...
    def _generate_exercise_notebook(self, nb_spec, curriculum, out_path, sol_path): ...
    def _generate_solution(self, exercise_response, nb_spec, curriculum, sol_path): ...
    def _detect_target_notebook(self, instruction, curriculum) -> int | None: ...
    def _persist_state(self) -> None:
        """Write self.state to .distill_state.json. Called after every step."""
```

### Source code extraction strategy

Replaces v1's "dump first 6000 chars" with smarter extraction:

```python
def _extract_relevant_source(self, nb_spec: dict) -> str:
    """
    Strategy:
    1. If nb_spec.key_symbols is non-empty:
        Use repo.read_files_for_concept(files, symbols=key_symbols, max_total_chars=...)
    2. Else:
        Use repo.read_files_for_concept(files, max_total_chars=...)
        (whole-file truncation as fallback)
    """
```

### Test (`tests/test_generator.py`)

- `test_generator_skips_completed_notebooks` — state.progress respected
- `test_generator_persists_after_each_notebook`
- `test_generator_uses_symbol_extraction_when_specified`
- `test_generator_falls_back_to_whole_files_when_no_symbols`
- `test_generator_generates_teaching_with_correct_metadata`
- `test_detect_target_notebook_by_id`
- `test_detect_target_notebook_by_title`
- `test_detect_target_notebook_by_concept`
- `test_detect_target_notebook_returns_none_for_ambiguous`

---

## 6.7 Capstone Agent (`agents/capstone.py`)

### Purpose

Generate `interfaces.py`, `test_capstone.py`, `implementation.py`, `README.md` for the
capstone project. **Crucially: validates the generated tests against a generated reference
implementation**, regenerating if tests fail on the reference.

This is the single most important quality mechanism in the project.

See `plan/04-validation.md` for the full validation loop spec.

### Public API

```python
class CapstoneAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: Path,
        state: dict,
    ): ...
    
    def generate_with_validation(self, curriculum: dict, max_attempts: int = 3) -> None:
        """
        Full capstone generation with validation loop.
        
        See plan/04-validation.md for algorithm details.
        """
    
    def refine(self, instruction: str, curriculum: dict) -> None:
        """Regenerate capstone with extra requirements injected."""
```

---

## 6.8 Cost Tracker (`llm/cost.py`)

### Purpose

Estimate cost before generation; track actual after.

### Public API

```python
# Per-million-token rates in USD. Update as providers change pricing.
RATES = {
    "claude-sonnet-4-20250514":     {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":       {"input": 15.0, "output": 75.0},
    "gpt-4o":                       {"input": 2.5,  "output": 10.0},
    "o3":                           {"input": 15.0, "output": 60.0},
    "gemini-2.5-pro-preview-05-06": {"input": 1.25, "output": 10.0},
    "gemini-2.0-flash":             {"input": 0.10, "output": 0.40},
}

def estimate_cost(
    findings: dict,
    curriculum: dict,
    model: str,
) -> dict:
    """
    Pre-generation estimate. Returns:
    {
      "input_tokens": int,
      "output_tokens": int,
      "total_usd": float,
      "breakdown": {
        "notebooks": float,
        "exercises": float,
        "capstone": float,
      }
    }
    
    Heuristic: per notebook ~3000 input + 4000 output tokens.
    Per exercise ~2000 input + 3500 output. Capstone ~4 LLM calls × 3000/3000.
    """

def compute_actual_cost(state: dict) -> float:
    """From state.cost.* fields, compute total USD."""

def format_cost_report(state: dict) -> str:
    """Human-readable cost summary for end-of-run print."""
```

### Test (`tests/test_cost.py`)

- `test_estimate_cost_scales_with_notebook_count`
- `test_compute_actual_cost_handles_unknown_model` (returns 0 with warning)
- `test_format_cost_report_shows_breakdown`
