# 06 — Edge Cases, Logging, Error Handling

## Logging (`infra/logging.py`)

Structured logging to both stdout and file. Used throughout the project.

```python
import logging
from pathlib import Path

def setup_logging(level: str = "INFO", output_dir: Path | None = None) -> None:
    """
    Configure root logger.
    - Console handler: level INFO+ (clean output for users)
    - File handler:    level DEBUG+ (everything, for debugging)
    
    File location: {output_dir}/distill.log if output_dir given,
                   else ./distill.log
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Clear any existing handlers (idempotent)
    root.handlers.clear()
    
    # Console: simple format, INFO+
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level))
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    
    # File: detailed format, DEBUG+
    log_path = (output_dir or Path(".")) / "distill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_path)
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(file_h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

What gets logged:

| Module | DEBUG | INFO | WARNING | ERROR |
|---|---|---|---|---|
| explorer | every tool call w/ args | step count, completion | malformed JSON | fallback called |
| curriculum | full prompt, response | notebook count | missing fields | parse failed |
| generator | prompts | each notebook complete | source extraction fallback | LLM error |
| capstone | prompts, full validation output | each step, validation result | validation needed retry | validation failed all attempts |
| llm.client | request/response | — | retry triggered | retries exhausted |

Always log to file at DEBUG. User-facing console output should be clean (INFO).

---

## Error Handling

### Categories

1. **User errors** (bad input) — print clear message, exit 1, no traceback
2. **Configuration errors** (missing API key, missing config) — same as #1
3. **Transient errors** (API rate limits, network timeouts) — retry with backoff
4. **Generation errors** (LLM returned malformed JSON, etc.) — log, attempt recovery
5. **Bugs** (unexpected exceptions) — print traceback, exit 1, log full stack

### Pattern

```python
# In distill.py main()
try:
    if args.repo: cmd_explore(args)
    # ...
except KeyboardInterrupt:
    print("\nInterrupted. Run --resume to continue.")
    sys.exit(130)
except (EnvironmentError, FileNotFoundError, ValueError) as e:
    # User-facing errors (categories 1-2)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Bugs (category 5)
    logger = logging.getLogger("distill")
    logger.exception("Unexpected error")
    print(f"\nUnexpected error: {e}", file=sys.stderr)
    print(f"See distill.log for details.", file=sys.stderr)
    sys.exit(1)
```

### Specific cases

#### Missing API key
```
Error: ANTHROPIC_API_KEY environment variable not set.
Set it with: export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Cloned repo too large to explore
```
Warning: Repository has 12,543 Python files. Explorer may struggle.
Consider:
  - Use a model with larger context (gemini-2.5-pro-preview-05-06 has 1M)
  - Set step_models.explore: gemini-2.5-pro-preview-05-06 in config.yaml
```

#### LLM returns non-JSON in explorer
- Log full response at DEBUG
- Send correction message to LLM
- Counts as a step toward max_steps
- If still failing at end: call `_extract_findings_fallback` (asks LLM to reformat)

#### LLM returns invalid Python in capstone
- Caught by validation loop's `pytest` subprocess
- Captured stderr is fed back into the regen prompt
- After max_attempts: ship with WARNING in capstone/README.md

---

## Known Edge Cases

### Large repos (vLLM, llama.cpp, kubernetes)

Symptom: Explorer hits max_steps without enough context.

Mitigations (already in spec):
- `get_priority_files()` returns curated list to seed exploration
- Force-finalize message if max_steps reached
- `_extract_findings_fallback` as last resort

User remedy:
```yaml
# config.yaml
step_models:
  explore: gemini-2.5-pro-preview-05-06   # 1M context
generation:
  max_explore_steps: 30                    # bump if needed
```

### Repos without examples/

Symptom: Explorer has nothing concrete to anchor concepts to.

Behavior: Explorer should still work via README + core source files. The priority list
falls back gracefully when examples/ doesn't exist.

### Monorepos (a single repo containing many sub-projects)

Symptom: Explorer wanders into unrelated subdirectories.

Mitigations:
- The CURRICULUM.md review step lets the user notice this
- User can edit `key_source_files` in YAML blocks to constrain regeneration
- Future: add `--subpath` CLI arg to scope exploration (NOT in v1)

### Repos with non-Python languages (llama.cpp = C++)

Symptom: Explorer's `search_code` defaults to `*.py` and finds nothing.

Behavior: Explorer can pass `file_pattern="*.cpp"` etc. The system prompt should
mention this option. For v1: make the strategy section in EXPLORER_SYSTEM mention
the file_pattern parameter.

### Repos requiring credentials (private)

Out of scope for v1. `git clone` will fail with clear error. Document in README.

### LLM response truncation

Symptom: notebook ends mid-cell because max_tokens reached.

Mitigation: 
- Default max_tokens are generous (6000 for notebooks, 5000 for exercises)
- `validate_notebook()` catches notebooks with bad structure
- For v1: log warning if response ends suspiciously (e.g., not with `===MARKDOWN===` and no closing markdown).

### CURRICULUM.md user edits break parsing

Symptom: User edits YAML block, breaks indentation.

Mitigation:
- `parse_curriculum_md` uses `try/except yaml.YAMLError` per block
- On parse error: log warning, fall back to original curriculum's data for that block
- Print clear message: `"Warning: notebook 03 YAML invalid, using original spec"`

### Network failure mid-generation

Symptom: 5 of 8 notebooks generated, then connection drops.

Mitigation:
- `_persist_state()` called after each notebook
- `progress.notebooks_complete` lists what's done
- `--resume` (or re-running `--continue`) skips already-done notebooks

### Cost overruns

Mitigation:
- `--estimate-cost` shows estimate before running
- `--continue` shows estimate and asks for confirmation (unless `--no-confirm`)
- Cost tracked per step, total shown at end

### Concurrent runs in same output directory

Behavior: `_save_state()` overwrites without locking. Last writer wins.

Mitigation: Document "don't run two distill processes in the same output dir at once."
File locking is overkill for v1.

### Unicode in repo content

Symptom: `read_file` fails on file with non-UTF-8 encoding.

Mitigation: All `read_text()` calls use `encoding="utf-8", errors="replace"`.

### Notebook generation produces invalid Python in code cells

Symptom: Generated notebook has syntax errors.

Mitigation for v1: 
- `validate_notebook()` checks JSON structure but NOT Python syntax
- We don't execute notebooks (would require kernel + dependencies)
- Document as a known limitation: "Generated code is best-effort. Review before running."

For v2 (out of scope): use `ast.parse()` to validate code cells.

---

## Testing strategy summary

```
tests/
├── conftest.py              # Shared fixtures: MockLLM, sample_repo, etc.
├── fixtures/
│   ├── sample_repo/         # Tiny repo for repo tool tests
│   └── responses/           # Canned LLM responses for agent tests
├── test_llm_client.py       # Provider routing, retries
├── test_repo.py             # File access, AST extraction
├── test_notebook.py         # Pure-function tests
├── test_curriculum_md.py    # Parser robustness with edits
├── test_explorer.py         # Tool-call loop with MockLLM
├── test_curriculum.py       # JSON parsing, defaults
├── test_generator.py        # Resumability, source extraction
├── test_capstone_validation.py  # The validation loop ⭐ critical
├── test_cost.py
├── test_cli.py
└── test_integration.py      # End-to-end with MockLLM
```

Run all: `pytest -v`
Just unit tests: `pytest -v -m "not integration"`
Just integration: `pytest -v -m integration`

---

## Definition of Done — by phase

### Phase 1 (Foundations)
- [ ] All files in `llm/`, `tools/` exist and importable
- [ ] `pytest tests/test_llm_client.py tests/test_repo.py tests/test_notebook.py -v` passes
- [ ] Coverage ≥ 80% on these modules

### Phase 2 (Agents)
- [ ] `agents/explorer.py`, `agents/curriculum.py` importable
- [ ] All tests in `test_explorer.py` and `test_curriculum.py` pass
- [ ] Smoke test on minGPT (with API key) produces valid findings + curriculum

### Phase 3 (Generator)
- [ ] `agents/generator.py` importable
- [ ] All tests in `test_generator.py` pass
- [ ] Generates valid `.ipynb` (passes `validate_notebook()`)
- [ ] Resumability test: kill mid-run, resume, no duplicates

### Phase 4 (Capstone Validation)
- [ ] `agents/capstone.py` importable
- [ ] `test_capstone_validation.py` passes — including the deliberately-bad-test case
- [ ] Manual test: generate capstone, verify reference impl is deleted, verify all tests pass against a hand-written student impl

### Phase 5 (CLI)
- [ ] `python distill.py --help` shows all commands
- [ ] Each command's happy path works (with MockLLM in tests, real API in manual)
- [ ] State persisted correctly across commands
- [ ] `--status` shows accurate progress

### Phase 6 (Polish)
- [ ] Cost tracking accurate within ±20% on a real run
- [ ] Logging to file works; DEBUG level captures everything
- [ ] All previous tests still pass
- [ ] README.md updated with full usage instructions
