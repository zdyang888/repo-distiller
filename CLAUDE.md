# CLAUDE.md — repo-distiller Implementation Guide

> **You are Claude Code.** Read this file first. It tells you what we're building, how the
> documentation is organized, and how to work through implementation. Other docs in `plan/`
> contain the detailed specifications.

---

## What We're Building

**repo-distiller** transforms complex GitHub repositories (LangChain, vLLM, nanoGPT) into
structured educational curricula: teaching notebooks, exercise notebooks, and a capstone
project with validated tests. Input: a GitHub URL. Output: a self-contained course directory.

**Why it matters:** beginners can't learn from production codebases — too many abstractions,
too much engineering complexity. This tool reconstructs the same content as a learnable
sequence.

---

## Documentation Map

Read in this order based on what you're working on:

| File | When to read |
|------|--------------|
| `plan/00-overview.md` | Always first — gives architecture and principles |
| `plan/01-data-models.md` | Working on any module that produces/consumes structured data |
| `plan/02-modules.md` | Implementing a specific module (Section 6.x) |
| `plan/03-prompts.md` | Implementing or modifying LLM prompts |
| `plan/04-validation.md` | Implementing the capstone validation loop (Phase 4) |
| `plan/05-cli.md` | Implementing the CLI or user-facing flows |
| `plan/06-edge-cases.md` | When you hit something the spec doesn't cover |

**Do not read all docs at once.** Each implementation phase below tells you which docs to read.

---

## Implementation Phases (Recommended Session Boundaries)

Implement in 6 phases. Each phase is one Claude Code session (~1-2 hours of work). At the
end of each phase, all tests in that phase must pass before moving on.

### Phase 1 — Foundations (estimated: 1 session)

**Read:** `plan/00-overview.md`, `plan/01-data-models.md`, `plan/02-modules.md` (sections 6.1, 6.2, 6.3)

**Build:**
- Project skeleton (directories, `requirements.txt`, `.gitignore`, basic `README.md`)
- `llm/client.py` — unified LLM client (Claude / OpenAI / Gemini)
- `tools/repo.py` — git clone + file access
- `tools/notebook.py` — .ipynb JSON construction
- `tests/test_notebook.py` — pure-function unit tests
- `tests/test_repo.py` — uses fixture directory, no clone needed

**Verify:** `pytest tests/test_notebook.py tests/test_repo.py -v` passes.

**Do not build:** any agent, the CLI, prompts, anything else. Stop after foundations work.

---

### Phase 2 — Agents: Explorer + Curriculum (estimated: 1 session)

**Read:** `plan/00-overview.md` (refresh), `plan/02-modules.md` (sections 6.4, 6.5),
         `plan/03-prompts.md` (Explorer + Curriculum prompts)

**Build:**
- `agents/explorer.py` — structured JSON tool-call loop
- `agents/curriculum.py` — single-call curriculum generation
- `tests/test_explorer.py` — uses a `MockLLM` fixture; no real API calls
- `tests/test_curriculum.py` — same approach

**Verify:** `pytest tests/test_explorer.py tests/test_curriculum.py -v` passes.

**Manual smoke test (optional, requires API key):**
```bash
python -c "
from llm.client import LLMClient
from tools.repo import RepoTool
from agents.explorer import ExplorerAgent
import tempfile, subprocess, os
# Clone a small repo
tmp = tempfile.mkdtemp()
subprocess.run(['git', 'clone', '--depth=1',
                'https://github.com/karpathy/minGPT', f'{tmp}/minGPT'])
rt = RepoTool('https://github.com/karpathy/minGPT', work_dir=tmp)
rt.repo_path = f'{tmp}/minGPT'
findings = ExplorerAgent(LLMClient(), rt).explore()
print(findings)
"
```

---

### Phase 3 — Generator (estimated: 1-2 sessions)

**Read:** `plan/02-modules.md` (section 6.6), `plan/03-prompts.md` (Notebook + Exercise + Solution prompts)

**Build:**
- `agents/generator.py` — but ONLY the notebook + exercise generation (NOT capstone yet)
- `tests/test_generator.py` — verify notebook structure parsing, slug generation, etc.

**Verify:** `pytest tests/test_generator.py -v` passes.

**Why split capstone off:** Capstone requires the validation loop, which is a substantial
sub-system. Doing it separately keeps each session focused.

---

### Phase 4 — Capstone Generation + Validation Loop (estimated: 1 session)

**Read:** `plan/04-validation.md` (entire file — this is the most important quality
mechanism), `plan/03-prompts.md` (Capstone prompts), `plan/02-modules.md` (section 6.7).

**Build:**
- `agents/capstone.py` — separate from generator.py because it has its own validation loop
- `validation/runner.py` — runs pytest in a subprocess against a generated reference impl
- `tests/test_capstone.py` — verify the validation loop catches bad tests

**Verify:** `pytest tests/test_capstone.py -v` passes. Optionally run end-to-end: generate
capstone → reference impl → run tests → verify they pass.

**This is the most important phase for product quality.** Take your time.

---

### Phase 5 — CLI + Integration (estimated: 1 session)

**Read:** `plan/05-cli.md`, `plan/01-data-models.md` (refresh on DistillState).

**Build:**
- `distill.py` — the CLI (one file, but use helper functions liberally)
- `tests/test_cli.py` — argparse and command routing tests
- Wire everything together end-to-end

**Verify:** Run `python distill.py --help` and check output. Then run a real end-to-end
test on a small repo (e.g., karpathy/minGPT) if you have an API key.

---

### Phase 6 — Polish: Logging + Cost Tracking (estimated: 1 session)

**Read:** `plan/02-modules.md` (section 6.8 Cost), `plan/06-edge-cases.md`.

**Build:**
- `llm/cost.py` — token tracking and cost estimation
- `infra/logging.py` — structured logging to file
- Update `distill.py` to wire in `--estimate-cost`, `--status`, `--log-level`

**Verify:** All previous tests still pass. Cost estimate within ±20% on a real run.

---

## Working Style Within a Session

When you start a session:

1. **Read CLAUDE.md (this file).**
2. **Read the docs listed for the current phase.** Don't read other docs.
3. **Check what's already built** (`git log`, `ls`, `pytest --collect-only`).
4. **State the plan.** Before writing code, summarize what you'll build in this session.
5. **Build incrementally.** One module at a time. Run tests after each.
6. **Commit at logical checkpoints.** Each working module = one commit.
7. **End with a status report.** What was built, what tests pass, what's left.

---

## Critical Quality Rules

These apply to every phase:

1. **Test before moving on.** Each module has unit tests in this spec. Run them. They must
   pass before you commit.
2. **No silent failures.** If an LLM returns malformed output, log it and either retry or
   fail loudly. Never paper over with a placeholder.
3. **No real API calls in tests.** Use the `MockLLM` pattern (see `plan/02-modules.md`).
4. **Validate your own work.** After implementing the capstone validator, actually run it
   against a generated example and check it catches a deliberately-bad test.
5. **Type hints everywhere.** Use Python 3.10+ syntax: `dict | None`, `list[str]`.
6. **Docstrings on all public functions.** One-line summary minimum, args/returns when
   non-obvious.

---

## When You're Stuck

- **Spec is ambiguous?** Ask the user before guessing. Better to clarify than to backtrack.
- **Spec contradicts your judgment?** Flag it explicitly. The spec is v1, not infallible.
- **Phase scope feels too big?** Split it. End the session with a partial commit and
  document what's left.
- **Tests are hard to write?** That's a sign the design is off. Refactor before adding
  more code.

---

## Definition of Done (overall project)

- [ ] All 6 phases implemented
- [ ] All tests pass: `pytest tests/ -v`
- [ ] End-to-end smoke test on karpathy/minGPT produces a valid course
- [ ] Capstone validation loop verified to catch a deliberately-bad test
- [ ] README.md explains installation, usage, and configuration
- [ ] Cost estimation accurate within ±20% on a real run
