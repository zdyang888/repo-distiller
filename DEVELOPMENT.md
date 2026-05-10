# Development Guide — repo-distiller

This project was built via **vibe coding**: a human wrote the full specification in advance,
then Claude Code implemented it phase by phase in a series of focused sessions. The spec
lives in `plan/` and serves as the ground truth for every implementation decision.

---

## How this was built

### The vibe coding approach

1. The complete architecture, data models, module specs, prompts, and edge-case handling
   were written up-front as structured Markdown docs in `plan/`.
2. `CLAUDE.md` was written as the agent's entry point — it tells Claude Code what to read,
   what to build, and in what order.
3. Each session: Claude Code reads the relevant plan docs, states its plan, implements
   incrementally, runs tests, and commits at logical checkpoints.
4. The human's job: review diffs, run tests, catch spec ambiguities, and refine the spec
   for the next session.

### Why write specs first

- Forces clarity before any code exists
- Makes sessions fast — no back-and-forth on "what should this do?"
- Makes output verifiable — tests come directly from the spec
- Creates a record of every design decision and tradeoff

---

## Plan document map

Read these when working on each phase:

| File | Contents |
|---|---|
| `plan/00-overview.md` | Architecture, data flow, design principles, tech stack |
| `plan/01-data-models.md` | All schemas: ExplorerFindings, Curriculum, DistillState, CURRICULUM.md format |
| `plan/02-modules.md` | Module-by-module specs: public API, implementation notes, test list |
| `plan/03-prompts.md` | All LLM prompt templates (verbatim, copy-paste ready) |
| `plan/04-validation.md` | Capstone validation loop — the most important quality mechanism |
| `plan/05-cli.md` | CLI commands, argument parsing, user-facing flows |
| `plan/06-edge-cases.md` | Logging, error handling, known failure modes |

---

## Implementation phases

The project is implemented in 6 phases. Each phase = one Claude Code session.

| Phase | What gets built | Tests to pass |
|---|---|---|
| 1 | LLM client, RepoTool, NotebookTool | `test_notebook.py`, `test_repo.py` |
| 2 | ExplorerAgent, CurriculumAgent | `test_explorer.py`, `test_curriculum.py` |
| 3 | GeneratorAgent (notebooks + exercises) | `test_generator.py` |
| 4 | CapstoneAgent + validation loop | `test_capstone.py` |
| 5 | CLI (`distill.py`) + end-to-end wiring | `test_cli.py` |
| 6 | Cost tracking, structured logging, polish | all tests |

**Rule:** all tests for a phase must pass before moving to the next.

---

## Starting a new session

```
> Read CLAUDE.md and check git status. We are starting Phase N.
> Confirm what's already built, then proceed with Phase N as specified.
```

This re-grounds Claude Code in the plan and prevents re-doing or skipping work.

---

## Critical quality rules (from CLAUDE.md)

- No real API calls in tests — use `MockLLM` (defined in `tests/conftest.py`)
- No silent failures — if an LLM returns malformed output, log and retry or fail loudly
- Type hints everywhere (`X | Y` syntax, Python 3.10+)
- Docstrings on all public functions
- Commit at each working module checkpoint

---

## Project structure (target, after all phases)

```
repo-distiller/
├── distill.py              # CLI entry point
├── config.yaml             # Model and generation config
├── requirements.txt
├── CLAUDE.md               # Agent entry point (instructions for Claude Code)
├── README.md               # User-facing docs
├── DEVELOPMENT.md          # This file
├── plan/                   # Full specification docs
│   ├── 00-overview.md
│   ├── 01-data-models.md
│   ├── 02-modules.md
│   ├── 03-prompts.md
│   ├── 04-validation.md
│   ├── 05-cli.md
│   └── 06-edge-cases.md
├── llm/
│   ├── client.py           # Unified LLM client (Claude/OpenAI/Gemini)
│   └── cost.py             # Token tracking and cost estimation
├── tools/
│   ├── repo.py             # Git clone + controlled file access
│   ├── notebook.py         # nbformat 4.5 construction and parsing
│   └── curriculum_md.py    # CURRICULUM.md serializer/parser
├── agents/
│   ├── explorer.py         # Tool-call loop for repo navigation
│   ├── curriculum.py       # Single-call curriculum generation
│   ├── generator.py        # Teaching + exercise notebook generation
│   └── capstone.py         # Capstone with validation loop
├── validation/
│   └── runner.py           # Runs pytest against generated reference impl
├── infra/
│   └── logging.py          # Structured file logging
└── tests/
    ├── conftest.py          # MockLLM + shared fixtures
    ├── fixtures/
    │   └── sample_repo/     # Fixture repo for RepoTool tests
    ├── test_notebook.py
    ├── test_repo.py
    ├── test_explorer.py
    ├── test_curriculum.py
    ├── test_generator.py
    ├── test_capstone.py
    └── test_cli.py
```
