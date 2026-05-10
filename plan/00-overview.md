# 00 — Overview & Architecture

## What this builds

A Python tool: `python distill.py --repo <github-url>` produces a structured educational
course in `output/<repo>_course/`:

```
output/<name>_course/
├── CURRICULUM.md         # editable course plan (user reviews before generation)
├── README.md             # learning path with linked notebook table
├── notebooks/            # teaching notebooks (concept + implementation)
├── exercises/            # 吴恩达-style exercises with auto-graded tests
└── capstone/             # mini-reimplementation project
    ├── interfaces.py     # ABC contracts students implement against
    ├── implementation.py # student starter file
    ├── test_capstone.py  # pytest tests (validated against reference impl!)
    └── README.md         # project guide with architecture diagram
```

## Three-step user workflow

```
1. EXPLORE   python distill.py --repo URL
                ↓ produces CURRICULUM.md
2. REVIEW    user opens CURRICULUM.md, optionally annotates
                ↓
3. GENERATE  python distill.py --continue
                ↓ produces all notebooks + capstone
4. REFINE    python distill.py --refine "instruction"   (optional, repeatable)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          distill.py                             │
│                       (CLI entry point)                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐   ┌────────────┐    ┌────────────┐
       │  Explorer  │   │ Curriculum │    │ Generator  │
       │   Agent    │   │   Agent    │    │   Agent    │
       └─────┬──────┘   └──────┬─────┘    └──────┬─────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ┌──────────────────────────────────────────────┐
       │                LLMClient                     │
       │  (Claude / OpenAI / Gemini, unified API)     │
       └──────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐   ┌────────────┐    ┌────────────┐
       │  RepoTool  │   │ Notebook   │    │ Validation │
       │ (git+files)│   │   Tool     │    │   Runner   │
       └────────────┘   └────────────┘    └────────────┘
```

## Data flow

```
GitHub URL
    │
    ├──► RepoTool.clone()
    │
    ├──► ExplorerAgent.explore()
    │       │
    │       │  Loop: tool calls (list_dir, read_file, search_code)
    │       │  Until: {"action": "done", "findings": {...}}
    │       │
    │       └──► ExplorerFindings (structured)
    │
    ├──► CurriculumAgent.generate(findings)
    │       │
    │       │  Single LLM call
    │       │
    │       └──► Curriculum (structured)
    │              │
    │              └──► serialized to CURRICULUM.md
    │                   sidecar JSON saved as .distill_state.json
    │
    [USER REVIEWS CURRICULUM.md, OPTIONALLY EDITS]
    │
    ├──► GeneratorAgent.generate_all(curriculum)
    │       │
    │       │  For each notebook:
    │       │    - extract relevant source via AST
    │       │    - generate teaching .ipynb
    │       │    - generate exercise .ipynb
    │       │    - generate solution .ipynb
    │       │
    │       └──► All notebooks written
    │
    └──► CapstoneAgent.generate_with_validation(curriculum)
            │
            │  1. Generate interfaces.py
            │  2. Generate test_capstone.py
            │  3. Generate reference_impl.py (hidden)
            │  4. Run pytest on reference_impl.py
            │  5. If pass: ship it.
            │     If fail: regenerate tests with failure context, retry (max 2x)
            │     If still fail: ship with WARNING in capstone/README.md
            │
            └──► capstone/ directory complete
```

## Design Principles

1. **Human-in-the-loop, not autonomous.** AI proposes the curriculum, human confirms or
   edits, then AI generates. We do not trust AI to make pedagogical decisions alone.

2. **Vertical depth over breadth.** 5–8 excellent notebooks beat 20 mediocre ones. The
   spec caps `max_notebooks` at 8.

3. **Validate, don't hope.** The capstone tests are validated by running them against a
   generated reference implementation. If tests fail on the reference, regenerate.

4. **Behavior over implementation.** Capstone tests check input/output contracts, not
   internal data structures. Students should be free to implement however they want.

5. **Resumable everywhere.** If generation fails halfway, resuming should skip already-done
   work. State is persisted aggressively.

6. **Predictable cost.** Estimate tokens before generation, track actuals, expose to user.
   No surprises at the API bill.

7. **Model-agnostic.** Works with Claude, GPT-4o, or Gemini. The most appropriate model
   per step can be configured (e.g., Gemini for huge repos due to 1M context).

## Tech stack

- **Python 3.10+** (uses `X | Y` type union syntax)
- **anthropic** SDK (Claude)
- **openai** SDK (OpenAI + Gemini via OpenAI-compatible endpoint)
- **pyyaml** (config)
- **pytest** (testing + capstone validation)
- **subprocess** for git (no gitpython dependency needed)

No frameworks (no LangChain, no FastAPI, no Click). Standard library + above only.

## What this is NOT

- Not a code-explanation tool. We don't annotate every line.
- Not an autonomous agent. There's a mandatory human review step.
- Not a code generator for production. Output is for learning, not deployment.
- Not for non-AI repos. Prompts are tuned for AI/ML domain (LLM frameworks, ML infra,
  inference engines, agent systems, quant frameworks). Other domains may work but are
  unsupported.

## Success Criterion

> A beginner who cannot read the original repository can understand and modify the
> distilled version within a few hours.

This is the only thing that matters. Everything else (architecture, prompts, validation)
exists to serve this goal.
