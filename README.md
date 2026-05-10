# repo-distiller — Implementation Plan Package

This directory contains everything needed to implement repo-distiller from scratch using
Claude Code.

## Files in this package

```
.
├── README.md                    ← you are here
├── CLAUDE.md                    ← MAIN ENTRY POINT — read first
└── plan/
    ├── 00-overview.md           ← Architecture & principles
    ├── 01-data-models.md        ← All schemas (ExplorerFindings, Curriculum, etc.)
    ├── 02-modules.md            ← Detailed spec for each Python module
    ├── 03-prompts.md            ← All LLM prompt templates (verbatim)
    ├── 04-validation.md         ← Capstone validation loop ⭐ critical
    ├── 05-cli.md                ← CLI commands & flow
    └── 06-edge-cases.md         ← Logging, error handling, known issues
```

## How to use this with Claude Code

### Quickstart

```bash
# 1. Set up a new project directory
mkdir repo-distiller && cd repo-distiller

# 2. Drop this entire planning package into the project root
cp -r /path/to/this/package/* .

# 3. Start Claude Code
claude

# 4. Tell Claude Code to begin
> Read CLAUDE.md and start with Phase 1.
```

Claude Code will then read `CLAUDE.md`, which tells it which `plan/*.md` files to consult
for the current phase.

### Recommended session structure

The implementation is split into **6 phases**, each ~1-2 hours of work. Each phase should
be its own Claude Code session for best results:

| Session | Phase | What gets built |
|---|---|---|
| 1 | Phase 1 | LLM client, repo tool, notebook tool + tests |
| 2 | Phase 2 | Explorer + Curriculum agents + tests |
| 3 | Phase 3 | Generator agent + tests |
| 4 | Phase 4 | Capstone with validation loop ⭐ |
| 5 | Phase 5 | CLI (distill.py) wiring everything |
| 6 | Phase 6 | Cost tracking, logging, polish |

**Why split sessions:** keeps context focused, each session has a clear deliverable, and
Claude Code can verify (`pytest`) before committing.

### How to start each session

For sessions 2-6, start with:

```
> Read CLAUDE.md and check git status. We are starting Phase N.
> Confirm what's already built, then proceed with Phase N as specified.
```

This re-grounds Claude Code in the plan and prevents it from re-doing or skipping work.

---

## What's intentionally NOT in this package

- **No reference implementation.** Earlier prototype code was a sketch; the spec
  supersedes it. Claude Code should implement to spec, not copy old code.
- **No screenshots / visual designs.** All output formats are described textually.
- **No deployment / packaging.** This is a developer tool run from source.

---

## After implementation

Once all 6 phases are done, the project will have this structure:

```
repo-distiller/
├── distill.py                  # CLI entry
├── config.yaml
├── requirements.txt
├── README.md
├── llm/
├── tools/
├── agents/
├── prompts/
├── export/
├── infra/
├── tests/
└── output/                     # Generated courses go here
```

Test it on a real repo:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python distill.py --repo https://github.com/karpathy/minGPT
# Review output/minGPT_course/CURRICULUM.md
python distill.py --continue --output output/minGPT_course
```

---

## Plan version

**v3** — final scope. PDFs are NOT a product feature (notebooks are used in their native
.ipynb form on a laptop). Key features vs. v1 prototype:
- ⭐ Capstone validation loop (generates reference impl, runs tests, regenerates if fail)
- Resumable generation (skip already-completed notebooks)
- AST-based source code extraction (cleaner than whole-file truncation)
- Structured CURRICULUM.md format (YAML blocks, robust parsing)
- Cost estimation + tracking
- `--preview` mode for single-notebook quality check
- `--status` and `--resume` commands
- File logging at DEBUG level
- Split into multiple plan files for focused sessions
