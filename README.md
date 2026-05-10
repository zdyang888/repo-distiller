# repo-distiller

Transform any GitHub repository into a structured, learnable course — teaching notebooks, exercises, and a validated capstone project.

**Problem:** Production codebases (LangChain, vLLM, nanoGPT) are too complex for beginners. Too many abstractions, too much infra, no learning path.

**Solution:** repo-distiller reads the repo, proposes a curriculum you review and edit, then generates:

```
output/<name>_course/
├── CURRICULUM.md         # editable course plan (you review before generation)
├── README.md             # learning path with notebook table
├── notebooks/            # teaching notebooks: concept + implementation
├── exercises/            # exercises with TODO stubs and auto-graded tests
└── capstone/             # mini-reimplementation project
    ├── interfaces.py     # ABC contracts for students to implement
    ├── implementation.py # student starter file
    ├── test_capstone.py  # pytest tests (validated against a reference impl)
    └── README.md         # project guide
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key (Claude by default)
export ANTHROPIC_API_KEY="sk-ant-..."

# Step 1: Explore the repo and generate a draft curriculum
python distill.py --repo https://github.com/karpathy/nanoGPT

# Step 2: Open and review (edit if you like)
open output/nanoGPT_course/CURRICULUM.md

# Step 3: Generate all notebooks and the capstone
python distill.py --continue --output output/nanoGPT_course
```

---

## Workflow

```
1. EXPLORE    python distill.py --repo <url>
                  └─► produces CURRICULUM.md for you to review

2. REVIEW     open CURRICULUM.md — toggle notebooks, edit descriptions,
              add requirements. Lines outside notebook blocks are ignored.

3. GENERATE   python distill.py --continue
                  └─► produces notebooks/, exercises/, capstone/

4. REFINE     python distill.py --refine "make the attention notebook deeper"
                  └─► regenerates targeted content (optional, repeatable)
```

---

## Configuration

Edit `config.yaml` to choose your model and generation settings:

```yaml
default_model: claude-sonnet-4-20250514

# Per-step overrides — e.g. use Gemini for large repos (1M context)
step_models:
  explore: gemini-2.0-flash
  notebooks: claude-sonnet-4-20250514

generation:
  max_explore_steps: 20
  max_notebooks: 8
```

Supported models:

| Provider | Model names | Env var |
|---|---|---|
| Anthropic | `claude-*` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-*`, `o1`, `o3`, `o4` | `OPENAI_API_KEY` |
| Google | `gemini-*` | `GOOGLE_API_KEY` |

---

## Other commands

```bash
# Check progress on an in-progress course
python distill.py --status --output output/nanoGPT_course

# Resume a failed or partial generation
python distill.py --resume --output output/nanoGPT_course

# Preview a single notebook before generating everything
python distill.py --preview-notebook 03 --output output/nanoGPT_course

# Estimate cost before committing to full generation
python distill.py --estimate-cost --output output/nanoGPT_course
```

---

## What repos work best

Designed for AI/ML codebases:

- LLM frameworks (LangChain, LlamaIndex)
- Agent systems
- ML infra (vLLM, tgi)
- Inference engines
- Small, readable ML implementations (nanoGPT, minGPT, micrograd)

Other domains may work but prompts are tuned for AI/ML.

---

## How it works

1. **Explorer agent** navigates the repo with tool calls (`list_dir`, `read_file`, `search_code`) until it has a structural understanding, then produces `ExplorerFindings`.
2. **Curriculum agent** turns findings into a prioritized notebook plan (single LLM call).
3. **Generator agent** uses AST-based symbol extraction to pull relevant source, then generates teaching + exercise notebooks per concept.
4. **Capstone agent** generates `interfaces.py` + `test_capstone.py` + a hidden reference implementation, runs `pytest` against the reference, and regenerates if tests fail. This validation loop is the main quality guarantee.

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- API key for at least one provider (see Configuration above)
