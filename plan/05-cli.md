# 05 — CLI Interface (`distill.py`)

Single file. All commands routed via argparse. Keep helper functions private (`_name`).

## Command spec

```
usage: distill.py [COMMAND] [options]

Commands (mutually exclusive):
  --repo URL                Step 1: explore repo and generate CURRICULUM.md
  --continue                Step 2: generate notebooks from CURRICULUM.md
  --refine "INSTRUCTION"    Step 3: refine specific content
  --preview NN              Generate ONE notebook (id NN) for quality check
  --resume                  Resume an interrupted --continue from saved state
  --estimate-cost           Show cost estimate without running
  --status                  Show progress of an output directory

Options:
  --output DIR              Output directory (auto-detected if omitted)
  --model MODEL             Override default_model for this run
  --config FILE             Config file path (default: config.yaml)
  --log-level LEVEL         DEBUG | INFO | WARNING (default: INFO)
  --no-confirm              Skip interactive confirmations (for scripting)
```

## Print conventions

```
# Step header (60-char divider line)
repo-distiller  |  exploring https://github.com/karpathy/nanoGPT
────────────────────────────────────────────────────────────

# Sub-steps (2-space indent)
  [1/3] Cloning repository...
  [2/3] Agent exploring repository...
        Found 6 concepts: Tokenization, Embeddings, Self-Attention, ...
  [3/3] Generating curriculum outline...
        Planned 6 notebooks
        Estimated cost: $0.42

# Completion / next steps block
────────────────────────────────────────────────────────────
Curriculum written to:  output/nanogpt_course/CURRICULUM.md

Next steps:
  1. Open and review CURRICULUM.md
  2. Run: python distill.py --continue --output output/nanogpt_course
```

Use plain ASCII for terminal output. No emoji in CLI text (some terminals don't render).
Emoji is fine in generated content (notebooks, README.md).

---

## Command implementations

### `--repo URL`

```python
def cmd_explore(args):
    """Step 1: clone, explore, write CURRICULUM.md."""
    setup_logging(args.log_level)
    
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model
    
    output_dir = Path(args.output or _default_output_dir(args.repo))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nrepo-distiller  |  exploring {args.repo}")
    print("─" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # 1. Clone
    print("[1/3] Cloning repository...")
    repo_tool = RepoTool(args.repo, work_dir=str(output_dir / "_repo_cache"))
    repo_path = repo_tool.clone()
    
    # 2. Explore
    print("[2/3] Agent exploring repository...")
    explorer = ExplorerAgent(client, repo_tool)
    findings = explorer.explore()
    print(f"      Found {len(findings.get('concepts', []))} concepts: "
          + ", ".join(c['name'] for c in findings.get('concepts', [])))
    
    # 3. Curriculum
    print("\n[3/3] Generating curriculum outline...")
    curriculum = CurriculumAgent(client).generate(findings)
    print(f"      Planned {len(curriculum.get('notebooks', []))} notebooks")
    
    # Estimate cost
    estimate = estimate_cost(findings, curriculum, client.model)
    print(f"      Estimated full-generation cost: ${estimate['total_usd']:.2f}")
    
    # Persist state
    state = _new_state(args.repo, repo_path, output_dir, findings, curriculum, client)
    _save_state(output_dir, state)
    
    # Write CURRICULUM.md
    write_curriculum_md(output_dir / "CURRICULUM.md", curriculum, args.repo)
    
    # Next steps message
    print("\n" + "─" * 60)
    print(f"Curriculum written to:  {output_dir / 'CURRICULUM.md'}")
    print("\nNext steps:")
    print("  1. Open and review CURRICULUM.md")
    print("  2. Edit YAML blocks to skip notebooks or add requirements")
    print(f"  3. Run: python distill.py --continue --output {output_dir}")
    print(f"\nOr preview one notebook first:")
    print(f"  python distill.py --preview 01 --output {output_dir}")
```

### `--continue`

```python
def cmd_continue(args):
    """Step 2: generate all notebooks + capstone."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    
    if state["step"] == "generated":
        print("Course already fully generated. Use --refine for changes.")
        if not args.no_confirm and not _confirm("Regenerate everything anyway?"):
            return
    
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model
    
    print(f"\nrepo-distiller  |  generating course")
    print("─" * 60)
    print(f"Repo: {state['repo_url']}")
    print(f"Output: {output_dir}\n")
    
    # Re-parse user-edited CURRICULUM.md
    user_md = (output_dir / "CURRICULUM.md").read_text()
    curriculum = parse_curriculum_md(user_md, state["curriculum"])
    state["curriculum"] = curriculum
    
    # Cost confirmation
    estimate = estimate_cost(state["findings"], curriculum, client.model)
    print(f"Estimated cost: ${estimate['total_usd']:.2f}")
    print(f"  - Notebooks ({len(curriculum['notebooks'])}): ${estimate['breakdown']['notebooks']:.2f}")
    print(f"  - Capstone:  ${estimate['breakdown']['capstone']:.2f}")
    
    if not args.no_confirm and not _confirm("Proceed?"):
        return
    
    state["step"] = "generating"
    _save_state(output_dir, state)
    
    # Re-build tools
    repo_tool = RepoTool(state["repo_url"])
    repo_tool.repo_path = state["repo_path"]
    
    try:
        # Generate notebooks
        gen = GeneratorAgent(client, repo_tool, output_dir, state)
        gen.generate_all(user_md, curriculum)
        
        # Generate capstone (with validation loop)
        cap = CapstoneAgent(client, output_dir, state)
        cap.generate_with_validation(curriculum)
        
        # Write README
        write_course_readme(output_dir / "README.md", curriculum)
        
        state["step"] = "generated"
        _save_state(output_dir, state)
        
    except Exception as e:
        state["step"] = "failed"
        state["progress"]["last_error"] = str(e)
        _save_state(output_dir, state)
        print(f"\nFAILED: {e}")
        print(f"Resume with: python distill.py --resume --output {output_dir}")
        raise
    
    if client.config.get("output", {}).get("init_git", True):
        _init_git(output_dir)
    
    # Final report
    print("\n" + "─" * 60)
    print(f"Course generated at: {output_dir}")
    print(format_cost_report(state))
    print("\nDirectory structure:")
    _print_tree(output_dir, exclude={"_repo_cache", ".distill_state.json"})
```

### `--refine "INSTRUCTION"`

```python
def cmd_refine(args):
    """Refine specific content based on user instruction."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    
    print(f"\nrepo-distiller  |  refine")
    print("─" * 60)
    print(f"Instruction: {args.refine}\n")
    
    repo_tool = RepoTool(state["repo_url"])
    repo_tool.repo_path = state["repo_path"]
    
    # Detect target
    if "capstone" in args.refine.lower():
        target = "capstone"
    else:
        gen = GeneratorAgent(client, repo_tool, output_dir, state)
        nb_idx = gen._detect_target_notebook(args.refine, state["curriculum"])
        if nb_idx is None:
            print("Could not identify target. Be more specific:")
            print('  - Reference a notebook: "notebook 03 needs ..."')
            print('  - Reference a concept: "the attention notebook needs ..."')
            print('  - Mention capstone:    "capstone tests should ..."')
            return
        target = nb_idx
    
    # Confirm
    if target == "capstone":
        print("Target: capstone")
    else:
        nb = state["curriculum"]["notebooks"][target]
        print(f"Target: notebook {nb['id']} — {nb['title']}")
    
    if not args.no_confirm and not _confirm("Regenerate?"):
        return
    
    if target == "capstone":
        cap = CapstoneAgent(client, output_dir, state)
        cap.refine(args.refine, state["curriculum"])
    else:
        gen = GeneratorAgent(client, repo_tool, output_dir, state)
        # Backup
        nb = state["curriculum"]["notebooks"][target]
        slug = _slugify(nb["title"])
        for path in [output_dir / f"notebooks/{nb['id']}_{slug}.ipynb",
                     output_dir / f"exercises/{nb['id']}_{slug}_exercise.ipynb"]:
            if path.exists():
                shutil.copy(path, path.with_suffix(".ipynb.bak"))
                print(f"  Backed up: {path.name}")
        # Inject and regenerate
        nb["description"] += f"\n\nAdditional requirement: {args.refine}"
        gen.generate_one(nb["id"], state["curriculum"])
    
    print(f"\nRefinement applied. Commit changes:")
    print(f"  cd {output_dir} && git add -A && git commit -m 'refine: {args.refine[:50]}'")
```

### `--preview NN`

```python
def cmd_preview(args):
    """Generate a single notebook for quality preview."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    
    nb_id = args.preview
    notebooks = state["curriculum"]["notebooks"]
    nb_idx = next((i for i, n in enumerate(notebooks) if n["id"] == nb_id), None)
    
    if nb_idx is None:
        print(f"No notebook with id '{nb_id}'. Available: "
              + ", ".join(n["id"] for n in notebooks))
        return
    
    nb = notebooks[nb_idx]
    print(f"\nPreviewing notebook {nb_id}: {nb['title']}")
    print("─" * 60)
    
    repo_tool = RepoTool(state["repo_url"])
    repo_tool.repo_path = state["repo_path"]
    
    gen = GeneratorAgent(client, repo_tool, output_dir, state)
    gen.generate_one(nb_id, state["curriculum"])
    
    slug = _slugify(nb["title"])
    print(f"\nGenerated: {output_dir / 'notebooks' / f'{nb_id}_{slug}.ipynb'}")
    print(f"Open it to check quality before running --continue for the full course.")
```

### `--resume`

```python
def cmd_resume(args):
    """Resume a previously failed --continue from saved progress."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    
    if state["step"] != "failed" and state["step"] != "generating":
        print(f"Nothing to resume. Current state: {state['step']}")
        return
    
    print(f"Resuming generation. Already complete:")
    print(f"  Notebooks: {state['progress']['notebooks_complete']}")
    print(f"  Exercises: {state['progress']['exercises_complete']}")
    print(f"  Capstone:  {state['progress']['capstone_complete']}")
    
    # Just call cmd_continue — it respects state.progress
    cmd_continue(args)
```

### `--estimate-cost`

```python
def cmd_estimate_cost(args):
    """Show estimate without generating."""
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model
    
    estimate = estimate_cost(state["findings"], state["curriculum"], client.model)
    
    print(f"\nCost estimate (model: {client.model})")
    print("─" * 60)
    print(f"Input tokens:  ~{estimate['input_tokens']:,}")
    print(f"Output tokens: ~{estimate['output_tokens']:,}")
    print(f"Total:         ${estimate['total_usd']:.2f}")
    print("\nBreakdown:")
    for category, cost in estimate["breakdown"].items():
        print(f"  {category:12s} ${cost:.2f}")
```

### `--status`

```python
def cmd_status(args):
    """Show progress and cost so far for an output directory."""
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    
    print(f"\nStatus: {output_dir}")
    print("─" * 60)
    print(f"Repo:    {state['repo_url']}")
    print(f"Step:    {state['step']}")
    print(f"Model:   {state['config_snapshot']['default_model']}")
    print()
    print("Progress:")
    nb_done = len(state["progress"]["notebooks_complete"])
    nb_total = len(state["curriculum"]["notebooks"])
    ex_done = len(state["progress"]["exercises_complete"])
    print(f"  Notebooks:  {nb_done}/{nb_total}")
    print(f"  Exercises:  {ex_done}/{nb_total}")
    print(f"  Capstone:   {'✓' if state['progress']['capstone_complete'] else '✗'}")
    
    if state["progress"].get("last_error"):
        print(f"\nLast error: {state['progress']['last_error']}")
    
    print()
    print(format_cost_report(state))
```

---

## Helpers

```python
def _confirm(prompt: str) -> bool:
    """Prompt y/N. Default no. Returns True only if user types 'y'."""
    response = input(f"{prompt} [y/N] ").strip().lower()
    return response == "y"


def _default_output_dir(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    return f"output/{name}_course"


def _find_latest_output() -> str:
    """Find most recently modified output/*/.distill_state.json"""
    candidates = sorted(
        Path("output").glob("*/.distill_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("No output directory found. Run --repo first.")
        sys.exit(1)
    return str(candidates[0].parent)


def _load_state(output_dir: Path) -> dict:
    state_file = output_dir / ".distill_state.json"
    if not state_file.exists():
        print(f"No state in {output_dir}. Run --repo first.")
        sys.exit(1)
    return json.loads(state_file.read_text())


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / ".distill_state.json").write_text(json.dumps(state, indent=2))


def _new_state(repo_url, repo_path, output_dir, findings, curriculum, client) -> dict:
    return {
        "schema_version": 1,
        "repo_url": repo_url,
        "repo_path": str(repo_path),
        "output_dir": str(output_dir),
        "findings": findings,
        "curriculum": curriculum,
        "step": "explored",
        "progress": {
            "notebooks_complete": [],
            "exercises_complete": [],
            "capstone_complete": False,
            "last_error": None,
        },
        "cost": {
            "explore_input_tokens": 0,
            "explore_output_tokens": 0,
            "curriculum_input_tokens": 0,
            "curriculum_output_tokens": 0,
            "notebooks_input_tokens": 0,
            "notebooks_output_tokens": 0,
            "capstone_input_tokens": 0,
            "capstone_output_tokens": 0,
            "total_estimated_usd": 0.0,
        },
        "config_snapshot": {
            "default_model": client.model,
            "step_models": client.config.get("step_models", {}),
        },
    }


def _print_tree(path: Path, prefix: str = "", exclude: set = None):
    """Print directory tree, skipping hidden and excluded items."""
    exclude = exclude or set()
    items = sorted(
        [p for p in path.iterdir() if p.name not in exclude and not p.name.startswith(".")],
        key=lambda p: (p.is_file(), p.name),
    )
    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item.name)
        if item.is_dir():
            ext = "    " if i == len(items) - 1 else "│   "
            _print_tree(item, prefix + ext, exclude)


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]


def _init_git(output_dir: Path):
    (output_dir / ".gitignore").write_text("_repo_cache/\n.distill_state.json\n*.bak\n")
    subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=output_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial course by repo-distiller"],
                   cwd=output_dir, capture_output=True)
```

---

## Main entry point

```python
def main():
    parser = argparse.ArgumentParser(
        description="Transform GitHub repos into educational curricula"
    )
    # Mutually exclusive commands
    cmd = parser.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--repo", help="GitHub URL")
    cmd.add_argument("--continue", dest="do_continue", action="store_true")
    cmd.add_argument("--refine", help="Refinement instruction")
    cmd.add_argument("--preview", help="Preview notebook by ID, e.g. '03'")
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--estimate-cost", dest="estimate_cost", action="store_true")
    cmd.add_argument("--status", action="store_true")
    
    # Options
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--no-confirm", action="store_true")
    
    args = parser.parse_args()
    
    if args.repo:           cmd_explore(args)
    elif args.do_continue:  cmd_continue(args)
    elif args.refine:       cmd_refine(args)
    elif args.preview:      cmd_preview(args)
    elif args.resume:       cmd_resume(args)
    elif args.estimate_cost: cmd_estimate_cost(args)
    elif args.status:       cmd_status(args)


if __name__ == "__main__":
    main()
```

---

## Test (`tests/test_cli.py`)

- `test_argument_parsing_repo` — parses --repo URL correctly
- `test_argument_parsing_mutually_exclusive` — --repo and --continue together raises
- `test_default_output_dir_derivation` — URL → directory name
- `test_find_latest_output_picks_most_recent`
- `test_state_save_load_roundtrip`
- `test_confirm_returns_true_on_y`
- `test_confirm_returns_false_on_anything_else`
