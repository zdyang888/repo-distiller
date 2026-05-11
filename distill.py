#!/usr/bin/env python3
"""repo-distiller CLI — transform GitHub repos into educational curricula."""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from agents.capstone import CapstoneAgent
from agents.curriculum import CurriculumAgent
from agents.explorer import ExplorerAgent
from agents.generator import GeneratorAgent
from llm.client import LLMClient
from tools.repo import RepoTool

logger = logging.getLogger(__name__)

# ── Cost estimation ───────────────────────────────────────────────────────────

# Rough token costs per 1M tokens (USD)
_TOKEN_COSTS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-20251001": {"input": 0.8, "output": 4.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.3},
}
_DEFAULT_TOKEN_COST: dict[str, float] = {"input": 3.0, "output": 15.0}


def estimate_cost(findings: dict, curriculum: dict, model: str) -> dict:
    """Rough cost estimate for full generation.

    Args:
        findings: ExplorerFindings dict.
        curriculum: Curriculum dict.
        model: Model name string.

    Returns:
        Dict with input_tokens, output_tokens, total_usd, breakdown keys.
    """
    n_notebooks = len(curriculum.get("notebooks", []))
    costs = _TOKEN_COSTS.get(model, _DEFAULT_TOKEN_COST)

    # Heuristic token estimates per notebook (teach + exercise + solution)
    notebooks_in = n_notebooks * 3000
    notebooks_out = n_notebooks * 6000
    capstone_in = 4000
    capstone_out = 3000

    total_in = notebooks_in + capstone_in
    total_out = notebooks_out + capstone_out

    notebooks_usd = (notebooks_in * costs["input"] + notebooks_out * costs["output"]) / 1_000_000
    capstone_usd = (capstone_in * costs["input"] + capstone_out * costs["output"]) / 1_000_000

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_usd": notebooks_usd + capstone_usd,
        "breakdown": {
            "notebooks": notebooks_usd,
            "capstone": capstone_usd,
        },
    }


def format_cost_report(state: dict) -> str:
    """Format a cost summary string from state dict."""
    cost = state.get("cost", {})
    total_in = sum(v for k, v in cost.items() if "input_tokens" in k)
    total_out = sum(v for k, v in cost.items() if "output_tokens" in k)
    total_usd = cost.get("total_estimated_usd", 0.0)
    return "\n".join([
        "Cost so far:",
        f"  Input tokens:  {total_in:,}",
        f"  Output tokens: {total_out:,}",
        f"  Estimated USD: ${total_usd:.4f}",
    ])


# ── Curriculum MD helpers ─────────────────────────────────────────────────────

def write_curriculum_md(path: Path, curriculum: dict, repo_url: str) -> None:
    """Write a human-readable CURRICULUM.md with embedded JSON.

    Args:
        path: Output path for CURRICULUM.md.
        curriculum: Curriculum dict from CurriculumAgent.
        repo_url: Source repository URL.
    """
    lines = [
        f"# Curriculum: {curriculum.get('title', 'Course')}",
        f"\nSource: {repo_url}\n",
        "## Overview\n",
        curriculum.get("mental_model", "") + "\n",
        "## Notebooks\n",
    ]
    for nb in curriculum.get("notebooks", []):
        lines.append(f"### {nb['id']}. {nb['title']}")
        lines.append(f"\n{nb.get('description', '')}\n")
        if nb.get("learning_objectives"):
            lines.append("**Learning objectives:**")
            for obj in nb["learning_objectives"]:
                lines.append(f"- {obj}")
            lines.append("")
        lines.append(f"**Exercise:** {nb.get('exercise_description', 'TBD')}\n")

    cap = curriculum.get("capstone", {})
    lines += [
        "## Capstone Project\n",
        f"**{cap.get('title', 'Capstone')}**\n",
        cap.get("description", "") + "\n",
        "---\n",
        "<!-- Edit the JSON below to skip notebooks (set skip: true) -->\n",
        "```json",
        json.dumps(curriculum, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_curriculum_md(user_md: str, fallback: dict) -> dict:
    """Parse CURRICULUM.md back to a curriculum dict.

    Extracts the embedded JSON block. Falls back to fallback if parsing fails.

    Args:
        user_md: Contents of user-edited CURRICULUM.md.
        fallback: Curriculum dict to return if JSON extraction fails.

    Returns:
        Parsed curriculum dict.
    """
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", user_md)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and "notebooks" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    logger.warning("parse_curriculum_md: could not parse embedded JSON, using state curriculum")
    return fallback


def write_course_readme(path: Path, curriculum: dict) -> None:
    """Write a README.md for the generated course directory.

    Args:
        path: Output path for README.md.
        curriculum: Curriculum dict.
    """
    lines = [
        f"# {curriculum.get('title', 'Course')}\n",
        curriculum.get("mental_model", "") + "\n",
        "## Contents\n",
        "| Notebook | Description |",
        "|----------|-------------|",
    ]
    for nb in curriculum.get("notebooks", []):
        slug = _slugify(nb["title"])
        nb_file = f"notebooks/{nb['id']}_{slug}.ipynb"
        lines.append(
            f"| [{nb['id']}. {nb['title']}]({nb_file}) | {nb.get('description', '')} |"
        )

    cap = curriculum.get("capstone", {})
    lines += [
        "\n## Capstone Project\n",
        f"**{cap.get('title', 'Capstone')}**: {cap.get('description', '')}\n",
        "See `capstone/` for instructions, starter code, and tests.\n",
        "## Getting Started\n",
        "```bash",
        "pip install -r requirements.txt",
        "jupyter notebook",
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str) -> None:
    """Configure root logger at the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state(output_dir: Path) -> dict:
    """Load .distill_state.json from output_dir. Exits if not found."""
    state_file = output_dir / ".distill_state.json"
    if not state_file.exists():
        print(f"No state in {output_dir}. Run --repo first.")
        sys.exit(1)
    return json.loads(state_file.read_text())


def _save_state(output_dir: Path, state: dict) -> None:
    """Write state dict to .distill_state.json in output_dir."""
    (output_dir / ".distill_state.json").write_text(json.dumps(state, indent=2))


def _new_state(
    repo_url: str,
    repo_path,
    output_dir: Path,
    findings: dict,
    curriculum: dict,
    client: LLMClient,
) -> dict:
    """Create a fresh state dict after exploration."""
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
            "step_models": client._config.get("step_models", {}),
        },
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

def _confirm(prompt: str) -> bool:
    """Prompt y/N. Default no. Returns True only if user types 'y'."""
    response = input(f"{prompt} [y/N] ").strip().lower()
    return response == "y"


def _default_output_dir(repo_url: str) -> str:
    """Derive a default output directory name from a repo URL."""
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    return f"output/{name}_course"


def _find_latest_output() -> str:
    """Find the most recently modified output directory with a state file."""
    candidates = sorted(
        Path("output").glob("*/.distill_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("No output directory found. Run --repo first.")
        sys.exit(1)
    return str(candidates[0].parent)


def _print_tree(path: Path, prefix: str = "", exclude: set | None = None) -> None:
    """Print a directory tree, skipping hidden files and excluded names."""
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
    """Convert text to a lowercase filesystem-safe slug (max 40 chars)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]


def _init_git(output_dir: Path) -> None:
    """Initialise a git repo in output_dir and make an initial commit."""
    (output_dir / ".gitignore").write_text("_repo_cache/\n.distill_state.json\n*.bak\n")
    subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=output_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial course by repo-distiller"],
        cwd=output_dir,
        capture_output=True,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_explore(args: argparse.Namespace) -> None:
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
    concepts = findings.get("concepts", [])
    print(f"      Found {len(concepts)} concepts: " + ", ".join(c["name"] for c in concepts))

    # 3. Curriculum
    print("\n[3/3] Generating curriculum outline...")
    curriculum = CurriculumAgent(client).generate(findings)
    print(f"      Planned {len(curriculum.get('notebooks', []))} notebooks")

    estimate = estimate_cost(findings, curriculum, client.model)
    print(f"      Estimated full-generation cost: ${estimate['total_usd']:.2f}")

    # Persist
    state = _new_state(args.repo, repo_path, output_dir, findings, curriculum, client)
    _save_state(output_dir, state)
    write_curriculum_md(output_dir / "CURRICULUM.md", curriculum, args.repo)

    print("\n" + "─" * 60)
    print(f"Curriculum written to:  {output_dir / 'CURRICULUM.md'}")
    print("\nNext steps:")
    print("  1. Open and review CURRICULUM.md")
    print("  2. Edit JSON block to skip notebooks or add requirements")
    print(f"  3. Run: python distill.py --continue --output {output_dir}")
    print(f"\nOr preview one notebook first:")
    print(f"  python distill.py --preview 01 --output {output_dir}")


def cmd_continue(args: argparse.Namespace) -> None:
    """Step 2: generate all notebooks + capstone from saved state."""
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
    curriculum_path = output_dir / "CURRICULUM.md"
    if curriculum_path.exists():
        user_md = curriculum_path.read_text()
        curriculum = parse_curriculum_md(user_md, state["curriculum"])
    else:
        user_md = ""
        curriculum = state["curriculum"]
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

    repo_tool = RepoTool(state["repo_url"])
    repo_tool.repo_path = state["repo_path"]

    try:
        gen = GeneratorAgent(client, repo_tool, output_dir, state)
        gen.generate_all(user_md, curriculum)

        cap = CapstoneAgent(client, output_dir, state)
        cap.generate_with_validation(curriculum)

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

    if client._config.get("output", {}).get("init_git", True):
        _init_git(output_dir)

    print("\n" + "─" * 60)
    print(f"Course generated at: {output_dir}")
    print(format_cost_report(state))
    print("\nDirectory structure:")
    _print_tree(output_dir, exclude={"_repo_cache", ".distill_state.json"})


def cmd_refine(args: argparse.Namespace) -> None:
    """Refine specific content based on a user instruction."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model

    print(f"\nrepo-distiller  |  refine")
    print("─" * 60)
    print(f"Instruction: {args.refine}\n")

    repo_tool = RepoTool(state["repo_url"])
    repo_tool.repo_path = state["repo_path"]

    # Detect target
    if "capstone" in args.refine.lower():
        target: str | int = "capstone"
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
        # Capstone refinement = full re-generation with validation loop
        cap = CapstoneAgent(client, output_dir, state)
        cap.generate_with_validation(state["curriculum"])
    else:
        gen = GeneratorAgent(client, repo_tool, output_dir, state)
        nb = state["curriculum"]["notebooks"][target]
        slug = _slugify(nb["title"])
        for backup_path in [
            output_dir / f"notebooks/{nb['id']}_{slug}.ipynb",
            output_dir / f"exercises/{nb['id']}_{slug}_exercise.ipynb",
        ]:
            if backup_path.exists():
                shutil.copy(backup_path, backup_path.with_suffix(".ipynb.bak"))
                print(f"  Backed up: {backup_path.name}")
        nb["description"] += f"\n\nAdditional requirement: {args.refine}"
        gen.generate_one(nb["id"], state["curriculum"])

    print(f"\nRefinement applied. Commit changes:")
    print(f"  cd {output_dir} && git add -A && git commit -m 'refine: {args.refine[:50]}'")


def cmd_preview(args: argparse.Namespace) -> None:
    """Generate a single notebook for quality preview."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model

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


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a previously failed --continue from saved progress."""
    setup_logging(args.log_level)
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)

    if state["step"] not in ("failed", "generating"):
        print(f"Nothing to resume. Current state: {state['step']}")
        return

    print(f"Resuming generation. Already complete:")
    print(f"  Notebooks: {state['progress']['notebooks_complete']}")
    print(f"  Exercises: {state['progress']['exercises_complete']}")
    print(f"  Capstone:  {state['progress']['capstone_complete']}")

    cmd_continue(args)


def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """Show cost estimate without generating anything."""
    output_dir = Path(args.output or _find_latest_output())
    state = _load_state(output_dir)
    client = LLMClient(args.config)
    if args.model:
        client.model = args.model

    est = estimate_cost(state["findings"], state["curriculum"], client.model)

    print(f"\nCost estimate (model: {client.model})")
    print("─" * 60)
    print(f"Input tokens:  ~{est['input_tokens']:,}")
    print(f"Output tokens: ~{est['output_tokens']:,}")
    print(f"Total:         ${est['total_usd']:.2f}")
    print("\nBreakdown:")
    for category, cost in est["breakdown"].items():
        print(f"  {category:12s} ${cost:.2f}")


def cmd_status(args: argparse.Namespace) -> None:
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
    print(f"  Capstone:   {'done' if state['progress']['capstone_complete'] else 'pending'}")

    if state["progress"].get("last_error"):
        print(f"\nLast error: {state['progress']['last_error']}")

    print()
    print(format_cost_report(state))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="Transform GitHub repos into educational curricula"
    )
    cmd = parser.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--repo", metavar="URL", help="GitHub URL to process")
    cmd.add_argument(
        "--continue", dest="do_continue", action="store_true",
        help="Generate notebooks from CURRICULUM.md",
    )
    cmd.add_argument("--refine", metavar="INSTRUCTION", help="Refine specific content")
    cmd.add_argument("--preview", metavar="NN", help="Generate ONE notebook for quality check")
    cmd.add_argument("--resume", action="store_true", help="Resume an interrupted --continue")
    cmd.add_argument(
        "--estimate-cost", dest="estimate_cost", action="store_true",
        help="Show cost estimate without running",
    )
    cmd.add_argument("--status", action="store_true", help="Show progress of an output directory")

    parser.add_argument("--output", metavar="DIR", help="Output directory (auto-detected if omitted)")
    parser.add_argument("--model", metavar="MODEL", help="Override default model for this run")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity",
    )
    parser.add_argument("--no-confirm", action="store_true", help="Skip interactive confirmations")

    args = parser.parse_args()

    if args.repo:
        cmd_explore(args)
    elif args.do_continue:
        cmd_continue(args)
    elif args.refine:
        cmd_refine(args)
    elif args.preview:
        cmd_preview(args)
    elif args.resume:
        cmd_resume(args)
    elif args.estimate_cost:
        cmd_estimate_cost(args)
    elif args.status:
        cmd_status(args)


if __name__ == "__main__":
    main()
