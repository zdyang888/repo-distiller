"""Generator agent: produce teaching, exercise, and solution notebooks from a Curriculum."""

import json
import logging
import os
import re
from pathlib import Path

from llm.client import LLMClient
from prompts.notebook import EXERCISE_SYSTEM, EXERCISE_USER, TEACH_SYSTEM, TEACH_USER
from prompts.solution import SOLUTION_USER
from tools.notebook import build_notebook, parse_llm_response, save_notebook
from tools.repo import RepoTool

logger = logging.getLogger(__name__)

STATE_FILE = ".distill_state.json"


def _slug(title: str) -> str:
    """Convert a title to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


class GeneratorAgent:
    """Generate teaching, exercise, and solution notebooks from a Curriculum."""

    def __init__(
        self,
        llm_client: LLMClient,
        repo_tool: RepoTool,
        output_dir: Path,
        state: dict,
    ) -> None:
        self.llm = llm_client
        self.repo = repo_tool
        self.output_dir = output_dir
        self.state = state
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, user_curriculum_md: str, curriculum: dict) -> None:
        """Run full generation pipeline for all notebooks in the curriculum.

        Skips notebooks already listed in state.progress.notebooks_complete /
        exercises_complete (resumability). Persists state after every step.

        Args:
            user_curriculum_md: Markdown representation of the curriculum (context only).
            curriculum: Curriculum dict from CurriculumAgent.
        """
        notebooks = curriculum.get("notebooks", [])
        progress = self.state.setdefault("progress", {})
        notebooks_complete: list[str] = progress.setdefault("notebooks_complete", [])
        exercises_complete: list[str] = progress.setdefault("exercises_complete", [])

        for i, nb_spec in enumerate(notebooks):
            nb_id = nb_spec["id"]
            prev_title = notebooks[i - 1]["title"] if i > 0 else "None"
            next_title = notebooks[i + 1]["title"] if i < len(notebooks) - 1 else "None"
            slug = _slug(nb_spec["title"])
            teach_path = self.output_dir / f"{nb_id}_{slug}.ipynb"
            ex_path = self.output_dir / f"{nb_id}_{slug}_exercise.ipynb"
            sol_path = self.output_dir / f"{nb_id}_{slug}_solution.ipynb"

            if nb_id not in notebooks_complete:
                try:
                    source_code = self._extract_relevant_source(nb_spec)
                    self._generate_teaching_notebook(
                        nb_spec, curriculum, source_code, prev_title, next_title, teach_path
                    )
                    notebooks_complete.append(nb_id)
                    self._persist_state()
                    logger.info("Generated teaching notebook: %s", teach_path)
                except Exception as exc:
                    self.state["step"] = "failed"
                    self.state["last_error"] = str(exc)
                    self._persist_state()
                    raise

            if nb_id not in exercises_complete:
                try:
                    self._generate_exercise_notebook(nb_spec, curriculum, ex_path, sol_path)
                    exercises_complete.append(nb_id)
                    self._persist_state()
                    logger.info("Generated exercise notebook: %s", ex_path)
                except Exception as exc:
                    self.state["step"] = "failed"
                    self.state["last_error"] = str(exc)
                    self._persist_state()
                    raise

        self._write_readme(curriculum)

    def generate_one(self, nb_id: str, curriculum: dict) -> None:
        """Generate (or regenerate) a single notebook by ID.

        Always overwrites existing files without backing up. Caller is responsible
        for backups when needed (e.g., refine()).

        Args:
            nb_id: Notebook ID string (e.g. "01").
            curriculum: Curriculum dict.
        """
        notebooks = curriculum.get("notebooks", [])
        for i, nb_spec in enumerate(notebooks):
            if nb_spec["id"] == nb_id:
                slug = _slug(nb_spec["title"])
                prev_title = notebooks[i - 1]["title"] if i > 0 else "None"
                next_title = notebooks[i + 1]["title"] if i < len(notebooks) - 1 else "None"
                teach_path = self.output_dir / f"{nb_id}_{slug}.ipynb"
                ex_path = self.output_dir / f"{nb_id}_{slug}_exercise.ipynb"
                sol_path = self.output_dir / f"{nb_id}_{slug}_solution.ipynb"

                source_code = self._extract_relevant_source(nb_spec)
                self._generate_teaching_notebook(
                    nb_spec, curriculum, source_code, prev_title, next_title, teach_path
                )
                self._generate_exercise_notebook(nb_spec, curriculum, ex_path, sol_path)
                return

        logger.warning("generate_one: notebook id %r not found in curriculum", nb_id)

    def refine(self, instruction: str, curriculum: dict) -> None:
        """Refine content based on a natural-language instruction.

        Detects the target notebook, confirms with the user, backs up existing
        files, then regenerates.

        Args:
            instruction: User instruction, e.g. "make notebook 01 use more examples".
            curriculum: Curriculum dict.
        """
        if "capstone" in instruction.lower():
            logger.warning("Capstone refinement should be handled by CapstoneAgent.")
            return

        idx = self._detect_target_notebook(instruction, curriculum)
        if idx is None:
            print("Could not determine which notebook to refine. Please be more specific.")
            return

        notebooks = curriculum.get("notebooks", [])
        nb_spec = notebooks[idx]
        print(f"Refining notebook {nb_spec['id']}: {nb_spec['title']}")
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

        # Back up existing files
        slug = _slug(nb_spec["title"])
        nb_id = nb_spec["id"]
        for suffix in ("", "_exercise", "_solution"):
            p = self.output_dir / f"{nb_id}_{slug}{suffix}.ipynb"
            if p.exists():
                p.rename(str(p) + ".bak")

        # Inject instruction into description and regenerate
        nb_spec_copy = dict(nb_spec)
        nb_spec_copy["description"] = (
            nb_spec.get("description", "") + f"\n\nRefinement request: {instruction}"
        )
        notebooks_copy = list(notebooks)
        notebooks_copy[idx] = nb_spec_copy
        curriculum_copy = dict(curriculum)
        curriculum_copy["notebooks"] = notebooks_copy

        self.generate_one(nb_id, curriculum_copy)

    # ── Internal ──

    def _generate_teaching_notebook(
        self,
        nb_spec: dict,
        curriculum: dict,
        source_code: str,
        prev_title: str,
        next_title: str,
        out_path: Path,
    ) -> None:
        """Call LLM to generate a teaching notebook and save it."""
        objectives = "\n".join(f"- {obj}" for obj in nb_spec.get("learning_objectives", []))
        prompt = TEACH_USER.format(
            project_title=curriculum.get("title", ""),
            mental_model=curriculum.get("mental_model", ""),
            nb_title=nb_spec.get("title", ""),
            concept=nb_spec.get("concept", ""),
            description=nb_spec.get("description", ""),
            objectives=objectives or "- Understand the core concept",
            prev_title=prev_title,
            next_title=next_title,
            source_code=source_code,
            visualization_idea=nb_spec.get("visualization_idea", ""),
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=TEACH_SYSTEM,
            model=self.llm.model_for_step("notebooks"),
        )
        cells = parse_llm_response(response)
        nb = build_notebook(cells, title=nb_spec.get("title", ""))
        save_notebook(nb, str(out_path))

    def _generate_exercise_notebook(
        self,
        nb_spec: dict,
        curriculum: dict,
        out_path: Path,
        sol_path: Path,
    ) -> None:
        """Call LLM to generate an exercise notebook and its solution, then save both."""
        objectives = "\n".join(f"- {obj}" for obj in nb_spec.get("learning_objectives", []))
        prompt = EXERCISE_USER.format(
            project_title=curriculum.get("title", ""),
            concept=nb_spec.get("concept", ""),
            exercise_description=nb_spec.get("exercise_description", ""),
            objectives=objectives or "- Practice the core concept",
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=EXERCISE_SYSTEM,
            model=self.llm.model_for_step("notebooks"),
        )
        
        # Save solution FIRST (keeps implementations)
        self._generate_solution(response, nb_spec, curriculum, sol_path)
        
        # Hollow out the response for the exercise version
        hollowed_response = _hollow_out_exercise(response)
        cells = parse_llm_response(hollowed_response)
        nb = build_notebook(cells, title=f"{nb_spec.get('title', '')} — Exercise")
        save_notebook(nb, str(out_path))

    def _generate_solution(
        self,
        exercise_response: str,
        nb_spec: dict,
        curriculum: dict,
        sol_path: Path,
    ) -> None:
        """Generate a solution notebook by completing all exercise stubs.

        Args:
            exercise_response: Raw LLM response text for the exercise notebook.
            nb_spec: Notebook spec dict.
            curriculum: Curriculum dict.
            sol_path: Output path for the solution notebook.
        """
        prompt = SOLUTION_USER.format(
            exercise_content=exercise_response,
            project_title=curriculum.get("title", ""),
            mental_model=curriculum.get("mental_model", ""),
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.llm.model_for_step("notebooks"),
        )
        solution_text = _apply_solutions(exercise_response, response)
        cells = parse_llm_response(solution_text)
        nb = build_notebook(cells, title=f"{nb_spec.get('title', '')} — Solution")
        save_notebook(nb, str(sol_path))

    def _extract_relevant_source(self, nb_spec: dict) -> str:
        """Extract source code for the notebook concept.

        Uses AST-based symbol extraction when key_symbols is non-empty,
        falls back to whole-file reading otherwise.

        Args:
            nb_spec: Notebook spec dict with key_source_files and key_symbols.

        Returns:
            Formatted source code string for use in the LLM prompt.
        """
        files = nb_spec.get("key_source_files", [])
        symbols = nb_spec.get("key_symbols", [])
        if not files:
            return ""
        if symbols:
            return self.repo.read_files_for_concept(files, symbols=symbols)
        return self.repo.read_files_for_concept(files)

    def _detect_target_notebook(self, instruction: str, curriculum: dict) -> int | None:
        """Detect which notebook an instruction refers to.

        Matches by notebook ID, title (case-insensitive), or concept name.

        Args:
            instruction: User instruction string.
            curriculum: Curriculum dict.

        Returns:
            0-based index of the matched notebook, or None if zero or multiple matches.
        """
        notebooks = curriculum.get("notebooks", [])
        instruction_lower = instruction.lower()
        matches: list[int] = []

        for i, nb in enumerate(notebooks):
            # Match by ID (e.g. "01" in "improve notebook 01")
            nb_id = nb.get("id", "")
            if nb_id and nb_id in instruction:
                matches.append(i)
                continue
            # Match by title
            title = nb.get("title", "").lower()
            if title and title in instruction_lower:
                matches.append(i)
                continue
            # Match by concept
            concept = nb.get("concept", "").lower()
            if concept and concept in instruction_lower:
                matches.append(i)
                continue

        if len(matches) == 1:
            return matches[0]
        return None

    def _persist_state(self) -> None:
        """Atomically write current state to .distill_state.json in output_dir.

        Writes to a temp file then renames to avoid corruption if the process
        is killed mid-write.
        """
        target = self.output_dir / STATE_FILE
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, target)

    def _write_readme(self, curriculum: dict) -> None:
        """Write a course README.md to output_dir."""
        title = curriculum.get("title", "Course")
        mental_model = curriculum.get("mental_model", "")
        notebooks = curriculum.get("notebooks", [])

        lines = [f"# {title}", "", mental_model, "", "## Notebooks", ""]
        for nb in notebooks:
            nb_id = nb.get("id", "")
            nb_title = nb.get("title", "")
            description = nb.get("description", "")
            slug = _slug(nb_title)
            lines += [
                f"### {nb_id}. {nb_title}",
                "",
                description,
                "",
                f"- Teaching: `{nb_id}_{slug}.ipynb`",
                f"- Exercise: `{nb_id}_{slug}_exercise.ipynb`",
                f"- Solution: `{nb_id}_{slug}_solution.ipynb`",
                "",
            ]

        readme_path = self.output_dir / "README.md"
        readme_path.write_text("\n".join(lines), encoding="utf-8")


def _hollow_out_exercise(text: str) -> str:
    """Replace code between START/END tags with a placeholder.

    Args:
        text: Raw LLM response text.

    Returns:
        Modified text with code implementations removed.
    """
    pattern = re.compile(
        r"(### START CODE HERE ###\n).*?(\n[ \t]*### END CODE HERE ###)",
        re.DOTALL,
    )
    # Replace with a comment and a pass statement, preserving indentation
    return pattern.sub(r"\1    # your code here\n    pass\2", text)


def _strip_func_header(impl: str) -> str:
    """Strip the leading ``def`` line and docstring from a function implementation.

    The LLM returns full function definitions, but the exercise template already
    has the signature and docstring. We only need the body.

    Args:
        impl: Full function implementation text from the LLM.

    Returns:
        Function body only, preserving indentation.
    """
    lines = impl.split("\n")
    i = 0

    # Skip leading blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Skip def line
    if i < len(lines) and lines[i].strip().startswith("def "):
        # Handle multi-line signatures (def foo(\n    arg1,\n    arg2\n):)
        while i < len(lines) and "):" not in lines[i]:
            i += 1
        i += 1  # skip the line with ):
    else:
        return impl  # No def line found, return as-is

    # Skip blank lines after def
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Skip docstring if present
    if i < len(lines):
        stripped = lines[i].strip()
        for quote in ('"""', "'''"):
            if stripped.startswith(quote):
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    # Single-line docstring like """some text"""
                    i += 1
                else:
                    # Multi-line docstring — find closing quotes
                    i += 1
                    while i < len(lines) and quote not in lines[i]:
                        i += 1
                    if i < len(lines):
                        i += 1  # skip closing quote line
                break

    return "\n".join(lines[i:])


def _apply_solutions(exercise_text: str, solution_response: str) -> str:
    """Apply solution implementations to an exercise template.

    Parses ``FUNCTION: name / ===CODE=== / def ...`` blocks from solution_response,
    strips the function signature/docstring (already in the template), and replaces
    the corresponding START/END CODE stubs in exercise_text with just the body.

    Args:
        exercise_text: Raw exercise LLM response (===MARKDOWN=== / ===CODE=== format).
        solution_response: Solution LLM response with FUNCTION: blocks.

    Returns:
        Modified exercise_text with stubs replaced by implementations.
    """
    func_pattern = re.compile(
        r"FUNCTION:\s*(\w+)\s*\n===CODE===\n(.*?)(?=\nFUNCTION:|\Z)",
        re.DOTALL,
    )
    result = exercise_text
    for match in func_pattern.finditer(solution_response):
        func_name = match.group(1)
        impl = match.group(2).strip()
        body = _strip_func_header(impl)

        func_loc = result.find(f"def {func_name}(")
        if func_loc < 0:
            continue
        after = result[func_loc:]
        stub_pattern = re.compile(
            r"[ \t]*### START CODE HERE ###\n.*?[ \t]*### END CODE HERE ###",
            re.DOTALL,
        )
        after_replaced = stub_pattern.sub(body, after, count=1)
        result = result[:func_loc] + after_replaced

    return result
