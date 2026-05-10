"""RepoTool: clone a GitHub repo and provide controlled file access."""

import ast
import fnmatch
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", ".github", ".gitlab", "__pycache__", "node_modules",
    "migrations", "alembic", ".pytest_cache", "dist", "build",
    ".eggs", ".tox", "venv", ".venv", "env", ".mypy_cache",
    ".ruff_cache", "site-packages",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".lock", ".bin", ".whl", ".pyc",
    ".pyo", ".so", ".dylib", ".dll", ".exe", ".db", ".sqlite",
    ".DS_Store", ".parquet", ".npy", ".npz", ".h5", ".ckpt",
    ".safetensors", ".pt", ".pth",
}


class RepoTool:
    """Clone a GitHub repo and expose controlled file access to agents."""

    def __init__(self, url: str, work_dir: str | None = None) -> None:
        self.url = url
        self.work_dir = work_dir or os.path.join(os.getcwd(), "work")
        self.repo_path: str | None = None

    def clone(self) -> str:
        """Shallow clone the repo. Reuses existing clone. Returns abs path."""
        repo_name = self.url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target = os.path.join(self.work_dir, repo_name)
        if os.path.isdir(os.path.join(target, ".git")):
            logger.info("Reusing existing clone at %s", target)
            self.repo_path = target
            return target

        os.makedirs(self.work_dir, exist_ok=True)
        logger.info("Cloning %s → %s", self.url, target)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.url, target],
            check=True,
            capture_output=True,
        )
        self.repo_path = target
        return target

    # ── Tools used by ExplorerAgent ──────────────────────────────────────────

    def list_dir(self, rel_path: str = "") -> str:
        """List directory contents, skipping junk dirs and binary extensions."""
        assert self.repo_path, "Call clone() first"
        abs_path = os.path.join(self.repo_path, rel_path) if rel_path else self.repo_path

        if not os.path.exists(abs_path):
            return f"Path not found: {rel_path}"
        if not os.path.isdir(abs_path):
            return f"Not a directory: {rel_path}"

        entries = sorted(os.scandir(abs_path), key=lambda e: (not e.is_dir(), e.name))
        display_path = rel_path or "/"
        lines = [f"Contents of {display_path}:"]

        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRS:
                    continue
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                ext = Path(entry.name).suffix.lower()
                if ext in SKIP_EXTENSIONS:
                    continue
                size = entry.stat().st_size
                if size >= 1024:
                    size_str = f"{size // 1024} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
                else:
                    size_str = f"{size} B"
                lines.append(f"  [FILE] {entry.name}  ({size_str})")

        return "\n".join(lines)

    def read_file(self, rel_path: str, max_chars: int = 8000) -> str:
        """Read file content, truncating at max_chars."""
        assert self.repo_path, "Call clone() first"
        abs_path = os.path.join(self.repo_path, rel_path)

        if not os.path.exists(abs_path):
            return f"File not found: {rel_path}"

        ext = Path(rel_path).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            return f"Binary file skipped: {rel_path}"

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
            return f"Error reading {rel_path}: {exc}"

        total = len(content)
        if total > max_chars:
            content = content[:max_chars]
            return f"=== {rel_path} ===\n{content}\n[Truncated: showing {max_chars:,} of {total:,} chars]"
        return f"=== {rel_path} ===\n{content}"

    def search_code(
        self, keyword: str, file_pattern: str = "*.py", max_results: int = 20
    ) -> str:
        """Case-insensitive recursive search returning matching lines."""
        assert self.repo_path, "Call clone() first"
        results: list[str] = []
        keyword_lower = keyword.lower()

        for root, dirs, files in os.walk(self.repo_path):
            # Prune skip dirs in-place
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_pattern):
                    continue
                ext = Path(fname).suffix.lower()
                if ext in SKIP_EXTENSIONS:
                    continue
                abs_file = os.path.join(root, fname)
                rel_file = os.path.relpath(abs_file, self.repo_path)
                try:
                    with open(abs_file, encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if keyword_lower in line.lower():
                                results.append(f"{rel_file}:{lineno}  {line.rstrip()}")
                                if len(results) >= max_results:
                                    break
                except OSError:
                    pass
                if len(results) >= max_results:
                    break

        if not results:
            return f"Search '{keyword}': no results"
        header = f"Search '{keyword}' ({len(results)} results):"
        return header + "\n" + "\n".join(results)

    # ── Helpers for generators ────────────────────────────────────────────────

    def get_priority_files(self) -> list[str]:
        """Return up to 12 high-value files for understanding architecture."""
        assert self.repo_path, "Call clone() first"
        root = Path(self.repo_path)
        found: list[str] = []

        # 1. README files
        for name in ("README.md", "README.rst", "README.txt", "readme.md"):
            p = root / name
            if p.exists():
                found.append(name)
                break

        # 2. Architecture docs
        for name in ("ARCHITECTURE.md", "DESIGN.md", "OVERVIEW.md",
                      "architecture.md", "design.md", "overview.md"):
            p = root / name
            if p.exists():
                found.append(name)

        # 3. Examples / tutorials (up to 3 each)
        for dir_name in ("examples", "example", "tutorials", "cookbook", "demo"):
            dir_path = root / dir_name
            if not dir_path.is_dir():
                continue
            count = 0
            for fpath in sorted(dir_path.rglob("*")):
                if fpath.suffix in (".py", ".ipynb") and fpath.is_file():
                    found.append(str(fpath.relative_to(root)))
                    count += 1
                    if count >= 3:
                        break

        # 4. Top-level __init__.py files (sorted by depth, max 6)
        init_files = sorted(root.rglob("__init__.py"), key=lambda p: len(p.parts))
        count = 0
        for init in init_files:
            rel = str(init.relative_to(root))
            if not any(part in SKIP_DIRS for part in init.parts):
                found.append(rel)
                count += 1
                if count >= 6:
                    break

        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for f in found:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result[:12]

    def extract_symbols(self, rel_path: str, symbols: list[str]) -> str:
        """
        AST-based extraction of specific class/function definitions.

        Supports "ClassName", "ClassName.method", "function_name".
        Falls back to read_file() for non-Python or parse failures.
        """
        assert self.repo_path, "Call clone() first"
        abs_path = os.path.join(self.repo_path, rel_path)

        if not rel_path.endswith(".py"):
            return self.read_file(rel_path)

        if not os.path.exists(abs_path):
            return f"File not found: {rel_path}"

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return self.read_file(rel_path)

        parts: list[str] = [f"=== {rel_path} ===\n"]

        for symbol in symbols:
            if "." in symbol:
                class_name, method_name = symbol.split(".", 1)
                extracted = _extract_method(tree, source, class_name, method_name)
                if extracted:
                    parts.append(f"# === {symbol} ===\n{extracted}\n")
                else:
                    parts.append(f"# {symbol} not found\n")
            else:
                extracted = _extract_top_level(tree, source, symbol)
                if extracted:
                    parts.append(f"# === {symbol} ===\n{extracted}\n")
                else:
                    parts.append(f"# {symbol} not found\n")

        return "\n".join(parts)

    def read_files_for_concept(
        self,
        file_paths: list[str],
        symbols: list[str] | None = None,
        max_total_chars: int = 6000,
    ) -> str:
        """Read multiple files with a shared char budget."""
        use_symbols = bool(symbols)
        chunks: list[str] = []
        remaining = max_total_chars

        for rel_path in file_paths:
            if remaining <= 0:
                break
            if use_symbols:
                chunk = self.extract_symbols(rel_path, symbols)  # type: ignore[arg-type]
            else:
                chunk = self.read_file(rel_path, max_chars=remaining)
            chunks.append(chunk)
            remaining -= len(chunk)

        return "\n\n".join(chunks)


# ── AST helpers ───────────────────────────────────────────────────────────────

def _extract_top_level(tree: ast.AST, source: str, name: str) -> str | None:
    """Extract a top-level class or function by name."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                seg = ast.get_source_segment(source, node)
                if seg:
                    return seg
    return None


def _extract_method(
    tree: ast.AST, source: str, class_name: str, method_name: str
) -> str | None:
    """Extract a method from a named class."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == method_name:
                        seg = ast.get_source_segment(source, child)
                        if seg:
                            return seg
    return None
