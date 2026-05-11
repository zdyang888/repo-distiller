"""Subprocess-based pytest runner for capstone validation."""

import subprocess
import tempfile
from pathlib import Path


def run_pytest(
    interfaces_content: str,
    tests_content: str,
    reference_impl: str,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Run test_capstone.py against a reference implementation in an isolated tempdir.

    Writes interfaces.py, implementation.py, and test_capstone.py into a fresh
    temporary directory, then invokes pytest in a subprocess. The reference
    implementation is never written to the final output directory.

    Args:
        interfaces_content: Python source for interfaces.py.
        tests_content: Python source for test_capstone.py.
        reference_impl: Python source for implementation.py (reference only).
        timeout: Maximum seconds to allow pytest to run before killing it.

    Returns:
        (passed, output) — passed is True iff pytest exit code is 0.
        output contains combined stdout + stderr from pytest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "interfaces.py").write_text(interfaces_content, encoding="utf-8")
        (tmp / "implementation.py").write_text(reference_impl, encoding="utf-8")
        (tmp / "test_capstone.py").write_text(tests_content, encoding="utf-8")
        # Ensure the temp dir is on sys.path so imports resolve
        (tmp / "conftest.py").write_text(
            "import sys; sys.path.insert(0, '.')", encoding="utf-8"
        )

        try:
            result = subprocess.run(
                [
                    "python", "-m", "pytest", "test_capstone.py",
                    "-v", "--tb=short", "--no-header", "-x",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + "\n" + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Validation timed out (exceeded {timeout}s)"
