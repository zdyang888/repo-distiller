"""Structured logging: console (INFO+) and rotating file (DEBUG+)."""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", output_dir: Path | None = None) -> None:
    """Configure root logger with a console handler and a file handler.

    - Console handler: user-specified level (default INFO), minimal format.
    - File handler: DEBUG level, timestamped format. Captures everything.

    Idempotent — clears existing handlers before adding new ones.

    Args:
        level: Console log level string ("DEBUG", "INFO", "WARNING").
        output_dir: Directory for distill.log. Defaults to current directory.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers so this is safe to call multiple times
    root.handlers.clear()

    # Console: clean output for users, respects --log-level
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # File: full debug detail for troubleshooting
    log_path = (output_dir or Path(".")) / "distill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_h)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Convenience wrapper around logging.getLogger."""
    return logging.getLogger(name)
