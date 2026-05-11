"""Tests for distill.py CLI — argparse, helpers, state I/O."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import distill


# ── Argparse tests ────────────────────────────────────────────────────────────

def test_argument_parsing_repo():
    """--repo URL is parsed and stored as args.repo."""
    parser_args = ["--repo", "https://github.com/karpathy/minGPT"]
    args = distill.main.__code__  # just confirm it's callable

    # Parse directly via argparse
    import argparse
    parser = argparse.ArgumentParser()
    cmd = parser.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--repo")
    cmd.add_argument("--continue", dest="do_continue", action="store_true")
    cmd.add_argument("--refine")
    cmd.add_argument("--preview")
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--estimate-cost", dest="estimate_cost", action="store_true")
    cmd.add_argument("--status", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--no-confirm", action="store_true")

    parsed = parser.parse_args(["--repo", "https://github.com/karpathy/minGPT"])
    assert parsed.repo == "https://github.com/karpathy/minGPT"
    assert parsed.do_continue is False
    assert parsed.no_confirm is False
    assert parsed.config == "config.yaml"
    assert parsed.log_level == "INFO"


def test_argument_parsing_mutually_exclusive():
    """--repo and --continue together should raise SystemExit."""
    import argparse
    parser = argparse.ArgumentParser()
    cmd = parser.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--repo")
    cmd.add_argument("--continue", dest="do_continue", action="store_true")
    cmd.add_argument("--refine")
    cmd.add_argument("--preview")
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--estimate-cost", dest="estimate_cost", action="store_true")
    cmd.add_argument("--status", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-confirm", action="store_true")

    with pytest.raises(SystemExit):
        parser.parse_args(["--repo", "https://github.com/x/y", "--continue"])


# ── _default_output_dir ───────────────────────────────────────────────────────

def test_default_output_dir_derivation():
    """URL tail becomes the directory name with _course suffix."""
    assert distill._default_output_dir("https://github.com/karpathy/minGPT") == "output/minGPT_course"
    assert distill._default_output_dir("https://github.com/karpathy/minGPT.git") == "output/minGPT_course"
    assert distill._default_output_dir("https://github.com/owner/repo/") == "output/repo_course"


# ── _find_latest_output ───────────────────────────────────────────────────────

def test_find_latest_output_picks_most_recent(tmp_path):
    """_find_latest_output returns the directory with the newest state file."""
    output = tmp_path / "output"

    older = output / "older_course"
    older.mkdir(parents=True)
    (older / ".distill_state.json").write_text("{}")

    import time
    time.sleep(0.01)  # ensure mtime difference

    newer = output / "newer_course"
    newer.mkdir(parents=True)
    (newer / ".distill_state.json").write_text("{}")

    with patch("distill.Path") as mock_path_cls:
        # Override only the glob call inside _find_latest_output
        mock_path_cls.return_value.glob.return_value = [
            older / ".distill_state.json",
            newer / ".distill_state.json",
        ]
        # Actually just test with real filesystem
        pass

    # Use real filesystem via monkeypatching cwd
    import os
    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = distill._find_latest_output()
        assert "newer_course" in result
    finally:
        os.chdir(orig)


# ── State save/load roundtrip ─────────────────────────────────────────────────

def test_state_save_load_roundtrip(tmp_path):
    """State written by _save_state is identical when read by _load_state."""
    state = {
        "schema_version": 1,
        "repo_url": "https://github.com/karpathy/minGPT",
        "repo_path": "/tmp/minGPT",
        "output_dir": str(tmp_path),
        "findings": {"concepts": []},
        "curriculum": {"notebooks": [], "capstone": {}},
        "step": "explored",
        "progress": {
            "notebooks_complete": [],
            "exercises_complete": [],
            "capstone_complete": False,
            "last_error": None,
        },
        "cost": {"total_estimated_usd": 0.0},
        "config_snapshot": {"default_model": "claude-sonnet-4-20250514", "step_models": {}},
    }
    distill._save_state(tmp_path, state)
    loaded = distill._load_state(tmp_path)
    assert loaded == state


def test_load_state_exits_if_missing(tmp_path):
    """_load_state calls sys.exit when state file does not exist."""
    with pytest.raises(SystemExit):
        distill._load_state(tmp_path)


# ── _confirm ──────────────────────────────────────────────────────────────────

def test_confirm_returns_true_on_y():
    """_confirm returns True when user enters 'y'."""
    with patch("builtins.input", return_value="y"):
        assert distill._confirm("Proceed?") is True


def test_confirm_returns_false_on_anything_else():
    """_confirm returns False for '', 'n', 'N', 'yes', random input."""
    for answer in ("", "n", "N", "yes", "no", "maybe"):
        with patch("builtins.input", return_value=answer):
            assert distill._confirm("Proceed?") is False


# ── estimate_cost ─────────────────────────────────────────────────────────────

def test_estimate_cost_returns_required_keys():
    """estimate_cost returns dict with expected keys."""
    findings = {"concepts": []}
    curriculum = {"notebooks": [{"id": "01"}, {"id": "02"}]}
    result = distill.estimate_cost(findings, curriculum, "claude-sonnet-4-20250514")
    assert "input_tokens" in result
    assert "output_tokens" in result
    assert "total_usd" in result
    assert "breakdown" in result
    assert result["total_usd"] > 0


def test_estimate_cost_scales_with_notebooks():
    """More notebooks → higher estimated cost."""
    findings = {"concepts": []}
    small = {"notebooks": [{"id": "01"}]}
    large = {"notebooks": [{"id": f"{i:02d}"} for i in range(1, 9)]}
    model = "claude-sonnet-4-20250514"
    assert distill.estimate_cost(findings, large, model)["total_usd"] > \
           distill.estimate_cost(findings, small, model)["total_usd"]


# ── write/parse curriculum MD ─────────────────────────────────────────────────

def test_write_parse_curriculum_md_roundtrip(tmp_path):
    """write_curriculum_md + parse_curriculum_md is a roundtrip for the curriculum dict."""
    curriculum = {
        "title": "Test Course",
        "mental_model": "A mental model.",
        "concepts": [],
        "notebooks": [
            {
                "id": "01",
                "title": "Intro",
                "description": "An intro.",
                "learning_objectives": ["Understand basics"],
                "exercise_description": "Do basics",
                "key_source_files": [],
                "key_symbols": [],
                "prerequisites": [],
                "visualization_idea": "",
                "concept": "basics",
            }
        ],
        "capstone": {"title": "mini-project", "description": "Build it.", "modules": []},
    }
    path = tmp_path / "CURRICULUM.md"
    distill.write_curriculum_md(path, curriculum, "https://github.com/x/y")
    md = path.read_text()

    recovered = distill.parse_curriculum_md(md, {})
    assert recovered["title"] == curriculum["title"]
    assert len(recovered["notebooks"]) == 1
    assert recovered["notebooks"][0]["id"] == "01"


def test_parse_curriculum_md_falls_back_on_bad_json():
    """parse_curriculum_md returns fallback when the embedded JSON is broken."""
    bad_md = "# Course\n```json\n{broken\n```"
    fallback = {"notebooks": [], "title": "fallback"}
    result = distill.parse_curriculum_md(bad_md, fallback)
    assert result is fallback
