"""Tests for llm/cost.py — estimation, actual cost, and formatting."""

import logging

import pytest

from llm.cost import (
    TOKEN_COSTS,
    compute_actual_cost,
    estimate_cost,
    format_cost_report,
)


# ── estimate_cost ─────────────────────────────────────────────────────────────

def test_estimate_cost_scales_with_notebook_count():
    """More notebooks → higher estimated cost."""
    findings = {"concepts": []}
    small = {"notebooks": [{"id": "01"}]}
    large = {"notebooks": [{"id": f"{i:02d}"} for i in range(1, 9)]}
    model = "claude-sonnet-4-20250514"
    assert estimate_cost(findings, large, model)["total_usd"] > \
           estimate_cost(findings, small, model)["total_usd"]


def test_estimate_cost_breakdown_has_three_categories():
    """Breakdown includes notebooks, exercises, and capstone keys."""
    findings = {"concepts": []}
    curriculum = {"notebooks": [{"id": "01"}, {"id": "02"}]}
    result = estimate_cost(findings, curriculum, "claude-sonnet-4-20250514")
    assert set(result["breakdown"].keys()) == {"notebooks", "exercises", "capstone"}


def test_estimate_cost_uses_default_for_unknown_model():
    """Unknown model falls back to default rates without raising."""
    findings = {"concepts": []}
    curriculum = {"notebooks": [{"id": "01"}]}
    result = estimate_cost(findings, curriculum, "unknown-model-xyz")
    assert result["total_usd"] > 0


def test_estimate_cost_tokens_match_breakdown_sum():
    """Total USD equals sum of breakdown values."""
    findings = {"concepts": []}
    curriculum = {"notebooks": [{"id": "01"}, {"id": "02"}, {"id": "03"}]}
    result = estimate_cost(findings, curriculum, "gpt-4o")
    expected = sum(result["breakdown"].values())
    assert abs(result["total_usd"] - expected) < 1e-9


# ── compute_actual_cost ───────────────────────────────────────────────────────

def _make_state(model: str, explore_in: int = 0, explore_out: int = 0,
                notebooks_in: int = 0, notebooks_out: int = 0) -> dict:
    return {
        "config_snapshot": {"default_model": model, "step_models": {}},
        "cost": {
            "explore_input_tokens": explore_in,
            "explore_output_tokens": explore_out,
            "curriculum_input_tokens": 0,
            "curriculum_output_tokens": 0,
            "notebooks_input_tokens": notebooks_in,
            "notebooks_output_tokens": notebooks_out,
            "capstone_input_tokens": 0,
            "capstone_output_tokens": 0,
        },
    }


def test_compute_actual_cost_known_model():
    """Actual cost is correctly computed from token counts for a known model."""
    state = _make_state("claude-sonnet-4-20250514", explore_in=1_000_000, explore_out=0)
    # 1M input tokens at $3.00/1M = $3.00
    assert abs(compute_actual_cost(state) - 3.0) < 1e-6


def test_compute_actual_cost_handles_unknown_model(caplog):
    """Returns 0 and logs a warning when model is not in TOKEN_COSTS."""
    state = _make_state("not-a-real-model", explore_in=1000, explore_out=500)
    with caplog.at_level(logging.WARNING, logger="llm.cost"):
        result = compute_actual_cost(state)
    assert result == 0.0
    assert any("unknown model" in r.message.lower() for r in caplog.records)


def test_compute_actual_cost_skips_zero_steps():
    """Steps with zero tokens contribute nothing to the total."""
    state = _make_state("claude-sonnet-4-20250514")  # all zeros
    assert compute_actual_cost(state) == 0.0


def test_compute_actual_cost_multi_step():
    """Costs accumulate correctly across multiple steps."""
    rates = TOKEN_COSTS["gpt-4o"]  # $2.50 in / $10.00 out
    state = _make_state("gpt-4o", explore_in=1_000_000, notebooks_out=1_000_000)
    expected = rates["input"] + rates["output"]  # $2.50 + $10.00
    assert abs(compute_actual_cost(state) - expected) < 1e-6


# ── format_cost_report ────────────────────────────────────────────────────────

def test_format_cost_report_shows_breakdown():
    """Report contains token counts and USD line."""
    state = _make_state("claude-sonnet-4-20250514", explore_in=5000, explore_out=2000)
    report = format_cost_report(state)
    assert "Input tokens" in report
    assert "Output tokens" in report
    assert "Actual USD" in report
    assert "5,000" in report  # formatted with comma


def test_format_cost_report_is_string():
    """format_cost_report always returns a string."""
    state = _make_state("claude-sonnet-4-20250514")
    assert isinstance(format_cost_report(state), str)
