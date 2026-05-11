"""Token cost estimation and actual cost tracking for supported models."""

import logging

logger = logging.getLogger(__name__)

# Cost per 1M tokens (USD). Source: provider pricing pages as of 2026-05.
# For tiered models (*) the standard tier (<=200K token context) is used.
TOKEN_COSTS: dict[str, dict[str, float]] = {
    # ── Anthropic Claude ──────────────────────────────────────────────────────
    "claude-sonnet-4-20250514": {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":   {"input": 15.0, "output": 75.0},
    "claude-haiku-4-20251001":  {"input": 0.8,  "output": 4.0},
    # ── OpenAI GPT-5.x flagship ──────────────────────────────────────────────
    "gpt-5.5":                  {"input": 5.0,   "output": 30.0},
    "gpt-5.5-pro":              {"input": 30.0,  "output": 180.0},
    "gpt-5.4":                  {"input": 2.5,   "output": 15.0},
    "gpt-5.4-mini":             {"input": 0.75,  "output": 4.5},
    "gpt-5.4-nano":             {"input": 0.20,  "output": 1.25},
    "gpt-5.4-pro":              {"input": 30.0,  "output": 180.0},
    "gpt-5.3-codex":            {"input": 1.75,  "output": 14.0},
    "gpt-5.2":                  {"input": 1.75,  "output": 14.0},
    "gpt-5.2-pro":              {"input": 21.0,  "output": 168.0},
    "gpt-5.1":                  {"input": 1.25,  "output": 10.0},
    "gpt-5":                    {"input": 1.25,  "output": 10.0},
    "gpt-5-mini":               {"input": 0.25,  "output": 2.0},
    "gpt-5-nano":               {"input": 0.05,  "output": 0.4},
    "gpt-5-pro":                {"input": 15.0,  "output": 120.0},
    # ── OpenAI GPT-4.1 ───────────────────────────────────────────────────────
    "gpt-4.1":                  {"input": 2.0,   "output": 8.0},
    "gpt-4.1-mini":             {"input": 0.4,   "output": 1.6},
    "gpt-4.1-nano":             {"input": 0.1,   "output": 0.4},
    # ── OpenAI GPT-4o ────────────────────────────────────────────────────────
    "gpt-4o":                   {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.6},
    # ── OpenAI o-series (reasoning) ──────────────────────────────────────────
    "o4-mini":                  {"input": 1.1,   "output": 4.4},
    "o3":                       {"input": 2.0,   "output": 8.0},
    "o3-mini":                  {"input": 1.1,   "output": 4.4},
    "o3-pro":                   {"input": 20.0,  "output": 80.0},
    "o1":                       {"input": 15.0,  "output": 60.0},
    "o1-mini":                  {"input": 1.1,   "output": 4.4},
    "o1-pro":                   {"input": 150.0, "output": 600.0},
    # ── Google Gemini 3.x ─────────────────────────────────────────────────────
    # * tiered: <=200K context; >200K is $4.00 in / $18.00 out
    "gemini-3.1-pro-preview":   {"input": 2.0,  "output": 12.0},
    "gemini-3.1-flash-lite":    {"input": 0.25, "output": 1.5},
    "gemini-3-flash-preview":   {"input": 0.5,  "output": 3.0},
    # ── Google Gemini 2.5 ─────────────────────────────────────────────────────
    # * tiered: <=200K context; >200K is $2.50 in / $15.00 out
    "gemini-2.5-pro":           {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash":         {"input": 0.3,  "output": 2.5},
    "gemini-2.5-flash-lite":    {"input": 0.1,  "output": 0.4},
    # ── Google Gemini 2.0 ─────────────────────────────────────────────────────
    "gemini-2.0-flash":         {"input": 0.1,  "output": 0.4},
    "gemini-2.0-flash-lite":    {"input": 0.1,  "output": 0.4},
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
    costs = TOKEN_COSTS.get(model, _DEFAULT_TOKEN_COST)

    # Heuristic token estimates (per spec section 6.8)
    # Teaching notebooks: ~3000 in + 4000 out each
    # Exercises:          ~2000 in + 3500 out each
    # Capstone:           ~4 LLM calls × 3000/3000 tokens
    notebooks_in = n_notebooks * 3000
    notebooks_out = n_notebooks * 4000
    exercises_in = n_notebooks * 2000
    exercises_out = n_notebooks * 3500
    capstone_in = 4 * 3000
    capstone_out = 4 * 3000

    total_in = notebooks_in + exercises_in + capstone_in
    total_out = notebooks_out + exercises_out + capstone_out

    notebooks_usd = (notebooks_in * costs["input"] + notebooks_out * costs["output"]) / 1_000_000
    exercises_usd = (exercises_in * costs["input"] + exercises_out * costs["output"]) / 1_000_000
    capstone_usd = (capstone_in * costs["input"] + capstone_out * costs["output"]) / 1_000_000

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_usd": notebooks_usd + exercises_usd + capstone_usd,
        "breakdown": {
            "notebooks": notebooks_usd,
            "exercises": exercises_usd,
            "capstone": capstone_usd,
        },
    }


def compute_actual_cost(state: dict) -> float:
    """Compute total actual USD cost from accumulated token counts in state.

    Uses TOKEN_COSTS to price the tokens logged per step. Falls back to 0 and
    warns when the model is unknown.

    Args:
        state: DistillState dict with cost and config_snapshot keys.

    Returns:
        Total cost in USD as a float.
    """
    cost = state.get("cost", {})
    config_snap = state.get("config_snapshot", {})
    step_models = config_snap.get("step_models", {})
    default_model = config_snap.get("default_model", "")

    total = 0.0
    for step in ("explore", "curriculum", "notebooks", "capstone"):
        in_tok = cost.get(f"{step}_input_tokens", 0)
        out_tok = cost.get(f"{step}_output_tokens", 0)
        if in_tok == 0 and out_tok == 0:
            continue
        model = step_models.get(step, default_model)
        rates = TOKEN_COSTS.get(model)
        if rates is None:
            logger.warning("compute_actual_cost: unknown model %r for step %s, skipping", model, step)
            continue
        total += (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000

    return total


def format_cost_report(state: dict) -> str:
    """Return a human-readable cost summary string for end-of-run printing.

    Args:
        state: DistillState dict.

    Returns:
        Multi-line string with token counts and USD estimate.
    """
    cost = state.get("cost", {})
    total_in = sum(v for k, v in cost.items() if "input_tokens" in k)
    total_out = sum(v for k, v in cost.items() if "output_tokens" in k)
    actual_usd = compute_actual_cost(state)
    return "\n".join([
        "Cost so far:",
        f"  Input tokens:  {total_in:,}",
        f"  Output tokens: {total_out:,}",
        f"  Actual USD:    ${actual_usd:.4f}",
    ])
