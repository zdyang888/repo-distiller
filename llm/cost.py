"""Token cost estimation for supported models."""

# Cost per 1M tokens (USD). Source: provider pricing pages as of 2026-05.
# For tiered models (*) the standard tier (<=200K token context) is used.
TOKEN_COSTS: dict[str, dict[str, float]] = {
    # ── Anthropic Claude ──────────────────────────────────────────────────────
    "claude-sonnet-4-20250514": {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":   {"input": 15.0, "output": 75.0},
    "claude-haiku-4-20251001":  {"input": 0.8,  "output": 4.0},
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o":                   {"input": 5.0,  "output": 15.0},
    "gpt-4o-mini":              {"input": 0.15, "output": 0.6},
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
