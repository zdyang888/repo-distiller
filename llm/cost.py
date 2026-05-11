"""Token cost estimation for supported models."""

# Cost per 1M tokens (USD). Update as pricing changes.
TOKEN_COSTS: dict[str, dict[str, float]] = {
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
