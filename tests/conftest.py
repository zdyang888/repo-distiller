"""Shared test fixtures, including MockLLM."""

import os
from pathlib import Path

import pytest


class MockLLM:
    """Test double for LLMClient. Returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        """responses: list of strings, returned one per chat() call."""
        self.responses = list(responses)
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return "mock-model"

    @property
    def gen(self) -> dict:
        return {"max_explore_steps": 20, "max_notebooks": 8}

    def model_for_step(self, step: str) -> str:
        return "mock-model"

    @property
    def last_usage(self) -> dict:
        return {"input_tokens": 0, "output_tokens": 0, "model": "mock-model"}

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        self.calls.append({"messages": messages, "system": system, "model": model})
        if not self.responses:
            raise RuntimeError("MockLLM ran out of canned responses")
        return self.responses.pop(0)


@pytest.fixture
def sample_repo_path() -> Path:
    """Return path to the sample_repo fixture directory."""
    return Path(__file__).parent / "fixtures" / "sample_repo"
