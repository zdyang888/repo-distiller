"""Unified LLM client routing to Claude / OpenAI / Gemini."""

import logging
import os
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "default_model": "claude-sonnet-4-20250514",
    "step_models": {},
    "generation": {"max_explore_steps": 20, "max_notebooks": 8},
}


class LLMClient:
    """Single entry point for LLM calls. Routes by model name."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config = dict(_DEFAULT_CONFIG)
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with cfg_file.open() as f:
                loaded = yaml.safe_load(f) or {}
            self._config.update(loaded)
            # Merge nested dicts
            if "generation" in loaded:
                self._config["generation"] = {
                    **_DEFAULT_CONFIG["generation"],
                    **loaded["generation"],
                }
            if "step_models" in loaded:
                self._config["step_models"] = loaded["step_models"] or {}

        self._model: str = self._config.get("default_model", "claude-sonnet-4-20250514")
        self._last_usage: dict = {}

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def gen(self) -> dict:
        """Return the generation section of config."""
        return self._config.get("generation", _DEFAULT_CONFIG["generation"])

    def model_for_step(self, step: str) -> str:
        """Return step-specific model or default."""
        step_models = self._config.get("step_models") or {}
        return step_models.get(step, self._model)

    @property
    def last_usage(self) -> dict:
        """Returns {"input_tokens": int, "output_tokens": int, "model": str}."""
        return self._last_usage

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        """Call the LLM and return assistant text. Tracks tokens via last_usage."""
        effective_model = model or self._model
        last_exc: Exception | None = None

        for attempt in range(retries):
            try:
                if "claude" in effective_model:
                    response_text, usage = self._call_anthropic(
                        messages, system, effective_model, max_tokens
                    )
                elif any(p in effective_model for p in ("gpt-", "o1", "o3", "o4")):
                    response_text, usage = self._call_openai(
                        messages, system, effective_model, max_tokens
                    )
                elif "gemini" in effective_model:
                    response_text, usage = self._call_gemini(
                        messages, system, effective_model, max_tokens
                    )
                else:
                    raise ValueError(f"Unknown model: {effective_model!r}")

                self._last_usage = {**usage, "model": effective_model}
                return response_text

            except Exception as exc:
                last_exc = exc
                wait = 5 * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries, exc, wait,
                )
                if attempt < retries - 1:
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ── Provider implementations ──────────────────────────────────────────────

    def _call_anthropic(
        self,
        messages: list[dict],
        system: str | None,
        model: str,
        max_tokens: int,
    ) -> tuple[str, dict]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install with: pip install anthropic")

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=key)
        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        text = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage

    def _call_openai(
        self,
        messages: list[dict],
        system: str | None,
        model: str,
        max_tokens: int,
    ) -> tuple[str, dict]:
        try:
            import openai
        except ImportError:
            raise ImportError("Install with: pip install openai")

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=key)
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        text = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        return text, usage

    def _call_gemini(
        self,
        messages: list[dict],
        system: str | None,
        model: str,
        max_tokens: int,
    ) -> tuple[str, dict]:
        try:
            import openai
        except ImportError:
            raise ImportError("Install with: pip install openai")

        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("GOOGLE_API_KEY not set")

        client = openai.OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        text = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        return text, usage
