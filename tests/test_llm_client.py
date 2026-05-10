"""Tests for llm/client.py — all LLM calls are mocked, no real API calls."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from llm.client import LLMClient


# ── Mock SDK factories ────────────────────────────────────────────────────────

def _anthropic_response(text: str, input_tokens: int = 10, output_tokens: int = 20):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def _openai_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _mock_anthropic_sdk(response=None, side_effect=None):
    """Return a fake `anthropic` module with a controllable Anthropic client."""
    sdk = MagicMock()
    create = sdk.Anthropic.return_value.messages.create
    if side_effect is not None:
        create.side_effect = side_effect
    elif response is not None:
        create.return_value = response
    return sdk


def _mock_openai_sdk(response=None, side_effect=None):
    """Return a fake `openai` module with a controllable OpenAI client."""
    sdk = MagicMock()
    create = sdk.OpenAI.return_value.chat.completions.create
    if side_effect is not None:
        create.side_effect = side_effect
    elif response is not None:
        create.return_value = response
    return sdk


def _client(model: str) -> LLMClient:
    c = LLMClient(config_path="nonexistent_config.yaml")
    c.model = model
    return c


# ── model_for_step ────────────────────────────────────────────────────────────

def test_model_for_step_returns_default_when_no_override(tmp_path):
    """With no step_models configured, model_for_step returns default_model."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("default_model: claude-sonnet-4-20250514\n")
    client = LLMClient(config_path=str(cfg))
    assert client.model_for_step("explore") == "claude-sonnet-4-20250514"
    assert client.model_for_step("capstone") == "claude-sonnet-4-20250514"


def test_model_for_step_returns_override_when_configured(tmp_path):
    """step_models overrides are returned for their specific step."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "default_model: claude-sonnet-4-20250514\n"
        "step_models:\n"
        "  explore: gemini-2.0-flash\n"
    )
    client = LLMClient(config_path=str(cfg))
    assert client.model_for_step("explore") == "gemini-2.0-flash"
    assert client.model_for_step("notebooks") == "claude-sonnet-4-20250514"


# ── Provider routing ──────────────────────────────────────────────────────────

def test_routing_claude_calls_anthropic():
    """A claude-* model name routes to the Anthropic SDK."""
    client = _client("claude-sonnet-4-20250514")
    fake_sdk = _mock_anthropic_sdk(response=_anthropic_response("hello"))

    with patch.dict(sys.modules, {"anthropic": fake_sdk}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"
    fake_sdk.Anthropic.assert_called_once_with(api_key="test-key")
    fake_sdk.Anthropic.return_value.messages.create.assert_called_once()


def test_routing_gpt_calls_openai():
    """A gpt-* model name routes to the OpenAI SDK without a custom base_url."""
    client = _client("gpt-4o")
    fake_sdk = _mock_openai_sdk(response=_openai_response("world"))

    with patch.dict(sys.modules, {"openai": fake_sdk}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "world"
    init_kwargs = fake_sdk.OpenAI.call_args.kwargs
    assert "base_url" not in init_kwargs
    assert init_kwargs["api_key"] == "test-key"


def test_routing_gemini_uses_compat_endpoint():
    """A gemini-* model name calls OpenAI SDK with the Gemini base_url."""
    client = _client("gemini-2.0-flash")
    fake_sdk = _mock_openai_sdk(response=_openai_response("gemini says hi"))

    with patch.dict(sys.modules, {"openai": fake_sdk}):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "gemini says hi"
    init_kwargs = fake_sdk.OpenAI.call_args.kwargs
    assert init_kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert init_kwargs["api_key"] == "test-key"


# ── Retry behaviour ───────────────────────────────────────────────────────────

def test_chat_retries_on_transient_error():
    """chat() retries on failure and succeeds on the third attempt."""
    client = _client("claude-sonnet-4-20250514")
    good_resp = _anthropic_response("success")
    fake_sdk = _mock_anthropic_sdk(
        side_effect=[RuntimeError("transient"), RuntimeError("transient"), good_resp]
    )

    with patch.dict(sys.modules, {"anthropic": fake_sdk}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("time.sleep"):
                result = client.chat([{"role": "user", "content": "hi"}], retries=3)

    assert result == "success"
    assert fake_sdk.Anthropic.return_value.messages.create.call_count == 3


# ── Missing API key ───────────────────────────────────────────────────────────

def test_missing_api_key_raises_envvar_error():
    """EnvironmentError is raised when the required API key env var is absent."""
    cases = [
        ("claude-sonnet-4-20250514", "ANTHROPIC_API_KEY", {"anthropic": _mock_anthropic_sdk()}),
        ("gpt-4o", "OPENAI_API_KEY", {"openai": _mock_openai_sdk()}),
        ("gemini-2.0-flash", "GOOGLE_API_KEY", {"openai": _mock_openai_sdk()}),
    ]
    for model, env_var, fake_modules in cases:
        client = _client(model)
        env = {k: v for k, v in os.environ.items() if k != env_var}
        with patch.dict(sys.modules, fake_modules):
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(EnvironmentError, match=env_var):
                    client.chat([{"role": "user", "content": "hi"}], retries=1)
