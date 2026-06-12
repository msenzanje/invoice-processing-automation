"""
LLM client wrapper — the only module that talks to a provider API directly.

Agents call ``get_llm_client().complete(prompt)`` and receive raw text back; they
have no knowledge of which provider, model, or credentials are in play. This
isolation is what makes the ingestion self-correction loop (which can call the
LLM up to three times in sequence) both testable and provider-swappable: changing
providers is a one-line change here, not a refactor across the codebase.

Provider selection follows PROJECT_CONTEXT.md: xAI Grok is primary, Anthropic
Claude is the fallback when no xAI key is configured. Everything is env-driven.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)

XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))


class LLMProtocol(Protocol):
    """The single method every LLM client (real or mock) must provide."""

    def complete(self, prompt: str) -> str:
        ...


class LLMClient:
    """Thin wrapper over the configured provider's completion endpoint."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or self._auto_provider()).lower()
        self.timeout = timeout
        if self.provider == "grok":
            self.api_key = api_key or os.environ.get("XAI_API_KEY")
            self.model = model or os.environ.get("XAI_MODEL", "grok-3")
        elif self.provider == "claude":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider!r}")
        if not self.api_key:
            raise ValueError(
                f"No API key configured for provider {self.provider!r}. "
                "Set XAI_API_KEY (or ANTHROPIC_API_KEY for the Claude fallback)."
            )

    @staticmethod
    def _auto_provider() -> str:
        """Grok is primary; fall back to Claude only if no xAI key is present."""
        if os.environ.get("XAI_API_KEY"):
            return "grok"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "claude"
        return "grok"

    def complete(self, prompt: str) -> str:
        """Send ``prompt`` to the configured model and return the raw text reply."""
        logger.debug(
            "LLM request: provider=%s model=%s prompt_chars=%d",
            self.provider, self.model, len(prompt),
        )
        if self.provider == "grok":
            text = self._complete_openai_compatible(prompt)
        else:
            text = self._complete_anthropic(prompt)
        logger.debug("LLM response: chars=%d", len(text))
        return text

    def _complete_openai_compatible(self, prompt: str) -> str:
        """Call an OpenAI-compatible /chat/completions endpoint (xAI Grok)."""
        url = f"{XAI_BASE_URL}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _complete_anthropic(self, prompt: str) -> str:
        """Call the Anthropic Messages API (Claude fallback)."""
        url = f"{ANTHROPIC_BASE_URL}/messages"
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


def get_llm_client() -> LLMClient:
    """Factory used by agents. Construction validates that a key is configured."""
    return LLMClient()


class MockLLMClient:
    """Test double for :class:`LLMClient` — never touches the network.

    Pass ``responses`` to script successive ``.complete()`` return values (e.g.
    invalid JSON twice then valid JSON, to exercise the self-correction loop). The
    last scripted response is repeated if ``.complete()`` is called more times than
    there are responses. With no ``responses`` it returns a single valid payload.
    """

    DEFAULT_RESPONSE = (
        '{"vendor": "Mock Vendor", "amount": 100.0, '
        '"items": [{"item": "MockItem", "quantity": 1, "unit_price": 100.0}], '
        '"due_date": "2026-01-01", "invoice_id": "MOCK-001"}'
    )

    def __init__(self, responses: Optional[Sequence[str]] = None) -> None:
        self._responses = list(responses) if responses is not None else None
        self._calls = 0

    def complete(self, prompt: str) -> str:
        self._calls += 1
        if self._responses is None:
            return self.DEFAULT_RESPONSE
        index = min(self._calls - 1, len(self._responses) - 1)
        return self._responses[index]

    @property
    def call_count(self) -> int:
        return self._calls


# Convenience module-level instance for quick imports in tests.
mock_llm_client = MockLLMClient()
