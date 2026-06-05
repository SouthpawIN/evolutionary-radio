"""
omni_client.py — HTTP client for OmniStep via llama-server.

Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint.
OmniStep takes a vibe string and produces an ACE-Step tag string.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

log = logging.getLogger("radio.omni_client")

DEFAULT_URL = "http://localhost:PORT"
DEFAULT_TIMEOUT = 30


class OmniClient:
    """Synchronous HTTP client for OmniStep (llama-server)."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = "omnisenter-6b",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            log.error("OmniStep request failed: %s", e)
            raise RuntimeError(f"OmniStep unreachable at {self.base_url}: {e}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error("OmniStep bad response: %s", e)
            raise RuntimeError(f"OmniStep returned unexpected response: {e}") from e

    def generate_tags(
        self,
        vibe: str,
        system_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """High-level: vibe → ACE-Step tag string.

        Uses the prompt_template module for message formatting.
        """
        # Lazy import to avoid circular deps
        from prompt_template import PromptTemplate

        tmpl = PromptTemplate(system_prompt=system_prompt, model=self.model)
        user_msg = tmpl.build_user_message(vibe)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        raw = self.chat(messages, temperature=temperature)
        return tmpl.parse(raw)

    def health_check(self) -> bool:
        """Check if the server is reachable."""
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
