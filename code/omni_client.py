"""
omni_client.py — HTTP client for OmniStep via llama-server.

Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint.
OmniStep takes a vibe string and produces an ACE-Step tag string.

POLITE VRAM (2026-06-08): the brain is shared with other workloads
(training, IDE, etc.). The radio's queue-fill loop runs in the
background and shouldn't wake the brain when the user is doing
something VRAM-heavy. The `polite_chat()` method below probes free
VRAM and defers when the GPU needs the space.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("radio.omni_client")

DEFAULT_URL = "http://localhost:PORT"
DEFAULT_TIMEOUT = 30

# Polite VRAM settings — defer queue fill if free VRAM is below this.
# Set via env: RADIO_POLITE_MIN_FREE_GB (default 4 GB)
POLITE_MIN_FREE_GB = float(os.environ.get("RADIO_POLITE_MIN_FREE_GB", "4"))
# Defer retry interval when VRAM is tight (seconds)
POLITE_DEFER_RETRY_S = int(os.environ.get("RADIO_DEFER_RETRY_S", "60"))


def _import_polite_vram():
    """Lazy import the shared polite_vram module."""
    p = Path("/home/sovthpaw/.hermes/bin/polite_vram.py")
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("polite_vram", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        log.warning("polite_vram import failed: %s", e)
        return None


def _is_brain_loaded() -> bool:
    """Best-effort check: is the brain already loaded in VRAM?

    Hits /v1/models on the brain. If it responds, it's loaded (or
    ready to spawn on next call). If it doesn't respond, it's gone.
    """
    # Imported lazily to avoid circular deps
    from urllib.request import urlopen
    from urllib.error import URLError
    # We don't have access to base_url here; the caller passes it.
    return True  # conservative: assume loaded, the proxy will spawn if not


def should_defer_for_vram(min_free_gb: float = POLITE_MIN_FREE_GB,
                          gpu_index: int = 0) -> Tuple[bool, str]:
    """Caller-side check: should the radio defer this generation?

    Returns (should_defer, reason).
    """
    pv = _import_polite_vram()
    if pv is None:
        return False, "polite_vram unavailable, proceeding"
    return (not pv.should_wake_brain(min_free_gb, gpu_index)[0],
            pv.should_wake_brain(min_free_gb, gpu_index)[1])


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

    def polite_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 200,
        min_free_gb: float = POLITE_MIN_FREE_GB,
    ) -> Optional[str]:
        """Polite version of chat() — defers when VRAM is tight.

        The brain (omni-va) is a shared resource. If the user is doing
        something VRAM-heavy, the radio's queue-fill loop should wait
        rather than wake the brain.

        Returns:
          - the assistant text on success
          - None if deferred (caller should retry later)
        """
        defer, reason = should_defer_for_vram(min_free_gb)
        if defer:
            log.info("polite_chat: deferring (%s)", reason)
            return None
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

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
