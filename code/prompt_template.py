"""
prompt_template.py — OmniStep prompt → ACE-Step tag string.

The OmniStep "brain" is a text LLM. It receives a user vibe (e.g. "chill lofi")
and produces an ACE-Step tag string — a comma-separated list of style/mood
tags that the voice synthesizer understands.

This module owns the *shape* of that exchange:
- ``build_user_prompt(vibe)``        : format the vibe into a chat-completions user message
- ``parse_assistant_response(text)`` : extract the tag string from the LLM's reply
- ``PromptTemplate``                  : convenience class that holds the system prompt

It does NOT call OmniStep itself — that is ``omni_client.py``. Keeping these
separate makes the template testable without a network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Default tag seed bank — used when OmniStep is unreachable, so the radio
# can still produce something coherent from a bare vibe string.
VIBE_SEEDS: dict[str, str] = {
    "chill lofi":      "lofi, chill, 80bpm, warm pads, dusty drums, vinyl hiss, mellow bass, relaxed",
    "aggressive metal":"metal, aggressive, 160bpm, distorted guitars, double kick, dark, intense",
    "dreamy ambient":  "ambient, dreamy, 60bpm, lush pads, reverb, ethereal, slow, atmospheric",
    "energetic edm":   "edm, energetic, 128bpm, synth lead, sidechain, punchy, euphoric",
    "jazz":            "jazz, smooth, 110bpm, brush drums, upright bass, sax, warm, intimate",
    "classical":       "classical, orchestral, 90bpm, strings, piano, cinematic, dynamic",
    "default":         "instrumental, balanced, 100bpm, warm, mid-tempo, mix of synth and acoustic",
}

_TAG_SPLIT_RE = re.compile(r"[,\n;|]+")


def _resolve_seed(vibe: str) -> str:
    """Look up a seed tag string from a bare vibe; case-insensitive."""
    key = vibe.strip().lower()
    return VIBE_SEEDS.get(key, VIBE_SEEDS["default"])


def build_user_prompt(vibe: str) -> str:
    """Format the user's vibe into the user-message text sent to OmniStep.

    We keep this short and concrete — OmniStep is asked to *transform*, not
    to *imagine from scratch*, which is more reliable at low temperature.
    """
    seed = _resolve_seed(vibe)
    return (
        f"Vibe: {vibe}\n"
        f"Seed tags: {seed}\n"
        "Output a refined comma-separated ACE-Step tag string, ≤ 200 chars. "
        "Tags only — no commentary, no quotes."
    )


def parse_assistant_response(text: str) -> str:
    """Extract the tag string from OmniStep's reply.

    OmniStep sometimes returns extra whitespace, code-fence markers, or even
    a single line of preamble. We strip those defensively and return the
    first non-empty line that looks like comma-separated tags.
    """
    if not text:
        return _resolve_seed("default")

    cleaned_lines = []
    for raw in text.splitlines():
        # Strip code fences and bullet markers
        line = raw.strip().strip("`").lstrip("-*•").strip()
        if not line:
            continue
        cleaned_lines.append(line)

    candidate = cleaned_lines[0] if cleaned_lines else ""

    # If the line has no commas and no spaces, it's likely garbage — fall back.
    if "," not in candidate and " " not in candidate and len(candidate) > 30:
        return _resolve_seed("default")

    # Normalize whitespace around tags
    parts = [p.strip().lower() for p in _TAG_SPLIT_RE.split(candidate) if p.strip()]
    return ", ".join(parts)


@dataclass
class PromptTemplate:
    """Holds the system prompt and a reference to the OmniStep model name.

    Usage::

        tmpl = PromptTemplate(system_prompt="...", model="omnisenter-6b")
        user_msg = tmpl.build_user_message("chill lofi")
        tags = tmpl.parse("lofi, chill, 80bpm, warm pads")
    """

    system_prompt: str
    model: str = "omnisenter-6b"

    def build_user_message(self, vibe: str) -> str:
        return build_user_prompt(vibe)

    def parse(self, assistant_text: str) -> str:
        return parse_assistant_response(assistant_text)


__all__ = [
    "PromptTemplate",
    "VIBE_SEEDS",
    "build_user_prompt",
    "parse_assistant_response",
]
