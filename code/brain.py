#!/usr/bin/env python3
"""
brain.py — the local model as the radio's brain.

Per Chris's epiphany (2026-06-08): the Evolutionary Radio is a "desk
pet" that ALSO runs the local model server. The same local model
that drives the inference is the "brain" that:
  1. Generates music (the existing Loops 1 + 2 in radio.py)
  2. Maintains the user-idea LLM Wiki (the note-taker role)
  3. Curates templates in the vault
  4. Compacts the wiki to a Wikipedia summary for Hermes preload

This module wraps the brain so the radio can call it cleanly. It
sits in code/ alongside the other radio modules (omni_client,
gepa, darwin, etc.) and is the **only** place the radio talks to
the local model. Any future code that wants to ask the local model
should go through Brain (not OmniClient directly).

The Brain class is also the bridge to the **Gold Judge** role
(/home/sovthpaw/.hermes/bin/gold_judge.py) — but for now, the brain
IS the judge (same model, same endpoint).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("radio.brain")

# Polite VRAM env (also used by omni_client and the proxy)
POLITE_MIN_FREE_GB = float(os.environ.get("RADIO_POLITE_MIN_FREE_GB", "4"))

# Lazy import of the omni_client (existing in the radio)
def _import_omni_client():
    p = Path(__file__).resolve().parent / "omni_client.py"
    spec = importlib.util.spec_from_file_location("omni_client", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy import of the gold_judge (in ~/.hermes/bin/)
def _import_gold_judge():
    p = Path("/home/sovthpaw/.hermes/bin/gold_judge.py")
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("gold_judge", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy import of the wiki_manager (in ~/.hermes/bin/)
def _import_wiki_manager():
    p = Path("/home/sovthpaw/.hermes/bin/wiki_manager.py")
    spec = importlib.util.spec_from_file_location("wiki_manager", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Brain:
    """The local model as the radio's brain.

    Use the brain for:
      - Music generation (existing omni_client.chat)
      - Idea clustering (the note-taker)
      - Template curation
      - Wikipedia compaction
      - Any other LLM call

    The brain is also the Gold Judge (same model, same endpoint).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082/v1",
        model: str = "carnice-35a3b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._oc = None
        self._wj = None
        self._gj = None

    @property
    def omni(self):
        if self._oc is None:
            self._oc = _import_omni_client().OmniClient(
                base_url=self.base_url, model=self.model, timeout=self.timeout
            )
        return self._oc

    @property
    def wiki(self):
        if self._wj is None:
            self._wj = _import_wiki_manager()
        return self._wj

    @property
    def judge(self):
        if self._gj is None:
            self._gj = _import_gold_judge()
        return self._gj

    # ---- Music generation (wraps OmniClient) ----

    def chat(self, messages: list, **kwargs) -> str:
        """Polite chat (defers if VRAM too tight). Falls back to
        omni_client.chat if gold-judge is unavailable.
        """
        defer = _import_omni_client().should_defer_for_vram(POLITE_MIN_FREE_GB)
        if defer[0]:
            log.info("brain: deferring chat (%s)", defer[1])
            return ""  # caller decides what to do
        return self.omni.chat(messages, **kwargs)

    def generate_music_tags(self, vibe: str, system_prompt: str) -> str:
        """Convenience: vibe → ACE-Step tag string (the radio's main use)."""
        return self.omni.generate_tags(vibe, system_prompt)

    # ---- Note-taker: maintain the wiki ----

    def curate_event(self, event: dict, related: list = None) -> Optional[dict]:
        """Take a raw event and decide if it becomes a wiki entry.

        Steps:
          1. Politely check VRAM
          2. Ask the brain to classify (new idea? update existing? skip?)
          3. If new idea, ask for slug + title + body + tags + importance
          4. Write to the wiki via wiki_manager

        The optional `related` list (from semantic search of the wiki) is
        included in the classification prompt so the brain can avoid
        creating duplicates and instead update the existing entry.

        Returns the written entry's slug, or None if deferred / skipped.
        """
        if not self.judge:
            log.warning("brain.curate_event: gold_judge unavailable, can't curate")
            return None

        # 1. Polite VRAM check
        defer = _import_omni_client().should_defer_for_vram(POLITE_MIN_FREE_GB)
        if defer[0]:
            log.info("brain.curate_event: deferring (%s)", defer[1])
            return None

        # 2. Classify
        raw = event.get("raw", "")
        source = event.get("source", "manual")

        # Build a related-ideas context block (if any were found)
        related_block = ""
        if related:
            lines = ["Related entries already in the user's LLM wiki (semantic search):"]
            for r in related[:5]:
                lines.append(f"  - [{r.get('kind', '?')}] {r.get('title', r.get('slug', '?'))} (score={r.get('score', 0):.2f})")
                snippet = (r.get("snippet") or "")[:120]
                if snippet:
                    lines.append(f"    snippet: {snippet}")
            lines.append("")
            lines.append("If this event is closely related to one of the above, prefer UPDATING that entry over creating a new one. Reply 'add: NO, update: <slug>' to update an existing one.")
            related_block = "\n".join(lines)

        classify_prompt = (
            f"You are the radio's note-taker. The user just had this event:\n\n"
            f"  source: {source}\n"
            f"  text: {raw}\n\n"
            f"{related_block}\n\n"
            f"Decide:\n"
            f"  1. Is this a new idea worth adding to the user's LLM wiki? "
            f"(answer YES or NO)\n"
            f"  2. If YES, give it a slug (kebab-case, 3-5 words), a one-line title, "
            f"and a 1-2 sentence body. Mark it 'idea' (general), 'project' "
            f"(long-running), or 'followup' (a question to ask later).\n\n"
            f"Format your reply as YAML:\n"
            f"  add: <YES|NO>\n"
            f"  kind: <idea|project|followup|->\n"
            f"  slug: <kebab-slug>\n"
            f"  title: <one-line title>\n"
            f"  body: <1-2 sentences>\n"
            f"  importance: <0.0-1.0>\n"
        )
        try:
            reply = self.judge.judge(classify_prompt, temperature=0.2, max_tokens=400)
        except Exception as e:
            log.error("brain.curate_event: judge failed: %s", e)
            return None
        parsed = _parse_yaml_reply(reply)
        if not parsed or parsed.get("add", "NO").upper() != "YES":
            log.info("brain.curate_event: skipped (not a new idea)")
            return None
        # 3. Write to wiki
        kind = parsed.get("kind", "idea").lower()
        if kind not in ("idea", "project", "followup"):
            kind = "idea"
        slug = parsed.get("slug", "").strip().lower().replace(" ", "-")
        if not slug:
            slug = f"i_{int(time.time())}_auto"
        try:
            self.wiki.write(
                slug=slug,
                kind=f"{kind}s" if not kind.endswith("s") else kind,
                title=parsed.get("title", slug)[:200],
                body=parsed.get("body", raw),
                tags=[source],
                concepts=[],
                importance=float(parsed.get("importance", 0.5)),
            )
            log.info("brain.curate_event: wrote %s/%s", kind, slug)
            return {"kind": kind, "slug": slug}
        except Exception as e:
            log.error("brain.curate_event: write failed: %s", e)
            return None

    def compact_wikipedia(self, max_entries: int = 30) -> Optional[str]:
        """Compact the wiki into a Wikipedia-style summary at
        ~/.hermes/wiki/index.md. Returns the path, or None if deferred.
        """
        if not self.judge:
            return None
        defer = _import_omni_client().should_defer_for_vram(POLITE_MIN_FREE_GB)
        if defer[0]:
            log.info("brain.compact_wikipedia: deferring (%s)", defer[1])
            return None

        all_entries = []
        for kind in ("ideas", "projects", "followups", "people"):
            for entry in self.wiki.list_entries(kind):
                data = self.wiki.read(entry["slug"], kind)
                body = (data.get("body") or "")[:300]
                all_entries.append({
                    "kind": kind,
                    "slug": entry["slug"],
                    "title": entry.get("title", entry["slug"]),
                    "importance": entry.get("importance", 0.5),
                    "body": body,
                })
        # Top N by importance
        all_entries.sort(key=lambda x: -float(x.get("importance") or 0))
        top = all_entries[:max_entries]
        if not top:
            return None

        # Ask the brain to compose a Wikipedia summary
        body_in = "\n\n".join(
            f"## [{e['kind']}] {e['title']}\n{e['body']}"
            for e in top
        )
        prompt = (
            "You are the Wikipedia compactor for the user's idea LLM wiki. "
            "Given these top entries by importance, produce a single "
            "concise Wikipedia-style summary that captures the user's "
            "current thinking. Organize by theme. Use bullet points. "
            "Preserve all key technical terms and project names verbatim. "
            "Output ONLY the summary, no preamble.\n\n"
            f"{body_in}"
        )
        try:
            summary = self.judge.judge(prompt, temperature=0.3, max_tokens=2000)
        except Exception as e:
            log.error("brain.compact_wikipedia: judge failed: %s", e)
            return None

        # Write to index.md
        today = time.strftime("%Y-%m-%d")
        header = (
            f"# User Idea Wiki — Compaction {today}\n\n"
            f"Auto-compacted from {len(top)} entries (top by importance) "
            f"of {len(all_entries)} total. Regenerate with "
            f"`brain.compact_wikipedia()` or via omni-va.\n\n"
            f"---\n\n"
        )
        path = Path("/home/sovthpaw/.hermes/wiki/index.md")
        path.write_text(header + summary.strip() + "\n")
        log.info("brain.compact_wikipedia: wrote %s (%d chars, %d entries)",
                 path, len(summary), len(top))
        return str(path)


def _parse_yaml_reply(reply: str) -> dict:
    """Naive YAML-like parser for the brain's structured replies.

    Handles simple `key: value` lines. Doesn't handle multiline or
    nesting. Good enough for the note-taker's classification replies.
    """
    out = {}
    for line in reply.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().lower()
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        out[k] = v
    return out
