"""
skip_logger.py — Append-only JSONL skip log.

Each played track gets one record. GEPA reads skip_rate per
prompt_template_id; Darwin reads clap_score /fad_score per voice_version.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("radio.skip_logger")

DEFAULT_LOG_PATH = os.path.expanduser("~/path/to/skip_log.jsonl")


class SkipLogger:
    """Writes per-track records to a JSONL file."""

    def __init__(self, path: str = DEFAULT_LOG_PATH):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_track(
        self,
        tags: str,
        audio_path: str,
        played_seconds: float,
        total_seconds: float,
        skipped: bool,
        voice_model: str = "ace-step-mlx-4bit",
        voice_version: str = "2026-06-05",
        prompt_template_id: str = "default",
        clap_score: Optional[float] = None,
        fad_score: Optional[float] = None,
        track_id: Optional[str] = None,
    ) -> str:
        """Append a record. Returns the track_id."""
        track_id = track_id or uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        skip_at = played_seconds if skipped else None

        record = {
            "track_id": track_id,
            "prompt_template_id": prompt_template_id,
            "prompt": tags,
            "generated_at": now,
            "played_seconds": played_seconds,
            "total_seconds": total_seconds,
            "skipped": skipped,
            "skip_at_seconds": skip_at,
            "clap_score": clap_score,
            "fad_score": fad_score,
            "voice_model": voice_model,
            "voice_version": voice_version,
        }

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
            log.debug("skip log: %s skipped=%s played=%.1f", track_id, skipped, played_seconds)
        except OSError as e:
            log.warning("failed to write skip log: %s", e)

        return track_id
