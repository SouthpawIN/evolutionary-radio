"""
feedback.py — Track sentiment signals (like/dislike/skip) for evolutionary feedback.

Each track played gets a feedback record. These records drive:
  - GEPA prompt evolution (gradient-free prompt optimization)
  - Darwin Family genome evolution (genetic recombination of prompt genomes)
  - Sentiment-aware queue filling (bias toward liked styles)

The feedback log is append-only JSONL. Each record has:
  track_id, sentiment, tags, prompt_template_id, played_seconds, timestamp
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("radio.feedback")

DEFAULT_FEEDBACK_PATH = Path("~/.local/share/evolutionary-radio/feedback.jsonl").expanduser()


@dataclass
class FeedbackRecord:
    track_id: str
    sentiment: str  # "like" | "dislike" | "skip" | "skip_next"
    tags: str
    prompt_template_id: str = "default"
    played_seconds: float = 0.0
    total_seconds: float = 60.0
    timestamp: float = field(default_factory=time.time)
    vibe: str = ""
    genome_id: str = ""


class FeedbackLogger:
    """Append-only JSONL logger for user sentiment signals."""

    def __init__(self, path: Path = DEFAULT_FEEDBACK_PATH):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: FeedbackRecord) -> None:
        """Append a feedback record."""
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps({
                    "track_id": record.track_id,
                    "sentiment": record.sentiment,
                    "tags": record.tags,
                    "prompt_template_id": record.prompt_template_id,
                    "played_seconds": record.played_seconds,
                    "total_seconds": record.total_seconds,
                    "timestamp": record.timestamp,
                    "vibe": record.vibe,
                    "genome_id": record.genome_id,
                }) + "\n")
            log.debug("feedback: %s sentiment=%s tags=%s", record.track_id, record.sentiment, record.tags[:50])
        except OSError as e:
            log.warning("failed to write feedback: %s", e)

    def read_all(self) -> list[dict]:
        """Read all feedback records."""
        records = []
        if not self.path.exists():
            return records
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_recent(self, n: int = 50) -> list[dict]:
        """Get the N most recent feedback records."""
        return self.read_all()[-n:]


def compute_sentiment_scores(records: list[dict]) -> dict[str, float]:
    """Compute per-tag sentiment scores from feedback history.

    Returns a dict of tag → score where:
      - +1.0 = always liked
      - 0.0 = neutral / no signal
      - -1.0 = always disliked

    Tags are extracted by splitting the 'tags' string on commas.
    """
    tag_signals: dict[str, list[float]] = defaultdict(list)

    sentiment_weights = {
        "like": 1.0,
        "dislike": -1.0,
        "skip": -0.5,
        "skip_next": -0.3,
    }

    for rec in records:
        weight = sentiment_weights.get(rec.get("sentiment", ""), 0.0)
        if weight == 0.0:
            continue
        tags_str = rec.get("tags", "")
        for tag in tags_str.split(","):
            tag = tag.strip().lower()
            if tag:
                tag_signals[tag].append(weight)

    # Average per tag
    scores = {}
    for tag, signals in tag_signals.items():
        scores[tag] = sum(signals) / len(signals)

    return scores


def get_top_liked_tags(records: list[dict], n: int = 10) -> list[tuple[str, float]]:
    """Return the top N most-liked tags with their scores."""
    scores = compute_sentiment_scores(records)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def get_top_disliked_tags(records: list[dict], n: int = 10) -> list[tuple[str, float]]:
    """Return the top N most-disliked tags with their scores."""
    scores = compute_sentiment_scores(records)
    return sorted(scores.items(), key=lambda x: x[1])[:n]
