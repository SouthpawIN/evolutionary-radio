"""
queue.py — asyncio.Queue wrapper with skip/pause/status API.

This is the data plane for Loops 1 and 2 of the radio:
  * Loop 2 (queue fill) is the **producer** — calls ``put_track``.
  * Loop 1 (playback)   is the **consumer** — calls ``get_track``.

The queue also exposes the small auxiliary state that the CLI's ``status``
command renders: depth, last N tracks, last generation latency, and a
pause event that the consumer respects.

We deliberately keep this layer thin — no mpv, no HTTP, no logging beyond
the Python logger. Everything here is unit-testable in isolation.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    """A unit of work moving through the radio.

    Attributes:
        tags:        The ACE-Step tag string that generated this track.
        audio_path:  Filesystem path to the rendered audio (WAV / MP3 / FLAC).
        generated_at: Unix timestamp when ACE-Step finished generating.
        duration_sec: Expected playback length (best-effort).
        source:      "seed" | "omnistep" | "fallback" — provenance for GEPA.
    """

    tags: str
    audio_path: str
    generated_at: float
    duration_sec: float = 60.0
    source: str = "omnistep"
    # Free-form metadata for the skip log / GEPA reflection
    meta: dict = field(default_factory=dict)


class RadioQueue:
    """Bounded asyncio.Queue with the radio's auxiliary bookkeeping.

    The class is *the* state container for the radio process. There is
    exactly one instance per running radio, shared by all four loops.
    """

    def __init__(self, maxsize: int = 5, history_size: int = 5):
        self._q: asyncio.Queue[Track] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused by default
        # Skip is implemented as a one-shot event: producer sets it, consumer awaits it.
        self._skip_event = asyncio.Event()
        # Track history (most-recent-first). Bounded to history_size.
        self._history: deque[Track] = deque(maxlen=history_size)
        # Generation stats
        self._last_gen_latency_sec: Optional[float] = None
        self._gens_completed: int = 0

    # ------------------------------------------------------------------ size
    def qsize(self) -> int:
        return self._q.qsize()

    def maxsize(self) -> int:
        return self._maxsize

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def skip_requested(self) -> bool:
        return self._skip_event.is_set()

    # ----------------------------------------------------------------- control
    def request_skip(self) -> None:
        """Ask the consumer to abandon the current track. Idempotent."""
        self._skip_event.set()

    def clear_skip(self) -> None:
        """Consumer calls this after honoring a skip request."""
        self._skip_event.clear()

    async def wait_for_skip(self) -> None:
        """Block until a skip is requested (or skip is already pending)."""
        if self._skip_event.is_set():
            return
        await self._skip_event.wait()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    async def wait_if_paused(self) -> None:
        await self._pause_event.wait()

    # ------------------------------------------------------- producer (Loop 2)
    async def put_track(self, track: Track, timeout: Optional[float] = 1.0) -> bool:
        """Put a track on the queue. Returns False if the queue is full.

        We use a short timeout so a stuck consumer doesn't deadlock the
        fill loop; the producer can simply drop and try again next tick.
        """
        try:
            await asyncio.wait_for(self._q.put(track), timeout=timeout)
            self._record_generation(track)
            return True
        except asyncio.TimeoutError:
            return False

    def _record_generation(self, track: Track) -> None:
        # If the producer set ``track.meta["gen_latency_sec"]``, capture it.
        latency = track.meta.get("gen_latency_sec")
        if isinstance(latency, (int, float)):
            self._last_gen_latency_sec = float(latency)
        self._gens_completed += 1

    # ------------------------------------------------------- consumer (Loop 1)
    async def get_track(self) -> Track:
        """Block for the next track. Respects pause."""
        await self.wait_if_paused()
        track = await self._q.get()
        return track

    def record_playback(self, track: Track, played_seconds: float, skipped: bool) -> None:
        """Append a track to history with playback metadata for the skip log."""
        # We avoid mutating the Track (it may still be referenced) and
        # just push it onto history. The skip_logger is responsible for
        # writing to the JSONL file.
        self._history.appendleft(track)
        track.meta.setdefault("played_seconds", played_seconds)
        track.meta.setdefault("skipped", skipped)
        track.meta.setdefault("played_at", time.time())

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        """Snapshot for the CLI ``status`` command and the state file."""
        return {
            "queue_depth": self.qsize(),
            "queue_max":   self._maxsize,
            "paused":      self.is_paused,
            "gens_completed": self._gens_completed,
            "last_gen_latency_sec": self._last_gen_latency_sec,
            "history": [
                {
                    "tags": t.tags,
                    "audio_path": t.audio_path,
                    "generated_at": t.generated_at,
                    "played_seconds": t.meta.get("played_seconds", 0.0),
                    "skipped": t.meta.get("skipped", False),
                }
                for t in self._history
            ],
        }


__all__ = ["RadioQueue", "Track"]
