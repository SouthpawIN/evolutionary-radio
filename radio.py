#!/usr/bin/env python3
"""
radio.py — Main entry point for Evolutionary Radio.

Two async loops for v1:
  Loop 1 (Playback):  Dequeue track → mpv → wait for EOF → next
  Loop 2 (Queue Fill): If queue < 5: vibe → OmniStep → ACE-Step → enqueue

CLI:
  python radio.py start --vibe="chill lofi beats for coding"
  python radio.py stop
  python radio.py skip
  python radio.py status
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add code/ to path so sibling modules import cleanly
_CODE_DIR = Path(__file__).resolve().parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import yaml
from track_queue import RadioQueue, Track
from mpv_player import MpvPlayer
from prompt_template import PromptTemplate
from omni_client import OmniClient
from acestep_client import AceStepClient
from skip_logger import SkipLogger
from feedback import FeedbackLogger, FeedbackRecord
from gepa import GEPAPool
from darwin import DarwinPopulation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "code" / "config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# State file (for CLI status/skip commands to talk to the running process)
# ---------------------------------------------------------------------------
def _state_dir() -> Path:
    return Path("~/.local/share/evolutionary-radio").expanduser()

def _pid_file() -> Path:
    return _state_dir() / "radio.pid"

def _state_file() -> Path:
    return _state_dir() / "state.json"

def _write_state(status: dict) -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    with open(_state_file(), "w") as f:
        json.dump(status, f, indent=2)

def _write_pid() -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    with open(_pid_file(), "w") as f:
        f.write(str(os.getpid()))

def _remove_pid() -> None:
    try:
        _pid_file().unlink(missing_ok=True)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Loop 1: Playback (consumer)
# ---------------------------------------------------------------------------
async def _poll_skip_file(skip_path: Path, queue: RadioQueue) -> None:
    """Periodically check for skip request file during playback."""
    while True:
        await asyncio.sleep(1.0)
        if skip_path.exists():
            skip_path.unlink(missing_ok=True)
            queue.request_skip()
            return

async def playback_loop(
    queue: RadioQueue,
    player: MpvPlayer,
    skip_logger: SkipLogger,
    cfg: dict,
    feedback_logger: FeedbackLogger = None,
) -> None:
    """Dequeue a track, play it via mpv, log the result."""
    log = logging.getLogger("radio.playback")
    log.info("playback loop started")

    while True:
        track = await queue.get_track()
        log.info("playing: %s (tags: %s)", track.audio_path, track.tags)

        start_time = time.time()
        skipped = False

        try:
            # Clear the end-file event BEFORE loading to prevent stale
            # events from the previous track causing immediate return.
            player._end_file_event.clear()
            await player.loadfile(track.audio_path)

            # Wait a moment for mpv to actually start playing the file.
            # Without this, a stale end-file event can fire immediately.
            await asyncio.sleep(1.0)

            # If end-file already fired during loadfile (race condition),
            # it means mpv rejected or immediately finished the file.
            # Log it and try the next track instead of hanging.
            if player._end_file_event.is_set():
                log.warning("end-file fired immediately after loadfile — file may be invalid: %s", track.audio_path)
                player._end_file_event.clear()
                # Don't wait — just move to next track
                continue

            # Check for skip request file (written by CLI cmd_skip)
            skip_req = _state_dir() / "skip_request"
            if skip_req.exists():
                skip_req.unlink(missing_ok=True)
                queue.request_skip()

            # Wait for either: track ends, or skip is requested
            # Poll for skip file every 1s while waiting for track to end
            done_task = asyncio.create_task(player.wait_for_end())
            skip_task = asyncio.create_task(queue.wait_for_skip())
            skip_poll = asyncio.create_task(_poll_skip_file(skip_req, queue))

            finished, pending = await asyncio.wait(
                [done_task, skip_task, skip_poll],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Log which task completed
            if done_task in finished:
                log.info("track ended naturally")
            if skip_task in finished:
                log.info("skip requested")

            # Cancel whichever didn't fire
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            if skip_task in finished:
                skipped = True
                log.info("skipped by user")
                queue.clear_skip()
                # Stop current playback
                await player.stop()
                # Re-spawn mpv for next track
                player.start()
                await player.connect()
                await player.wait_ready()

        except Exception as e:
            log.error("playback error: %s", e)

        played = time.time() - start_time
        queue.record_playback(track, played_seconds=played, skipped=skipped)

        # Log to skip file
        skip_logger.log_track(
            tags=track.tags,
            audio_path=track.audio_path,
            played_seconds=played,
            total_seconds=track.duration_sec,
            skipped=skipped,
        )

        # Update state file
        _write_state(queue.status())

        # Poll for feedback requests (like/dislike) written by CLI
        feedback_req = _state_dir() / "feedback_request"
        if feedback_req.exists():
            try:
                sentiment = feedback_req.read_text().strip()
                feedback_req.unlink(missing_ok=True)
                if sentiment in ("like", "dislike") and feedback_logger:
                    record = FeedbackRecord(
                        track_id=getattr(track, "meta", {}).get("track_id", ""),
                        sentiment=sentiment,
                        tags=track.tags,
                        played_seconds=played,
                        total_seconds=track.duration_sec,
                        vibe=getattr(queue_fill_loop, "_vibe", ""),
                        genome_id=getattr(track, "meta", {}).get("genome_id", ""),
                    )
                    feedback_logger.log(record)
                    log.info("feedback: %s for track %s", sentiment, record.track_id)
            except (OSError, ValueError) as e:
                log.warning("failed to read feedback request: %s", e)

# ---------------------------------------------------------------------------
# Loop 2: Queue Fill (producer)
# ---------------------------------------------------------------------------
async def queue_fill_loop(
    queue: RadioQueue,
    omni: OmniClient,
    voice: AceStepClient,
    prompt_cfg: dict,
    queue_cfg: dict,
    gepa_pool: GEPAPool = None,
    darwin_pop: DarwinPopulation = None,
    feedback_logger: FeedbackLogger = None,
) -> None:
    """If queue is below target, generate a new track."""
    log = logging.getLogger("radio.queue_fill")
    log.info("queue fill loop started")

    system_prompt = prompt_cfg.get("system_prompt", "")
    target_depth = queue_cfg.get("target_depth", 3)
    fill_sleep = queue_cfg.get("fill_sleep_sec", 5)
    duration = prompt_cfg.get("duration_sec", 60)

    # Default vibe if none set
    vibe = getattr(queue_fill_loop, "_vibe", "chill lofi beats for coding")
    current_genome_id = "default"

    while True:
        try:
            if queue.qsize() < target_depth:
                log.info("queue depth %d/%d — generating new track", queue.qsize(), queue.maxsize())

                # Step 1: Get tags from OmniStep
                t0 = time.time()
                try:
                    # Use GEPA genome to bias the prompt if available
                    effective_prompt = system_prompt
                    if gepa_pool and gepa_pool.genomes:
                        genome = gepa_pool.select_genome()
                        current_genome_id = genome.genome_id
                        effective_prompt = gepa_pool.build_system_prompt(genome, system_prompt)
                        log.info("using GEPA genome: %s (fitness=%.3f)", genome.genome_id, genome.fitness)
                    tags = omni.generate_tags(vibe, effective_prompt)
                    log.info("OmniStep tags: %s (%.1fs)", tags, time.time() - t0)
                except Exception as e:
                    log.warning("OmniStep failed, using seed fallback: %s", e)
                    from prompt_template import _resolve_seed
                    tags = _resolve_seed(vibe)

                # Step 2: Generate audio via ACE-Step
                t0 = time.time()
                try:
                    audio_path, gen_time = voice.generate(tags, duration_sec=duration)
                except Exception as e:
                    log.error("ACE-Step generation failed: %s", e)
                    await asyncio.sleep(fill_sleep)
                    continue

                # Step 3: Create track and enqueue
                track = Track(
                    tags=tags,
                    audio_path=audio_path,
                    generated_at=time.time(),
                    duration_sec=duration,
                    source="omnistep",
                    meta={"gen_latency_sec": gen_time, "genome_id": current_genome_id},
                )

                ok = await queue.put_track(track)
                if ok:
                    log.info("track enqueued (depth: %d)", queue.qsize())
                else:
                    log.warning("queue full, dropping track")

            await asyncio.sleep(fill_sleep)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("queue fill error: %s", e)
            await asyncio.sleep(fill_sleep)

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def run_radio(vibe: str, cfg: dict) -> None:
    """Set up all components and run the two loops."""
    log = logging.getLogger("radio.main")

    # Initialize components
    queue = RadioQueue(maxsize=cfg["queue"]["maxsize"])

    player = MpvPlayer(
        binary=cfg["player"]["binary"],
        socket_dir=cfg["player"]["socket_dir"],
        volume=cfg["player"]["volume"],
        extra_args=cfg["player"].get("extra_args"),
    )

    omni = OmniClient(
        base_url=cfg["brain"]["url"],
        model=cfg["brain"]["model"],
        timeout=cfg["brain"]["timeout_sec"],
    )

    voice = AceStepClient()

    skip_logger = SkipLogger(path=cfg.get("skip_log", os.path.expanduser("~/path/to/skip_log.jsonl")))

    # Feedback + evolution systems
    feedback_logger = FeedbackLogger()
    gepa_pool = GEPAPool(feedback_logger)
    darwin_pop = DarwinPopulation(feedback_logger)

    # Write PID file
    _write_pid()

    # Shutdown event
    shutdown = asyncio.Event()

    def handle_signal(*_):
        log.info("received shutdown signal")
        shutdown.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Check OmniStep health
    if omni.health_check():
        log.info("OmniStep is reachable at %s", cfg["brain"]["url"])
    else:
        log.warning("OmniStep not reachable at %s — will use seed fallback", cfg["brain"]["url"])

    # Start mpv
    player.start()
    await player.connect()
    await player.wait_ready()
    log.info("mpv ready")

    # Set the vibe on the fill loop
    queue_fill_loop._vibe = vibe  # type: ignore

    # Start both loops
    playback_task = asyncio.create_task(
        playback_loop(queue, player, skip_logger, cfg, feedback_logger=feedback_logger),
        name="playback",
    )
    fill_task = asyncio.create_task(
        queue_fill_loop(queue, omni, voice, cfg["brain"], cfg["queue"],
                        gepa_pool=gepa_pool, darwin_pop=darwin_pop, feedback_logger=feedback_logger),
        name="queue_fill",
    )

    # Loop 3: Evolution (GEPA + Darwin) — runs every 10 minutes
    async def evolution_loop():
        evo_log = logging.getLogger("radio.evolution")
        evo_interval = cfg.get("evolution", {}).get("interval_sec", 600)
        while True:
            await asyncio.sleep(evo_interval)
            try:
                evo_log.info("running GEPA evolution...")
                gepa_pool.evolve()
                evo_log.info("running Darwin evolution...")
                darwin_pop.evolve_generation()
                evo_log.info("evolution complete")
            except Exception as e:
                evo_log.error("evolution error: %s", e)

    evo_task = asyncio.create_task(evolution_loop(), name="evolution")

    log.info("radio started — vibe: %s", vibe)
    print(f"🎶 Radio started — vibe: {vibe}")
    print(f"   Queue: 0/{cfg['queue']['maxsize']}")
    print(f"   Brain: {cfg['brain']['url']}")
    print(f"   Voice: ACE-Step MLX")
    print(f"   Press Ctrl+C to stop")

    # Wait for shutdown
    await shutdown.wait()

    # Clean shutdown
    log.info("shutting down...")
    playback_task.cancel()
    fill_task.cancel()
    evo_task.cancel()

    for task in [playback_task, fill_task, evo_task]:
        try:
            await task
        except asyncio.CancelledError:
            pass

    await player.stop()
    _remove_pid()
    log.info("radio stopped")
    print("🛑 Radio stopped")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_start(args):
    cfg = load_config()
    vibe = args.vibe or "chill lofi beats for coding"
    asyncio.run(run_radio(vibe, cfg))

def cmd_stop(_args):
    pid_file = _pid_file()
    if not pid_file.exists():
        print("No running radio found")
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to radio (pid {pid})")
    except ProcessLookupError:
        print(f"Process {pid} not found — cleaning up stale PID")
        _remove_pid()

def cmd_skip(_args):
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    # Write a skip request file — the running radio checks for this
    skip_req = _state_dir() / "skip_request"
    skip_req.touch()
    print("Skip requested")

def cmd_skip_next(args):
    """Skip the next N tracks in the queue."""
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    n = getattr(args, "count", 1) or 1
    # Write skip_next request with count
    skip_req = _state_dir() / "skip_next_request"
    with open(skip_req, "w") as f:
        f.write(str(n))
    print(f"Skip next {n} track(s) requested")

def cmd_like(_args):
    """Like the currently playing track."""
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    feedback_req = _state_dir() / "feedback_request"
    with open(feedback_req, "w") as f:
        f.write("like")
    print("👍 Liked!")

def cmd_dislike(_args):
    """Dislike the currently playing track."""
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    feedback_req = _state_dir() / "feedback_request"
    with open(feedback_req, "w") as f:
        f.write("dislike")
    print("👎 Disliked")

def cmd_queue(_args):
    """Show what's in the queue."""
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    with open(state) as f:
        status = json.load(f)
    history = status.get("history", [])
    depth = status.get("depth", 0)
    maxsize = status.get("maxsize", 5)
    print(f"Queue: {depth}/{maxsize}")
    if history:
        print("\nRecent tracks:")
        for i, t in enumerate(history[-5:]):
            tags = t.get("tags", "?")[:60]
            skipped = " ⏭" if t.get("skipped") else ""
            print(f"  {i+1}. {tags}{skipped}")
    else:
        print("  (empty)")

def cmd_evolve(_args):
    """Trigger an immediate evolution step."""
    feedback = FeedbackLogger()
    gepa_pool = GEPAPool(feedback)
    darwin_pop = DarwinPopulation(feedback)
    print("Running GEPA evolution...")
    gepa_pool.evolve()
    print("Running Darwin evolution...")
    darwin_pop.evolve_generation()
    best_gepa = gepa_pool.genomes[0] if gepa_pool.genomes else None
    best_darwin = darwin_pop.get_best()
    print(f"\nBest GEPA genome: {best_gepa.genome_id} (fitness={best_gepa.fitness:.3f})" if best_gepa else "No GEPA genomes")
    print(f"Best Darwin genome: {best_darwin.genome_id} (fitness={best_darwin.fitness:.3f})")

def cmd_status(_args):
    state = _state_file()
    if not state.exists():
        print("No running radio found")
        return
    with open(state) as f:
        status = json.load(f)
    print(json.dumps(status, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Evolutionary Radio")
    sub = parser.add_subparsers(dest="command")

    start_p = sub.add_parser("start", help="Start the radio")
    start_p.add_argument("--vibe", "-v", type=str, help="Vibe string for music generation")

    sub.add_parser("stop", help="Stop the radio")
    sub.add_parser("skip", help="Skip current track")
    skip_next_p = sub.add_parser("skip-next", help="Skip next N tracks in queue")
    skip_next_p.add_argument("count", type=int, nargs="?", default=1, help="Number of tracks to skip")
    sub.add_parser("like", help="Like the current track")
    sub.add_parser("dislike", help="Dislike the current track")
    sub.add_parser("queue", help="Show queue contents")
    sub.add_parser("evolve", help="Trigger immediate evolution")
    sub.add_parser("status", help="Show radio status")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "skip":
        cmd_skip(args)
    elif args.command == "skip-next":
        cmd_skip_next(args)
    elif args.command == "like":
        cmd_like(args)
    elif args.command == "dislike":
        cmd_dislike(args)
    elif args.command == "queue":
        cmd_queue(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
