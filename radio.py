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
async def playback_loop(
    queue: RadioQueue,
    player: MpvPlayer,
    skip_logger: SkipLogger,
    cfg: dict,
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
            await player.loadfile(track.audio_path)

            # Wait for either: track ends, or skip is requested
            done_task = asyncio.create_task(player.wait_for_end())
            skip_task = asyncio.create_task(queue.wait_for_skip())

            finished, pending = await asyncio.wait(
                [done_task, skip_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

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

# ---------------------------------------------------------------------------
# Loop 2: Queue Fill (producer)
# ---------------------------------------------------------------------------
async def queue_fill_loop(
    queue: RadioQueue,
    omni: OmniClient,
    voice: AceStepClient,
    prompt_cfg: dict,
    queue_cfg: dict,
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

    while True:
        try:
            if queue.qsize() < target_depth:
                log.info("queue depth %d/%d — generating new track", queue.qsize(), queue.maxsize())

                # Step 1: Get tags from OmniStep
                t0 = time.time()
                try:
                    tags = omni.generate_tags(vibe, system_prompt)
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
                    meta={"gen_latency_sec": gen_time},
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
        playback_loop(queue, player, skip_logger, cfg),
        name="playback",
    )
    fill_task = asyncio.create_task(
        queue_fill_loop(queue, omni, voice, cfg["brain"], cfg["queue"]),
        name="queue_fill",
    )

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

    for task in [playback_task, fill_task]:
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
    sub.add_parser("status", help="Show radio status")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "skip":
        cmd_skip(args)
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
