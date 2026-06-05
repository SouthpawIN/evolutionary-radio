"""
mpv_player.py — mpv IPC control via a Unix domain socket.

Single-reader architecture: one background task reads ALL messages from
the mpv socket and routes them to either command responses or event handlers.

This eliminates the race condition where two readers compete for socket
data and events get consumed by the wrong reader.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("radio.mpv")

DEFAULT_BINARY = "mpv"
SOCKET_PREFIX = "hermes-mpv-"


class MpvPlayer:
    """An async wrapper around an mpv subprocess with Unix-socket IPC."""

    def __init__(
        self,
        binary: str = DEFAULT_BINARY,
        socket_dir: str = "/tmp",
        volume: int = 70,
        extra_args: Optional[list[str]] = None,
    ):
        self.binary = binary
        self.socket_dir = Path(socket_dir).expanduser()
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = self.socket_dir / f"{SOCKET_PREFIX}{uuid.uuid4().hex[:8]}.sock"
        self.volume = max(0, min(int(volume), 130))  # mpv range is 0-130
        self.extra_args = list(extra_args or [
            "--no-video", "--no-terminal", "--quiet", "--audio-display=no", "--idle=yes",
        ])
        self._proc: Optional[subprocess.Popen] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._sock: Optional[socket.socket] = None
        self._connected = asyncio.Event()
        self._end_file_event = asyncio.Event()
        # Pending command responses: request_id -> Future
        self._pending_responses: dict[int, asyncio.Future] = {}
        self._request_counter = 0

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Spawn the mpv subprocess. Does not block."""
        if self._proc is not None:
            log.warning("mpv already started (pid=%s)", self._proc.pid)
            return

        # Clean up stale socket file from a previous run.
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        args = [
            self.binary,
            f"--input-ipc-server={self.socket_path}",
            "--pause",  # we start paused, then loadfile() unpauses
            f"--volume={self.volume}",
            *self.extra_args,
        ]
        log.info("starting mpv: %s", " ".join(args))
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("mpv spawned pid=%s, socket=%s", self._proc.pid, self.socket_path)

    async def connect(self, timeout_sec: float = 5.0) -> None:
        """Open the IPC socket. Polls until the file exists."""
        deadline = asyncio.get_event_loop().time() + timeout_sec
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"mpv exited with code {self._proc.returncode} before IPC connect")
            if self.socket_path.exists():
                try:
                    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self._sock.setblocking(False)
                    self._sock.connect(str(self.socket_path))
                    self._connected.set()
                    # Start SINGLE background reader
                    self._reader_task = asyncio.create_task(self._read_loop(), name="mpv-reader")
                    log.info("mpv IPC connected at %s", self.socket_path)
                    return
                except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                    log.debug("mpv connect retry: %s", e)
                    if self._sock is not None:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                        self._sock = None
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"mpv did not open IPC socket at {self.socket_path} within {timeout_sec}s")
            await asyncio.sleep(0.05)

    async def _read_loop(self) -> None:
        """Single reader: dispatch command responses and events."""
        assert self._sock is not None
        loop = asyncio.get_event_loop()
        buf = b""
        try:
            while True:
                try:
                    chunk = await loop.sock_recv(self._sock, 4096)
                except ConnectionError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue

                    # Route: command response vs event
                    if "request_id" in msg and msg["request_id"] is not None:
                        req_id = msg["request_id"]
                        fut = self._pending_responses.pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(msg)
                    elif "event" in msg:
                        self._handle_event(msg)
        except Exception as e:
            log.warning("mpv read loop terminated: %s", e)
        finally:
            log.debug("mpv read loop exited")

    def _handle_event(self, msg: dict) -> None:
        """Route mpv events to the right handler."""
        event = msg.get("event")
        if event == "end-file":
            log.info("mpv end-file: reason=%s", msg.get("reason"))
            self._end_file_event.set()
        elif event in ("start-file", "file-loaded", "playback-restart"):
            log.debug("mpv event: %s", event)
        # Ignore other events (audio-reconfig, idle, etc.)

    async def wait_ready(self, timeout_sec: float = 5.0) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout=timeout_sec)

    async def stop(self) -> None:
        """Tell mpv to quit, then unlink the socket file."""
        try:
            if self._sock is not None and self._proc is not None and self._proc.poll() is None:
                await self._send_command(["quit"])
        except Exception as e:
            log.debug("mpv quit command failed (ok if already gone): %s", e)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                log.warning("mpv did not exit cleanly, killing")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        # Clean up the socket file.
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
                log.info("removed mpv socket %s", self.socket_path)
        except OSError as e:
            log.debug("could not unlink mpv socket: %s", e)
        self._proc = None

    # ----------------------------------------------------------- IPC helpers
    async def _send_command(self, command: list[Any]) -> Optional[dict]:
        """Send a JSON command to mpv and wait for its response.

        Uses request_id so the single reader can route the response back.
        """
        if self._sock is None:
            raise RuntimeError("mpv not connected")

        self._request_counter += 1
        req_id = self._request_counter
        payload = (json.dumps({"command": command, "request_id": req_id}) + "\n").encode("utf-8")

        # Create a future for the response
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[req_id] = fut

        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendall(self._sock, payload)
            # Wait for the response with a timeout
            return await asyncio.wait_for(fut, timeout=2.0)
        except asyncio.TimeoutError:
            log.debug("command timed out: %s", command)
            self._pending_responses.pop(req_id, None)
            return None
        except (ConnectionError, OSError) as e:
            log.debug("command failed: %s", e)
            self._pending_responses.pop(req_id, None)
            return None

    # ----------------------------------------------------------- public API
    async def loadfile(self, path: str) -> None:
        """Load a new file and unpause."""
        self._end_file_event.clear()
        # ``replace`` ensures we don't append to a stale playlist.
        await self._send_command(["loadfile", path, "replace"])
        await self._send_command(["set_property", "pause", False])
        log.info("mpv playing: %s", path)

    async def pause(self) -> None:
        await self._send_command(["set_property", "pause", True])

    async def resume(self) -> None:
        await self._send_command(["set_property", "pause", False])

    async def set_volume(self, volume: int) -> None:
        v = max(0, min(int(volume), 130))
        await self._send_command(["set_property", "volume", v])
        self.volume = v

    async def wait_for_end(self) -> None:
        """Block until the current track ends or is replaced."""
        await self._end_file_event.wait()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def current_socket(self) -> str:
        return str(self.socket_path)


__all__ = ["MpvPlayer"]
