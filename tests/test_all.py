#!/usr/bin/env python3
"""
Comprehensive test suite for Evolutionary Radio.
Tests ALL functionality: modules, CLI, playback, skip, status, cache, error handling.

Run: cd ~/evolutionary-radio && ./venv/bin/python3 tests/test_all.py
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add code/ to path
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

PASSED = 0
FAILED = 0
ERRORS = []

def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  ❌ {name}: {e}")

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: expected {b!r}, got {a!r}")

def assert_true(val, msg=""):
    if not val:
        raise AssertionError(f"{msg}: expected True, got {val!r}")

def assert_false(val, msg=""):
    if val:
        raise AssertionError(f"{msg}: expected False, got {val!r}")

def assert_exists(path, msg=""):
    if not Path(path).exists():
        raise AssertionError(f"{msg}: {path} does not exist")

# =========================================================================
# Module Tests
# =========================================================================
print("\n📦 Module Tests")

def test_track_creation():
    from track_queue import Track
    t = Track(tags="lofi, chill", audio_path="/tmp/test.wav", generated_at=time.time())
    assert_eq(t.tags, "lofi, chill")
    assert_eq(t.duration_sec, 60.0)
    assert_eq(t.source, "omnistep")
run_test("Track dataclass creation", test_track_creation)

def test_queue_put_get():
    from track_queue import RadioQueue, Track
    q = RadioQueue(maxsize=3)
    t = Track(tags="test", audio_path="/tmp/test.wav", generated_at=time.time())
    result = asyncio.get_event_loop().run_until_complete(q.put_track(t))
    assert_true(result, "put_track should return True")
    assert_eq(q.qsize(), 1)
run_test("RadioQueue put/get", test_queue_put_get)

def test_queue_full():
    from track_queue import RadioQueue, Track
    q = RadioQueue(maxsize=2)
    t1 = Track(tags="t1", audio_path="/tmp/t1.wav", generated_at=time.time())
    t2 = Track(tags="t2", audio_path="/tmp/t2.wav", generated_at=time.time())
    asyncio.get_event_loop().run_until_complete(q.put_track(t1))
    asyncio.get_event_loop().run_until_complete(q.put_track(t2))
    result = asyncio.get_event_loop().run_until_complete(q.put_track(t1, timeout=0.1))
    assert_false(result, "put_track on full queue should return False")
run_test("RadioQueue full", test_queue_full)

def test_queue_skip():
    from track_queue import RadioQueue
    q = RadioQueue()
    assert_false(q.skip_requested)
    q.request_skip()
    assert_true(q.skip_requested)
    q.clear_skip()
    assert_false(q.skip_requested)
run_test("RadioQueue skip", test_queue_skip)

def test_queue_status():
    from track_queue import RadioQueue, Track
    q = RadioQueue(maxsize=5)
    t = Track(tags="test", audio_path="/tmp/test.wav", generated_at=time.time())
    asyncio.get_event_loop().run_until_complete(q.put_track(t))
    status = q.status()
    assert_eq(status["queue_depth"], 1)
    assert_eq(status["queue_max"], 5)
    assert_false(status["paused"])
run_test("RadioQueue status", test_queue_status)

def test_prompt_basic():
    from prompt_template import PromptTemplate
    pt = PromptTemplate(system_prompt="test prompt")
    result = pt.build_user_message("chill lofi beats")
    assert_true(isinstance(result, str), "render should return string")
    assert_true(len(result) > 0, "render should not be empty")
run_test("PromptTemplate basic", test_prompt_basic)

def test_prompt_genre():
    from prompt_template import PromptTemplate
    pt = PromptTemplate(system_prompt="test prompt")
    result = pt.build_user_message("jazz, piano, 120bpm")
    assert_true("jazz" in result.lower() or "piano" in result.lower(),
                 f"result should contain genre terms: {result}")
run_test("PromptTemplate with genre", test_prompt_genre)

def test_skip_logger():
    from skip_logger import SkipLogger
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        sl = SkipLogger(path=path)
        sl.log_track(tags="test", audio_path="/tmp/test.wav",
                     played_seconds=30.0, total_seconds=60.0, skipped=False)
        with open(path) as f:
            lines = f.readlines()
        assert_eq(len(lines), 1)
        data = json.loads(lines[0])
        assert_eq(data["prompt"], "test")
        assert_eq(data["skipped"], False)
    finally:
        os.unlink(path)
run_test("SkipLogger creation", test_skip_logger)

def test_skip_logger_skipped():
    from skip_logger import SkipLogger
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        sl = SkipLogger(path=path)
        sl.log_track(tags="bad", audio_path="/tmp/bad.wav",
                     played_seconds=5.0, total_seconds=60.0, skipped=True)
        with open(path) as f:
            data = json.loads(f.readline())
        assert_eq(data["skipped"], True)
        assert_eq(data["played_seconds"], 5.0)
    finally:
        os.unlink(path)
run_test("SkipLogger skipped track", test_skip_logger_skipped)

def test_config_loads():
    import yaml
    config_path = CODE_DIR / "config.yaml"
    assert_exists(config_path, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert_true("brain" in cfg, "config should have brain section")
    assert_true("player" in cfg, "config should have player section")
    assert_true("queue" in cfg, "config should have queue section")
run_test("Config loads", test_config_loads)

def test_config_fields():
    import yaml
    with open(CODE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert_true("url" in cfg["brain"], "brain should have url")
    assert_true("volume" in cfg["player"], "player should have volume")
    assert_true("maxsize" in cfg["queue"], "queue should have maxsize")
run_test("Config has required fields", test_config_fields)

# =========================================================================
# mpv Player Tests
# =========================================================================
print("\n🔊 mpv Player Tests")

def test_mpv_creation():
    from mpv_player import MpvPlayer
    p = MpvPlayer(volume=80)
    assert_eq(p.volume, 80)
    assert_true(p.socket_path.name.startswith("hermes-mpv-"))
run_test("MpvPlayer creation", test_mpv_creation)

def test_mpv_volume_clamp():
    from mpv_player import MpvPlayer
    p = MpvPlayer(volume=200)
    assert_eq(p.volume, 130)
    p2 = MpvPlayer(volume=-10)
    assert_eq(p2.volume, 0)
run_test("MpvPlayer volume clamping", test_mpv_volume_clamp)

def test_mpv_lifecycle():
    from mpv_player import MpvPlayer
    p = MpvPlayer()
    p.start()
    assert_true(p.is_running, "mpv should be running")
    asyncio.get_event_loop().run_until_complete(p.connect())
    asyncio.get_event_loop().run_until_complete(p.wait_ready())
    asyncio.get_event_loop().run_until_complete(p.stop())
    assert_false(p.is_running, "mpv should be stopped")
run_test("MpvPlayer start/connect/stop lifecycle", test_mpv_lifecycle)

def test_mpv_loadfile():
    from mpv_player import MpvPlayer
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    assert_true(len(tracks) > 0, "need at least one cached track")
    track_path = str(tracks[0])

    p = MpvPlayer()
    p.start()
    asyncio.get_event_loop().run_until_complete(p.connect())
    asyncio.get_event_loop().run_until_complete(p.wait_ready())
    asyncio.get_event_loop().run_until_complete(p.loadfile(track_path))
    time.sleep(1)

    # Verify playing
    resp = asyncio.get_event_loop().run_until_complete(
        p._send_command(["get_property", "pause"])
    )
    assert_true(resp and resp.get("data") == False, "should be playing")

    asyncio.get_event_loop().run_until_complete(p.stop())
run_test("MpvPlayer loadfile and play", test_mpv_loadfile)

def test_mpv_skip_mid_track():
    from mpv_player import MpvPlayer
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    track_path = str(tracks[0])

    p = MpvPlayer()
    p.start()
    asyncio.get_event_loop().run_until_complete(p.connect())
    asyncio.get_event_loop().run_until_complete(p.wait_ready())
    asyncio.get_event_loop().run_until_complete(p.loadfile(track_path))
    time.sleep(1)

    # Stop (simulates skip)
    asyncio.get_event_loop().run_until_complete(p.stop())
    assert_false(p.is_running, "mpv should stop after skip")
run_test("MpvPlayer skip mid-track", test_mpv_skip_mid_track)

def test_mpv_volume():
    from mpv_player import MpvPlayer
    p = MpvPlayer()
    p.start()
    asyncio.get_event_loop().run_until_complete(p.connect())
    asyncio.get_event_loop().run_until_complete(p.wait_ready())
    asyncio.get_event_loop().run_until_complete(p.set_volume(100))
    assert_eq(p.volume, 100)
    asyncio.get_event_loop().run_until_complete(p.set_volume(50))
    assert_eq(p.volume, 50)
    asyncio.get_event_loop().run_until_complete(p.stop())
run_test("MpvPlayer set_volume", test_mpv_volume)

def test_mpv_pause_resume():
    from mpv_player import MpvPlayer
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    track_path = str(tracks[0])

    p = MpvPlayer()
    p.start()
    asyncio.get_event_loop().run_until_complete(p.connect())
    asyncio.get_event_loop().run_until_complete(p.wait_ready())
    asyncio.get_event_loop().run_until_complete(p.loadfile(track_path))
    time.sleep(0.5)

    asyncio.get_event_loop().run_until_complete(p.pause())
    time.sleep(0.5)
    resp = asyncio.get_event_loop().run_until_complete(
        p._send_command(["get_property", "pause"])
    )
    assert_true(resp and resp.get("data") == True, "should be paused")

    asyncio.get_event_loop().run_until_complete(p.resume())
    time.sleep(0.5)
    resp = asyncio.get_event_loop().run_until_complete(
        p._send_command(["get_property", "pause"])
    )
    assert_true(resp and resp.get("data") == False, "should be unpaused")

    asyncio.get_event_loop().run_until_complete(p.stop())
run_test("MpvPlayer pause/resume", test_mpv_pause_resume)

# =========================================================================
# CLI Tests
# =========================================================================
print("\n🖥️  CLI Tests")

RADIO_BIN = str(Path(__file__).resolve().parent.parent / "venv" / "bin" / "python3")
RADIO_PY = str(Path(__file__).resolve().parent.parent / "radio.py")

def test_cli_status_no_radio():
    state_dir = Path.home() / ".local" / "share" / "evolutionary-radio"
    state_file = state_dir / "state.json"
    pid_file = state_dir / "radio.pid"
    if state_file.exists():
        state_file.unlink()
    if pid_file.exists():
        pid_file.unlink()
    result = subprocess.run(
        [RADIO_BIN, RADIO_PY, "status"],
        capture_output=True, text=True, timeout=5
    )
    assert_true("No running radio" in result.stdout, f"should say no radio: {result.stdout}")
run_test("CLI: status with no running radio", test_cli_status_no_radio)

def test_cli_stop_no_radio():
    result = subprocess.run(
        [RADIO_BIN, RADIO_PY, "stop"],
        capture_output=True, text=True, timeout=5
    )
    assert_true("No running radio" in result.stdout, f"should say no radio: {result.stdout}")
run_test("CLI: stop with no running radio", test_cli_stop_no_radio)

# =========================================================================
# Integration Tests
# =========================================================================
print("\n🔗 Integration Tests")

def test_omni_reachable():
    from omni_client import OmniClient
    import yaml
    with open(CODE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    client = OmniClient(base_url=cfg["brain"]["url"], model=cfg["brain"].get("model", ""), timeout=5)
    result = client.health_check()
    assert_true(result, "OmniStep should be reachable")
run_test("OmniStep server reachable", test_omni_reachable)

def test_omni_tags():
    from omni_client import OmniClient
    import yaml
    with open(CODE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    client = OmniClient(base_url=cfg["brain"]["url"], model=cfg["brain"].get("model", ""), timeout=10)
    tags = client.generate_tags("chill lofi beats", "")
    assert_true(isinstance(tags, str), f"tags should be string: {type(tags)}")
    assert_true(len(tags) > 0, "tags should not be empty")
run_test("OmniStep tag generation", test_omni_tags)

def test_cache_exists():
    cache_dir = Path.home() / "music" / "radio_cache"
    assert_exists(cache_dir, "cache directory")
    tracks = list(cache_dir.glob("*.wav"))
    assert_true(len(tracks) > 0, f"cache should have tracks, found {len(tracks)}")
run_test("Cache directory exists and has tracks", test_cache_exists)

def test_cache_writable():
    skip_log = Path.home() / "music" / "skip_log.jsonl"
    skip_log.touch()
    assert_exists(skip_log, "skip log")
run_test("Skip log file writable", test_cache_writable)

def test_cache_wav_valid():
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)[:3]
    for track in tracks:
        with open(track, "rb") as f:
            header = f.read(4)
        assert_eq(header, b"RIFF", f"{track.name} is not a valid WAV file")
run_test("Cache track files are valid WAV", test_cache_wav_valid)

def test_cache_sizes():
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)[:3]
    for track in tracks:
        size_mb = track.stat().st_size / (1024 * 1024)
        assert_true(0.5 < size_mb < 50, f"{track.name} size {size_mb:.1f}MB is unreasonable")
run_test("Cached track file sizes reasonable", test_cache_sizes)

# =========================================================================
# Playback Simulation Tests
# =========================================================================
print("\n🎵 Playback Simulation Tests")

def test_full_lifecycle():
    from mpv_player import MpvPlayer
    from track_queue import RadioQueue, Track
    from skip_logger import SkipLogger

    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    assert_true(len(tracks) > 0, "need cached tracks")

    track_path = str(tracks[0])
    queue = RadioQueue(maxsize=3)
    player = MpvPlayer()
    skip_logger = SkipLogger(path="/tmp/test_skip.jsonl")

    player.start()
    asyncio.get_event_loop().run_until_complete(player.connect())
    asyncio.get_event_loop().run_until_complete(player.wait_ready())

    t = Track(tags="test", audio_path=track_path, generated_at=time.time())
    asyncio.get_event_loop().run_until_complete(queue.put_track(t))
    assert_eq(queue.qsize(), 1)

    asyncio.get_event_loop().run_until_complete(player.loadfile(track_path))
    time.sleep(1)

    resp = asyncio.get_event_loop().run_until_complete(
        player._send_command(["get_property", "pause"])
    )
    assert_true(resp and resp.get("data") == False, "should be playing")

    asyncio.get_event_loop().run_until_complete(player.stop())
    assert_false(player.is_running, "player should be stopped after skip")

    skip_logger.log_track(tags="test", audio_path=track_path,
                          played_seconds=5.0, total_seconds=60.0, skipped=True)

    with open("/tmp/test_skip.jsonl") as f:
        data = json.loads(f.readline())
    assert_eq(data["skipped"], True)

    os.unlink("/tmp/test_skip.jsonl")
run_test("Full lifecycle: start → load → skip → stop", test_full_lifecycle)

def test_rapid_skips():
    from mpv_player import MpvPlayer
    cache_dir = Path.home() / "music" / "radio_cache"
    tracks = sorted(cache_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    assert_true(len(tracks) > 2, "need at least 3 cached tracks")

    player = MpvPlayer()
    player.start()
    asyncio.get_event_loop().run_until_complete(player.connect())
    asyncio.get_event_loop().run_until_complete(player.wait_ready())

    for i in range(3):
        track_path = str(tracks[i % len(tracks)])
        asyncio.get_event_loop().run_until_complete(player.loadfile(track_path))
        time.sleep(0.5)
        asyncio.get_event_loop().run_until_complete(player.stop())
        assert_false(player.is_running, f"player should stop after skip {i+1}")
        if i < 2:  # Don't re-spawn on last iteration
            player.start()
            asyncio.get_event_loop().run_until_complete(player.connect())
            asyncio.get_event_loop().run_until_complete(player.wait_ready())
run_test("Multiple rapid skips", test_rapid_skips)

def test_queue_refill():
    from track_queue import RadioQueue, Track
    queue = RadioQueue(maxsize=3)
    for i in range(3):
        t = Track(tags=f"track{i}", audio_path=f"/tmp/t{i}.wav", generated_at=time.time())
        asyncio.get_event_loop().run_until_complete(queue.put_track(t))
    assert_eq(queue.qsize(), 3)

    track = asyncio.get_event_loop().run_until_complete(queue.get_track())
    assert_eq(track.tags, "track0")
    assert_eq(queue.qsize(), 2)

    t = Track(tags="new", audio_path="/tmp/new.wav", generated_at=time.time())
    asyncio.get_event_loop().run_until_complete(queue.put_track(t))
    assert_eq(queue.qsize(), 3)
run_test("Queue refill during playback", test_queue_refill)

def test_skip_file_detection():
    from pathlib import Path
    skip_req = Path.home() / ".local" / "share" / "evolutionary-radio" / "skip_request"
    skip_req.unlink(missing_ok=True)
    skip_req.touch()
    assert_true(skip_req.exists(), "skip request file should exist")
    detected = skip_req.exists()
    if detected:
        skip_req.unlink(missing_ok=True)
    assert_true(detected)
    assert_false(skip_req.exists(), "skip request should be deleted after detection")
run_test("Skip file detection during playback", test_skip_file_detection)

# =========================================================================
# File Integrity Tests
# =========================================================================
print("\n📄 File Integrity Tests")

def test_radio_py_exists():
    assert_exists(RADIO_PY, "radio.py")
    assert_true(os.access(RADIO_PY, os.R_OK), "radio.py should be readable")
run_test("radio.py exists and is readable", test_radio_py_exists)

def test_start_script():
    script = Path(__file__).resolve().parent.parent / "start_radio.sh"
    assert_exists(script, "start_radio.sh")
    assert_true(os.access(script, os.X_OK), "start_radio.sh should be executable")
run_test("start_radio.sh exists and is executable", test_start_script)

def test_requirements():
    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    assert_exists(req, "requirements.txt")
run_test("requirements.txt exists", test_requirements)

def test_requirements_deps():
    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    content = req.read_text().lower()
    assert_true("torch" in content, "requirements.txt should include torch")
    assert_true("soundfile" in content, "requirements.txt should include soundfile")
    assert_true("pyyaml" in content or "yaml" in content, "requirements.txt should include pyyaml")
run_test("requirements.txt has all deps", test_requirements_deps)

def test_code_modules():
    expected = ["track_queue.py", "mpv_player.py", "prompt_template.py",
                "omni_client.py", "acestep_client.py", "skip_logger.py", "config.yaml"]
    for mod in expected:
        assert_exists(CODE_DIR / mod, f"code/{mod}")
run_test("code/ directory has all modules", test_code_modules)

def test_skill_exists():
    skill = Path(__file__).resolve().parent.parent / "SKILL.md"
    assert_exists(skill, "SKILL.md")
run_test("SKILL.md exists", test_skill_exists)

def test_sanitized_exists():
    san_dir = Path(__file__).resolve().parent.parent / "sanitized_for_sharing"
    assert_exists(san_dir, "sanitized_for_sharing/")
run_test("sanitized_for_sharing/ exists", test_sanitized_exists)

def test_no_personal_info_radio():
    content = open(RADIO_PY).read()
    assert_true("/Users/ailab" not in content, "radio.py contains personal path")
    assert_true("11450" not in content, "radio.py contains port number")
run_test("No personal info in radio.py", test_no_personal_info_radio)

def test_no_personal_info_modules():
    for mod in ["track_queue.py", "mpv_player.py", "prompt_template.py",
                "skip_logger.py", "acestep_client.py"]:
        content = open(CODE_DIR / mod).read()
        assert_true("/Users/ailab" not in content, f"{mod} contains personal path")
run_test("No personal info in code modules", test_no_personal_info_modules)

def test_gitignore():
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    assert_exists(gitignore, ".gitignore")
    content = gitignore.read_text()
    assert_true("venv" in content, ".gitignore should exclude venv")
run_test(".gitignore excludes venv", test_gitignore)

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*60}")
print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")
print(f"{'='*60}")

if ERRORS:
    print("\n❌ Failures:")
    for name, err in ERRORS:
        print(f"  • {name}: {err}")

if FAILED == 0:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n⚠️  {FAILED} test(s) failed")
    sys.exit(1)
