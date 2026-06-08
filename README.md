# Evolutionary Radio

> **One prompt in → infinite generative music out.** Every track is freshly rendered by a voice model whose **weights** (Darwin) and **prompts** (GEPA) are continuously improving in the background.

A continuous-play generative music radio where **OmniStep 6B is the brain** and **ACE-Step 2B is the voice**. Four concurrent loops run side-by-side: real-time playback, queue fill, prompt evolution, and weight evolution.

**Naming:** "Evolutionary Radio" or "OmniStep Radio" — never "jam" or "jamming".

---

## What is this?

You give it a vibe — *chill lofi beats for coding*, *aggressive metal, 140bpm*, *dark ambient drone* — and it just keeps playing. The system is built on a single idea: **a model that gets better while you listen**. Two independent evolution loops run in the background:

- **GEPA** (prompt evolution, fast — minutes) rewrites the prompt template the brain sends to the voice.
- **Darwin** (weight evolution, slow — hours/overnight) re-merges the voice's DiT weights against another diffusion family via CMA-ES.

The primary user-facing fitness signal is **skip rate**. Lower = better.

---

## The four loops

```
┌────────────────────────────────────────────────────────────────────────┐
│  Evolutionary Radio (single Python process, asyncio)                   │
│                                                                        │
│  ┌─ Loop 1: PLAYBACK ─────────────────────────────────────────────┐    │
│  │  asyncio.Queue → mpv (--input-ipc-server=...) → EOF/skip → ... │    │
│  │  HARD real-time: must dequeue < track length                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌─ Loop 2: QUEUE FILL ───────────────────────────────────────────┐    │
│  │  if qsize < 5: POST /release_task to ACE-Step → poll result   │    │
│  │  → put track on queue. Sleep 5s, check again.                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌─ Loop 3: GEPA PROMPT EVOLUTION (slow) ──────────────────────────┐    │
│  │  read skip_log + CLAP scores → GEPA reflect → new prompt      │    │
│  │  cadence: every 50 generations OR 1 hour of listening          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  ┌─ Loop 4: DARWIN WEIGHT EVOLUTION (slowest) ─────────────────────┐    │
│  │  CMA-ES on ACE-Step × Stable Audio 2 → FAD/CLAP → update      │    │
│  │  cadence: nightly or weekend                                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

| Loop | Cadence | Block other loops? |
|---|---|---|
| 1. Playback (mpv) | Hard real-time | No — must dequeue < track length |
| 2. Queue fill (ACE-Step gen) | ~7-10s/track on RTX 3090 | No |
| 3. GEPA prompt evolve | Every 50 tracks or 1 hour | No (LLM API calls) |
| 4. Darwin weight evolve | Nightly/weekend | No (full merge, hours) |

---

## The brain/voice split

| Component | Model | Port | Role | VRAM |
|---|---|---|---|---|
| **Brain** | OmniStep 6B (Darwin-merged text LLM) | `:11450` | Generates ACE-Step prompt from vibe | ~5GB Q4_K_M |
| **Voice** | ACE-Step v1.5 SFT 2B (DiT) | `:7860` | Renders audio from prompt | ~8-12GB FP16 |
| **Player** | mpv (Unix socket IPC) | `/tmp/hermes-mpv-<uuid>` | Plays audio | 0 (CPU) |
| **Queue** | `asyncio.Queue(maxsize=5)` | in-process | Track buffer | 0 |
| **GEPA** | Prompt evolver | background | Reads `skip_log.jsonl`, mutates prompt | 0 (CPU) |
| **Darwin** | CMA-ES nightly | background | Re-merges ACE-Step DiT weights | GPU during runs |

The brain can be evolved without touching the voice's audio output. The voice can be hot-swapped (2B → 4B XL) without re-mergeing the brain. GEPA evolves the prompt template between them without touching either.

---

## Voice-model latency budget

Any voice that generates a 60s track in under 300s keeps the queue full (5 tracks × 60s = 300s of gen budget per 60s of playback).

| Voice model | Gen latency (60s song) | VRAM | Radio-friendly? |
|---|---|---|---|
| **ACE-Step v1.5 SFT 2B** ← default | ~7-10s on RTX 3090 | ~8-12GB | Yes — fill rate ≫ playback rate |
| ACE-Step v1.5 XL 4B DiT | ~20-30s | ~20-24GB | Tight — still keeps queue if target=5 |
| MusicGen-stereo-large 3.3B | ~30-60s | ~10GB | Slow but viable |
| Stable Audio 2 | ~10-20s | ~6GB | Yes (also a Darwin merge partner) |

---

## Quick start

```bash
# 1. Brain must be running
ss -tlnp | grep ":11450"          # OmniStep 6B

# 2. Voice must be running
curl -s http://localhost:7860/health

# 3. Clone + run
git clone https://github.com/SouthpawIN/evolutionary-radio ~/.hermes/skills/media/evolutionary-radio
cd ~/.hermes/skills/media/evolutionary-radio
python radio.py start --vibe="chill lofi beats for coding"

# 4. In another terminal
python radio.py status
python radio.py skip
python radio.py log
python radio.py stop
```

### Slash commands

| Command | Action |
|---|---|
| `radio start --vibe="..."` | Begin all 4 loops; pass vibe to OmniStep prompt template |
| `radio start --prompt="..."` | Use literal prompt instead of vibe |
| `radio stop` | Stop playback, drain queue, kill ACE-Step |
| `radio skip` | Skip current track, dequeue next |
| `radio pause` / `radio resume` | Pause / resume playback (queue fill continues) |
| `radio status` | Queue depth, last 5 tracks, last gen latency, prompt template, GEPA/Darwin gen count |
| `radio evolve` | Force a GEPA prompt generation now |
| `radio voice <model>` | Hot-swap DiT (2B → 4B XL); planned, requires VRAM check |
| `radio log` | Tail `~/music/skip_log.jsonl` |

---

## Architecture: how it works

### 1. The vibe-to-prompt pipeline

The brain (OmniStep 6B) takes a vibe string and turns it into an ACE-Step tag string:

```
"chill lofi beats for coding"
  → OmniStep (text reasoning)
  → "lofi, chillhop, 80bpm, jazzy chords, vinyl crackle, mellow keys, rainy mood"
  → ACE-Step (text → audio)
  → 60s .wav
```

The **prompt template** that OmniStep uses is the thing GEPA evolves. Start with a hand-written template, then GEPA mutates it.

### 2. The queue race

```
Time ─────────────────────────────────────────────────────────────►

[generate t0]  [generate t1]  [generate t2]  [generate t3]  [generate t4]
  7s            7s             7s             7s             7s
  ↓             ↓              ↓              ↓              ↓
[──play t0 60s──][──play t1 60s──][──play t2 60s──][──play t3 60s──]
                                                
queue: 4 3 4 3 4 5 4 3 4 3 4 ...
        ↑         ↑
        still    refilled
        playing  before empty
```

**5-track buffer** means the user can skip 5 times in a row and the music still keeps flowing. Below 3 → starvation risk. Below 1 → dead air.

### 3. The skip log (the fitness signal)

`~/music/skip_log.jsonl` — append-only JSONL, one record per played track:

```json
{"track_id": "uuid-1", "prompt_template_id": "vibe-lofi-v3",
 "prompt": "chill lofi beats for coding, 80bpm, jazzy chords",
 "generated_at": "2026-06-03T04:00:00Z", "played_seconds": 47,
 "total_seconds": 60, "skipped": false, "skip_at_seconds": null,
 "clap_score": 0.78, "fad_score": 12.4,
 "voice_model": "ace-step-v15-sft-2b", "voice_version": "2026-01-28"}
```

- **GEPA** reads `skip_rate` per `prompt_template_id`
- **Darwin** reads `clap_score` and `fad_score` per `voice_version`

### 4. VRAM plan (RTX 3090 24GB)

| Component | VRAM |
|---|---|
| OmniStep 6B (Q4_K_M, brain) | ~5GB |
| ACE-Step 2B SFT (FP16 DiT, voice) | ~8-12GB |
| mpv player | 0 (CPU) |
| **Total** | **~13-17GB** |

To start the radio: kill `llama-main` (the local 27B Q4_K_M on GPU 0) first, then start ACE-Step.

---

## Skill contents

```
evolutionary-radio/
├── SKILL.md                         # the master orchestrator doc
├── README.md                        # this file
├── LICENSE                          # MIT
├── references/
│   └── omnisenter-integration.md    # port assignments, brain/voice split, kill order
└── code/
    ├── config.yaml                  # ports, queue size, voice, brain, GEPA cadence
    ├── queue.py                     # asyncio.Queue wrapper with skip/pause API
    ├── mpv_player.py                # mpv IPC control (loadfile/pause/quit/volume)
    ├── prompt_template.py           # OmniStep prompt → ACE-Step tag string
    ├── darwin/                      # CMA-ES nightly evolver (TBD)
    ├── gepa/                        # prompt evolution loop (TBD)
    └── tests/                       # test suite (TBD)
```

---

## Key design decisions

1. **OmniStep is the brain, not a music model.** It generates the *prompt* sent to ACE-Step. This is what makes the system "evolutionary" in two independent dimensions simultaneously.
2. **ACE-Step 2B is the default voice, not 4B XL.** Speed matters more than quality for an infinite radio. The 2B's 7-10s generation is what makes the queue race winnable.
3. **The queue is the only hard constraint.** 3-5 tracks ahead means the user can skip 5 times in a row and music keeps flowing.
4. **Skip rate is the primary fitness signal.** Not CLAP, not FAD. The user is the ground truth.
5. **GEPA + Darwin are independent loops.** GEPA is fast (minutes). Darwin is slow (hours). They don't block each other. GEPA runs on the current voice; Darwin replaces the voice when ready.

---

## Pitfalls (must-read)

1. **mpv IPC socket cleanup** — `os.unlink(socket_path)` on shutdown. Use UUID-prefixed paths to avoid collisions.
2. **ACE-Step VRAM footprint** — don't launch ACE-Step if the brain + something else is already on GPU. Always check `nvidia-smi` first.
3. **mpv cannot use socat** — use Python's `socket` module directly. See `tui_gateway/server.py` for the working pattern.
4. **ACE-Step API status codes** — `0`=pending, `1`=complete, `-1`=failed (retry with different prompt; don't infinite-loop on the same prompt).
5. **Queue starvation under sustained skip** — 10 skips in a row drains the queue in 1-2 min. Drop the Loop 2 sleep from 5s to 1s during heavy skip.
6. **GEPA reflection quality** — a bad reflection prompt produces bad mutations. Use: *"This prompt produced a track the user skipped after N seconds. Given the user-skip rate, what tag combination in the prompt would have produced a more compelling track? Be specific."*
7. **Darwin evals are NOT real-time** — full FAD + CLAP on 100 FMA clips is 5-10 min/candidate. 20 candidates × 20 generations = **67-200 GPU-hours**. Run Darwin overnight, not interactively.
8. **mpv flags** — always pass `--no-video --no-terminal --quiet`. The radio process owns stdout.
9. **`signal.SIGTERM` propagation** — radio must propagate SIGTERM to mpv, ACE-Step subprocess, and GEPA loop. Use `asyncio.create_subprocess_exec`.

---

## Related skills

- [`evolutionary-model-merging`](../evolutionary-model-merging) — Darwin weight evolution (Loop 4). This skill calls into it.
- `gepa-prompt-evolution` (wiki entity) — the prompt evolution method (Loop 3).
- `herm-tui-radio` — the TUI bottom-bar widget that surfaces radio state.
- `media/heartmula` — Suno-like song gen; could be a swap-in voice.
- `creative/songwriting-and-ai-music` — lyrics + Suno prompting techniques for evolving the OmniStep prompt template.

---

## References

- Paper: Kim, T. et al. (2026). *Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning.* arXiv:2605.14386.
- Paper: Agrawal, L. et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:2507.19457.
- ACE-Step: `ACE-Step/ACE-Step-v1-3.5B` on HuggingFace
- OmniStep 6B: Darwin-merged Omni + ACE-Step text bodies (see `evolutionary-model-merging` skill)

---

## Author

Chris (SouthpawIN) — Senter Dev Discord, Nous Research
See also: [`evolutionary-model-merging`](../evolutionary-model-merging), [`multimodal-expansion`](../multimodal-expansion)

---

## Related: the OmniSenter system

Evolutionary Radio is one application of the OmniSenter architecture. The broader system:
- **OmniSenter-MoE-32A8B** — the main 5-stage pipeline. Synthesia adds cross-modal memory; Ohm adds self-evolution
- **Design post**: [OmniSenter: The Self-Evolving Multimodal Auxiliary for Hermes](https://github.com/SouthpawIN/evolutionary-training/blob/master/blog/omnisenter-self-evolving.md)
- **Architecture wiki**: `~/wiki/concepts/omnisenter-architecture.md`
- **Synthesia wiki** (cross-modal memory): `~/wiki/concepts/synthesia.md`
- **Ohm wiki** (self-evolving model): `~/wiki/concepts/omnisenter-ohm.md`
