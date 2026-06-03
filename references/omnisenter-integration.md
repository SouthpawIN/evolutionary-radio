# OmniSenter Integration — Evolutionary Radio as Plugin of the Broader System

**Date:** 2026-06-03
**Related:** `~/.hermes/skills/media/evolutionary-radio/SKILL.md`, `~/.hermes/skills/mlops/evolutionary-model-merging/SKILL.md`, `~/.hermes/skills/herm-tui-radio/SKILL.md`

## How the radio plugs into OmniSenter

The [[evolutionary-radio]] is **not** a standalone system — it's the **music-modality plugin** of the [[omnisenter-architecture]] (wiki: `concepts/omnisenter-architecture`). It owns the playback + queue + generation + evolution loops for music; the rest of OmniSenter owns text reasoning, vision, speech-in, speech-out, video, image.

```
OmniSenter 6A3B
├── core/nemotron/        :11400  Nemotron Nano 30B A3B (1M context, 3.5B active)
│                                  Central MoE + text reasoning + multimodal IN
├── experts/text/         :11450  OmniStep 6B (Darwin-merged Omni+ACE-Step text body)
│   └── :11452            OmniLance 6B (Darwin-merged Omni+Lance text body)
├── experts/multimodal/   :11451  Qwen2.5-Omni-3B (vision+audio IN, speech OUT via talker)
├── plugins/music/        :7860   ACE-Step 2B SFT (text → music)  ← EVOLUTIONARY RADIO LIVES HERE
│                                  ├── radio.py (this skill, 4 loops)
│                                  ├── mpv player (Unix socket IPC)
│                                  ├── asyncio queue (5 tracks)
│                                  ├── GEPA loop (background, slow)
│                                  └── Darwin CMA-ES (nightly, slowest)
├── plugins/video/        :7861   Lance_3B_Video DiT (text → video)
└── router/omnisenter     :11400  Intent-based dispatch
```

## Port assignments (as of 2026-06-03)

| Service | Port | VRAM | Status |
|---|---|---|---|
| Nemotron Nano 30B A3B | :11400 | 31GB FP8 (needs Blackwell) | Downloaded, not yet serving |
| OmniStep 6B (the radio's brain) | :11450 | ~5GB Q4_K_M | Running |
| OmniLance 6B | :11452 | ~5GB Q4_K_M | Running |
| Qwen2.5-Omni-3B | :11451 | ~6GB | Running |
| ACE-Step 2B SFT (the radio's voice) | :7860 | ~8-12GB | Installing |
| mpv player IPC | Unix socket `/tmp/hermes-mpv-<uuid>` | 0 | Per-radio-process |
| TUI music bar | n/a (publishes to state) | 0 | Wired in Ink TUI |

## The brain/voice split — why it works

The radio splits cleanly into:

- **Brain (text LLM, OmniStep 6B):** generates the *prompt* sent to ACE-Step. Knows about music genres, BPM, structure, vibes. Has ACE-Step's text encoder (Qwen2.5-3B class) merged into its weights via Darwin.
- **Voice (DiT decoder, ACE-Step 2B):** turns the prompt into audio. Doesn't need any text reasoning — pure diffusion. Same Qwen2.5-3B text encoder family as the brain, so the merge is shape-compatible.

This split means:
1. The brain can be evolved via Darwin without touching the voice's audio output
2. The voice can be evolved via Darwin (Loop 4) without touching the brain's text reasoning
3. The voice can be hot-swapped (e.g., 2B to 4B XL) without re-mergeing the brain
4. GEPA can evolve the *prompt template* between the two without touching either

## Skip-log format (the primary fitness signal)

`~/music/skip_log.jsonl` — append-only JSONL, one record per played track.

```jsonl
{"track_id": "uuid-1", "prompt_template_id": "vibe-lofi-v3", "prompt": "chill lofi beats for coding, 80bpm, jazzy chords", "generated_at": "2026-06-03T04:00:00Z", "played_seconds": 47, "total_seconds": 60, "skipped": false, "skip_at_seconds": null, "clap_score": 0.78, "fad_score": 12.4, "voice_model": "ace-step-v15-sft-2b", "voice_version": "2026-01-28"}
{"track_id": "uuid-2", "prompt_template_id": "vibe-lofi-v3", "prompt": "chill lofi beats for coding, 80bpm, jazzy chords", "generated_at": "2026-06-03T04:01:30Z", "played_seconds": 8, "total_seconds": 60, "skipped": true, "skip_at_seconds": 8, "clap_score": 0.42, "fad_score": 28.1, "voice_model": "ace-step-v15-sft-2b", "voice_version": "2026-01-28"}
```

GEPA reads this to compute `skip_rate` per `prompt_template_id`. Darwin reads `clap_score` and `fad_score` per `voice_version`.

## VRAM pressure and kill order

When the radio is active, the following is loaded on GPU 0 (RTX 3090, 24GB):

| Component | VRAM |
|---|---|
| OmniStep 6B (Q4_K_M) | ~5GB |
| ACE-Step 2B SFT (FP16 DiT) | ~8-12GB |
| mpv player | 0 (CPU) |
| **Total** | **~13-17GB** |

This leaves ~7-11GB for OS / other processes. The local llama servers (llama-main :8080, llama-aux :8081) each use ~6-8GB when awakened. To run the radio:

1. **Kill llama-main** (the local qwen3.6-27B Q4_K_M on GPU 0)
2. **Optionally keep llama-aux** (35B A3B on GPU 1, separate)
3. **Start ACE-Step 2B** on GPU 0
4. **Verify** with `nvidia-smi` before launching the radio

When the radio stops:
1. `radio stop` drains queue, kills mpv, unloads ACE-Step via SIGTERM
2. **Restart llama-main** on GPU 0 (the auto-start proxy will wake it on next ping)

## Related entities / wikilinks

- `concepts/evolutionary-radio` — the master concept page
- `concepts/omnisenter-architecture` — where the radio plugs into
- `concepts/generative-darwin-evolution` — the per-modality Darwin plan
- `concepts/darwin-family-paper` — the methodology
- `entities/ace-step-v15-sft-2b` — the default voice
- `entities/ace-step-1-5-xl-4b-dit` — the future HQ mode voice
- `entities/gepa-prompt-evolution` — the prompt evolution method
- `entities/acestep-music-pipeline` — the existing LoRA catalog
- `entities/nvidia-partnership` — Nemotron Nano access (the central MoE)

## Versioning of this reference

- 0.1.0 (2026-06-03) — initial draft. Captures port assignments, brain/voice split, skip-log schema, VRAM plan.
