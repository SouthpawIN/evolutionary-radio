# Evolutionary Radio — code/ directory

**Stripped-down (2026-06-09, per Chris):** Darwin merge, GEPA prompt
evolution, feedback logger, and skip logger are moved to `advanced/`
and disabled by default. The default radio is now: play, generate,
polite-chat. Simple. The advanced systems are still available —
opt-in via `RADIO_ENABLE_DARWIN=1` and `RADIO_ENABLE_GEPA=1` env vars.

## Default (stripped-down) files

| File | Lines | What |
|---|---|---|
| `omni_client.py` | ~180 | HTTP client for the omni-va slot at `:8082/v1`. Polite VRAM. |
| `acestep_client.py` | ~95 | HTTP client for ACE-Step music generation. |
| `mpv_player.py` | ~260 | Single-reader mpv wrapper. The audio playback engine. |
| `queue.py` | ~165 | The data plane for Loops 1+2 (track queue). |
| `track_queue.py` | ~165 | Track queue data structure. |
| `prompt_template.py` | ~110 | The prompt template the OmniStep brain uses. |
| `brain.py` | ~305 | The OmniStep brain wrapper. Drives music + note-taker. |

**Total: ~1280 lines** (down from ~2115 — 40% smaller).

## advanced/ files (opt-in)

| File | Lines | What |
|---|---|---|
| `advanced/darwin.py` | ~336 | CMA-ES weight evolution (overnight). **Disabled by default.** |
| `advanced/gepa.py` | ~297 | Prompt template evolution (real-time). **Disabled by default.** |
| `advanced/feedback.py` | ~133 | User feedback records (skips, likes, dislikes). |
| `advanced/skip_logger.py` | ~70 | Per-track skip records. |

**Total advanced: ~835 lines.** Available via lazy-import when
`RADIO_ENABLE_DARWIN=1` or `RADIO_ENABLE_GEPA=1` is set.

## Entry point

`radio.py` (in the parent dir) is the CLI:
```
python3 radio.py start --vibe "chill lofi beats for coding"
python3 radio.py stop
python3 radio.py skip
python3 radio.py status
python3 radio.py evolve   # advanced — requires RADIO_ENABLE_GEPA + RADIO_ENABLE_DARWIN
```

## The chat interface is Hermes

The radio is just a music engine. The chat interface with the
OmniStep agent is Hermes itself. To chat:

```bash
# Spawn Hermes with the OmniStep agent as the system prompt
hermes launch --from-md ~/.hermes/hub/agents/omni-step.md
```

Or via the omni-va proxy:
```bash
curl -X POST http://127.0.0.1:8082/hermes/launch \
  -H "Content-Type: application/json" \
  -d '{"agent": "omni-step", "query": "what should I play next?"}'
```

No custom TUI needed. Hermes IS the chat interface.
