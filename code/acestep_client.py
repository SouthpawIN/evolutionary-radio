"""
acestep_client.py — ACE-Step audio generation via PyTorch + MPS.

Uses the official ace_step package with MPS backend on Apple Silicon.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

log = logging.getLogger("radio.acestep_client")

DEFAULT_OUTPUT_DIR = "~/path/to/audio_cache"


class AceStepClient:
    """Generate audio from text prompts using ACE-Step."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None

    def _ensure_model(self):
        """Lazy-load the ACE-Step pipeline."""
        if self._pipeline is not None:
            return

        log.info("loading ACE-Step model (may download on first use)")
        t0 = time.time()

        from acestep.pipeline_ace_step import ACEStepPipeline

        # Determine device and dtype
        if torch.backends.mps.is_available():
            device_id = 0
            dtype = "float32"  # MPS doesn't support bfloat16
        else:
            device_id = 0
            dtype = "bfloat16"

        self._pipeline = ACEStepPipeline(
            checkpoint_dir=self.checkpoint_path,
            device_id=device_id,
            dtype=dtype,
            cpu_offload=False,
            torch_compile=False,
        )

        log.info("ACE-Step loaded in %.1fs", time.time() - t0)

    def generate(
        self,
        tags: str,
        duration_sec: float = 60.0,
        num_inference_steps: int = 27,
        output_name: Optional[str] = None,
    ) -> tuple[str, float]:
        """Generate an audio file from a tag string.

        Returns (audio_path, generation_time_sec).
        """
        self._ensure_model()

        t0 = time.time()
        log.info("generating %.0fs audio from tags: %s", duration_sec, tags)

        # Save to a temp path, then copy to our output dir
        if output_name is None:
            output_name = f"track_{int(time.time())}.wav"
        output_path = self.output_dir / output_name

        self._pipeline(
            prompt=tags,
            lyrics="",  # instrumental — no lyrics
            audio_duration=duration_sec,
            infer_step=num_inference_steps,
            guidance_scale=0.0,  # ACE-Step uses CFG=0 for best instrumental quality
            save_path=str(output_path),
        )

        gen_time = time.time() - t0
        log.info("generated %s in %.1fs", output_path.name, gen_time)
        return str(output_path), gen_time
