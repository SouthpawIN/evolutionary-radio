"""
gepa.py — GEPA: Gradient-Free Prompt Evolution for the radio.

GEPA evolves prompt templates based on user feedback (likes/dislikes/skips).
It's a population-based approach: maintain N prompt variants, score them
by recent feedback, select the best, and mutate to create new variants.

This is Loop B (Prompt Evolution) in the radio architecture:
  1. Read feedback log → compute per-template scores
  2. Select top-K templates by rolling average
  3. Mutate: swap tags, adjust weights, blend templates
  4. Inject mutations into the template pool
  5. Queue fill loop samples from the pool

No gradients, no backprop — just evolutionary pressure from human ears.
"""
from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from feedback import FeedbackLogger, compute_sentiment_scores

log = logging.getLogger("radio.gepa")

# Tag categories for structured mutation
TAG_CATEGORIES = {
    "genre": ["lofi", "jazz", "ambient", "edm", "metal", "classical", "hip-hop", "rnb", "soul", "funk", "blues", "rock", "pop", "country", "reggae", "dub"],
    "mood": ["chill", "aggressive", "dreamy", "energetic", "melancholic", "uplifting", "dark", "warm", "ethereal", "intense", "relaxed", "groovy"],
    "tempo": ["60bpm", "80bpm", "100bpm", "110bpm", "120bpm", "128bpm", "140bpm", "160bpm"],
    "instrument": ["synth", "piano", "guitar", "strings", "drums", "bass", "sax", "pads", "vocals", "brass", "flute", "organ"],
    "texture": ["warm", "cold", "gritty", "clean", "distorted", "reverb", "delay", "vinyl hiss", "lo-fi", "hi-fi", "compressed"],
    "energy": ["low", "medium", "high", "building", "dropping", "steady"],
}


@dataclass
class PromptGenome:
    """A prompt template represented as evolvable parameters.

    The genome controls HOW the OmniStep brain generates ACE-Step tags.
    Different genomes produce different musical styles from the same vibe.
    """
    genome_id: str = ""
    # Tag preferences: category → list of preferred tags
    tag_weights: dict[str, list[str]] = field(default_factory=dict)
    # Temperature for OmniStep (higher = more creative, lower = more consistent)
    temperature: float = 0.7
    # System prompt modifier (injected into the OmniStep system prompt)
    style_modifier: str = ""
    # Fitness score (rolling average of feedback)
    fitness: float = 0.0
    # Number of tracks generated with this genome
    tracks_generated: int = 0
    # Lineage: parent genome IDs
    parents: list[str] = field(default_factory=list)
    # Birth timestamp
    born_at: float = field(default_factory=time.time)


# Default seed genomes — the founding population
SEED_GENOMES = [
    PromptGenome(
        genome_id="default",
        tag_weights={},
        temperature=0.7,
        style_modifier="",
    ),
    PromptGenome(
        genome_id="warm-analog",
        tag_weights={
            "texture": ["warm", "vinyl hiss", "lo-fi", "compressed"],
            "instrument": ["pads", "piano", "bass"],
            "energy": ["low", "steady"],
        },
        temperature=0.6,
        style_modifier="Prefer warm, analog textures. Vintage feel.",
    ),
    PromptGenome(
        genome_id="energetic-synth",
        tag_weights={
            "instrument": ["synth", "drums", "bass"],
            "energy": ["high", "building"],
            "mood": ["energetic", "uplifting"],
        },
        temperature=0.8,
        style_modifier="Prefer energetic, modern synth sounds.",
    ),
    PromptGenome(
        genome_id="ambient-space",
        tag_weights={
            "mood": ["dreamy", "ethereal", "melancholic"],
            "texture": ["reverb", "delay", "clean"],
            "instrument": ["pads", "strings", "piano"],
            "energy": ["low"],
        },
        temperature=0.5,
        style_modifier="Prefer spacious, ambient, atmospheric textures.",
    ),
]


class GEPAPool:
    """Population of prompt genomes with selection and mutation."""

    def __init__(
        self,
        feedback_logger: FeedbackLogger,
        pool_path: Optional[Path] = None,
        population_size: int = 8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
    ):
        self.feedback = feedback_logger
        self.pool_path = pool_path or Path("~/.local/share/evolutionary-radio/gepa_pool.json").expanduser()
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.genomes: list[PromptGenome] = []
        self._load_or_seed()

    def _load_or_seed(self):
        """Load pool from disk, or seed with defaults."""
        if self.pool_path.exists():
            try:
                with open(self.pool_path) as f:
                    data = json.load(f)
                self.genomes = [PromptGenome(**g) for g in data]
                log.info("loaded %d genomes from %s", len(self.genomes), self.pool_path)
                return
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("corrupt pool file, re-seeding: %s", e)

        self.genomes = copy.deepcopy(SEED_GENOMES)
        self._save()
        log.info("seeded GEPA pool with %d genomes", len(self.genomes))

    def _save(self):
        """Persist pool to disk."""
        data = []
        for g in self.genomes:
            data.append({
                "genome_id": g.genome_id,
                "tag_weights": g.tag_weights,
                "temperature": g.temperature,
                "style_modifier": g.style_modifier,
                "fitness": g.fitness,
                "tracks_generated": g.tracks_generated,
                "parents": g.parents,
                "born_at": g.born_at,
            })
        with open(self.pool_path, "w") as f:
            json.dump(data, f, indent=2)

    def select_genome(self) -> PromptGenome:
        """Tournament selection: pick the best from a random subset."""
        if not self.genomes:
            return copy.deepcopy(SEED_GENOMES[0])

        # 80% exploitation (tournament), 20% exploration (random)
        if random.random() < 0.8 and len(self.genomes) >= self.tournament_size:
            tournament = random.sample(self.genomes, min(self.tournament_size, len(self.genomes)))
            return max(tournament, key=lambda g: g.fitness)
        else:
            return random.choice(self.genomes)

    def update_fitness(self, genome_id: str, feedback_records: list[dict]):
        """Recalculate fitness for a genome based on recent feedback."""
        # Filter feedback for this genome
        genome_feedback = [r for r in feedback_records if r.get("genome_id") == genome_id]
        if not genome_feedback:
            return

        # Compute sentiment score
        sentiment_map = {"like": 1.0, "dislike": -1.0, "skip": -0.5, "skip_next": -0.3}
        scores = [sentiment_map.get(r.get("sentiment", ""), 0.0) for r in genome_feedback]

        # Rolling average with decay (recent feedback matters more)
        decay = 0.95
        weighted_sum = 0.0
        weight_total = 0.0
        for i, score in enumerate(scores):
            w = decay ** (len(scores) - 1 - i)
            weighted_sum += score * w
            weight_total += w

        new_fitness = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Update genome
        for g in self.genomes:
            if g.genome_id == genome_id:
                # Smooth update (don't jump too fast)
                g.fitness = 0.7 * g.fitness + 0.3 * new_fitness
                g.tracks_generated += len(genome_feedback)
                break

    def mutate(self, parent: PromptGenome) -> PromptGenome:
        """Create a mutated copy of a genome."""
        child = PromptGenome(
            genome_id=f"gen-{int(time.time())}-{random.randint(100,999)}",
            tag_weights=copy.deepcopy(parent.tag_weights),
            temperature=parent.temperature,
            style_modifier=parent.style_modifier,
            parents=[parent.genome_id],
            born_at=time.time(),
        )

        # Mutate tag weights (30% chance per category)
        for category, options in TAG_CATEGORIES.items():
            if random.random() < self.mutation_rate:
                # Add or replace tags in this category
                n_tags = random.randint(1, min(3, len(options)))
                child.tag_weights[category] = random.sample(options, n_tags)

        # Mutate temperature (±0.1)
        if random.random() < self.mutation_rate:
            child.temperature = max(0.1, min(1.5, parent.temperature + random.uniform(-0.1, 0.1)))

        # Mutate style modifier (20% chance)
        if random.random() < 0.2:
            modifiers = [
                "Prefer warm, analog textures.",
                "Go for something unexpected.",
                "Keep it minimal and sparse.",
                "Make it lush and layered.",
                "Focus on rhythm and groove.",
                "Explore dissonance and tension.",
                "Aim for cinematic atmosphere.",
                "Keep the energy steady and hypnotic.",
            ]
            child.style_modifier = random.choice(modifiers)

        log.info("mutated %s → %s (temp=%.2f)", parent.genome_id, child.genome_id, child.temperature)
        return child

    def evolve(self):
        """Run one generation of GEPA evolution.

        1. Update fitness scores from recent feedback
        2. Select parents via tournament
        3. Mutate to fill empty slots
        4. Prune weakest if over population size
        """
        records = self.feedback.get_recent(200)

        # Update fitness for all genomes
        for g in self.genomes:
            self.update_fitness(g.genome_id, records)

        # Sort by fitness
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

        # Prune to population size (keep top performers)
        if len(self.genomes) > self.population_size:
            self.genomes = self.genomes[:self.population_size]

        # Fill empty slots with mutations
        while len(self.genomes) < self.population_size:
            parent = self.select_genome()
            child = self.mutate(parent)
            self.genomes.append(child)

        # Occasionally create new mutations even when full (diversity injection)
        if random.random() < 0.15 and len(self.genomes) >= 2:
            parent = self.select_genome()
            child = self.mutate(parent)
            # Replace the weakest
            weakest = min(self.genomes, key=lambda g: g.fitness)
            if weakest.fitness < child.fitness or random.random() < 0.1:
                self.genomes.remove(weakest)
                self.genomes.append(child)
                log.info("replaced weak genome %s with %s", weakest.genome_id, child.genome_id)

        self._save()
        log.info("GEPA evolved: %d genomes, best=%.3f", len(self.genomes), self.genomes[0].fitness if self.genomes else 0)

    def build_system_prompt(self, genome: PromptGenome, base_prompt: str) -> str:
        """Inject genome preferences into the OmniStep system prompt."""
        parts = [base_prompt]

        if genome.style_modifier:
            parts.append(f"\nStyle preference: {genome.style_modifier}")

        if genome.tag_weights:
            prefs = []
            for category, tags in genome.tag_weights.items():
                prefs.append(f"{category}: {', '.join(tags)}")
            parts.append(f"\nPreferred tags by category:\n" + "\n".join(prefs))

        return "\n".join(parts)
