"""
darwin.py — Darwin Family Evolution for the radio's prompt genomes.

While GEPA evolves prompt TEMPLATES (which tags to suggest), Darwin evolves
the underlying GENOME VECTOR — the 14-dimensional parameter space that
controls HOW templates are combined and weighted.

From arXiv:2605.14386 (Kim et al., 2026):
  genome g = (γ, α_attn, α_ffn, α_emb, ρA, ρB, r0..r5, τ, λ)

For the radio, these map to:
  γ    — global creativity ratio (how much to deviate from seed)
  α_*  — per-aspect weights (genre, mood, tempo, instrument, texture, energy)
  ρA/ρB — parent density (how much to trust parent A vs B in merges)
  r0..r5 — block ratios (6 tag categories: genre/mood/tempo/instrument/texture/energy)
  τ    — MRI-Trust coefficient (blend between diagnostic and genome signals)
  λ    — regularization (penalize extreme values)

The Darwin loop:
  1. Maintain a population of genome vectors
  2. Score each genome by rolling feedback average
  3. Select parents via tournament
  4. Merge via MRI-Trust Fusion
  5. Benchmark child vs parents
  6. Keep the winner
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from feedback import FeedbackLogger, compute_sentiment_scores

log = logging.getLogger("radio.darwin")


@dataclass
class DarwinGenome:
    """14-dimensional genome vector for evolutionary prompt optimization.

    Maps the Darwin Family paper's genome to radio prompt parameters.
    """
    genome_id: str = ""

    # Core parameters (6)
    gamma: float = 0.5        # Global creativity ratio [0, 1]
    alpha_attn: float = 0.5   # Genre attention weight [0, 1]
    alpha_ffn: float = 0.5    # Mood/feedforward weight [0, 1]
    alpha_emb: float = 0.5    # Embedding/tempo weight [0, 1]
    rho_a: float = 0.5        # Parent A density [0, 1]
    rho_b: float = 0.5        # Parent B density [0, 1]

    # Block ratios (6) — one per tag category
    r_genre: float = 0.5
    r_mood: float = 0.5
    r_tempo: float = 0.5
    r_instrument: float = 0.5
    r_texture: float = 0.5
    r_energy: float = 0.5

    # Hyper parameters (2)
    tau: float = 0.4          # MRI-Trust coefficient [0, 1]
    lam: float = 0.01         # Regularization [0, 0.1]

    # Metadata
    fitness: float = 0.0
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    born_at: float = field(default_factory=time.time)
    tracks_scored: int = 0

    def to_vector(self) -> list[float]:
        """Flatten to a 14-dimensional vector."""
        return [
            self.gamma, self.alpha_attn, self.alpha_ffn, self.alpha_emb,
            self.rho_a, self.rho_b,
            self.r_genre, self.r_mood, self.r_tempo, self.r_instrument, self.r_texture, self.r_energy,
            self.tau, self.lam,
        ]

    @classmethod
    def from_vector(cls, vec: list[float], genome_id: str = "", **kwargs) -> "DarwinGenome":
        """Reconstruct from a 14-dimensional vector."""
        return cls(
            genome_id=genome_id,
            gamma=vec[0], alpha_attn=vec[1], alpha_ffn=vec[2], alpha_emb=vec[3],
            rho_a=vec[4], rho_b=vec[5],
            r_genre=vec[6], r_mood=vec[7], r_tempo=vec[8],
            r_instrument=vec[9], r_texture=vec[10], r_energy=vec[11],
            tau=vec[12], lam=vec[13],
            **kwargs,
        )

    def clamp(self):
        """Keep all values in valid ranges."""
        self.gamma = max(0.0, min(1.0, self.gamma))
        self.alpha_attn = max(0.0, min(1.0, self.alpha_attn))
        self.alpha_ffn = max(0.0, min(1.0, self.alpha_ffn))
        self.alpha_emb = max(0.0, min(1.0, self.alpha_emb))
        self.rho_a = max(0.0, min(1.0, self.rho_a))
        self.rho_b = max(0.0, min(1.0, self.rho_b))
        self.r_genre = max(0.0, min(1.0, self.r_genre))
        self.r_mood = max(0.0, min(1.0, self.r_mood))
        self.r_tempo = max(0.0, min(1.0, self.r_tempo))
        self.r_instrument = max(0.0, min(1.0, self.r_instrument))
        self.r_texture = max(0.0, min(1.0, self.r_texture))
        self.r_energy = max(0.0, min(1.0, self.r_energy))
        self.tau = max(0.0, min(1.0, self.tau))
        self.lam = max(0.0, min(0.1, self.lam))


# Seed genomes
SEED_DARWIN_GENOMES = [
    DarwinGenome(genome_id="darwin-balanced", gamma=0.5, generation=0),
    DarwinGenome(genome_id="darwin-creative", gamma=0.8, tau=0.6, generation=0),
    DarwinGenome(genome_id="darwin-conservative", gamma=0.2, tau=0.2, generation=0),
    DarwinGenome(genome_id="darwin-genre-heavy", alpha_attn=0.8, r_genre=0.8, generation=0),
    DarwinGenome(genome_id="darwin-mood-heavy", alpha_ffn=0.8, r_mood=0.8, generation=0),
]


def mri_trust_fusion(
    parent_a: DarwinGenome,
    parent_b: DarwinGenome,
    child_id: str,
    diagnostic_signal: Optional[dict] = None,
) -> DarwinGenome:
    """MRI-Trust Fusion: blend two parent genomes using per-tensor mixing.

    θM(T) = (1 - r_final(T)) · θA(T) + r_final(T) · θB(T)
    r_final(T) = τ · r_MRI(T) + (1 - τ) · r_genome(T)

    For the radio, the "diagnostic signal" comes from feedback:
      - entropy: diversity of liked tags
      - variance: consistency of sentiment
      - cosine distance: how different parent styles are
    """
    vec_a = parent_a.to_vector()
    vec_b = parent_b.to_vector()

    # Default diagnostic: uniform mixing
    r_mri = [0.5] * 14
    if diagnostic_signal:
        # Use feedback to bias the mixing ratios
        tag_scores = diagnostic_signal.get("tag_scores", {})
        # If parent A's style tags are more liked, bias toward A
        a_score = diagnostic_signal.get("parent_a_fitness", 0.5)
        b_score = diagnostic_signal.get("parent_b_fitness", 0.5)
        total = a_score + b_score + 1e-8
        bias = a_score / total  # How much to favor parent A
        r_mri = [1.0 - bias] * 14  # r_MRI = proportion from parent B

    # Genome-specified mixing ratio
    tau = (parent_a.tau + parent_b.tau) / 2
    r_genome = [(a + b) / 2 for a, b in zip(
        [parent_a.rho_a] * 14,
        [parent_b.rho_b] * 14,
    )]

    # Final mixing: r_final = τ · r_MRI + (1-τ) · r_genome
    r_final = [tau * m + (1 - tau) * g for m, g in zip(r_mri, r_genome)]

    # Blend vectors
    child_vec = [(1 - r) * a + r * b for r, a, b in zip(r_final, vec_a, vec_b)]

    child = DarwinGenome.from_vector(
        child_vec,
        genome_id=child_id,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parents=[parent_a.genome_id, parent_b.genome_id],
    )
    child.clamp()

    # Apply regularization (penalize extreme values)
    for i in range(14):
        v = child.to_vector()[i]
        if v < 0.1 or v > 0.9:
            # Nudge toward center
            center = 0.5
            child_vec[i] = v + child.lam * (center - v)

    child = DarwinGenome.from_vector(
        child_vec,
        genome_id=child_id,
        generation=child.generation,
        parents=child.parents,
    )
    child.clamp()

    log.info("MRI-Trust fusion: %s + %s → %s (τ=%.2f)",
             parent_a.genome_id, parent_b.genome_id, child_id, tau)
    return child


class DarwinPopulation:
    """Population of Darwin genomes with evolution loop."""

    def __init__(
        self,
        feedback_logger: FeedbackLogger,
        pop_path: Optional[Path] = None,
        population_size: int = 8,
        tournament_size: int = 3,
    ):
        self.feedback = feedback_logger
        self.pop_path = pop_path or Path("~/.local/share/evolutionary-radio/darwin_pop.json").expanduser()
        self.pop_path.parent.mkdir(parents=True, exist_ok=True)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.genomes: list[DarwinGenome] = []
        self._load_or_seed()

    def _load_or_seed(self):
        if self.pop_path.exists():
            try:
                with open(self.pop_path) as f:
                    data = json.load(f)
                self.genomes = [DarwinGenome(**g) for g in data]
                log.info("loaded %d Darwin genomes", len(self.genomes))
                return
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("corrupt Darwin pop, re-seeding: %s", e)

        self.genomes = copy.deepcopy(SEED_DARWIN_GENOMES)
        self._save()

    def _save(self):
        data = []
        for g in self.genomes:
            data.append({
                "genome_id": g.genome_id, "gamma": g.gamma,
                "alpha_attn": g.alpha_attn, "alpha_ffn": g.alpha_ffn, "alpha_emb": g.alpha_emb,
                "rho_a": g.rho_a, "rho_b": g.rho_b,
                "r_genre": g.r_genre, "r_mood": g.r_mood, "r_tempo": g.r_tempo,
                "r_instrument": g.r_instrument, "r_texture": g.r_texture, "r_energy": g.r_energy,
                "tau": g.tau, "lam": g.lam,
                "fitness": g.fitness, "generation": g.generation,
                "parents": g.parents, "born_at": g.born_at, "tracks_scored": g.tracks_scored,
            })
        with open(self.pop_path, "w") as f:
            json.dump(data, f, indent=2)

    def update_fitness(self, genome_id: str, records: list[dict]):
        """Update fitness from feedback."""
        genome_recs = [r for r in records if r.get("genome_id") == genome_id]
        if not genome_recs:
            return
        sentiment_map = {"like": 1.0, "dislike": -1.0, "skip": -0.5, "skip_next": -0.3}
        scores = [sentiment_map.get(r.get("sentiment", ""), 0.0) for r in genome_recs]
        decay = 0.95
        weighted = sum(s * decay ** (len(scores) - 1 - i) for i, s in enumerate(scores))
        weight_sum = sum(decay ** i for i in range(len(scores)))
        new_fitness = weighted / weight_sum if weight_sum > 0 else 0.0
        for g in self.genomes:
            if g.genome_id == genome_id:
                g.fitness = 0.7 * g.fitness + 0.3 * new_fitness
                g.tracks_scored += len(genome_recs)
                break

    def tournament_select(self) -> DarwinGenome:
        """Tournament selection."""
        if random.random() < 0.8:
            pool = random.sample(self.genomes, min(self.tournament_size, len(self.genomes)))
            return max(pool, key=lambda g: g.fitness)
        return random.choice(self.genomes)

    def evolve_generation(self):
        """Run one generation of Darwin evolution."""
        records = self.feedback.get_recent(200)

        # Update all fitness scores
        for g in self.genomes:
            self.update_fitness(g.genome_id, records)

        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

        # Keep top performers
        survivors = self.genomes[:max(2, self.population_size // 2)]

        # Create children via MRI-Trust Fusion
        children = []
        while len(survivors) + len(children) < self.population_size:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()
            while parent_b.genome_id == parent_a.genome_id and len(self.genomes) > 1:
                parent_b = self.tournament_select()

            child_id = f"darwin-gen{max(g.generation for g in self.genomes)+1}-{random.randint(100,999)}"

            # Build diagnostic signal from feedback
            diagnostic = {
                "parent_a_fitness": parent_a.fitness,
                "parent_b_fitness": parent_b.fitness,
                "tag_scores": compute_sentiment_scores(records),
            }

            child = mri_trust_fusion(parent_a, parent_b, child_id, diagnostic)
            children.append(child)

        self.genomes = survivors + children
        self._save()

        log.info("Darwin generation complete: %d genomes, best=%.3f (%s)",
                 len(self.genomes), self.genomes[0].fitness, self.genomes[0].genome_id)

    def get_best(self) -> DarwinGenome:
        """Get the highest-fitness genome."""
        if not self.genomes:
            return SEED_DARWIN_GENOMES[0]
        return max(self.genomes, key=lambda g: g.fitness)

    def genome_to_vibe_bias(self, genome: DarwinGenome) -> dict:
        """Convert a Darwin genome into bias parameters for the queue fill loop.

        Returns a dict that can be injected into the OmniStep prompt.
        """
        return {
            "creativity": genome.gamma,
            "genre_weight": genome.alpha_attn,
            "mood_weight": genome.alpha_ffn,
            "tempo_weight": genome.alpha_emb,
            "genre_ratio": genome.r_genre,
            "mood_ratio": genome.r_mood,
            "tempo_ratio": genome.r_tempo,
            "instrument_ratio": genome.r_instrument,
            "texture_ratio": genome.r_texture,
            "energy_ratio": genome.r_energy,
            "trust": genome.tau,
        }
