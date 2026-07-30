"""
Composite scoring engine (§6.1–6.2).

Two-pass matching:
  1. Filter pass — drop candidates with zero tag overlap + incompatible constraints
  2. Scoring pass — weighted composite score with per-component breakdown

The scoring is exhaustive and deterministic: same query + same dataset = same output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Default weights (replaced by learned ranker in Phase 3) ─────────────────
DEFAULT_WEIGHTS = {
    "primary_technique": 0.30,
    "secondary_overlap": 0.15,
    "composition_sim": 0.20,
    "core_insight_sim": 0.25,
    "constraint_match": 0.10,
}


@dataclass
class CandidateFeatures:
    """Pre-computed features for a candidate problem."""
    problem_id: int
    primary_technique: str
    secondary_techniques: set[str]
    constraint_fingerprint: str
    core_insight_embedding: np.ndarray | None = None
    composition_embedding: np.ndarray | None = None

    # Metadata for display
    title: str = ""
    url: str = ""
    platform: str = ""
    core_insight: str = ""


@dataclass
class ScoreBreakdown:
    """Per-component breakdown of the composite score — keeps ranking auditable."""
    problem_id: int
    total_score: float
    primary_technique_score: float
    secondary_overlap_score: float
    composition_sim_score: float
    core_insight_sim_score: float
    constraint_match_score: float

    # Metadata
    title: str = ""
    url: str = ""
    platform: str = ""
    core_insight: str = ""
    primary_technique: str = ""
    secondary_techniques: set[str] = field(default_factory=set)


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Returns 0.0 if either vector is None.
    """
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


class Scorer:
    """
    Composite scoring engine.

    Usage:
        scorer = Scorer()
        results = scorer.score(query_features, candidate_features_list, top_n=20)
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def score(
        self,
        query: CandidateFeatures,
        candidates: list[CandidateFeatures],
        *,
        top_n: int = 20,
        min_score: float = 0.05,
    ) -> list[ScoreBreakdown]:
        """
        Two-pass matching:
          1. Filter: drop candidates with zero primary/secondary tag overlap
          2. Score: compute weighted composite + sort descending

        Returns top_n results with full score breakdowns.
        """
        query_tags = {query.primary_technique} | query.secondary_techniques

        results: list[ScoreBreakdown] = []

        for c in candidates:
            # Skip self-match
            if c.problem_id == query.problem_id:
                continue

            # ── Filter pass ─────────────────────────────────────────────
            candidate_tags = {c.primary_technique} | c.secondary_techniques
            if not query_tags & candidate_tags:
                continue  # zero overlap — skip

            # ── Scoring pass ────────────────────────────────────────────
            primary_score = float(query.primary_technique == c.primary_technique)
            secondary_score = jaccard(query.secondary_techniques, c.secondary_techniques)
            composition_score = cosine_similarity(
                query.composition_embedding, c.composition_embedding
            )
            insight_score = cosine_similarity(
                query.core_insight_embedding, c.core_insight_embedding
            )
            constraint_score = float(
                query.constraint_fingerprint == c.constraint_fingerprint
            )

            total = (
                self.weights["primary_technique"] * primary_score
                + self.weights["secondary_overlap"] * secondary_score
                + self.weights["composition_sim"] * composition_score
                + self.weights["core_insight_sim"] * insight_score
                + self.weights["constraint_match"] * constraint_score
            )

            if total < min_score:
                continue

            results.append(ScoreBreakdown(
                problem_id=c.problem_id,
                total_score=total,
                primary_technique_score=primary_score,
                secondary_overlap_score=secondary_score,
                composition_sim_score=composition_score,
                core_insight_sim_score=insight_score,
                constraint_match_score=constraint_score,
                title=c.title,
                url=c.url,
                platform=c.platform,
                core_insight=c.core_insight,
                primary_technique=c.primary_technique,
                secondary_techniques=c.secondary_techniques,
            ))

        # Sort by total score descending (deterministic — stable sort on equal scores)
        results.sort(key=lambda r: (-r.total_score, r.problem_id))

        # Apply "no strong match" threshold
        if results and results[0].total_score < 0.15:
            logger.info(
                f"Best match score is {results[0].total_score:.3f} — "
                f"below confidence threshold. Returning empty."
            )
            return []

        return results[:top_n]


def build_candidate_features(
    problem_id: int,
    primary_technique: str,
    secondary_techniques: list[str],
    constraint_fingerprint: str,
    core_insight_embedding: np.ndarray | None = None,
    composition_embedding: np.ndarray | None = None,
    title: str = "",
    url: str = "",
    platform: str = "",
    core_insight: str = "",
) -> CandidateFeatures:
    """Convenience factory for CandidateFeatures."""
    return CandidateFeatures(
        problem_id=problem_id,
        primary_technique=primary_technique,
        secondary_techniques=set(secondary_techniques),
        constraint_fingerprint=constraint_fingerprint,
        core_insight_embedding=core_insight_embedding,
        composition_embedding=composition_embedding,
        title=title,
        url=url,
        platform=platform,
        core_insight=core_insight,
    )
