"""
Reranking + explanation generation (§6.3).

Two modes:
  - With APIs (Voyage + Anthropic): full reranking + LLM explanations
  - Without APIs (Ollama / local): template-based explanations only

The system works fully without paid APIs — explanations are just less polished.
"""

from __future__ import annotations

import logging

from config import get_settings
from matching.score import ScoreBreakdown

logger = logging.getLogger(__name__)


class Reranker:
    """
    Optional reranking + explanation layer.
    Degrades gracefully when APIs aren't available.
    """

    def __init__(self):
        self._settings = get_settings()

    def rerank(
        self,
        query_text: str,
        candidates: list[ScoreBreakdown],
        *,
        top_n: int = 10,
    ) -> list[ScoreBreakdown]:
        """
        Rerank candidates. Uses Voyage rerank API if available,
        otherwise returns the original order (already sorted by composite score).
        """
        if not candidates:
            return []

        # Try Voyage rerank if available
        if self._settings.embedding_provider == "voyage" and self._settings.voyage_api_key:
            try:
                return self._voyage_rerank(query_text, candidates, top_n)
            except Exception as e:
                logger.warning(f"Voyage rerank failed, using original order: {e}")

        return candidates[:top_n]

    def generate_explanation(
        self,
        query_insight: str,
        query_technique: str,
        match: ScoreBreakdown,
    ) -> str:
        """
        Generate a human-readable explanation of why the match is similar.
        Uses LLM if available, otherwise generates a template-based explanation.
        """
        # Try LLM-based explanation
        if self._settings.llm_provider == "ollama":
            try:
                return self._ollama_explain(query_insight, query_technique, match)
            except Exception as e:
                logger.debug(f"Ollama explanation failed: {e}")
        elif self._settings.llm_provider == "anthropic" and self._settings.anthropic_api_key:
            try:
                return self._anthropic_explain(query_insight, query_technique, match)
            except Exception as e:
                logger.debug(f"Anthropic explanation failed: {e}")
        elif self._settings.llm_provider == "gemini" and self._settings.gemini_api_key:
            try:
                return self._gemini_explain(query_insight, query_technique, match)
            except Exception as e:
                logger.debug(f"Gemini explanation failed: {e}")

        # Fallback: template-based
        return self._template_explanation(query_technique, match)

    def _build_explain_prompt(
        self, query_insight: str, query_technique: str, match: ScoreBreakdown
    ) -> str:
        return (
            f"You are explaining why two competitive programming problems are similar.\n\n"
            f'Query problem\'s core insight: "{query_insight}"\n'
            f"Query problem\'s primary technique: {query_technique}\n\n"
            f'Matched problem\'s core insight: "{match.core_insight}"\n'
            f"Matched problem\'s primary technique: {match.primary_technique}\n\n"
            f"Score breakdown:\n"
            f"- Primary technique match: {match.primary_technique_score:.0%}\n"
            f"- Secondary technique overlap: {match.secondary_overlap_score:.0%}\n"
            f"- Composition similarity: {match.composition_sim_score:.0%}\n"
            f"- Core insight similarity: {match.core_insight_sim_score:.0%}\n"
            f"- Constraint match: {match.constraint_match_score:.0%}\n\n"
            f"Write 1-2 sentences explaining the conceptual similarity. "
            f"Be specific about the shared algorithmic structure."
        )

    def _ollama_explain(
        self, query_insight: str, query_technique: str, match: ScoreBreakdown
    ) -> str:
        import httpx
        prompt = self._build_explain_prompt(query_insight, query_technique, match)
        response = httpx.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={
                "model": self._settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _anthropic_explain(
        self, query_insight: str, query_technique: str, match: ScoreBreakdown
    ) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        prompt = self._build_explain_prompt(query_insight, query_technique, match)
        response = client.messages.create(
            model=self._settings.get_extraction_models()[1],
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _gemini_explain(
        self, query_insight: str, query_technique: str, match: ScoreBreakdown
    ) -> str:
        import httpx
        prompt = self._build_explain_prompt(query_insight, query_technique, match)
        model = self._settings.get_extraction_models()[0]
        model_name = model if model.startswith("models/") else f"models/{model}"
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent",
            params={"key": self._settings.gemini_api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 200},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _voyage_rerank(
        self, query_text: str, candidates: list[ScoreBreakdown], top_n: int
    ) -> list[ScoreBreakdown]:
        import voyageai
        client = voyageai.Client(api_key=self._settings.voyage_api_key)
        documents = [
            f"{c.primary_technique}: {c.core_insight}" for c in candidates
        ]
        result = client.rerank(
            query=query_text,
            documents=documents,
            model="rerank-2",
            top_k=min(top_n, len(candidates)),
        )
        return [candidates[item.index] for item in result.results]

    def _template_explanation(self, query_technique: str, match: ScoreBreakdown) -> str:
        """Generate a template-based explanation (no LLM needed)."""
        parts = []
        if match.primary_technique_score > 0:
            parts.append(f"Both require {match.primary_technique}")
        if match.core_insight_sim_score > 0.5:
            parts.append("sharing a similar core algorithmic insight")
        if match.constraint_match_score > 0:
            parts.append("under similar constraints")
        if match.secondary_overlap_score > 0:
            parts.append("with overlapping secondary techniques")

        if parts:
            return ". ".join(parts) + "."
        return "Shares overlapping technique tags and constraint profile."
