"""
LLM-based extraction pipeline (§4.2–4.5).

Supports three LLM providers:
  - Ollama (free, local) — default
  - Google Gemini (free tier: 15 RPM, 1M tokens/day)
  - Anthropic Claude (paid, highest quality)

Handles:
  - Single problem extraction (synchronous, for query-time)
  - Self-consistency: multiple runs + agreement check
  - Model routing: fast model first, escalate ambiguous/hard cases
  - Cross-checking against the deterministic parser output
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from config import get_settings
from extraction.parser import ParsedConstraints, parse_constraints
from extraction.schema import ProblemSchema, TagEvidence, get_tool_schema
from extraction.taxonomy import (
    get_taxonomy_prompt_block,
    is_valid_archetype,
    is_valid_framing,
    validate_or_flag,
)

logger = logging.getLogger(__name__)

# Load the decomposition prompt template
_PROMPT_PATH = Path(__file__).parent / "prompts" / "decomposition.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


# ── LLM Provider Abstraction ───────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    def extract_schema(self, problem_text: str, model: str) -> ProblemSchema:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """
    Uses Ollama for local, free LLM inference.
    Install: https://ollama.com
    Pull a model: ollama pull qwen2.5:7b
    """

    def __init__(self):
        import httpx
        settings = get_settings()
        self._base_url = settings.ollama_base_url
        self._client = httpx.Client(timeout=120.0)

    def extract_schema(self, problem_text: str, model: str) -> ProblemSchema:
        taxonomy_block = get_taxonomy_prompt_block()
        prompt = _PROMPT_TEMPLATE.format(
            taxonomy_block=taxonomy_block,
            problem=problem_text,
        )

        # Add JSON instruction since Ollama doesn't have tool_choice
        json_instruction = self._get_json_instruction()
        full_prompt = prompt + "\n\n" + json_instruction

        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 2000,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "")

        return self._parse_response(raw_text)

    def _get_json_instruction(self) -> str:
        schema = get_tool_schema()
        fields = list(schema.get("properties", {}).keys())
        return (
            "IMPORTANT: You MUST respond with a valid JSON object containing these fields: "
            f"{fields}\n\n"
            "Respond ONLY with the JSON object, no other text. Example structure:\n"
            '{\n'
            '  "primary_technique": "dp-bitmask",\n'
            '  "secondary_techniques": ["binary-search-answer"],\n'
            '  "composition_pattern": "Use bitmask DP to enumerate subsets, then binary search for optimal threshold",\n'
            '  "archetype": ["observation-reduces-search-space"],\n'
            '  "framing": "optimization",\n'
            '  "nearest_classical_analogue": "variant of TSP with bitmask DP",\n'
            '  "constraint_fingerprint": "n<=20 -> O(2^n * n)",\n'
            '  "core_insight": "The small N constraint signals bitmask enumeration",\n'
            '  "concept_count": 2,\n'
            '  "evidence": [{"tag": "dp-bitmask", "evidence": "n <= 20 suggests exponential approach"}],\n'
            '  "schema_version": 1\n'
            '}'
        )

    def _parse_response(self, raw_text: str) -> ProblemSchema:
        """Parse the raw LLM response into a ProblemSchema."""
        # Try to extract JSON from the response
        raw_text = raw_text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(raw_text)
            return ProblemSchema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Try to find JSON in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return ProblemSchema.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Try to find any JSON object in the response
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return ProblemSchema.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        raise ExtractionError(f"Could not parse LLM response as JSON: {raw_text[:500]}")


class GeminiProvider(LLMProvider):
    """
    Uses Google Gemini API (free tier: 15 RPM, 1M tokens/day).
    Get key: https://aistudio.google.com/apikey
    """

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.gemini_api_key
        if not self._api_key:
            raise ExtractionError(
                "GEMINI_API_KEY not set. Get one free at https://aistudio.google.com/apikey"
            )

    def extract_schema(self, problem_text: str, model: str) -> ProblemSchema:
        import httpx

        taxonomy_block = get_taxonomy_prompt_block()
        prompt = _PROMPT_TEMPLATE.format(
            taxonomy_block=taxonomy_block,
            problem=problem_text,
        )

        schema = get_tool_schema()
        json_instruction = (
            "Respond with a JSON object matching this schema. "
            "Output ONLY valid JSON, nothing else.\n"
            f"Schema: {json.dumps(schema, indent=2)}"
        )

        full_prompt = prompt + "\n\n" + json_instruction

        import time
        model_name = model if model.startswith("models/") else f"models/{model}"

        for attempt in range(4):
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent",
                params={"key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 2000,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=60.0,
            )
            if response.status_code == 429 and attempt < 3:
                wait_time = (attempt + 1) * 5
                logger.warning(f"Gemini API rate limited (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            break
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        return ProblemSchema.model_validate(parsed)


class AnthropicProvider(LLMProvider):
    """
    Uses Anthropic Claude API (paid, highest quality).
    Uses tool_choice for guaranteed schema-valid output.
    """

    def __init__(self):
        import anthropic
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ExtractionError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def extract_schema(self, problem_text: str, model: str) -> ProblemSchema:
        taxonomy_block = get_taxonomy_prompt_block()
        prompt = _PROMPT_TEMPLATE.format(
            taxonomy_block=taxonomy_block,
            problem=problem_text,
        )

        tool_def = {
            "name": "extract_problem_schema",
            "description": "Extract the structured algorithmic schema of a competitive programming problem.",
            "input_schema": get_tool_schema(),
        }

        response = self._client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "extract_problem_schema"},
            messages=[{"role": "user", "content": prompt}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_problem_schema":
                return ProblemSchema.model_validate(block.input)

        raise ExtractionError("Claude did not return a tool_use block")


class OpenAIProvider(LLMProvider):
    """
    Uses OpenAI API client. Compatible with OpenAI, Groq, DeepSeek, OpenRouter, etc.
    """
    def __init__(self):
        import openai
        settings = get_settings()
        if not settings.openai_api_key:
            raise ExtractionError("OPENAI_API_KEY not set")
        
        # If openai_base_url is empty, it defaults to OpenAI's official URL
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url if settings.openai_base_url else None
        )

    def extract_schema(self, problem_text: str, model: str) -> ProblemSchema:
        taxonomy_block = get_taxonomy_prompt_block()
        prompt = _PROMPT_TEMPLATE.format(
            taxonomy_block=taxonomy_block,
            problem=problem_text,
        )

        tool_def = {
            "type": "function",
            "function": {
                "name": "extract_problem_schema",
                "description": "Extract the structured algorithmic schema of a competitive programming problem.",
                "parameters": get_tool_schema(),
            }
        }

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": "extract_problem_schema"}},
            temperature=0,
            max_tokens=1500,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise ExtractionError("OpenAI did not return a tool_call")
            
        args = message.tool_calls[0].function.arguments
        try:
            data = json.loads(args)
            return ProblemSchema.model_validate(data)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"OpenAI returned invalid JSON: {e}")


def _get_provider() -> LLMProvider:
    """Factory: return the configured LLM provider."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    elif settings.llm_provider == "gemini":
        return GeminiProvider()
    elif settings.llm_provider == "anthropic":
        return AnthropicProvider()
    elif settings.llm_provider == "openai_compatible":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# ── Main Extractor ──────────────────────────────────────────────────────────

class Extractor:
    """
    Orchestrates LLM-based extraction with self-consistency and model routing.

    Usage:
        extractor = Extractor()
        result = extractor.extract(problem_text)
    """

    def __init__(self):
        settings = get_settings()
        self._provider = _get_provider()
        self._fast_model, self._strong_model = settings.get_extraction_models()
        self._consistency_runs = settings.self_consistency_runs

    def extract(
        self,
        problem_text: str,
        *,
        use_strong_model: bool = False,
        consistency_runs: int | None = None,
    ) -> ExtractionResult:
        """
        Full extraction pipeline for a single problem:
          1. Run deterministic constraint parsing
          2. Run LLM extraction (possibly multiple times for self-consistency)
          3. Cross-check LLM output against deterministic parse
          4. Validate tags against fixed vocabulary
          5. Return merged result with confidence score
        """
        runs = consistency_runs or self._consistency_runs

        # Step 1: Deterministic parse
        det_constraints = parse_constraints(problem_text)

        # Step 2: LLM extraction(s)
        model = self._strong_model if use_strong_model else self._fast_model
        schemas: list[ProblemSchema] = []

        for i in range(runs):
            logger.info(f"Extraction run {i + 1}/{runs} with {model}")
            try:
                schema = self._provider.extract_schema(problem_text, model)
                schemas.append(schema)
            except Exception as e:
                logger.error(f"Extraction run {i + 1} failed: {e}")

        if not schemas:
            raise ExtractionError("All extraction runs failed")

        # Step 3: Self-consistency merge
        merged, confidence = self._merge_schemas(schemas)

        # Step 4: Validate tags
        validated = self._validate_tags(merged)

        # Step 5: Cross-check constraints
        constraint_match = self._cross_check_constraints(validated, det_constraints)

        # Step 6: Decide if escalation is needed (only if we have a different strong model)
        needs_escalation = (
            not use_strong_model
            and self._fast_model != self._strong_model
            and (confidence < 0.7 or validated.concept_count >= 2 or not constraint_match)
        )

        if needs_escalation:
            logger.info(
                f"Escalating to strong model (confidence={confidence:.2f}, "
                f"concept_count={validated.concept_count}, constraint_match={constraint_match})"
            )
            return self.extract(
                problem_text,
                use_strong_model=True,
                consistency_runs=max(runs, 2),
            )

        return ExtractionResult(
            schema=validated,
            confidence=confidence,
            deterministic_constraints=det_constraints,
            constraint_match=constraint_match,
            model_used=model,
            runs=len(schemas),
        )

    def _merge_schemas(
        self, schemas: list[ProblemSchema]
    ) -> tuple[ProblemSchema, float]:
        """
        Merge multiple extraction runs via majority vote.
        Returns (merged_schema, confidence_score).
        """
        if len(schemas) == 1:
            return schemas[0], 1.0

        # Vote on primary_technique
        primary_votes = Counter(s.primary_technique for s in schemas)
        best_primary, primary_count = primary_votes.most_common(1)[0]

        # Vote on framing
        framing_votes = Counter(s.framing for s in schemas)
        best_framing, framing_count = framing_votes.most_common(1)[0]

        # Merge secondary_techniques: keep tags that appear in majority
        threshold = len(schemas) / 2
        all_secondary = Counter()
        for s in schemas:
            all_secondary.update(s.secondary_techniques)
        best_secondary = [
            tag for tag, count in all_secondary.items() if count >= threshold
        ][:3]

        # Merge archetypes similarly
        all_archetypes = Counter()
        for s in schemas:
            all_archetypes.update(s.archetype)
        best_archetypes = [
            a for a, count in all_archetypes.items() if count >= threshold
        ]

        base = schemas[0]
        best_insight = max(schemas, key=lambda s: len(s.core_insight)).core_insight
        concept_votes = Counter(s.concept_count for s in schemas)
        best_concept_count = concept_votes.most_common(1)[0][0]

        merged = ProblemSchema(
            primary_technique=best_primary,
            secondary_techniques=best_secondary,
            composition_pattern=base.composition_pattern,
            archetype=best_archetypes,
            framing=best_framing,
            nearest_classical_analogue=base.nearest_classical_analogue,
            constraint_fingerprint=base.constraint_fingerprint,
            core_insight=best_insight,
            concept_count=best_concept_count,
            evidence=base.evidence,
            schema_version=base.schema_version,
        )

        n = len(schemas)
        agreement_scores = [primary_count / n, framing_count / n]
        confidence = sum(agreement_scores) / len(agreement_scores)

        return merged, confidence

    def _validate_tags(self, schema: ProblemSchema) -> ProblemSchema:
        """Validate all tags against the fixed vocabulary, flagging unknowns."""
        primary, primary_valid = validate_or_flag(schema.primary_technique)
        if not primary_valid:
            logger.warning(f"Unknown primary technique: {schema.primary_technique}")

        secondary = []
        for tag in schema.secondary_techniques:
            validated, valid = validate_or_flag(tag)
            if not valid:
                logger.warning(f"Unknown secondary technique: {tag}")
            secondary.append(validated)

        archetypes = []
        for a in schema.archetype:
            if is_valid_archetype(a):
                archetypes.append(a)
            else:
                logger.warning(f"Unknown archetype: {a}")
                archetypes.append(f"other:{a}")

        framing = schema.framing
        if not is_valid_framing(framing):
            logger.warning(f"Unknown framing: {framing}")
            framing = f"other:{framing}"

        return schema.model_copy(update={
            "primary_technique": primary,
            "secondary_techniques": secondary,
            "archetype": archetypes,
            "framing": framing,
        })

    def _cross_check_constraints(
        self, schema: ProblemSchema, det: ParsedConstraints
    ) -> bool:
        """Cross-check LLM constraint_fingerprint against deterministic parse."""
        if not det.fingerprint or det.fingerprint == "unknown":
            return True

        llm_fp = schema.constraint_fingerprint.lower()

        if det.complexity_class:
            det_class = det.complexity_class.lower()
            det_terms = set(det_class.replace("(", "").replace(")", "").split())
            llm_terms = set(llm_fp.replace("(", "").replace(")", "").split())
            if det_terms & llm_terms:
                return True

        if det.max_bound:
            if str(det.max_bound) in llm_fp:
                return True

        logger.info(
            f"Constraint mismatch: LLM='{schema.constraint_fingerprint}' "
            f"vs DET='{det.fingerprint}'"
        )
        return False


class ExtractionResult:
    """Result of the full extraction pipeline."""

    def __init__(
        self,
        schema: ProblemSchema,
        confidence: float,
        deterministic_constraints: ParsedConstraints,
        constraint_match: bool,
        model_used: str,
        runs: int,
    ):
        self.schema = schema
        self.confidence = confidence
        self.deterministic_constraints = deterministic_constraints
        self.constraint_match = constraint_match
        self.model_used = model_used
        self.runs = runs

    def to_dict(self) -> dict:
        return {
            "schema": self.schema.model_dump(),
            "confidence": self.confidence,
            "constraint_match": self.constraint_match,
            "model_used": self.model_used,
            "runs": self.runs,
            "deterministic_fingerprint": self.deterministic_constraints.fingerprint,
        }


class ExtractionError(Exception):
    """Raised when extraction fails irrecoverably."""
    pass
