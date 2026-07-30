"""
Pydantic models for the structured extraction schema (§4.3).

ProblemSchema is what the LLM must return via tool use.
PROBLEM_SCHEMA_JSON is the JSON Schema exported for Claude's tool_choice.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from extraction.taxonomy import FRAMINGS


# ── Sub-models ──────────────────────────────────────────────────────────────

class TagEvidence(BaseModel):
    """A single tag + the sentence from the problem statement that justifies it."""
    tag: str = Field(description="Technique tag from the fixed vocabulary")
    evidence: str = Field(description="Short quote or pointer to the statement text justifying this tag")


# ── Main schema ─────────────────────────────────────────────────────────────

class ProblemSchema(BaseModel):
    """
    The structured algorithmic fingerprint of a competitive programming problem.
    Produced by the LLM extraction pipeline and cross-checked against the
    deterministic constraint parser.
    """
    primary_technique: str = Field(
        description="The single most important technique required to solve this problem. Must be from the fixed vocabulary."
    )
    secondary_techniques: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="0–3 additional techniques required, from the fixed vocabulary."
    )
    composition_pattern: str = Field(
        description="One sentence describing how the primary and secondary techniques combine to form the full solution."
    )
    archetype: list[str] = Field(
        default_factory=list,
        description="Higher-level solution patterns, from the fixed vocabulary. E.g. 'offline-processed-in-reverse'."
    )
    framing: str = Field(
        description="What kind of answer the problem asks for: counting, optimization, decision, construction, etc."
    )
    nearest_classical_analogue: str = Field(
        description="The closest well-known problem. E.g. 'variant of 0/1 knapsack', 'modified Dijkstra'."
    )
    constraint_fingerprint: str = Field(
        description="Complexity-class bucket derived from constraints. E.g. 'n≤2e5 → O(n log n)'."
    )
    core_insight: str = Field(
        description="The single 'aha' insight that unlocks the solution — this is what gets embedded for similarity matching."
    )
    concept_count: int = Field(
        ge=1,
        le=5,
        description="How many load-bearing techniques the problem requires (1, 2, or 3+)."
    )
    evidence: list[TagEvidence] = Field(
        default_factory=list,
        description="Evidence grounding: one entry per assigned tag, pointing to the statement text that justifies it."
    )
    schema_version: int = Field(
        default=1,
        description="Version of this schema. Bump on taxonomy changes."
    )


# ── JSON Schema for Claude tool use ─────────────────────────────────────────

def get_tool_schema() -> dict:
    """
    Return the JSON Schema dict suitable for passing as Claude's
    `tools[].input_schema`.  This is what enforces structured output
    via `tool_choice`.
    """
    schema = ProblemSchema.model_json_schema()
    # Clean up pydantic's $defs — Claude handles flat schemas better
    return _flatten_refs(schema)


def _flatten_refs(schema: dict) -> dict:
    """
    Inline any $ref / $defs that Pydantic generates so the schema
    is a single flat object — Claude's tool use handles this more
    reliably than nested $ref chains.
    """
    defs = schema.pop("$defs", {})
    return _resolve(schema, defs)


def _resolve(obj, defs: dict):
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            return _resolve(defs[ref_name], defs)
        return {k: _resolve(v, defs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(item, defs) for item in obj]
    return obj
