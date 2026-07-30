from pydantic import BaseModel
from typing import List, Optional


class ScoreComponents(BaseModel):
    """Per-component breakdown of similarity score — tells the user WHY a match was made."""
    primary_technique: float
    secondary_overlap: float
    composition_sim: float
    core_insight_sim: float
    constraint_match: float


class MatchResult(BaseModel):
    problem_id: int
    title: str
    url: str
    platform: str
    primary_technique: str
    secondary_techniques: List[str]
    core_insight: str
    total_score: float
    score_breakdown: ScoreComponents
    relationship_reason: str  # Human-readable sentence explaining WHY this is similar


class AnalyzeRequest(BaseModel):
    query: str
    top_n: int = 10
    explain: bool = False


class AnalyzeResponse(BaseModel):
    query_schema: dict
    matches: List[MatchResult]


class StatusResponse(BaseModel):
    total_problems: int
    with_statements: int
    with_extractions: int
