import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from api.schemas import AnalyzeRequest, AnalyzeResponse, StatusResponse, MatchResult, ScoreComponents
from db.models import Problem, Extraction
from db.session import get_db
from extraction.extractor import Extractor
from extraction.embedder import Embedder
from matching.score import Scorer, CandidateFeatures
from cli import _resolve_query

logger = logging.getLogger(__name__)

app = FastAPI(title="CP Finder API")

# Allow CORS for local frontend development (Vite runs on port 5173 usually)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status", response_model=StatusResponse)
def get_status():
    with get_db() as db:
        total_problems = db.query(Problem).count()
        with_statements = db.query(Problem).filter(Problem.raw_statement != "").count()
        total_extractions = db.query(Extraction).count()
    
    return StatusResponse(
        total_problems=total_problems,
        with_statements=with_statements,
        with_extractions=total_extractions
    )


def _build_relationship_reason(r) -> str:
    """Generate a human-readable sentence explaining WHY this problem is similar."""
    reasons = []
    if r.primary_technique_score >= 1.0:
        reasons.append(f"shares the same primary technique ({r.primary_technique})")
    if r.secondary_overlap_score > 0.3:
        reasons.append("overlapping secondary techniques")
    if r.core_insight_sim_score > 0.7:
        reasons.append("very similar core algorithmic insight")
    elif r.core_insight_sim_score > 0.4:
        reasons.append("related core insight")
    if r.composition_sim_score > 0.7:
        reasons.append("similar problem structure")
    if r.constraint_match_score >= 1.0:
        reasons.append("identical constraint profile")

    if not reasons:
        reasons.append("partial technique overlap")

    return "Related because it " + ", ".join(reasons) + "."


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_problem(request: AnalyzeRequest):
    # Step 1: Resolve query (handles Codeforces URL or raw text)
    problem_text = await _resolve_query(request.query)
    if not problem_text:
        raise HTTPException(status_code=400, detail="Could not resolve query to problem text")

    # Step 2: Extract schema using LLM
    extractor = Extractor()
    try:
        query_result = extractor.extract(problem_text, consistency_runs=2)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract problem schema: {e}")

    # Step 3: Embed the query's core insight
    embedder = Embedder()
    query_insight_emb = embedder.embed_query(query_result.schema.core_insight)
    query_comp_emb = embedder.embed_query(query_result.schema.composition_pattern)

    # Step 4: Load candidates from database
    with get_db() as db:
        rows = db.query(Problem, Extraction).join(Extraction).all()

    if not rows:
        return AnalyzeResponse(query_schema=query_result.schema.model_dump(), matches=[])

    candidates = []
    for problem, extraction in rows:
        candidates.append(CandidateFeatures(
            problem_id=problem.id,
            primary_technique=extraction.primary_technique,
            secondary_techniques=set(extraction.secondary_techniques or []),
            constraint_fingerprint=extraction.constraint_fingerprint or "",
            core_insight_embedding=(
                np.array(extraction.core_insight_embedding, dtype=np.float32)
                if extraction.core_insight_embedding is not None else None
            ),
            composition_embedding=(
                np.array(extraction.composition_embedding, dtype=np.float32)
                if extraction.composition_embedding is not None else None
            ),
            title=problem.title or "",
            url=problem.url,
            platform=problem.platform,
            core_insight=extraction.core_insight or "",
        ))

    # Step 5: Score candidates against the query
    query_features = CandidateFeatures(
        problem_id=-1,
        primary_technique=query_result.schema.primary_technique,
        secondary_techniques=set(query_result.schema.secondary_techniques),
        constraint_fingerprint=query_result.schema.constraint_fingerprint,
        core_insight_embedding=query_insight_emb,
        composition_embedding=query_comp_emb,
        core_insight=query_result.schema.core_insight,
    )

    scorer = Scorer()
    results = scorer.score(query_features, candidates, top_n=request.top_n)

    # Format results for API response
    matches = [
        MatchResult(
            problem_id=r.problem_id,
            title=r.title,
            url=r.url,
            platform=r.platform,
            primary_technique=r.primary_technique,
            secondary_techniques=list(r.secondary_techniques),
            core_insight=r.core_insight,
            total_score=r.total_score,
            score_breakdown=ScoreComponents(
                primary_technique=r.primary_technique_score,
                secondary_overlap=r.secondary_overlap_score,
                composition_sim=r.composition_sim_score,
                core_insight_sim=r.core_insight_sim_score,
                constraint_match=r.constraint_match_score,
            ),
            relationship_reason=_build_relationship_reason(r),
        )
        for r in results
    ]

    return AnalyzeResponse(
        query_schema=query_result.schema.model_dump(),
        matches=matches
    )
