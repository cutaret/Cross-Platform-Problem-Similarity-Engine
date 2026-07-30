"""
Codeforces API client — handles ingestion of problems from the official API.

API docs: https://codeforces.com/apiHelp
Rate limit: 1 request per 2 seconds (enforced by this client).

This client handles:
  - Fetching the full problem list (tags + ratings)
  - Fetching individual problem statements (HTML) from contest pages
  - Rate limiting to respect API ToS
  - Canonical ID generation: (codeforces, contest_id, problem_index)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from curl_cffi.requests import AsyncSession, Response

logger = logging.getLogger(__name__)

CF_API_BASE = "https://codeforces.com/api"
CF_SITE_BASE = "https://codeforces.com"
RATE_LIMIT_SECONDS = 2.0


@dataclass
class CFProblemMeta:
    """Metadata for a Codeforces problem from the API."""
    contest_id: int
    index: str              # e.g. "A", "B", "C1"
    name: str
    rating: int | None
    tags: list[str]
    url: str
    external_id: str        # e.g. "1900A"


@dataclass
class CFProblemFull(CFProblemMeta):
    """Full problem with statement text."""
    statement_html: str = ""
    time_limit_ms: int | None = None
    memory_limit_kb: int | None = None


class CodeforcesClient:
    """
    Async Codeforces API client with built-in rate limiting.

    Usage:
        async with CodeforcesClient() as client:
            problems = await client.fetch_problem_list()
            full = await client.fetch_problem_statement(problems[0])
    """

    def __init__(self, requests_per_second: float = 1 / RATE_LIMIT_SECONDS):
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._client: AsyncSession | None = None

    async def __aenter__(self):
        self._client = AsyncSession(
            impersonate="chrome",
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.close()

    async def _rate_limited_get(self, url: str, params: dict | None = None, retries: int = 3) -> Response:
        """GET with rate limiting — waits if needed to respect the 1 req/2sec limit."""
        import curl_cffi
        
        for attempt in range(retries):
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)

            self._last_request_time = time.monotonic()
            
            try:
                response = await self._client.get(url, params=params)
                
                # Codeforces sometimes returns 50x errors during heavy load
                if response.status_code >= 500:
                    raise RuntimeError(f"HTTP Error {response.status_code}")
                    
                response.raise_for_status()
                return response
                
            except Exception as e:
                if attempt == retries - 1:
                    raise
                
                wait_time = (attempt + 1) * 4
                logger.warning(f"Fetch failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
                # Re-create the session to clear any poisoned connection pool state
                if self._client:
                    await self._client.close()
                self._client = AsyncSession(impersonate="chrome", timeout=30.0)

    # ── Problem list (metadata only) ────────────────────────────────────

    async def fetch_problem_list(self) -> list[CFProblemMeta]:
        """
        Fetch all problems from Codeforces via `problemset.problems`.
        Returns metadata only (no statement text).
        """
        logger.info("Fetching Codeforces problem list...")
        resp = await self._rate_limited_get(f"{CF_API_BASE}/problemset.problems")
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Codeforces API error: {data.get('comment', 'unknown')}")

        problems = []
        # Build a rating lookup from problemStatistics
        rating_map: dict[tuple[int, str], int] = {}
        for stat in data["result"].get("problemStatistics", []):
            key = (stat.get("contestId"), stat.get("index"))
            # solvedCount isn't the rating — rating is on the problem object itself

        for p in data["result"]["problems"]:
            contest_id = p.get("contestId")
            index = p.get("index", "")
            if contest_id is None:
                continue  # skip gym/unofficial problems without contest ID

            rating = p.get("rating")
            tags = p.get("tags", [])
            name = p.get("name", "")
            external_id = f"{contest_id}{index}"
            url = f"{CF_SITE_BASE}/problemset/problem/{contest_id}/{index}"

            problems.append(CFProblemMeta(
                contest_id=contest_id,
                index=index,
                name=name,
                rating=rating,
                tags=tags,
                url=url,
                external_id=external_id,
            ))

        logger.info(f"Fetched {len(problems)} problems from Codeforces API")
        return problems

    # ── Individual problem statement ────────────────────────────────────

    async def fetch_problem_statement(self, meta: CFProblemMeta) -> CFProblemFull:
        """
        Fetch the full problem statement HTML from the contest page.
        Extracts the problem statement div, time limit, and memory limit.
        """
        url = f"{CF_SITE_BASE}/contest/{meta.contest_id}/problem/{meta.index}"
        logger.debug(f"Fetching statement for {meta.external_id} from {url}")

        resp = await self._rate_limited_get(url)
        html = resp.text

        from ingestion.normalizer import extract_problem_html

        parsed = extract_problem_html(html)

        return CFProblemFull(
            contest_id=meta.contest_id,
            index=meta.index,
            name=meta.name,
            rating=meta.rating,
            tags=meta.tags,
            url=meta.url,
            external_id=meta.external_id,
            statement_html=parsed.statement_html,
            time_limit_ms=parsed.time_limit_ms,
            memory_limit_kb=parsed.memory_limit_kb,
        )

    # ── Contest-specific fetching ───────────────────────────────────────

    async def fetch_contest_problems(self, contest_id: int) -> list[CFProblemMeta]:
        """Fetch all problems from a specific contest."""
        logger.info(f"Fetching problems for contest {contest_id}...")
        resp = await self._rate_limited_get(
            f"{CF_API_BASE}/contest.standings",
            params={"contestId": contest_id, "from": 1, "count": 1},
        )
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Codeforces API error: {data.get('comment', 'unknown')}")

        problems = []
        for p in data["result"].get("problems", []):
            index = p.get("index", "")
            external_id = f"{contest_id}{index}"
            problems.append(CFProblemMeta(
                contest_id=contest_id,
                index=index,
                name=p.get("name", ""),
                rating=p.get("rating"),
                tags=p.get("tags", []),
                url=f"{CF_SITE_BASE}/problemset/problem/{contest_id}/{index}",
                external_id=external_id,
            ))

        logger.info(f"Found {len(problems)} problems in contest {contest_id}")
        return problems

    async def fetch_recent_contests(self, count: int = 10) -> list[dict]:
        """Fetch recent finished contests."""
        resp = await self._rate_limited_get(
            f"{CF_API_BASE}/contest.list",
            params={"gym": "false"},
        )
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Codeforces API error: {data.get('comment', 'unknown')}")

        finished = [
            c for c in data["result"]
            if c.get("phase") == "FINISHED"
        ]
        return finished[:count]
