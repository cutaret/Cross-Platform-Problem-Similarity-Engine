"""
CodeChef API client — handles ingestion of problems from CodeChef.

This uses the undocumented frontend API endpoints:
- List problems in a contest: https://www.codechef.com/api/contests/{contest_code}
- Get problem details: https://www.codechef.com/api/contests/PRACTICE/problems/{problem_code}
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from curl_cffi.requests import AsyncSession, Response

logger = logging.getLogger(__name__)

CC_API_BASE = "https://www.codechef.com/api"
CC_SITE_BASE = "https://www.codechef.com"
RATE_LIMIT_SECONDS = 2.0


@dataclass
class CCProblemMeta:
    """Metadata for a CodeChef problem."""
    contest_id: str         # e.g. "START120A"
    problem_code: str       # e.g. "WATERCONS"
    name: str
    url: str
    external_id: str        # e.g. "WATERCONS" (problem codes are globally unique on CodeChef)


@dataclass
class CCProblemFull(CCProblemMeta):
    """Full problem with statement text."""
    statement_html: str = ""
    time_limit_ms: int | None = None
    memory_limit_kb: int | None = None


class CodeChefClient:
    """
    Async CodeChef API client with built-in rate limiting.
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
        """GET with rate limiting."""
        for attempt in range(retries):
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)

            self._last_request_time = time.monotonic()
            
            try:
                response = await self._client.get(url, params=params)
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
                
                # Re-create session
                if self._client:
                    await self._client.close()
                    self._client = AsyncSession(impersonate="chrome", timeout=30.0)

    async def fetch_contest_problems(self, contest_id: str) -> list[CCProblemMeta]:
        """Fetch all problems from a specific CodeChef contest."""
        logger.info(f"Fetching problems for CodeChef contest {contest_id}...")
        resp = await self._rate_limited_get(f"{CC_API_BASE}/contests/{contest_id}")
        data = resp.json()

        if data.get("status") != "success":
            logger.warning(f"Failed to fetch CodeChef contest {contest_id}: {data}")
            return []

        problems = []
        for p_code, p_data in data.get("problems", {}).items():
            problems.append(CCProblemMeta(
                contest_id=contest_id,
                problem_code=p_code,
                name=p_data.get("name", ""),
                url=f"{CC_SITE_BASE}/problems/{p_code}",
                external_id=p_code,
            ))

        logger.info(f"Found {len(problems)} problems in contest {contest_id}")
        return problems

    async def fetch_problem_statement(self, meta: CCProblemMeta) -> CCProblemFull:
        """Fetch the full problem statement JSON and extract details."""
        logger.debug(f"Fetching statement for {meta.external_id}")
        # Always fetch from PRACTICE contest to get public problems
        url = f"{CC_API_BASE}/contests/PRACTICE/problems/{meta.problem_code}"
        resp = await self._rate_limited_get(url)
        data = resp.json()

        statement_html = data.get("body", "")
        # CodeChef time limits are often strings like "1.0" or "0.5"
        time_limit_str = data.get("max_timelimit", "1")
        try:
            time_limit_ms = int(float(time_limit_str) * 1000)
        except (ValueError, TypeError):
            time_limit_ms = None
            
        return CCProblemFull(
            contest_id=meta.contest_id,
            problem_code=meta.problem_code,
            name=meta.name,
            url=meta.url,
            external_id=meta.external_id,
            statement_html=statement_html,
            time_limit_ms=time_limit_ms,
            memory_limit_kb=None, # Codechef API doesn't seem to provide memory limit directly in this endpoint
        )
