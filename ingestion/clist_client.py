"""
Clist API client — handles fetching recent contests to trigger recency sync.

API docs: https://clist.by/api/v4/doc/
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from curl_cffi.requests import AsyncSession, Response

logger = logging.getLogger(__name__)

CLIST_API_BASE = "https://clist.by/api/v4"
RATE_LIMIT_SECONDS = 1.0


@dataclass
class ClistContestMeta:
    """Metadata for a contest from Clist."""
    id: int
    resource_id: int
    resource: str           # e.g. "codeforces.com", "codechef.com"
    event: str              # e.g. "Codeforces Round 900"
    start: str              # ISO8601 string
    end: str                # ISO8601 string
    parsed_at: str | None


class ClistClient:
    """
    Async Clist API client with built-in rate limiting.

    Usage:
        async with ClistClient("username:apikey") as client:
            contests = await client.fetch_recent_contests()
    """

    def __init__(self, api_key: str, requests_per_second: float = 1 / RATE_LIMIT_SECONDS):
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._client: AsyncSession | None = None
        self.api_key = api_key

    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
            
        self._client = AsyncSession(
            impersonate="chrome",
            timeout=30.0,
            headers=headers
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
                wait_time = (attempt + 1) * 2
                logger.warning(f"Fetch failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
                # Re-create session
                if self._client:
                    headers = self._client.headers
                    await self._client.close()
                    self._client = AsyncSession(impersonate="chrome", timeout=30.0, headers=headers)

    async def fetch_recent_contests(self, resource: str, hours_ago: int = 24) -> list[ClistContestMeta]:
        """
        Fetch contests for a specific resource that ended within the last `hours_ago`.
        `resource` examples: codeforces.com, codechef.com
        """
        logger.info(f"Fetching recent {resource} contests from Clist...")
        
        from datetime import datetime, timedelta
        import urllib.parse
        
        # Calculate time threshold in UTC
        start_time = datetime.utcnow() - timedelta(hours=hours_ago)
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_time = datetime.utcnow()
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        params = {
            "resource": resource,
            "end__gte": start_str,
            "end__lte": end_str,
            "order_by": "-end",
            "limit": 50,
        }

        resp = await self._rate_limited_get(f"{CLIST_API_BASE}/contest/", params=params)
        data = resp.json()
        
        contests = []
        for obj in data.get("objects", []):
            contests.append(ClistContestMeta(
                id=obj.get("id"),
                resource_id=obj.get("resource_id"),
                resource=obj.get("resource"),
                event=obj.get("event"),
                start=obj.get("start"),
                end=obj.get("end"),
                parsed_at=obj.get("parsed_at"),
            ))
            
        logger.info(f"Found {len(contests)} contests that ended in the last {hours_ago} hours on {resource}")
        return contests
