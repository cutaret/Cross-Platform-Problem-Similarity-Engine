"""
Deterministic constraint parser (§4.1) — no LLM involved.

Extracts numeric bounds (n, m, q, t, k, etc.) from the problem statement's
constraints section via regex, then maps them to a complexity-class bucket.

Also extracts time_limit_ms and memory_limit_kb from the statement header.
The resulting `constraint_fingerprint` is 100% reproducible and serves as
a cross-check against the LLM extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Complexity class buckets ────────────────────────────────────────────────
# Maps upper bounds to expected algorithmic complexity.

COMPLEXITY_BUCKETS = [
    (20,        "exponential/bitmask"),      # 2^n feasible
    (100,       "O(n^3) or O(n^2 log n)"),
    (500,       "O(n^3)"),
    (5_000,     "O(n^2)"),
    (100_000,   "O(n sqrt(n)) or O(n log^2 n)"),
    (500_000,   "O(n log n)"),
    (2_000_000, "O(n log n) or O(n)"),
    (10_000_000, "O(n)"),
    (float("inf"), "O(n) or O(1) per query"),
]


def classify_bound(value: int) -> str:
    """Map a numeric upper bound to its complexity-class bucket."""
    for threshold, label in COMPLEXITY_BUCKETS:
        if value <= threshold:
            return label
    return "O(1) per query"


# ── Regex patterns ──────────────────────────────────────────────────────────

# Matches patterns like:  1 ≤ n ≤ 200000,  n ≤ 2·10^5,  n <= 2*10^5,  n≤2e5
_BOUND_PATTERNS = [
    # n ≤ 2·10^5  or  n ≤ 2*10^5  or  n ≤ 2×10^5
    re.compile(
        r"(?P<var>[a-zA-Z]\w*)\s*[≤<=]+\s*(?P<coeff>\d+)\s*[·*×]\s*10\s*\^?\s*\{?(?P<exp>\d+)\}?",
        re.IGNORECASE,
    ),
    # n ≤ 200000  (plain integer)
    re.compile(
        r"(?P<var>[a-zA-Z]\w*)\s*[≤<=]+\s*(?P<value>\d[\d,_]*)",
        re.IGNORECASE,
    ),
    # 1 ≤ n ≤ 200000
    re.compile(
        r"\d+\s*[≤<=]+\s*(?P<var>[a-zA-Z]\w*)\s*[≤<=]+\s*(?P<value>\d[\d,_]*)",
        re.IGNORECASE,
    ),
    # 1 ≤ n ≤ 2·10^5
    re.compile(
        r"\d+\s*[≤<=]+\s*(?P<var>[a-zA-Z]\w*)\s*[≤<=]+\s*(?P<coeff>\d+)\s*[·*×]\s*10\s*\^?\s*\{?(?P<exp>\d+)\}?",
        re.IGNORECASE,
    ),
]

# Time limit: "time limit per test: 2 seconds" or "Time Limit: 2000ms"
_TIME_LIMIT_PATTERNS = [
    re.compile(r"time\s*limit[^:]*:\s*(\d+)\s*second", re.IGNORECASE),
    re.compile(r"time\s*limit[^:]*:\s*(\d+)\s*ms", re.IGNORECASE),
]

# Memory limit: "memory limit per test: 256 megabytes"
_MEMORY_LIMIT_PATTERNS = [
    re.compile(r"memory\s*limit[^:]*:\s*(\d+)\s*megabyte", re.IGNORECASE),
    re.compile(r"memory\s*limit[^:]*:\s*(\d+)\s*mb", re.IGNORECASE),
    re.compile(r"memory\s*limit[^:]*:\s*(\d+)\s*kb", re.IGNORECASE),
]


@dataclass
class ParsedConstraints:
    """Result of deterministic constraint parsing."""
    bounds: dict[str, int] = field(default_factory=dict)  # var_name → upper_bound
    max_bound: int | None = None
    complexity_class: str | None = None
    time_limit_ms: int | None = None
    memory_limit_kb: int | None = None
    fingerprint: str = ""


def _parse_scientific(match: re.Match) -> int | None:
    """Extract integer value from a scientific notation match."""
    groups = match.groupdict()
    if "coeff" in groups and groups["coeff"] and "exp" in groups and groups["exp"]:
        return int(groups["coeff"]) * (10 ** int(groups["exp"]))
    if "value" in groups and groups["value"]:
        return int(groups["value"].replace(",", "").replace("_", ""))
    return None


def parse_constraints(statement: str) -> ParsedConstraints:
    """
    Parse a problem statement and extract:
      - Variable bounds (n, m, q, etc.)
      - Complexity class bucket
      - Time and memory limits
      - A deterministic fingerprint string

    This is the core of §4.1 — entirely regex-based, no LLM.
    """
    result = ParsedConstraints()

    # ── Extract variable bounds ─────────────────────────────────────────
    seen_vars: dict[str, int] = {}

    for pattern in _BOUND_PATTERNS:
        for match in pattern.finditer(statement):
            var_name = match.group("var").lower()
            value = _parse_scientific(match)
            if value is not None and value > 0:
                # Keep the largest bound seen for each variable
                if var_name not in seen_vars or value > seen_vars[var_name]:
                    seen_vars[var_name] = value

    result.bounds = seen_vars

    # ── Determine the dominant bound ────────────────────────────────────
    # Priority: n > m > q > t > any other, then largest value
    priority_vars = ["n", "m", "q", "t", "k"]
    dominant_var = None
    dominant_val = 0

    for pv in priority_vars:
        if pv in seen_vars and seen_vars[pv] > dominant_val:
            dominant_var = pv
            dominant_val = seen_vars[pv]

    # If none of the priority vars found, take the largest bound
    if dominant_var is None and seen_vars:
        dominant_var = max(seen_vars, key=seen_vars.get)
        dominant_val = seen_vars[dominant_var]

    if dominant_val > 0:
        result.max_bound = dominant_val
        result.complexity_class = classify_bound(dominant_val)

    # ── Extract time limit ──────────────────────────────────────────────
    for pattern in _TIME_LIMIT_PATTERNS:
        m = pattern.search(statement)
        if m:
            val = int(m.group(1))
            # If matched "seconds", convert to ms
            if "second" in pattern.pattern:
                val *= 1000
            result.time_limit_ms = val
            break

    # ── Extract memory limit ────────────────────────────────────────────
    for pattern in _MEMORY_LIMIT_PATTERNS:
        m = pattern.search(statement)
        if m:
            val = int(m.group(1))
            if "megabyte" in pattern.pattern or "mb" in pattern.pattern.lower():
                val *= 1024  # convert to KB
            result.memory_limit_kb = val
            break

    # ── Build fingerprint ───────────────────────────────────────────────
    parts = []
    if result.max_bound:
        parts.append(f"{dominant_var}≤{_format_bound(result.max_bound)}")
    if result.complexity_class:
        parts.append(f"→ {result.complexity_class}")
    if result.time_limit_ms:
        parts.append(f"TL={result.time_limit_ms}ms")
    if result.memory_limit_kb:
        parts.append(f"ML={result.memory_limit_kb}KB")

    result.fingerprint = " | ".join(parts) if parts else "unknown"

    return result


def _format_bound(value: int) -> str:
    """Format a bound as a human-readable string: 200000 → '2e5'."""
    if value <= 0:
        return str(value)
    # Try to express as a·10^b
    s = str(value)
    # Check if it's a clean power of 10 times a small coefficient
    for exp in range(9, 0, -1):
        base = 10 ** exp
        if value % base == 0:
            coeff = value // base
            if coeff < 10:
                return f"{coeff}e{exp}"
    # Fall back to raw number with underscores for readability
    if value >= 1000:
        return f"{value:,}".replace(",", "_")
    return str(value)
