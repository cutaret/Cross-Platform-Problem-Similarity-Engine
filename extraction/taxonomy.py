"""
Fixed vocabulary for technique tags, archetypes, and framings.

This is the canonical taxonomy — every tag the extraction pipeline can assign
MUST exist here.  Anything outside this vocabulary gets tagged as 'other:___'
and logged for periodic review (§7.3).

The taxonomy is organized as a two-level tree:
    category → [tag, tag, ...]

Expanded from Codeforces' ~40 official tags into finer-grained sub-techniques
that capture the distinctions that matter for similarity matching.
"""

from __future__ import annotations

# ── Two-level technique taxonomy ────────────────────────────────────────────
# category → list of tags within that category
TECHNIQUE_TAXONOMY: dict[str, list[str]] = {
    "dp": [
        "dp-linear",
        "dp-knapsack",
        "dp-bitmask",
        "dp-digit",
        "dp-tree",
        "dp-interval",
        "dp-probability",
        "dp-game",
        "dp-sos",             # sum over subsets
        "dp-convex-hull-trick",
        "dp-divide-and-conquer",
        "dp-profile",         # broken profile / plug DP
        "dp-other",
    ],
    "graphs": [
        "graph-bfs",
        "graph-dfs",
        "graph-shortest-path",
        "graph-mst",
        "graph-topological-sort",
        "graph-bipartite",
        "graph-euler-path",
        "graph-scc",          # strongly connected components
        "graph-bridge-articulation",
        "graph-flow-maxflow",
        "graph-flow-mincost",
        "graph-matching",
        "graph-coloring",
        "graph-cycle-detection",
        "graph-other",
    ],
    "trees": [
        "tree-traversal",
        "tree-lca",
        "tree-hld",           # heavy-light decomposition
        "tree-centroid",
        "tree-euler-tour",
        "tree-rerooting",
        "tree-diameter",
        "tree-other",
    ],
    "data-structures": [
        "ds-segment-tree",
        "ds-fenwick-tree",    # BIT
        "ds-sparse-table",
        "ds-dsu",
        "ds-dsu-rollback",
        "ds-trie",
        "ds-balanced-bst",    # treap, splay, etc.
        "ds-stack-monotonic",
        "ds-deque-monotonic",
        "ds-sqrt-decomposition",
        "ds-persistent",
        "ds-merge-sort-tree",
        "ds-other",
    ],
    "strings": [
        "string-hashing",
        "string-kmp",
        "string-z-function",
        "string-suffix-array",
        "string-suffix-automaton",
        "string-aho-corasick",
        "string-manacher",
        "string-trie",
        "string-other",
    ],
    "math": [
        "math-number-theory",
        "math-modular-arithmetic",
        "math-combinatorics",
        "math-probability",
        "math-game-theory",   # Sprague-Grundy, nim
        "math-linear-algebra",
        "math-fft-ntt",
        "math-geometry",
        "math-other",
    ],
    "greedy": [
        "greedy-exchange",
        "greedy-scheduling",
        "greedy-sorting",
        "greedy-other",
    ],
    "binary-search": [
        "binary-search-answer",
        "binary-search-predicate",
        "binary-search-other",
    ],
    "divide-and-conquer": [
        "dnc-merge-sort",
        "dnc-cdq",            # CDQ divide and conquer
        "dnc-other",
    ],
    "brute-force": [
        "brute-bitmask-enumeration",
        "brute-backtracking",
        "brute-meet-in-middle",
        "brute-other",
    ],
    "constructive": [
        "constructive-greedy",
        "constructive-invariant",
        "constructive-parity",
        "constructive-other",
    ],
    "two-pointers": [
        "two-pointers-sliding-window",
        "two-pointers-merge",
        "two-pointers-other",
    ],
    "interactive": [
        "interactive-binary-search",
        "interactive-adaptive",
        "interactive-other",
    ],
    "implementation": [
        "implementation-simulation",
        "implementation-parsing",
        "implementation-other",
    ],
    "sortings": [
        "sorting-custom-comparator",
        "sorting-coordinate-compression",
        "sorting-other",
    ],
}

# Flat set of all valid tags for O(1) lookup
ALL_TAGS: frozenset[str] = frozenset(
    tag for tags in TECHNIQUE_TAXONOMY.values() for tag in tags
)

# Reverse map: tag → category
TAG_TO_CATEGORY: dict[str, str] = {
    tag: category
    for category, tags in TECHNIQUE_TAXONOMY.items()
    for tag in tags
}


# ── Archetypes ──────────────────────────────────────────────────────────────
# Higher-level solution patterns that recur across many problems.
ARCHETYPES: frozenset[str] = frozenset([
    "offline-processed-in-reverse",
    "answer-on-tree-path",
    "sweep-line",
    "coordinate-compression-then-structure",
    "virtual-node-trick",
    "contribution-counting",
    "reduction-to-known-problem",
    "exchange-argument",
    "invariant-maintenance",
    "complement-counting",
    "small-to-large-merging",
    "parallel-binary-search",
    "observation-reduces-search-space",
    "transform-domain",            # e.g. XOR basis, polynomial
    "amortized-analysis-structure",
    "randomized-approach",
    "fractional-relaxation",
])


# ── Framings ────────────────────────────────────────────────────────────────
# What kind of answer the problem asks for.
FRAMINGS: frozenset[str] = frozenset([
    "counting",
    "optimization",          # minimize / maximize
    "decision",              # yes / no
    "construction",          # output a valid object
    "interactive",
    "game",
    "shortest-path",
    "string-matching",
    "simulation",
    "enumeration",
    "partitioning",
    "scheduling",
])


# ── Validation helpers ──────────────────────────────────────────────────────

def is_valid_tag(tag: str) -> bool:
    """Check whether a tag belongs to the fixed vocabulary."""
    return tag in ALL_TAGS


def is_valid_archetype(archetype: str) -> bool:
    """Check whether an archetype belongs to the fixed vocabulary."""
    return archetype in ARCHETYPES


def is_valid_framing(framing: str) -> bool:
    """Check whether a framing belongs to the fixed vocabulary."""
    return framing in FRAMINGS


def validate_or_flag(tag: str) -> tuple[str, bool]:
    """
    Return (tag, True) if valid, or ('other:<tag>', False) if unknown.
    The False flag signals that this needs periodic review (§7.3).
    """
    if is_valid_tag(tag):
        return tag, True
    return f"other:{tag}", False


def get_taxonomy_prompt_block() -> str:
    """
    Produce a formatted text block listing all valid tags, archetypes,
    and framings — meant to be injected into the LLM extraction prompt
    so the model knows the exact vocabulary it must use.
    """
    lines = ["## Valid Technique Tags (use ONLY these)\n"]
    for category, tags in sorted(TECHNIQUE_TAXONOMY.items()):
        lines.append(f"### {category}")
        for tag in tags:
            lines.append(f"  - {tag}")
        lines.append("")

    lines.append("## Valid Archetypes (use ONLY these)\n")
    for a in sorted(ARCHETYPES):
        lines.append(f"  - {a}")

    lines.append("\n## Valid Framings (use ONLY these)\n")
    for f in sorted(FRAMINGS):
        lines.append(f"  - {f}")

    lines.append(
        "\nIf none of the above tags fit, use 'other:<your-label>' — "
        "but this should be rare. Prefer the closest existing tag."
    )
    return "\n".join(lines)
