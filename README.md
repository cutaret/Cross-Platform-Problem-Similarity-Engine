# CP Similar Problem Finder

**Find conceptually similar competitive programming problems — by algorithmic structure, not surface text.**

Paste a Codeforces problem URL or raw problem text and get ranked matches from ~9,000+ problems, with full score breakdowns and explanations of *why* each match is similar.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/frontend-React%2019-61DAFB.svg)](https://react.dev/)
[![SQLite / PostgreSQL](https://img.shields.io/badge/db-SQLite%20%7C%20PostgreSQL-003B57.svg)](#database)

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How It Works — The Big Picture](#how-it-works--the-big-picture)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Web Interface](#web-interface)
- [Scoring Algorithm Explained](#scoring-algorithm-explained)
- [Extraction Pipeline In Detail](#extraction-pipeline-in-detail)
- [The Technique Taxonomy](#the-technique-taxonomy)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Roadmap](#roadmap)
- [Development](#development)

---

## Why This Exists

Competitive programming practice is most effective when you solve *conceptually related* problems — not just problems with the same Codeforces tag. A problem tagged "DP" + "graphs" on Codeforces could be anything from a trivial tree DP to a complex knapsack-on-a-DAG with bitmask optimization. Existing tag-based search treats these as interchangeable. They are not.

This tool solves a specific problem: **given a hard CP problem, find the most structurally similar problems to practice on**, where "structurally similar" means:

- Same primary algorithmic technique (e.g., segment tree with lazy propagation, not just "data structures")
- Similar *composition* of techniques (how multiple techniques interact)
- Same type of core insight (the "aha" moment that unlocks the solution)
- Compatible complexity-class constraints (e.g., both require O(n log n) solutions for n up to 2x10^5)

The system handles multi-concept problems — the hardest and most interesting ones — by representing *how* techniques combine, not just which tags co-occur.

---

## How It Works — The Big Picture

The system operates as a five-stage pipeline:

```
Problem Text  -->  Ingestion  -->  Extraction  -->  Storage  -->  Matching  -->  Ranked Results
                   (fetch +       (LLM + regex     (SQLite/      (5-signal
                    normalize)     decompose)       Postgres)      composite)
```

**Stage 1 — Ingestion:** Problems are fetched from the Codeforces API (official, rate-limited to 1 req/2sec). HTML statements are parsed with BeautifulSoup, preserving LaTeX math notation that carries important signal for extraction.

**Stage 2 — Extraction:** Each problem goes through two parallel analysis paths:
- A *deterministic constraint parser* (pure regex, no LLM) extracts variable bounds like `n <= 2*10^5` and maps them to complexity-class buckets like "O(n log n)".
- An *LLM extraction pipeline* decomposes the problem into a structured 11-field schema: primary technique, secondary techniques, composition pattern, core insight, archetypes, framing, and more.

The LLM output is cross-checked against the deterministic parse. Disagreements are flagged.

**Stage 3 — Storage:** Problems, extractions, embeddings, and technique tags are stored in SQLite (default, zero setup) or PostgreSQL with pgvector (production).

**Stage 4 — Matching:** A two-pass scoring engine:
1. *Filter pass* — eliminates candidates with zero technique tag overlap (cuts ~9,000 down to hundreds)
2. *Scoring pass* — computes a weighted composite of 5 signals: primary technique match, secondary technique overlap (Jaccard), core insight embedding similarity (cosine), composition pattern embedding similarity, and constraint fingerprint match

**Stage 5 — Results:** Ranked matches are returned with full per-component score breakdowns and human-readable explanations of why each match was selected. The system explicitly returns "no strong match found" rather than forcing bad recommendations.

---

## Architecture Deep Dive

```
+--------------+     +------------------+     +----------------+     +--------------+
|  Codeforces  |---->|    Ingestion     |---->|   Extraction   |---->|   Storage    |
|     API      |     | codeforces_client|     | parser (regex) |     |  SQLAlchemy  |
|              |     | normalizer (HTML)|     | extractor (LLM)|     |  SQLite/PG   |
+--------------+     +------------------+     | embedder       |     +------+-------+
                                              | taxonomy       |            |
                                              +----------------+            v
+--------------+     +------------------+     +----------------+     +--------------+
|   Frontend   |<----|    FastAPI API    |<----|   Matching     |<----|  Candidates  |
| React 19     |     | /api/analyze     |     | score.py       |     |  loaded from |
| Three.js 3D  |     | /api/status      |     | rerank.py      |     |  database    |
| D3 graphs    |     |                  |     | (5-signal)     |     |              |
+--------------+     +------------------+     +----------------+     +--------------+
```

### Technology Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **Ingestion** | `curl_cffi` (Cloudflare bypass) + `BeautifulSoup` + `lxml` | Codeforces uses Cloudflare; `curl_cffi` impersonates Chrome to get through. `lxml` is the fastest HTML parser. |
| **LLM Extraction** | Ollama / Gemini / Claude / OpenAI-compatible | Four providers for flexibility: Ollama for free local use, Gemini free tier for cloud, Claude for highest quality, OpenAI-compatible for Groq/DeepSeek/etc. |
| **Embeddings** | `sentence-transformers` (local) or Voyage AI (paid) | Local embeddings (BGE-small, 384-dim) are free and fast on CPU. Voyage AI is Anthropic's recommended embedding partner for higher quality. |
| **Database** | SQLite (default) or PostgreSQL + pgvector | SQLite needs zero setup. At ~9-15K problems, exhaustive search over embeddings takes <10ms on CPU — no vector DB needed. |
| **Scoring** | NumPy | Pure matrix ops. Cosine similarity via dot product on L2-normalized vectors. |
| **API** | FastAPI + Uvicorn | Async Python, auto-generated OpenAPI docs, Pydantic validation. |
| **Frontend** | React 19 + Vite + Three.js + `react-force-graph-3d` | Interactive 3D force-directed graph for visualizing problem similarity clusters. |
| **CLI** | Click + Rich | Beautiful terminal UI with progress bars, colored tables, and status panels. |

---

## Project Structure

```
cp-finder/
|
|-- ingestion/                     # Stage 1: Data acquisition
|   |-- codeforces_client.py       # Async CF API client, rate-limited (1 req/2sec)
|   |                              # Uses curl_cffi for Cloudflare bypass
|   |                              # Fetches problem lists + individual statements
|   |-- normalizer.py              # HTML -> clean text preserving LaTeX/MathJax
|                                  # Extracts time/memory limits from page headers
|
|-- extraction/                    # Stage 2: Problem analysis
|   |-- parser.py                  # Deterministic constraint parser (regex only)
|   |                              # Maps bounds to complexity buckets
|   |                              # e.g. n<=20 -> "exponential/bitmask"
|   |-- schema.py                  # Pydantic ProblemSchema (11 structured fields)
|   |                              # Also exports JSON Schema for LLM tool_choice
|   |-- extractor.py               # LLM extraction orchestrator
|   |                              # 4 providers: Ollama, Gemini, Claude, OpenAI
|   |                              # Self-consistency (multiple runs + majority vote)
|   |                              # Model routing (cheap first, escalate hard cases)
|   |-- embedder.py                # Embedding provider (local or Voyage AI)
|   |                              # L2-normalized vectors for cosine via dot product
|   |-- taxonomy.py                # Fixed technique vocabulary (~80 tags, 15 categories)
|   |                              # Archetypes (17) and framings (12)
|   |-- prompts/
|       |-- decomposition.md       # The extraction prompt template
|                                  # Forces step-by-step decomposition before tagging
|
|-- matching/                      # Stage 4: Similarity scoring
|   |-- score.py                   # Two-pass composite scorer
|   |                              # 5 weighted signals, deterministic ranking
|   |-- rerank.py                  # Optional Voyage rerank + LLM explanations
|                                  # Degrades gracefully without paid APIs
|
|-- db/                            # Stage 3: Persistence
|   |-- models.py                  # SQLAlchemy ORM: Problem, Extraction, Feedback,
|   |                              # TechniqueTag, ProblemTechniqueTag
|   |-- session.py                 # Engine + session factory (SQLite WAL mode)
|   |-- init.sql                   # pgvector extension init for Postgres
|
|-- api/                           # REST API
|   |-- main.py                    # FastAPI app: /api/analyze, /api/status
|   |-- schemas.py                 # Request/response Pydantic models
|
|-- frontend/                      # Web UI (React 19 + Vite)
|   |-- src/
|       |-- App.tsx                # Main: search bar + results + view toggle
|       |-- components/
|           |-- ResultCard.tsx      # Match card: score bars, tags, core insight
|           |-- SimilarityGraph.tsx # 3D force-directed graph (Three.js)
|                                  # Auto-rotating, clickable nodes, glow effects
|
|-- cli.py                         # CLI entry point (5 commands)
|-- config.py                      # Pydantic-settings: validated at startup
|-- pyproject.toml                 # Dependencies + build config
|-- docker-compose.yml             # Postgres + pgvector (optional)
|-- .env.example                   # All config options documented
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **One LLM provider** (choose one):
  - [Ollama](https://ollama.com) — free, local *(default)*
  - [Google Gemini](https://aistudio.google.com/apikey) — free tier (15 RPM, 1M tokens/day)
  - [Anthropic Claude](https://console.anthropic.com/) — paid, highest quality
  - OpenAI-compatible APIs (Groq, DeepSeek, OpenRouter, etc.)
- **Node.js 18+** (only if you want the web frontend)

### 1. Clone and Install

```bash
git clone https://github.com/cutaret/cp-finder.git
cd cp-finder

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e .
pip install sentence-transformers   # for local embeddings (free, ~30MB model)
```

### 2. Configure

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

The defaults use **SQLite + Ollama + local embeddings** — fully free, no API keys needed.

If using Ollama, install it from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull qwen2.5:7b
```

### 3. Initialize the Database

```bash
cp-finder init-db          # Creates all tables
cp-finder seed-taxonomy    # Populates ~80 technique tags
cp-finder config           # Verify everything is connected
```

### 4. Ingest Problems

```bash
# Full backfill: metadata + statements + LLM extraction (~9,000 problems)
cp-finder backfill

# Or incrementally:
cp-finder backfill --limit 100                # start with 100 problems
cp-finder backfill --skip-extraction          # ingest only, extract later
cp-finder backfill --skip-statements          # metadata only (fastest)
```

### 5. Find Similar Problems

```bash
cp-finder find-similar "https://codeforces.com/contest/1900/problem/D"
cp-finder find-similar "https://codeforces.com/contest/1900/problem/D" --explain -n 20
cp-finder find-similar problem.txt
```

---

## CLI Reference

The CLI (`cp-finder`) has five commands:

| Command | Description |
|---|---|
| `cp-finder config` | Display current configuration and verify provider connectivity |
| `cp-finder init-db` | Create all database tables (auto-detects SQLite vs Postgres) |
| `cp-finder seed-taxonomy` | Populate the `technique_tags` table with the fixed vocabulary |
| `cp-finder backfill` | Full pipeline: ingest Codeforces problems, fetch statements, run LLM extraction |
| `cp-finder find-similar QUERY` | Find similar problems (accepts URL, file path, or raw text >100 chars) |
| `cp-finder status` | Show ingestion/extraction progress statistics |

### find-similar options

| Flag | Default | Description |
|---|---|---|
| `--top-n`, `-n` | 10 | Number of results to return |
| `--explain`, `-e` | off | Generate LLM-powered explanations for top 5 matches |

### backfill options

| Flag | Description |
|---|---|
| `--limit N` | Only ingest the first N problems |
| `--skip-statements` | Skip fetching HTML statements (metadata only) |
| `--skip-extraction` | Skip the LLM extraction step |

---

## Web Interface

### Starting the Backend

```bash
uvicorn api.main:app --reload --port 8000
```

### Starting the Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

### Features

- **Search bar** — paste a Codeforces URL or raw problem text
- **Algorithm signature display** — shows the extracted primary technique, constraint fingerprint, and core insight for your query
- **Result cards** — each match shows:
  - Similarity percentage and primary technique tag
  - Secondary technique tags
  - Human-readable explanation of *why* this problem is similar
  - Core insight text
  - Expandable per-signal score breakdown with colored progress bars
  - Direct link to the problem on Codeforces
- **3D similarity graph** — interactive Three.js force-directed visualization:
  - Your query problem as the central node
  - Matched problems as orbiting nodes, sized by similarity score
  - Color-coded by technique category
  - Animated particles flowing along edges proportional to match strength
  - Auto-rotating camera, stops on user interaction
  - Click any node to open the problem in a new tab
  - Hover tooltips with technique, score, and insight preview

---

## Scoring Algorithm Explained

### Two-Pass Architecture

**Pass 1 — Filter:** Build the set of all tags (primary + secondary) for the query. Drop every candidate that has zero overlap with this set. This typically reduces ~9,000 candidates to a few hundred, making the scoring pass fast.

**Pass 2 — Score:** For each surviving candidate, compute five signals and combine them with learned weights:

| Signal | Weight | Computation | What It Captures |
|---|---|---|---|
| **Primary technique** | 30% | Exact string match (1.0 or 0.0) | Same core algorithm needed |
| **Core insight similarity** | 25% | Cosine similarity of embedded "aha" insights | Same fundamental observation |
| **Composition similarity** | 20% | Cosine similarity of embedded composition patterns | Techniques interact the same way |
| **Secondary overlap** | 15% | Jaccard similarity of secondary tag sets | Supporting techniques overlap |
| **Constraint match** | 10% | Exact match on fingerprint (1.0 or 0.0) | Same complexity regime |

### Determinism

The scoring is fully deterministic: same query + same database = identical output every time. Results are sorted by `(-total_score, problem_id)` for stable ordering on ties.

### No-Match Threshold

If the best match scores below 0.15, the system returns an empty result set with a message explaining that no strong match was found. This prevents forcing bad recommendations.

---

## Extraction Pipeline In Detail

The extraction pipeline is the core intelligence layer. It runs in five steps:

### Step 1: Deterministic Constraint Parsing (No LLM)

`parser.py` uses regex to extract variable bounds from the problem statement:

- Recognizes patterns like `1 <= n <= 200000`, `n <= 2*10^5`, `n <= 2e5`
- Prioritizes variables by importance: n > m > q > t > k
- Maps the dominant bound to a complexity bucket:

| Upper Bound | Complexity Class |
|---|---|
| n <= 20 | exponential/bitmask (2^n feasible) |
| n <= 500 | O(n^3) |
| n <= 5,000 | O(n^2) |
| n <= 100,000 | O(n sqrt(n)) or O(n log^2 n) |
| n <= 500,000 | O(n log n) |
| n <= 2,000,000 | O(n log n) or O(n) |
| n <= 10,000,000 | O(n) |

Also extracts time and memory limits from the statement header.

### Step 2: LLM Extraction (2-3 runs)

The LLM receives a decomposition-first prompt that forces it to:
1. Write out the solution's sub-steps *before* assigning any tags
2. Assign tags from the fixed vocabulary only
3. Provide evidence (statement quotes) for every tag
4. Describe the composition pattern
5. State the core insight

Supported providers: Ollama (local/free), Gemini (free tier), Anthropic Claude (paid/best), OpenAI-compatible (Groq, DeepSeek, etc.)

### Step 3: Self-Consistency Merge

Multiple extraction runs are merged via majority vote:
- Primary technique: most common across runs
- Secondary techniques: tags appearing in >= 50% of runs
- Archetypes: same threshold
- Core insight: longest response selected
- Confidence score: average agreement rate across fields

### Step 4: Tag Validation

Every tag is checked against the fixed vocabulary (~80 tags across 15 categories). Unknown tags are prefixed with `other:` and logged for periodic taxonomy review.

### Step 5: Cross-Check and Escalation

The LLM's constraint fingerprint is compared against the deterministic parser's output. If they disagree, or if confidence is below 70%, or if the problem has 2+ load-bearing techniques, the system escalates to the stronger model (e.g., from Qwen to Claude) and re-runs with more consistency checks.

---

## The Technique Taxonomy

The fixed vocabulary has **~80 technique tags** organized into **15 categories**, plus **17 archetypes** and **12 framings**.

### Categories and Example Tags

| Category | Example Tags |
|---|---|
| **dp** | dp-linear, dp-knapsack, dp-bitmask, dp-digit, dp-tree, dp-interval, dp-sos, dp-convex-hull-trick |
| **graphs** | graph-bfs, graph-dfs, graph-shortest-path, graph-mst, graph-scc, graph-flow-maxflow, graph-matching |
| **trees** | tree-lca, tree-hld, tree-centroid, tree-euler-tour, tree-rerooting |
| **data-structures** | ds-segment-tree, ds-fenwick-tree, ds-dsu, ds-dsu-rollback, ds-trie, ds-sqrt-decomposition |
| **strings** | string-hashing, string-kmp, string-z-function, string-suffix-array, string-aho-corasick |
| **math** | math-number-theory, math-combinatorics, math-game-theory, math-fft-ntt, math-geometry |
| **greedy** | greedy-exchange, greedy-scheduling, greedy-sorting |
| **binary-search** | binary-search-answer, binary-search-predicate |
| **brute-force** | brute-bitmask-enumeration, brute-backtracking, brute-meet-in-middle |
| **constructive** | constructive-greedy, constructive-invariant, constructive-parity |

### Archetypes (Higher-Level Patterns)

offline-processed-in-reverse, sweep-line, contribution-counting, exchange-argument, complement-counting, small-to-large-merging, parallel-binary-search, observation-reduces-search-space, transform-domain, and more.

### Framings

counting, optimization, decision, construction, interactive, game, shortest-path, simulation, enumeration, partitioning, scheduling, string-matching.

Tags outside the vocabulary are flagged as `other:<label>` and logged for periodic review and potential promotion into the taxonomy.

---

## Configuration Reference

All settings live in `.env`, validated by Pydantic at startup (typos blow up immediately):

| Variable | Default | Description |
|---|---|---|
| `DB_BACKEND` | `sqlite` | `sqlite` (zero setup) or `postgres` (Docker) |
| `SQLITE_PATH` | `cp_finder.db` | SQLite database file path |
| `DATABASE_URL` | auto-generated | Override for explicit DB connection string |
| `LLM_PROVIDER` | `ollama` | `ollama`, `gemini`, `anthropic`, or `openai_compatible` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model for extraction (good at structured JSON) |
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio key (free: 15 RPM) |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key |
| `OPENAI_API_KEY` | *(empty)* | OpenAI / Groq / DeepSeek key |
| `OPENAI_BASE_URL` | *(empty)* | Custom endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `OPENAI_FAST_MODEL` | `gpt-4o-mini` | Fast model for bulk extraction |
| `OPENAI_STRONG_MODEL` | `gpt-4o` | Strong model for escalation |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers, free) or `voyage` (paid) |
| `LOCAL_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local model (384-dim, 33MB) |
| `EMBEDDING_DIMENSION` | `384` | Must match the model's output dimension |
| `VOYAGE_API_KEY` | *(empty)* | Voyage AI key (Anthropic's embedding partner) |
| `VOYAGE_EMBED_MODEL` | `voyage-3.5` | Voyage embedding model (1024-dim) |
| `SELF_CONSISTENCY_RUNS` | `2` | Independent extraction runs per problem |
| `CODECHEF_ENABLED` | `false` | Toggle CodeChef ingestion (Phase 2) |
| `LEETCODE_ENABLED` | `false` | Toggle LeetCode ingestion (Phase 2) |

---

## API Reference

### POST /api/analyze

Analyze a problem and return ranked similar problems.

**Request:**
```json
{
  "query": "https://codeforces.com/contest/1900/problem/D",
  "top_n": 10,
  "explain": false
}
```

**Response:**
```json
{
  "query_schema": {
    "primary_technique": "dp-bitmask",
    "secondary_techniques": ["binary-search-answer"],
    "core_insight": "Small N constraint signals bitmask enumeration...",
    "constraint_fingerprint": "n<=20 -> exponential/bitmask",
    "...": "..."
  },
  "matches": [
    {
      "problem_id": 42,
      "title": "Problem Name",
      "url": "https://codeforces.com/...",
      "platform": "codeforces",
      "primary_technique": "dp-bitmask",
      "secondary_techniques": ["greedy-exchange"],
      "core_insight": "...",
      "total_score": 0.73,
      "score_breakdown": {
        "primary_technique": 1.0,
        "secondary_overlap": 0.33,
        "composition_sim": 0.65,
        "core_insight_sim": 0.72,
        "constraint_match": 1.0
      },
      "relationship_reason": "Related because it shares the same primary technique (dp-bitmask), very similar core algorithmic insight, identical constraint profile."
    }
  ]
}
```

### GET /api/status

Returns database statistics.

```json
{
  "total_problems": 9247,
  "with_statements": 8103,
  "with_extractions": 7856
}
```

---

## Database Schema

Five tables, supporting both SQLite and PostgreSQL:

### problems
Stores raw problem data from all platforms.

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| platform | TEXT | "codeforces", "codechef", "leetcode" |
| external_id | TEXT | Platform-specific ID (e.g. "1900D") |
| url | TEXT | Direct link to the problem |
| title | TEXT | Problem name |
| raw_statement | TEXT | Cleaned statement text (LaTeX preserved) |
| time_limit_ms | INTEGER | Parsed from statement header |
| memory_limit_kb | INTEGER | Parsed from statement header |
| native_rating | INTEGER | Platform's difficulty rating |
| native_tags_json | TEXT | Platform's tags as JSON array |
| contest_id | TEXT | Contest identifier |

### extractions
LLM-extracted algorithmic fingerprints, one per problem.

| Column | Type | Description |
|---|---|---|
| problem_id | INTEGER PK FK | Links to problems.id |
| primary_technique | TEXT | From fixed vocabulary |
| secondary_techniques_json | TEXT | JSON array of tags |
| composition_pattern | TEXT | How techniques combine |
| core_insight | TEXT | The "aha" insight (gets embedded) |
| constraint_fingerprint | TEXT | e.g. "n<=2e5 -> O(n log n)" |
| core_insight_embedding_json | TEXT | 384-dim vector as JSON |
| composition_embedding_json | TEXT | 384-dim vector as JSON |
| confidence | FLOAT | Self-consistency agreement rate |
| schema_version | SMALLINT | Bumped on taxonomy changes |

### technique_tags
Fixed vocabulary (~80 tags in 15 categories).

### problem_technique_tags
Many-to-many link with role ('primary' or 'secondary').

### feedback
User thumbs-up/down on suggested matches (for future learned ranking).

---

## Roadmap

| Phase | Status | Scope | Exit Criteria |
|---|---|---|---|
| **0** | Done | Codeforces backfill + CLI query | Paste a problem, get ranked matches from ~9K CF problems |
| **1** | Done | Web frontend + API + 3D viz | Non-technical paste-a-URL flow works end to end |
| **2** | Planned | CodeChef + clist.by recency | New contest problems appear within ~1 hour of contest end |
| **3** | Planned | Feedback + learned ranking | Hand-tuned weights replaced by LightGBM ranker |
| **4** | Planned | Taxonomy curation + eval harness | Gold-set precision@5 tracked release over release |
| **5** | Optional | Embedding fine-tuning | Only after thousands of confirmed similar/dissimilar pairs |

---

## Development

### Setup

```bash
pip install -e ".[dev]"    # includes pytest, ruff, mypy
```

### Commands

```bash
ruff check .       # Lint
mypy .             # Type check
pytest             # Run tests
cp-finder status   # Check pipeline progress
```

### Using PostgreSQL

For production or larger datasets:

```bash
docker compose up -d       # Starts pgvector/pgvector:pg16

# Update .env:
DB_BACKEND=postgres
DATABASE_URL=postgresql+psycopg://cpfinder:cpfinder_dev@localhost:5432/cpfinder

cp-finder init-db          # Creates tables + pgvector extension
```

### Key Design Decisions

- **SQLite as default** — Zero-dependency setup. At ~15K problems, exhaustive search is <10ms. No vector DB needed.
- **WAL mode** — SQLite runs in Write-Ahead Logging mode for better concurrent read performance.
- **JSON columns for portability** — Arrays and vectors are stored as JSON text in SQLite, native ARRAY/vector types in Postgres. Property accessors on the ORM models handle serialization transparently.
- **Deterministic scoring** — No randomness anywhere in the matching pipeline. Reproducibility is a core requirement.
- **Graceful degradation** — Every paid API (Claude, Voyage, Gemini) has a free alternative (Ollama, sentence-transformers). The system works fully offline.

---

## License

MIT

---

*Built for competitive programmers who want to practice smarter, not harder.*
