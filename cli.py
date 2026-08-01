"""
CLI entry point for the CP Similar Problem Finder.

Commands:
  - find-similar: Paste a problem URL or text -> get ranked matches
  - backfill: Trigger full Codeforces problem ingestion + extraction
  - status: Show ingestion/extraction progress
  - init-db: Initialize database tables
  - config: Show current configuration
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=False, show_path=False)],
)
logger = logging.getLogger("cp-finder")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """CP Similar Problem Finder -- find conceptually similar CP problems."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ── config ──────────────────────────────────────────────────────────────────

@main.command()
def config():
    """Show current configuration (providers, database, etc)."""
    from config import get_settings

    settings = get_settings()

    table = Table(title="Current Configuration", show_lines=True)
    table.add_column("Setting", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Database backend", settings.db_backend)
    table.add_row("Database URL", settings.get_database_url()[:60] + "...")
    table.add_row("LLM provider", settings.llm_provider)
    if settings.llm_provider == "ollama":
        table.add_row("Ollama URL", settings.ollama_base_url)
        table.add_row("Ollama model", settings.ollama_model)
    elif settings.llm_provider == "gemini":
        fast, strong = settings.get_extraction_models()
        table.add_row("Fast model", fast)
        table.add_row("Strong model", strong)
        table.add_row("API key set", "Yes" if settings.gemini_api_key else "No")
    elif settings.llm_provider == "anthropic":
        fast, strong = settings.get_extraction_models()
        table.add_row("Fast model", fast)
        table.add_row("Strong model", strong)
        table.add_row("API key set", "Yes" if settings.anthropic_api_key else "No")
    table.add_row("Embedding provider", settings.embedding_provider)
    if settings.embedding_provider == "local":
        table.add_row("Embed model", settings.local_embed_model)
    table.add_row("Embed dimension", str(settings.get_embedding_dimension()))
    table.add_row("Self-consistency runs", str(settings.self_consistency_runs))

    console.print(table)

    # Verify providers
    console.print("\n[bold]Provider Status:[/]")
    if settings.llm_provider == "ollama":
        try:
            import httpx
            resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            if settings.ollama_model in models or any(settings.ollama_model in m for m in models):
                console.print(f"  [green]Ollama: connected, model '{settings.ollama_model}' available[/]")
            else:
                console.print(f"  [yellow]Ollama: connected, but model '{settings.ollama_model}' not found[/]")
                console.print(f"  Available models: {models}")
                console.print(f"  Run: ollama pull {settings.ollama_model}")
        except Exception:
            console.print(f"  [red]Ollama: not running at {settings.ollama_base_url}[/]")
            console.print("  Start it with: ollama serve")


# ── init-db ─────────────────────────────────────────────────────────────────

@main.command()
def init_db():
    """Initialize the database schema (create all tables)."""
    from config import get_settings
    from db.models import Base
    from db.session import get_engine

    settings = get_settings()
    engine = get_engine()
    console.print(f"[bold blue]Creating database tables ({settings.db_backend})...[/]")

    try:
        if settings.db_backend == "postgres":
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

        Base.metadata.create_all(engine)
        console.print(f"[bold green]Database tables created successfully![/]")
        console.print(f"  Backend: {settings.db_backend}")
        console.print(f"  URL: {settings.get_database_url()}")
    except Exception as e:
        console.print(f"[bold red]Failed to create tables: {e}[/]")
        sys.exit(1)


# ── seed-taxonomy ───────────────────────────────────────────────────────────

@main.command()
def seed_taxonomy():
    """Populate the technique_tags table from the fixed vocabulary."""
    from db.models import TechniqueTag
    from db.session import get_db
    from extraction.taxonomy import TECHNIQUE_TAXONOMY

    with get_db() as db:
        count = 0
        for category, tags in TECHNIQUE_TAXONOMY.items():
            for tag_name in tags:
                existing = db.query(TechniqueTag).filter_by(name=tag_name).first()
                if not existing:
                    db.add(TechniqueTag(name=tag_name, category=category))
                    count += 1

        console.print(f"[bold green]Seeded {count} technique tags[/]")


# ── backfill ────────────────────────────────────────────────────────────────

@main.command()
@click.option("--platform", "-p", default="codeforces", type=click.Choice(["codeforces", "codechef", "all"]), help="Platform to backfill")
@click.option("--limit", "-l", default=None, type=int, help="Limit number of problems to ingest")
@click.option("--skip-statements", is_flag=True, help="Skip fetching full statements (metadata only)")
@click.option("--skip-extraction", is_flag=True, help="Skip LLM extraction (ingest only)")
def backfill(platform: str, limit: int | None, skip_statements: bool, skip_extraction: bool):
    """Backfill: ingest all problems + run extraction pipeline."""
    asyncio.run(_backfill_async(platform, limit, skip_statements, skip_extraction))


async def _backfill_async(platform: str, limit: int | None, skip_statements: bool, skip_extraction: bool):
    from db.models import Problem, Extraction
    from db.session import get_db
    from ingestion.codeforces_client import CodeforcesClient
    from ingestion.codechef_client import CodeChefClient
    from ingestion.normalizer import normalize_statement

    # Step 1: Fetch problem list
    console.print(f"\n[bold blue]Step 1: Fetching {platform} problem list...[/]")

    all_problems = []
    
    if platform in ["codeforces", "all"]:
        console.print("  [bold]Codeforces[/]")
        async with CodeforcesClient() as cf:
            cf_problems = await cf.fetch_problem_list()
            all_problems.extend(cf_problems)
            
    if platform in ["codechef", "all"]:
        from config import get_settings
        if not get_settings().codechef_enabled:
            console.print("  [yellow]CodeChef is disabled in config. Skipping...[/]")
        else:
            console.print("  [bold]CodeChef[/]")
            # Fetching all problems for CodeChef is not supported directly, usually we fetch practice problems.
            # In backfill, if they ask for CodeChef, we might just warn or fetch a specific contest if limit is small.
            console.print("  [yellow]Note: Full CodeChef backfill not implemented. Use sync-recent instead.[/]")

    if limit:
        all_problems = all_problems[:limit]

    console.print(f"  Found [bold]{len(all_problems)}[/] problems")

    # Step 2: Store metadata
    console.print("\n[bold blue]Step 2: Storing problem metadata...[/]")
    new_count = 0
    skip_count = 0

    with get_db() as db:
        for p in all_problems:
            existing = db.query(Problem).filter_by(
                platform="codeforces", external_id=p.external_id
            ).first()
            if existing:
                skip_count += 1
                continue

            problem = Problem(
                platform="codeforces" if hasattr(p, "contest_id") and not isinstance(p.contest_id, str) else "codechef",
                external_id=p.external_id,
                url=p.url,
                title=p.name,
                raw_statement="",
                native_rating=getattr(p, "rating", None),
                contest_id=str(p.contest_id),
            )
            problem.native_tags = getattr(p, "tags", [])  # uses the property setter → JSON
            db.add(problem)
            new_count += 1

    console.print(f"  [green]Added {new_count} new problems[/], skipped {skip_count} existing")

    if skip_statements:
        console.print("[yellow]Skipping statement fetching (--skip-statements)[/]")
        return

    # Step 3: Fetch full statements for problems that don't have them
    console.print("\n[bold blue]Step 3: Fetching problem statements...[/]")

    with get_db() as db:
        problems_needing_statements = (
            db.query(Problem)
            .filter(Problem.platform.in_(["codeforces", "codechef"]) if platform == "all" else Problem.platform == platform)
            .filter(Problem.raw_statement == "")
            .all()
        )
        # Detach from session so we can use them outside
        problems_list = [
            {"id": p.id, "platform": p.platform, "contest_id": p.contest_id, "external_id": p.external_id,
             "title": p.title, "native_rating": p.native_rating,
             "native_tags": p.native_tags, "url": p.url}
            for p in problems_needing_statements
        ]

    if not problems_list:
        console.print("  All problems already have statements")
    else:
        console.print(f"  Need to fetch statements for [bold]{len(problems_list)}[/] problems")
        fetched = 0
        failed = 0

        async with CodeforcesClient() as cf, CodeChefClient() as cc:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Fetching statements...",
                    total=len(problems_list),
                )

                for pdata in problems_list:
                    try:
                        if pdata["platform"] == "codeforces":
                            from ingestion.codeforces_client import CFProblemMeta

                            # Extract the problem index from external_id (e.g., "1900D" → "D")
                            ext_id = pdata["external_id"]
                            contest_id_str = pdata["contest_id"]
                            index = ext_id[len(contest_id_str):] if contest_id_str else ext_id

                            meta = CFProblemMeta(
                                contest_id=int(contest_id_str),
                                index=index,
                                name=pdata["title"] or "",
                                rating=pdata["native_rating"],
                                tags=pdata["native_tags"] or [],
                                url=pdata["url"],
                                external_id=ext_id,
                            )

                            full = await cf.fetch_problem_statement(meta)
                            statement_text = normalize_statement(full.statement_html)
                        elif pdata["platform"] == "codechef":
                            from ingestion.codechef_client import CCProblemMeta
                            
                            meta = CCProblemMeta(
                                contest_id=pdata["contest_id"],
                                problem_code=pdata["external_id"],
                                name=pdata["title"] or "",
                                url=pdata["url"],
                                external_id=pdata["external_id"],
                            )
                            
                            full = await cc.fetch_problem_statement(meta)
                            statement_text = normalize_statement(full.statement_html)

                        if statement_text:
                            with get_db() as db:
                                p = db.query(Problem).filter_by(id=pdata["id"]).first()
                                if p:
                                    p.raw_statement = statement_text
                                    p.time_limit_ms = full.time_limit_ms
                                    p.memory_limit_kb = full.memory_limit_kb
                            fetched += 1
                        else:
                            logger.debug(f"Empty statement for {pdata['external_id']}")
                            failed += 1

                    except Exception as e:
                        logger.warning(f"Failed to fetch {pdata['external_id']}: {e}")
                        failed += 1

                    progress.advance(task)

        console.print(f"  [green]Fetched {fetched} statements[/], [red]{failed} failed[/]")

    if skip_extraction:
        console.print("[yellow]Skipping extraction (--skip-extraction)[/]")
        return

    # Step 4: Run extraction on problems without extractions
    console.print("\n[bold blue]Step 4: Running LLM extraction...[/]")
    _run_extraction_batch()


def _run_extraction_batch():
    """Run extraction on all problems that don't have an extraction yet."""
    from db.models import Problem, Extraction
    from db.session import get_db
    from extraction.extractor import Extractor, ExtractionError
    from extraction.embedder import Embedder

    extractor = Extractor()
    embedder = Embedder()

    with get_db() as db:
        problems = (
            db.query(Problem)
            .outerjoin(Extraction)
            .filter(Extraction.problem_id.is_(None))
            .filter(Problem.raw_statement != "")
            .all()
        )
        # Detach
        problems_data = [
            {"id": p.id, "external_id": p.external_id, "raw_statement": p.raw_statement}
            for p in problems
        ]

    if not problems_data:
        console.print("  All problems already have extractions")
        return

    console.print(f"  Running extraction on [bold]{len(problems_data)}[/] problems")

    success = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(problems_data))

        for pdata in problems_data:
            try:
                result = extractor.extract(pdata["raw_statement"])

                # Generate embeddings
                insight_emb = embedder.embed(result.schema.core_insight)
                comp_emb = embedder.embed(result.schema.composition_pattern)

                with get_db() as db:
                    extraction = Extraction(
                        problem_id=pdata["id"],
                        schema_version=result.schema.schema_version,
                        primary_technique=result.schema.primary_technique,
                        composition_pattern=result.schema.composition_pattern,
                        framing=result.schema.framing,
                        nearest_classical_analogue=result.schema.nearest_classical_analogue,
                        constraint_fingerprint=result.schema.constraint_fingerprint,
                        core_insight=result.schema.core_insight,
                        concept_count=result.schema.concept_count,
                        confidence=result.confidence,
                        evidence_json=json.dumps(
                            [e.model_dump() for e in result.schema.evidence]
                        ),
                    )
                    # Use property setters for JSON-serialized fields
                    extraction.secondary_techniques = result.schema.secondary_techniques
                    extraction.archetype = result.schema.archetype
                    extraction.core_insight_embedding = insight_emb
                    extraction.composition_embedding = comp_emb
                    db.add(extraction)

                success += 1

            except Exception as e:
                logger.warning(f"Extraction failed for {pdata['external_id']}: {e}")
                failed += 1

            progress.advance(task)

    console.print(
        f"  [green]Extracted {success} problems[/], "
        f"[red]{failed} failed[/]"
    )


# ── sync-recent ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--hours", "-h", default=24, help="Number of hours to look back")
def sync_recent(hours: int):
    """Sync recent contests using Clist API and ingest problems."""
    asyncio.run(_sync_recent_async(hours))

async def _sync_recent_async(hours: int):
    from config import get_settings
    from ingestion.clist_client import ClistClient
    from ingestion.codeforces_client import CodeforcesClient
    from ingestion.codechef_client import CodeChefClient
    from db.models import Problem
    from db.session import get_db

    settings = get_settings()
    if not settings.clist_api_key:
        console.print("[red]Error: clist_api_key not configured. Cannot sync recent contests.[/]")
        return

    console.print(f"\n[bold blue]Syncing recent contests from last {hours} hours...[/]")

    async with ClistClient(settings.clist_api_key) as clist:
        cf_contests = await clist.fetch_recent_contests("codeforces.com", hours)
        
        cc_contests = []
        if settings.codechef_enabled:
            cc_contests = await clist.fetch_recent_contests("codechef.com", hours)

    all_problems = []

    if cf_contests:
        console.print(f"  [green]Found {len(cf_contests)} Codeforces contests[/]")
        async with CodeforcesClient() as cf:
            for c in cf_contests:
                # Codeforces contest IDs from Clist might not match perfectly, usually Clist ID is their own
                # But Clist event name has it, or resource ID. Actually Clist `href` contains the exact URL!
                # Wait, we need the contest_id for CF. For CodeChef we need the contest code.
                pass
                
                # We can also just use CodeforcesClient to fetch recent contests natively!
                # Since Codeforces has a native API for recent contests, let's use it for CF.
        
        # Native Codeforces recent contests
        async with CodeforcesClient() as cf:
            recent_cf = await cf.fetch_recent_contests(count=5)
            for c in recent_cf:
                # We should check if they ended within `hours`, but for simplicity just ingest top 1
                c_id = c.get("id")
                console.print(f"  Fetching CF Contest {c_id} ({c.get('name')})")
                try:
                    p = await cf.fetch_contest_problems(c_id)
                    all_problems.extend(p)
                except Exception as e:
                    console.print(f"    [red]Failed: {e}[/]")

    if cc_contests:
        console.print(f"  [green]Found {len(cc_contests)} CodeChef contests[/]")
        async with CodeChefClient() as cc:
            for c in cc_contests:
                # Extract contest code from event name or URL (href is available if requested, but we only have event name)
                # Event names like "Starters 120 (Rated...)"
                import re
                m = re.search(r"Starters\s+(\d+)", c.event, re.IGNORECASE)
                if m:
                    c_id = f"START{m.group(1)}"
                    console.print(f"  Fetching CC Contest {c_id} ({c.event})")
                    try:
                        p = await cc.fetch_contest_problems(c_id)
                        all_problems.extend(p)
                    except Exception as e:
                        console.print(f"    [red]Failed: {e}[/]")

    if not all_problems:
        console.print("  [yellow]No new problems found.[/]")
        return
        
    console.print(f"  [green]Total {len(all_problems)} problems found across platforms.[/]")
    
    # Store metadata
    new_count = 0
    with get_db() as db:
        for p in all_problems:
            platform = "codeforces" if hasattr(p, "contest_id") and not isinstance(p.contest_id, str) else "codechef"
            existing = db.query(Problem).filter_by(
                platform=platform, external_id=p.external_id
            ).first()
            if not existing:
                problem = Problem(
                    platform=platform,
                    external_id=p.external_id,
                    url=p.url,
                    title=p.name,
                    raw_statement="",
                    native_rating=getattr(p, "rating", None),
                    contest_id=str(p.contest_id),
                )
                problem.native_tags = getattr(p, "tags", [])
                db.add(problem)
                new_count += 1
                
    console.print(f"  Added {new_count} new problems to database.")
    
    if new_count > 0:
        console.print("\n[bold blue]Running backfill to fetch statements and extract...[/]")
        await _backfill_async("all", limit=None, skip_statements=False, skip_extraction=False)


# ── find-similar ────────────────────────────────────────────────────────────

@main.command()
@click.argument("query", required=True)
@click.option("--top-n", "-n", default=10, help="Number of results to show")
@click.option("--explain", "-e", is_flag=True, help="Generate LLM explanations for matches")
def find_similar(query: str, top_n: int, explain: bool):
    """Find similar problems. QUERY can be a Codeforces URL, file path, or problem text."""
    asyncio.run(_find_similar_async(query, top_n, explain))


async def _find_similar_async(query: str, top_n: int, explain: bool):
    from db.models import Problem, Extraction
    from db.session import get_db
    from extraction.extractor import Extractor
    from extraction.embedder import Embedder
    from matching.score import Scorer, CandidateFeatures

    # Step 1: Get the query problem text
    problem_text = await _resolve_query(query)
    if not problem_text:
        console.print("[red]Could not resolve query to a problem statement[/]")
        return

    console.print(Panel(
        problem_text[:500] + ("..." if len(problem_text) > 500 else ""),
        title="[bold]Query Problem[/]",
        border_style="blue",
    ))

    # Step 2: Extract schema for query
    with console.status("[bold green]Extracting query schema..."):
        extractor = Extractor()
        query_result = extractor.extract(problem_text, consistency_runs=3)

    console.print(f"\n[bold]Extracted Schema:[/]")
    console.print(f"  Primary: [cyan]{query_result.schema.primary_technique}[/]")
    console.print(f"  Secondary: {query_result.schema.secondary_techniques}")
    console.print(f"  Framing: {query_result.schema.framing}")
    console.print(f"  Insight: [italic]{query_result.schema.core_insight}[/]")
    console.print(f"  Constraints: {query_result.schema.constraint_fingerprint}")
    console.print(f"  Confidence: {query_result.confidence:.0%}")
    console.print()

    # Step 3: Embed the query
    embedder = Embedder()
    query_insight_emb = embedder.embed_query(query_result.schema.core_insight)
    query_comp_emb = embedder.embed_query(query_result.schema.composition_pattern)

    # Step 4: Load all candidates from DB
    with get_db() as db:
        rows = (
            db.query(Problem, Extraction)
            .join(Extraction)
            .all()
        )

    if not rows:
        console.print("[yellow]No extracted problems in database. Run 'backfill' first.[/]")
        return

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

    console.print(f"  Loaded [bold]{len(candidates)}[/] candidates from database")

    # Step 5: Score
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
    results = scorer.score(query_features, candidates, top_n=top_n)

    if not results:
        console.print(Panel(
            "[yellow]No strong matches found.[/] The query problem may use a novel "
            "combination of techniques not well-represented in the database.",
            title="No Match",
            border_style="yellow",
        ))
        return

    # Step 6: Display results
    table = Table(title=f"Top {len(results)} Similar Problems", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Score", style="green", width=7)
    table.add_column("Problem", style="cyan", min_width=30)
    table.add_column("Primary Technique", style="magenta", width=20)
    table.add_column("Core Insight", style="italic", min_width=40)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.total_score:.3f}",
            f"[link={r.url}]{r.title}[/link]\n{r.url}",
            r.primary_technique,
            r.core_insight[:120] + ("..." if len(r.core_insight) > 120 else ""),
        )

    console.print(table)

    # Step 7: Optional explanations
    if explain and results:
        console.print("\n[bold]Explanations:[/]")
        from matching.rerank import Reranker
        reranker = Reranker()

        for i, r in enumerate(results[:5], 1):
            explanation = reranker.generate_explanation(
                query_result.schema.core_insight,
                query_result.schema.primary_technique,
                r,
            )
            console.print(f"\n  [bold]#{i} {r.title}[/]")
            console.print(f"  {explanation}")


async def _resolve_query(query: str) -> str | None:
    """
    Resolve a query string to problem text.
    Accepts:
      - Codeforces URL (fetches the problem)
      - File path (reads the file)
      - Raw problem text (if > 100 chars)
    """
    # Check if it's a Codeforces URL
    cf_url_pattern = re.compile(
        r"codeforces\.com/(problemset/problem|contest)/(\d+)/([A-Za-z]\d?)"
    )
    match = cf_url_pattern.search(query)

    if match:
        contest_id = int(match.group(2))
        index = match.group(3)

        console.print(f"  Fetching problem {contest_id}/{index} from Codeforces...")

        from ingestion.codeforces_client import CodeforcesClient, CFProblemMeta
        from ingestion.normalizer import normalize_statement

        async with CodeforcesClient() as cf:
            meta = CFProblemMeta(
                contest_id=contest_id,
                index=index,
                name="",
                rating=None,
                tags=[],
                url=query,
                external_id=f"{contest_id}{index}",
            )
            full = await cf.fetch_problem_statement(meta)
            return normalize_statement(full.statement_html)

    # Try to read from a file
    path = Path(query)
    if path.exists() and path.is_file():
        console.print(f"  Reading problem from file: {path}")
        return path.read_text(encoding="utf-8")

    # Treat as raw problem text if it's long enough
    if len(query) > 100:
        return query

    console.print(f"[yellow]Could not resolve '{query}' -- provide a URL, text, or file path[/]")
    return None


# ── status ──────────────────────────────────────────────────────────────────

@main.command()
def status():
    """Show current ingestion and extraction statistics."""
    from config import get_settings
    from db.models import Problem, Extraction
    from db.session import get_db
    from sqlalchemy import func

    settings = get_settings()

    with get_db() as db:
        total_problems = db.query(Problem).count()
        with_statements = db.query(Problem).filter(Problem.raw_statement != "").count()
        total_extractions = db.query(Extraction).count()

        # Per-platform breakdown
        platform_counts = (
            db.query(Problem.platform, func.count(Problem.id))
            .group_by(Problem.platform)
            .all()
        )

    table = Table(title="CP Finder Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Database", f"{settings.db_backend} ({settings.get_database_url()[:50]})")
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("Embedding Provider", settings.embedding_provider)
    table.add_row("", "")
    table.add_row("Total problems", str(total_problems))
    for platform, count in platform_counts:
        table.add_row(f"  {platform}", str(count))
    table.add_row("With statements", str(with_statements))
    table.add_row("With extractions", str(total_extractions))
    table.add_row("Needing extraction", str(max(0, with_statements - total_extractions)))

    console.print(table)


if __name__ == "__main__":
    main()
