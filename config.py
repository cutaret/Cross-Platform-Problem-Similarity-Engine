"""
Centralized configuration loaded from environment variables / .env file.

Uses pydantic-settings so every value is validated at startup — a typo in .env
blows up immediately rather than silently producing wrong behaviour at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ─────────────────────────────────────────
    # "sqlite" for local testing (no Docker needed), "postgres" for production
    db_backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = ""  # auto-set based on db_backend if empty
    sqlite_path: str = "cp_finder.db"

    # ── LLM Provider ────────────────────────────────────
    # "ollama", "gemini", "anthropic", "openai_compatible"
    llm_provider: Literal["ollama", "gemini", "anthropic", "openai_compatible"] = "ollama"

    # OpenAI / Groq / DeepSeek / OpenRouter etc.
    openai_api_key: str = ""
    openai_base_url: str = ""  # leave empty for default OpenAI
    openai_fast_model: str = "gpt-4o-mini"
    openai_strong_model: str = "gpt-4o"

    # Anthropic (paid)
    anthropic_api_key: str = ""

    # Google Gemini (free tier: 15 RPM, 1M tokens/day)
    gemini_api_key: str = ""

    # Ollama (free, local — requires `ollama` installed)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"  # good at structured output

    # ── Extraction ──────────────────────────────────────
    extraction_fast_model: str = ""   # auto-set based on provider
    extraction_strong_model: str = "" # auto-set based on provider
    self_consistency_runs: int = 2

    # ── Embedding Provider ──────────────────────────────
    # "local" (sentence-transformers, free), "voyage" (paid)
    embedding_provider: Literal["local", "voyage"] = "local"

    # Local embedding model (sentence-transformers)
    local_embed_model: str = "BAAI/bge-small-en-v1.5"  # 384-dim, fast, good quality
    embedding_dimension: int = 384  # matches bge-small; set to 1024 for voyage

    # Voyage AI (paid)
    voyage_api_key: str = ""
    voyage_embed_model: str = "voyage-3.5"

    # ── Ingestion ───────────────────────────────────────
    clist_api_key: str = ""
    codechef_enabled: bool = False
    leetcode_enabled: bool = False

    def get_database_url(self) -> str:
        """Return the database URL, auto-setting from backend if not explicit."""
        if self.database_url:
            return self.database_url
        if self.db_backend == "sqlite":
            db_path = Path(self.sqlite_path).resolve()
            return f"sqlite:///{db_path}"
        return "postgresql+psycopg://cpfinder:cpfinder_dev@localhost:5432/cpfinder"

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension based on the provider and model."""
        if self.embedding_provider == "voyage":
            return 1024
        # Common local model dimensions
        model_dims = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        return model_dims.get(self.local_embed_model, self.embedding_dimension)

    def get_extraction_models(self) -> tuple[str, str]:
        """Return (fast_model, strong_model) based on provider."""
        if self.extraction_fast_model and self.extraction_strong_model:
            return self.extraction_fast_model, self.extraction_strong_model

        if self.llm_provider == "ollama":
            return self.ollama_model, self.ollama_model
        elif self.llm_provider == "gemini":
            return "gemini-2.0-flash", "gemini-2.0-flash"
        elif self.llm_provider == "openai_compatible":
            return self.openai_fast_model, self.openai_strong_model
        else:  # anthropic
            return "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250514"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
