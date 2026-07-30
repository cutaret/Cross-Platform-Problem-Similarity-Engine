"""
SQLAlchemy ORM models — mirrors the DDL from the design doc (§5.2)
with the additions from the implementation plan:
  - composition_embedding on extractions
  - normalized_difficulty on problems

Supports both SQLite (testing) and PostgreSQL (production).
When using SQLite, vector columns store JSON arrays and ARRAY columns
store JSON strings — the application layer handles the conversion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import get_settings

# ── Backend-aware column types ──────────────────────────────────────────────
# SQLite doesn't support ARRAY or vector types, so we conditionally import them.

_settings = get_settings()

if _settings.db_backend == "postgres":
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY

    def ArrayColumn():
        return Column(PG_ARRAY(Text), nullable=False, default=list)

    def NullableArrayColumn():
        return Column(PG_ARRAY(Text))

    def VectorColumn(dim: int):
        return Column(Vector(dim))
else:
    # SQLite fallback: store arrays as JSON text, vectors as JSON text
    def ArrayColumn():
        return Column(Text, nullable=False, default="[]")

    def NullableArrayColumn():
        return Column(Text)

    def VectorColumn(dim: int):
        return Column(Text)  # store as JSON array string


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Problem(Base):
    """
    A single competitive-programming problem from any supported platform.
    Canonical key: (platform, external_id).
    """
    __tablename__ = "problems"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    raw_statement: Mapped[str] = mapped_column(Text, nullable=False)
    time_limit_ms: Mapped[int | None] = mapped_column(Integer)
    memory_limit_kb: Mapped[int | None] = mapped_column(Integer)
    native_rating: Mapped[int | None] = mapped_column(Integer)
    normalized_difficulty: Mapped[float | None] = mapped_column(Float)
    contest_id: Mapped[str | None] = mapped_column(Text)
    contest_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    native_tags_json: Mapped[str | None] = mapped_column(Text)  # JSON array for cross-DB compat
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    extraction: Mapped[Extraction | None] = relationship(back_populates="problem", uselist=False)
    technique_links: Mapped[list[ProblemTechniqueTag]] = relationship(back_populates="problem")

    __table_args__ = (
        UniqueConstraint("platform", "external_id", name="uq_platform_external_id"),
    )

    @property
    def native_tags(self) -> list[str]:
        if not self.native_tags_json:
            return []
        return json.loads(self.native_tags_json)

    @native_tags.setter
    def native_tags(self, value: list[str]):
        self.native_tags_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<Problem {self.platform}:{self.external_id} '{self.title}'>"


class Extraction(Base):
    """
    Structured algorithmic extraction for a problem — produced by the
    LLM extraction pipeline (§4) and cross-checked against deterministic parsing.
    """
    __tablename__ = "extractions"

    problem_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("problems.id"), primary_key=True
    )
    schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)

    # Core fields from ProblemSchema
    primary_technique: Mapped[str] = mapped_column(Text, nullable=False)
    secondary_techniques_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    composition_pattern: Mapped[str | None] = mapped_column(Text)
    archetype_json: Mapped[str | None] = mapped_column(Text)
    framing: Mapped[str | None] = mapped_column(Text)
    nearest_classical_analogue: Mapped[str | None] = mapped_column(Text)
    constraint_fingerprint: Mapped[str | None] = mapped_column(Text)
    core_insight: Mapped[str | None] = mapped_column(Text)
    concept_count: Mapped[int | None] = mapped_column(SmallInteger)

    # Embeddings — stored as JSON text for SQLite compat, native vector for Postgres
    core_insight_embedding_json: Mapped[str | None] = mapped_column(Text)
    composition_embedding_json: Mapped[str | None] = mapped_column(Text)

    # Quality metadata
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_json: Mapped[str | None] = mapped_column(Text)  # JSON list[TagEvidence]

    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    problem: Mapped[Problem] = relationship(back_populates="extraction")

    # ── Convenience properties for list/array fields ────────────────────

    @property
    def secondary_techniques(self) -> list[str]:
        return json.loads(self.secondary_techniques_json) if self.secondary_techniques_json else []

    @secondary_techniques.setter
    def secondary_techniques(self, value: list[str]):
        self.secondary_techniques_json = json.dumps(value)

    @property
    def archetype(self) -> list[str]:
        return json.loads(self.archetype_json) if self.archetype_json else []

    @archetype.setter
    def archetype(self, value: list[str]):
        self.archetype_json = json.dumps(value)

    @property
    def core_insight_embedding(self) -> list[float] | None:
        if not self.core_insight_embedding_json:
            return None
        return json.loads(self.core_insight_embedding_json)

    @core_insight_embedding.setter
    def core_insight_embedding(self, value):
        if value is None:
            self.core_insight_embedding_json = None
        elif hasattr(value, 'tolist'):
            self.core_insight_embedding_json = json.dumps(value.tolist())
        else:
            self.core_insight_embedding_json = json.dumps(list(value))

    @property
    def composition_embedding(self) -> list[float] | None:
        if not self.composition_embedding_json:
            return None
        return json.loads(self.composition_embedding_json)

    @composition_embedding.setter
    def composition_embedding(self, value):
        if value is None:
            self.composition_embedding_json = None
        elif hasattr(value, 'tolist'):
            self.composition_embedding_json = json.dumps(value.tolist())
        else:
            self.composition_embedding_json = json.dumps(list(value))

    def __repr__(self) -> str:
        return f"<Extraction problem_id={self.problem_id} primary='{self.primary_technique}'>"


class TechniqueTag(Base):
    """
    A technique in the fixed vocabulary — organized as a two-level tree
    (category → tag).  E.g. category='dp', tag='dp-bitmask'.
    """
    __tablename__ = "technique_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<TechniqueTag '{self.name}' ({self.category})>"


class ProblemTechniqueTag(Base):
    """
    Many-to-many link between a problem and its assigned technique tags,
    with a role indicating whether this is a primary or secondary technique.
    """
    __tablename__ = "problem_technique_tags"

    problem_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("problems.id"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("technique_tags.id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    problem: Mapped[Problem] = relationship(back_populates="technique_links")
    tag: Mapped[TechniqueTag] = relationship()

    __table_args__ = (
        CheckConstraint("role IN ('primary', 'secondary')", name="ck_tag_role"),
    )


class Feedback(Base):
    """
    User-submitted thumbs-up/down on a suggested match.
    Used to train the learned ranking weights (Phase 3).
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_problem_id: Mapped[int] = mapped_column(Integer, ForeignKey("problems.id"))
    candidate_problem_id: Mapped[int] = mapped_column(Integer, ForeignKey("problems.id"))
    label: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
