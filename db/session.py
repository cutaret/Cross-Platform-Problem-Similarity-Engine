"""
Database session management and engine creation.

Supports both SQLite (for local testing, no Docker needed)
and PostgreSQL (for production with pgvector).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings

_engine = None
_session_factory = None


def get_engine():
    """Create and return a SQLAlchemy engine from settings."""
    global _engine
    if _engine is not None:
        return _engine

    settings = get_settings()
    db_url = settings.get_database_url()

    if settings.db_backend == "sqlite":
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for better concurrent read performance
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        _engine = create_engine(
            db_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a session factory bound to the default engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session and handles
    commit/rollback automatically.

    Usage:
        with get_db() as db:
            db.add(problem)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
