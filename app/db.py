"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def make_engine(url: str | None = None) -> Engine:
    """Create an engine for the given URL (or the configured default)."""
    db_url = url or settings.database_url
    connect_args: dict = {}
    if db_url.startswith("sqlite"):
        # SQLite needs check_same_thread=False for multi-threaded use
        # (benchmark runner + FastAPI dependencies). The timeout lets
        # concurrent writers wait on the write lock rather than failing fast.
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = 30
    return create_engine(db_url, future=True, connect_args=connect_args)


engine: Engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
