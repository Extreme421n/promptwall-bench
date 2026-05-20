"""Test fixtures.

Uses a fresh SQLite database per test session so tests run without any Docker
or external service. Production uses Postgres via docker-compose.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import make_engine
from app.models import Base
from app.seed import reset_schema, seed


@pytest.fixture(autouse=True)
def _isolate_llm_settings() -> Iterator[None]:
    """Force a clean LLM config for every test.

    Without this, a developer-local ``OPENAI_API_KEY`` in the environment
    would change the behaviour of factory tests and any /chat call routed
    through a non-mock model.
    """
    original_api_key = settings.openai_api_key
    original_base_url = settings.openai_base_url
    original_default_model = settings.default_model
    settings.openai_api_key = None
    settings.openai_base_url = None
    settings.default_model = "mock-1"
    try:
        yield
    finally:
        settings.openai_api_key = original_api_key
        settings.openai_base_url = original_base_url
        settings.default_model = original_default_model


@pytest.fixture(scope="session")
def test_engine() -> Iterator[Engine]:
    """A single SQLite engine for the whole test session."""
    url = os.environ.get("TEST_DATABASE_URL", "sqlite:///./test.db")
    # Clean slate every run
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        path = url.replace("sqlite:///", "", 1)
        if os.path.exists(path):
            os.remove(path)

    engine = make_engine(url)
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def db(test_engine: Engine) -> Iterator[Session]:
    """Per-test session wrapped in a transaction that is rolled back.

    Tests that trigger an ``IntegrityError`` cause the inner transaction to be
    auto-rolled back; guard the outer rollback so we don't emit the
    "transaction already deassociated" SAWarning.
    """
    Session_ = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False, future=True)
    connection = test_engine.connect()
    trans = connection.begin()
    session = Session_(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if trans.is_active:
            trans.rollback()
        connection.close()


@pytest.fixture(scope="session")
def seeded_engine(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Engine, dict[str, int]]]:
    """A fully seeded SQLite DB shared across the session.

    Used by both ``tests/test_seed.py`` and the per-tool tests. The seed is
    deterministic, so any test can rely on stable counts and ids.
    """
    db_path = tmp_path_factory.mktemp("seeded") / "seeded.db"
    engine = make_engine(f"sqlite:///{db_path}")
    reset_schema(engine)
    summary = seed(engine, scale="small")
    try:
        yield engine, summary
    finally:
        engine.dispose()


@pytest.fixture()
def seeded_session(seeded_engine: tuple[Engine, dict[str, int]]) -> Iterator[Session]:
    """Per-test session against the seeded DB. Read-only tests should be safe."""
    engine, _ = seeded_engine
    Session_ = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    session = Session_()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def api_client(seeded_engine: tuple[Engine, dict[str, int]]) -> Iterator:
    """A FastAPI TestClient bound to the seeded SQLite engine."""
    from fastapi.testclient import TestClient

    from app.db import get_session
    from app.main import app

    engine, _ = seeded_engine
    Session_ = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

    def _override_get_session() -> Iterator[Session]:
        s = Session_()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_session] = _override_get_session
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
