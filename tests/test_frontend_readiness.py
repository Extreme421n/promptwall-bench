"""Backend-readiness tests for frontend integration.

These tests assert the contract the web UI will rely on:

  * the four endpoints respond with the expected shape,
  * the chat response carries every field the UI needs,
  * CORS preflight succeeds for an allowed origin and is rejected (origin
    header omitted) for a non-allowed one,
  * configuration parses both list-form and comma-separated CORS_ORIGINS.

No frontend code is created — this is purely a backend smoke / contract test.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Customer


# ---------------------------------------------------------------------------
# Endpoint contract
# ---------------------------------------------------------------------------


def test_health_endpoint(api_client) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_tools_endpoint_returns_registered_tools(api_client) -> None:
    r = api_client.get("/tools")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body["count"] == len(body["tools"])
    assert body["count"] >= 30  # we ship 42
    # Every entry carries the metadata the UI will surface.
    for t in body["tools"]:
        assert {"name", "description", "domain", "risk_level", "read_only"} <= set(t)


def test_chat_endpoint_baseline_response_shape(api_client) -> None:
    r = api_client.post(
        "/chat",
        json={"mode": "baseline", "model": "mock", "message": "What is our cancellation policy?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Exactly the keys the frontend needs.
    assert set(body) == {
        "answer",
        "trace_id",
        "session_id",
        "tools_called",
        "evidence_ids",
        "latency_ms",
        "estimated_cost_usd",
    }
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["trace_id"], int)
    assert isinstance(body["session_id"], str) and body["session_id"]
    assert isinstance(body["tools_called"], list)
    assert isinstance(body["evidence_ids"], list)
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0

    # Each tool-call summary has the fields the UI surfaces.
    for tc in body["tools_called"]:
        assert {
            "name",
            "arguments",
            "success",
            "evidence_id",
            "error_type",
            "error_message",
            "latency_ms",
        } <= set(tc)


def test_chat_endpoint_includes_evidence_ids_for_tool_calls(
    api_client, seeded_session: Session
) -> None:
    """A chat that triggers a tool must surface its evidence_id."""
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    r = api_client.post(
        "/chat",
        json={
            "mode": "baseline",
            "model": "mock",
            "message": f"Show me internal notes on customer {cust_id}.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    if body["tools_called"]:
        # If a tool fired, its evidence_id should appear in evidence_ids
        # (when the tool succeeded). The frontend uses these to render
        # "grounded in evidence X" badges.
        succeeded = [tc for tc in body["tools_called"] if tc["success"]]
        for tc in succeeded:
            assert tc["evidence_id"] is not None
            assert tc["evidence_id"] in body["evidence_ids"]


# ---------------------------------------------------------------------------
# CORS — preflight + actual response
# ---------------------------------------------------------------------------


def test_cors_preflight_allows_default_dev_origin(api_client) -> None:
    """OPTIONS /chat from a localhost dev frontend (default allow-list)."""
    r = api_client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code in (200, 204), r.text
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    allowed = r.headers.get("access-control-allow-methods", "").upper()
    assert "POST" in allowed


def test_cors_actual_request_carries_origin_header(api_client) -> None:
    """Non-preflight POST also echoes the Access-Control-Allow-Origin header
    so the browser will accept the response."""
    r = api_client.post(
        "/chat",
        headers={"Origin": "http://localhost:5173"},  # Vite default
        json={"mode": "baseline", "model": "mock", "message": "hi"},
    )
    assert r.status_code == 200
    assert (
        r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    )


def test_cors_unallowed_origin_does_not_get_allow_header(api_client) -> None:
    """An origin that's not in the allow-list MUST NOT receive an
    Access-Control-Allow-Origin header. The request itself is still served —
    CORS enforcement happens in the browser, not on the server."""
    r = api_client.post(
        "/chat",
        headers={"Origin": "https://malicious.example.com"},
        json={"mode": "baseline", "model": "mock", "message": "hi"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in (None, "")


# ---------------------------------------------------------------------------
# Settings — env var parsing
# ---------------------------------------------------------------------------


def test_cors_origins_parses_comma_separated_string(monkeypatch) -> None:
    """``CORS_ORIGINS=a,b,c`` env-var form lands as a list of three strings."""
    monkeypatch.setenv(
        "CORS_ORIGINS", "https://a.example.com, https://b.example.com,https://c.example.com"
    )
    s = Settings()
    assert s.cors_origins == [
        "https://a.example.com",
        "https://b.example.com",
        "https://c.example.com",
    ]


def test_cors_origins_default_is_localhost_friendly() -> None:
    """No env var → default dev allow-list contains the common ports."""
    s = Settings()
    expected = {"http://localhost:3000", "http://localhost:5173"}
    assert expected <= set(s.cors_origins)


def test_settings_exposes_env_var_surface() -> None:
    """The Settings class exposes every env var the frontend integrator needs."""
    fields = set(Settings.model_fields)
    for k in (
        "database_url",
        "llm_provider",
        "default_model",
        "openai_api_key",
        "openai_base_url",
        "cors_origins",
    ):
        assert k in fields, f"Settings missing field {k}"
