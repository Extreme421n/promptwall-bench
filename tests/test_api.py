"""HTTP-level tests for the FastAPI app."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Customer, ToolInvocation


def test_health(api_client) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_tools(api_client) -> None:
    r = api_client.get("/tools")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] >= 30
    assert payload["count"] == 42
    names = sorted(t["name"] for t in payload["tools"])
    assert names == sorted(
        [
            # Phase 1D
            "get_customer_profile",
            "get_booking_details",
            "get_flight_status",
            "search_available_flights",
            "get_refund_status",
            "get_baggage_policy",
            "get_support_ticket_status",
            "search_kb_articles",
            # Phase 2E
            "search_available_seats",
            "calculate_change_fee",
            "search_change_options",
            "get_loyalty_balance",
            "get_policy_clause",
            "get_customer_open_issues",
            "search_customer_records",
            # Phase C1 — SaaS / billing
            "get_subscription_status",
            "get_plan_limits",
            "get_invoice_status",
            "calculate_usage_overage",
            "get_api_usage_summary",
            "get_saas_seat_allocation",
            # Phase C1 — Commerce
            "search_products",
            "get_product_details",
            "check_product_inventory",
            "get_commerce_order_status",
            "get_commerce_refund_status",
            "get_shipment_status",
            # Phase C2 — Support extras
            "search_support_tickets",
            "get_escalation_policy",
            "create_support_ticket_draft",
            # Phase C2 — KB / policy extras
            "search_policy_documents",
            "get_latest_policy_version",
            # Phase C2 — Commerce extras
            "calculate_bundle_price",
            "get_commerce_return_status",
            # Phase C2 — CRM extras
            "get_customer_segment",
            # Phase 6B-4 — textual retrieval
            "search_return_rules",
            "get_product_warranty_terms",
            "search_internal_agent_notes",
            "search_operational_incidents",
            "get_support_resolution_template",
            "list_policy_versions",
            "get_active_policy",
        ]
    )
    # Each summary has the metadata the chatbot/PromptWall layer will need.
    for t in payload["tools"]:
        assert {"name", "description", "domain", "risk_level", "read_only"} <= set(t)


def test_describe_tool_includes_schemas(api_client) -> None:
    r = api_client.get("/tools/get_customer_profile")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "get_customer_profile"
    assert body["input_schema"]["type"] == "object"
    assert body["output_schema"]["type"] == "object"


def test_describe_tool_unknown_returns_404(api_client) -> None:
    r = api_client.get("/tools/does_not_exist")
    assert r.status_code == 404


def test_execute_tool_happy_path(api_client, seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    before = seeded_session.execute(select(func.count()).select_from(ToolInvocation)).scalar_one()

    r = api_client.post(
        "/tools/get_customer_profile/execute",
        json={"input": {"customer_id": cust_id}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["evidence_id"] is not None
    assert body["evidence_id"].startswith("ev_")
    assert body["output"]["customer_id"] == cust_id
    assert body["error_type"] is None

    # And the row was persisted.
    seeded_session.expire_all()  # forget cached counts
    after = seeded_session.execute(select(func.count()).select_from(ToolInvocation)).scalar_one()
    assert after == before + 1


def test_execute_tool_validation_error_returns_200_with_failure(api_client) -> None:
    # The endpoint returns 200 with success=False because the executor logs the
    # failure as a tool_invocations row. This is the trace-friendly contract.
    r = api_client.post(
        "/tools/get_customer_profile/execute",
        json={"input": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_type"] == "ToolValidationError"
    assert "exactly one" in body["error_message"]


def test_execute_tool_not_found_returns_200_with_failure(api_client) -> None:
    r = api_client.post(
        "/tools/get_customer_profile/execute",
        json={"input": {"customer_id": 99_999_999}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_type"] == "ResourceNotFoundError"


def test_execute_unknown_tool_returns_404(api_client) -> None:
    r = api_client.post("/tools/no_such_tool/execute", json={"input": {}})
    assert r.status_code == 404


def test_execute_tool_with_explicit_trace_id(
    api_client, seeded_session: Session
) -> None:
    from app.services import TraceService

    svc = TraceService(seeded_session)
    chat = svc.create_chat_session(channel="web")
    trace = svc.create_trace(session_id=chat.id, user_message="manual", mode="baseline")
    seeded_session.commit()
    trace_id = trace.id

    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    r = api_client.post(
        "/tools/get_customer_profile/execute",
        json={"trace_id": trace_id, "input": {"customer_id": cust_id}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True

    # Look up the invocation we just made and confirm it's attached to that trace.
    persisted = seeded_session.execute(
        select(ToolInvocation)
        .where(ToolInvocation.evidence_id == body["evidence_id"])
    ).scalar_one()
    assert persisted.trace_id == trace_id
