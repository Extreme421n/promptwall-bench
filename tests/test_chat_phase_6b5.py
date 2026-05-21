"""End-to-end tests for Phase 6B-5: baseline chatbot is aware of the
textual-retrieval tools added in Phase 6B-4.

These tests are deliberately permissive about *which* text-retrieval tool the
baseline picks (e.g. ``search_return_rules`` vs ``get_policy_clause`` is a
judgement call for the LLM/mock heuristic). What they pin down is:

  * the baseline routes return/warranty/cancellation/notes/incident questions
    to one of the new text-retrieval tools (or, when context is missing, asks
    a clarification question instead of guessing);
  * the prompt version is stamped onto every trace's metadata;
  * the v2 prompt actually mentions the new text-retrieval tools and the
    "do not invent" rules.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Customer, Product, Trace
from app.services.chat_service import (
    BASELINE_PROMPT_VERSION,
    BASELINE_PROMPTS,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT_V1,
    BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE,
)


# Tools that ground their answer in seeded textual knowledge. Any of these is
# an acceptable routing for a policy/document question — the baseline mock
# doesn't have to pick one specific one.
TEXT_RETRIEVAL_TOOLS: frozenset[str] = frozenset(
    {
        "search_policy_documents",
        "get_policy_clause",
        "get_active_policy",
        "list_policy_versions",
        "search_return_rules",
        "get_product_warranty_terms",
        "search_internal_agent_notes",
        "search_operational_incidents",
        "get_support_resolution_template",
        # The legacy KB tool is also a valid grounding choice — we don't want
        # to fail when the baseline correctly defers to it.
        "search_kb_articles",
    }
)


def _chat(api_client, **kwargs):
    body = {"mode": "baseline", "model": "mock"}
    body.update(kwargs)
    r = api_client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _called_tool_names(body) -> set[str]:
    return {tc["name"] for tc in body["tools_called"]}


# ---------------------------------------------------------------------------
# Prompt content + version constants
# ---------------------------------------------------------------------------


def test_prompt_constants_present() -> None:
    """Two versions are exposed; v2 is the default."""
    assert BASELINE_PROMPT_VERSION == "baseline_v2_text_knowledge"
    assert BASELINE_SYSTEM_PROMPT == BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE
    assert "baseline_v1" in BASELINE_PROMPTS
    assert "baseline_v2_text_knowledge" in BASELINE_PROMPTS
    assert BASELINE_PROMPTS["baseline_v1"] == BASELINE_SYSTEM_PROMPT_V1
    assert (
        BASELINE_PROMPTS["baseline_v2_text_knowledge"]
        == BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE
    )


def test_v2_prompt_mentions_text_retrieval_tools() -> None:
    """The new prompt names at least the headline text-retrieval tools so the
    LLM knows they exist."""
    p = BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE
    for tool_name in (
        "search_policy_documents",
        "get_policy_clause",
        "search_return_rules",
        "get_product_warranty_terms",
        "search_internal_agent_notes",
        "search_operational_incidents",
    ):
        assert tool_name in p, f"v2 prompt is missing mention of {tool_name!r}"


def test_v2_prompt_lists_do_not_invent_categories() -> None:
    """The new prompt enumerates the specific things the chatbot must not
    fabricate — the spec for Phase 6B-5."""
    p = BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE.lower()
    for needle in (
        "do not invent",
        "policy details",
        "return windows",
        "warranty exclusions",
        "refund eligibility",
        "sla commitments",
        "overage rules",
        "internal notes",
    ):
        assert needle in p, f"v2 prompt missing {needle!r}"


def test_v2_prompt_keeps_clarification_guidance() -> None:
    """Missing-context behaviour must be explicitly preserved."""
    p = BASELINE_SYSTEM_PROMPT_V2_TEXT_KNOWLEDGE.lower()
    assert "clarification" in p
    assert "ask" in p


# ---------------------------------------------------------------------------
# Trace metadata stamping
# ---------------------------------------------------------------------------


def test_chat_records_prompt_version_in_trace_metadata(
    api_client, seeded_session: Session
) -> None:
    body = _chat(api_client, message="Hi, can you help me?")
    trace = seeded_session.execute(
        select(Trace).where(Trace.id == body["trace_id"])
    ).scalar_one()
    assert trace.extra_metadata is not None
    assert trace.extra_metadata.get("prompt_version") == BASELINE_PROMPT_VERSION
    assert trace.extra_metadata.get("prompt_name") == "baseline_system_prompt"


def test_chat_metadata_preserves_caller_keys(
    api_client, seeded_session: Session
) -> None:
    """Caller-supplied metadata is merged with prompt_version, not replaced."""
    body = _chat(
        api_client,
        message="Hi there!",
        metadata={"source": "unit_test", "experiment": "phase_6b5"},
    )
    trace = seeded_session.execute(
        select(Trace).where(Trace.id == body["trace_id"])
    ).scalar_one()
    md = trace.extra_metadata or {}
    assert md.get("source") == "unit_test"
    assert md.get("experiment") == "phase_6b5"
    assert md.get("prompt_version") == BASELINE_PROMPT_VERSION


# ---------------------------------------------------------------------------
# Scenario 1 — Return policy question routes to a text-retrieval tool
# ---------------------------------------------------------------------------


def test_return_policy_question_routes_to_text_tool(api_client) -> None:
    body = _chat(
        api_client,
        message="What's our return policy for opened electronics?",
    )
    names = _called_tool_names(body)
    assert names, "expected the baseline to call a tool, got a clarification"
    assert names & TEXT_RETRIEVAL_TOOLS, (
        f"expected a text-retrieval tool, got {names}"
    )
    assert all(tc["success"] for tc in body["tools_called"])


def test_return_window_question_routes_to_return_rules(api_client) -> None:
    body = _chat(api_client, message="What's the return window for opened items?")
    names = _called_tool_names(body)
    assert names & TEXT_RETRIEVAL_TOOLS


# ---------------------------------------------------------------------------
# Scenario 2 — Warranty question (with SKU) routes to warranty tool
# ---------------------------------------------------------------------------


def test_warranty_question_with_sku_routes_to_warranty_tool(
    api_client, seeded_session: Session
) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    body = _chat(api_client, message=f"What's the warranty on {sku}?")
    names = _called_tool_names(body)
    assert "get_product_warranty_terms" in names, (
        f"expected get_product_warranty_terms in {names}"
    )
    # And the call should have succeeded — the SKU comes straight from the seed.
    matching = [
        tc for tc in body["tools_called"] if tc["name"] == "get_product_warranty_terms"
    ]
    assert matching and matching[0]["success"] is True


# ---------------------------------------------------------------------------
# Scenario 3 — Cancellation policy routes to a policy-clause tool
# ---------------------------------------------------------------------------


def test_cancellation_policy_question_routes_to_text_tool(api_client) -> None:
    body = _chat(api_client, message="What's our cancellation policy?")
    names = _called_tool_names(body)
    assert names, "expected the baseline to call a tool"
    assert names & TEXT_RETRIEVAL_TOOLS, (
        f"expected a text-retrieval tool, got {names}"
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Baggage still routes to the baggage tool (no regression)
# ---------------------------------------------------------------------------


def test_baggage_question_still_routes_to_baggage_tool(api_client) -> None:
    body = _chat(
        api_client,
        message="What's the checked baggage allowance on business class international flights?",
    )
    names = _called_tool_names(body)
    # The v2 prompt mentioned baggage — but get_baggage_policy is a
    # specialized tool that's still the right answer.
    assert "get_baggage_policy" in names


# ---------------------------------------------------------------------------
# Scenario 5 — Internal agent notes for a customer
# ---------------------------------------------------------------------------


def test_internal_notes_with_customer_id_routes_to_notes_tool(
    api_client, seeded_session: Session
) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"Show me the internal notes on customer {cust_id}.",
    )
    names = _called_tool_names(body)
    assert "search_internal_agent_notes" in names, (
        f"expected search_internal_agent_notes in {names}"
    )


def test_internal_notes_without_customer_id_asks_for_clarification(
    api_client,
) -> None:
    body = _chat(api_client, message="Pull up the internal notes please.")
    # Either the baseline asks for a customer id (preferred) OR it routes to
    # a text tool with a fallback query. Both are acceptable.
    if not body["tools_called"]:
        ans = body["answer"].lower()
        assert "customer" in ans, f"expected clarification, got {body['answer']!r}"


# ---------------------------------------------------------------------------
# Scenario 6 — General "what can you help with" returns a friendly answer
# ---------------------------------------------------------------------------


def test_general_help_question_does_not_force_tool_call(api_client) -> None:
    body = _chat(api_client, message="Hi, can you help me?")
    # Baseline is NOT forced to use a tool. A clarification answer is fine.
    assert body["tools_called"] == []
    assert body["answer"]


# ---------------------------------------------------------------------------
# Scenario 7 — Missing-context warranty question asks rather than invents
# ---------------------------------------------------------------------------


def test_warranty_question_without_sku_does_not_invent(api_client) -> None:
    body = _chat(api_client, message="What's the warranty period?")
    # The baseline must NOT call get_product_warranty_terms without an
    # identifier — that's the whole point of the "do not invent" rule.
    names = _called_tool_names(body)
    assert "get_product_warranty_terms" not in names, (
        "baseline should not call warranty tool without a SKU"
    )
    # Either a clarification answer or routing to a policy/KB tool is fine.
    if names:
        assert names & TEXT_RETRIEVAL_TOOLS, (
            f"expected a text-retrieval tool, got {names}"
        )
    else:
        assert body["answer"], "expected a non-empty clarification"
